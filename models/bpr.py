import numpy as np
import pandas as pd

from implicit.cpu.bpr import BayesianPersonalizedRanking


# ==========================================
# implicit BPR 모델 생성
# ==========================================

def create_bpr_model(
    factors=40,
    learning_rate=0.05,
    regularization=0.001,
    iterations=10,
    verify_negative_samples=True,
    num_threads=0,
    random_state=42
):
    """
    implicit CPU BPR 모델을 생성한다.

    실제 학습 데이터(matrix)는 main.py에서 준비하고,
    여기서는 모델 객체만 생성한다.
    """

    return BayesianPersonalizedRanking(
        factors=factors,
        learning_rate=learning_rate,
        regularization=regularization,
        iterations=iterations,
        verify_negative_samples=verify_negative_samples,
        num_threads=num_threads,
        random_state=random_state
    )


# ==========================================
# implicit BPR
# -> 기존 MF 평가 pipeline 연결 Adapter
# ==========================================

class ImplicitBPRAdapter:
    """
    implicit BPR의 recommend() 인터페이스를
    기존 evaluation.py의 MF 평가 방식에 맞춰주는 Adapter.

    evaluation.py에서는:

        recommender.recommend(
            user_id=user_id,
            app_id_list=played_app_ids,
            top_n=10
        )

    형태를 기대하지만,

    implicit에서는:

        model.recommend(
            userid=user_idx,
            user_items=user_items[user_idx],
            N=10
        )

    형태를 사용한다.

    따라서 실제 Steam user_id/app_id와
    implicit 내부 index 사이를 변환한다.
    """

    def __init__(
        self,
        model,
        user_items,
        user_ids,
        item_ids
    ):
        self.model = model
        self.user_items = user_items
        self.user_ids = np.asarray(user_ids)
        self.item_ids = np.asarray(item_ids)


    def recommend(
        self,
        user_id,
        app_id_list=None,
        top_n=10
    ):
        # ======================================
        # 실제 user_id
        # -> implicit 내부 user index
        # ======================================

        user_idx = np.searchsorted(
            self.user_ids,
            user_id
        )

        # 존재하지 않는 사용자
        if (
            user_idx >= len(self.user_ids)
            or self.user_ids[user_idx] != user_id
        ):
            return pd.DataFrame(
                columns=[
                    "app_id",
                    "score"
                ]
            )

        # ======================================
        # Top-N 추천
        # ======================================

        item_indices, scores = self.model.recommend(
            userid=int(user_idx),
            user_items=self.user_items[user_idx],
            N=top_n,

            # Train에서 이미 상호작용한 게임 제외
            filter_already_liked_items=True
        )

        # ======================================
        # implicit 내부 item index
        # -> 실제 Steam app_id
        # ======================================

        app_ids = self.item_ids[item_indices]

        return pd.DataFrame(
            {
                "app_id": app_ids,
                "score": scores
            }
        )


# ==========================================
# Explicit False BPR Fine-Tuning
# ==========================================
#
# implicit BPR은 sparse matrix의 non-zero 값을
# 모두 positive interaction으로 취급한다.
#
# 따라서 False를 -1로 넣는 것만으로는
# explicit negative로 학습되지 않는다.
#
# 현재 프로젝트에서는:
#
# 1) True만 positive로 implicit BPR 학습
#    -> True > Unseen
#
# 2) 같은 유저의 True item(i) / False item(j)
#    pair를 이용해 추가 BPR 학습
#    -> True > False
#
# 두 단계로 학습한다.
# ==========================================

def fine_tune_with_explicit_false(
    model,
    positive_items,
    negative_items,
    epochs=1,
    learning_rate=0.01,
    regularization=0.001,
    batch_size=131072,
    random_state=42
):
    """
    이미 학습된 implicit BPR factor를 그대로 사용하여
    실제 False interaction을 explicit negative로 추가 학습한다.

    Parameters
    ----------
    model
        학습된 implicit BayesianPersonalizedRanking 모델.

    positive_items
        User x Item CSR matrix.
        True interaction만 1로 들어 있음.

    negative_items
        User x Item CSR matrix.
        False interaction만 1로 들어 있음.

    epochs
        Explicit False fine-tuning 반복 횟수.

    learning_rate
        Fine-tuning learning rate.

    regularization
        Fine-tuning L2 regularization.

    batch_size
        한 번에 처리할 False pair 수.

    random_state
        재현성을 위한 seed.
    """

    print(
        "\n===== Explicit False Fine-Tuning 준비 ====="
    )

    positive_items = positive_items.tocsr()
    negative_items = negative_items.tocsr()

    # ------------------------------------------
    # 유저별 True interaction 개수
    # ------------------------------------------

    positive_counts = np.diff(
        positive_items.indptr
    )

    # ------------------------------------------
    # False interaction을
    # (user, false_item) pair로 변환
    # ------------------------------------------

    negative_coo = negative_items.tocoo()

    negative_users = negative_coo.row.astype(
        np.int32,
        copy=False
    )

    negative_item_ids = negative_coo.col.astype(
        np.int32,
        copy=False
    )

    # True item이 하나도 없는 유저는
    # True > False pair를 만들 수 없으므로 제외
    valid_mask = (
        positive_counts[negative_users] > 0
    )

    negative_users = negative_users[
        valid_mask
    ]

    negative_item_ids = negative_item_ids[
        valid_mask
    ]

    n_pairs = len(
        negative_users
    )

    print(
        "Explicit False 학습 pair 수:",
        n_pairs
    )

    if n_pairs == 0:
        print(
            "True와 False를 동시에 가진 유저가 없어 "
            "Explicit False Fine-Tuning을 건너뜁니다."
        )

        return model

    rng = np.random.default_rng(
        random_state
    )

    # implicit BPR에서 설정한 latent factor 수
    latent_dim = model.factors

    user_factors = model.user_factors
    item_factors = model.item_factors

    if (
        user_factors is None
        or item_factors is None
    ):
        raise ValueError(
            "학습된 BPR factor가 없습니다. "
            "model.fit() 이후에 fine-tuning을 실행해야 합니다."
        )

    if (
        user_factors.shape[1] < latent_dim
        or item_factors.shape[1] < latent_dim
    ):
        raise ValueError(
            "BPR factor shape가 예상과 다릅니다."
        )

    # implicit CPU BPR은 item factor 마지막 차원을
    # bias 용도로 추가해서 사용할 수 있음.
    has_item_bias = (
        item_factors.shape[1]
        > latent_dim
    )

    # ==========================================
    # Fine-Tuning
    # ==========================================

    for epoch in range(epochs):

        # False interaction 순서를 랜덤하게 섞음
        order = rng.permutation(
            n_pairs
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for start in range(
            0,
            n_pairs,
            batch_size
        ):
            batch_indices = order[
                start:
                start + batch_size
            ]

            users = negative_users[
                batch_indices
            ]

            false_ids = negative_item_ids[
                batch_indices
            ]

            # ==================================
            # 같은 유저의 True item 하나 샘플링
            # ==================================

            starts = positive_items.indptr[
                users
            ]

            counts = positive_counts[
                users
            ]

            offsets = (
                rng.random(
                    len(users)
                )
                * counts
            ).astype(
                np.int64
            )

            positive_positions = (
                starts + offsets
            )

            true_ids = positive_items.indices[
                positive_positions
            ]

            # ==================================
            # 현재 factor 복사
            # ==================================

            p_u = user_factors[
                users,
                :latent_dim
            ].copy()

            q_i = item_factors[
                true_ids,
                :latent_dim
            ].copy()

            q_j = item_factors[
                false_ids,
                :latent_dim
            ].copy()

            # ==================================
            # x_uij
            #
            # score(True) - score(False)
            # ==================================

            score_diff = np.einsum(
                "ij,ij->i",
                p_u,
                q_i - q_j
            )

            if has_item_bias:
                b_i = item_factors[
                    true_ids,
                    latent_dim
                ].copy()

                b_j = item_factors[
                    false_ids,
                    latent_dim
                ].copy()

                score_diff += (
                    b_i - b_j
                )

            # ==================================
            # sigmoid(-x_uij)
            #
            # BPR gradient weight
            # ==================================

            clipped = np.clip(
                score_diff,
                -35.0,
                35.0
            )

            gradient_weight = (
                1.0
                /
                (
                    1.0
                    + np.exp(clipped)
                )
            ).astype(
                np.float32,
                copy=False
            )

            # ==================================
            # BPR Gradient
            # ==================================

            grad_p = (
                gradient_weight[:, None]
                * (q_i - q_j)
                - regularization * p_u
            )

            grad_q_i = (
                gradient_weight[:, None]
                * p_u
                - regularization * q_i
            )

            grad_q_j = (
                -gradient_weight[:, None]
                * p_u
                - regularization * q_j
            )

            # 같은 user/item이 batch 안에서
            # 여러 번 등장할 수 있으므로
            # np.add.at으로 gradient 누적
            np.add.at(
                user_factors[
                    :, :latent_dim
                ],
                users,
                learning_rate * grad_p
            )

            np.add.at(
                item_factors[
                    :, :latent_dim
                ],
                true_ids,
                learning_rate * grad_q_i
            )

            np.add.at(
                item_factors[
                    :, :latent_dim
                ],
                false_ids,
                learning_rate * grad_q_j
            )

            # ==================================
            # Item Bias Update
            # ==================================

            if has_item_bias:
                grad_b_i = (
                    gradient_weight
                    - regularization * b_i
                )

                grad_b_j = (
                    -gradient_weight
                    - regularization * b_j
                )

                np.add.at(
                    item_factors[
                        :, latent_dim
                    ],
                    true_ids,
                    learning_rate * grad_b_i
                )

                np.add.at(
                    item_factors[
                        :, latent_dim
                    ],
                    false_ids,
                    learning_rate * grad_b_j
                )

            # ==================================
            # 학습 상태 확인
            # ==================================

            batch_loss = np.logaddexp(
                0.0,
                -score_diff
            ).sum()

            total_loss += float(
                batch_loss
            )

            total_correct += int(
                np.sum(
                    score_diff > 0
                )
            )

            total_samples += len(
                users
            )

        # implicit BPR user factor의
        # 마지막 차원이 bias 계산용 상수라면
        # 다시 1.0으로 유지
        if (
            user_factors.shape[1]
            > latent_dim
        ):
            user_factors[
                :, latent_dim
            ] = 1.0

        mean_loss = (
            total_loss
            / max(
                total_samples,
                1
            )
        )

        pair_accuracy = (
            total_correct
            / max(
                total_samples,
                1
            )
        )

        print(
            f"Explicit False Epoch "
            f"{epoch + 1}/{epochs} "
            f"- pairs: {total_samples:,} "
            f"- loss: {mean_loss:.5f} "
            f"- pair_accuracy: {pair_accuracy:.4f}"
        )

    # factor가 변경됐으므로
    # recommend()에서 사용할 norm cache 초기화
    if hasattr(
        model,
        "_item_norms"
    ):
        model._item_norms = None

    if hasattr(
        model,
        "_user_norms"
    ):
        model._user_norms = None

    print(
        "\n===== Explicit False Fine-Tuning 완료 ====="
    )

    return model
