from implicit.cpu.bpr import BayesianPersonalizedRanking

from scipy.sparse import csr_matrix, save_npz, load_npz

from data_split import load_mf_split

from evaluation import (
    # BPR 평가용
    build_mf_user_review_groups,
    stratified_sample_users,
    run_mf_evaluation,
    print_evaluation_report
)

import numpy as np
import pandas as pd
import os
import gc


print(os.getcwd())


# ==========================================
# BPR Experiment Config
# ==========================================
#
# 앞으로 BPR 실험 파라미터는
# 이 구역에서만 수정.
#
# 예) 15 iteration 실험
# -> BPR_ITERATIONS = 15
#
# MODEL_PATH도 iteration 값에 맞춰
# 자동으로 바뀌므로 따로 수정할 필요 없음.
# ==========================================

BPR_ITERATIONS = 10
BPR_FACTORS = 40
BPR_LEARNING_RATE = 0.05
BPR_REGULARIZATION = 0.001

BPR_VERIFY_NEGATIVE_SAMPLES = True
BPR_NUM_THREADS = 0

# False:
# 기존 실험 그대로 사용
# True + False interaction을 모두 positive/seen으로 학습
#
# True:
# True만 implicit BPR의 positive로 학습한 뒤,
# False를 explicit negative로 사용하는 pairwise fine-tuning을 추가.
#
# 즉 True를 False보다 높은 점수로 두도록 실제로 학습함.
BPR_USE_EXPLICIT_FALSE = True

# Explicit False fine-tuning 설정
# 처음에는 1 epoch만 돌려서 효과부터 확인하는 것을 권장.
BPR_EXPLICIT_FALSE_EPOCHS = 1
BPR_EXPLICIT_FALSE_LEARNING_RATE = 0.01
BPR_EXPLICIT_FALSE_REGULARIZATION = 0.001
BPR_EXPLICIT_FALSE_BATCH_SIZE = 131072

RANDOM_STATE = 42


# ==========================================
# 추천 / 평가 설정
# ==========================================

TOP_N = 10

LOWER_BOUND = 10
UPPER_BOUND = 78

SAMPLE_PER_GROUP = 100


# False:
# Test의 True + False interaction 전체 평가
#
# True:
# Test의 True interaction만 평가
#
# 기본 = False
# 이후 True 추가 실험

POSITIVE_ONLY = False


# ==========================================
# 저장 경로
# ==========================================
#
# MODEL_PATH
# -> iteration별 학습 모델
#
# MATRIX_PATH / MAPPING_PATH
# -> Train 데이터가 같으면 계속 재사용
# ==========================================

# 실험 모드에 따라 모델 파일명을 자동으로 분리.
# 기존 baseline 모델과 explicit-negative 모델이 덮어써지지 않음.
if BPR_USE_EXPLICIT_FALSE:
    MODEL_PATH = (
        f"models/implicit_bpr_"
        f"{BPR_ITERATIONS}epoch_explicit_false_"
        f"{BPR_EXPLICIT_FALSE_EPOCHS}ft_model.npz"
    )
else:
    MODEL_PATH = (
        f"models/implicit_bpr_"
        f"{BPR_ITERATIONS}epoch_model.npz"
    )

# 전체 Train interaction(True + False)
# 추천 시 이미 본 게임을 제외하기 위한 seen matrix.
MATRIX_PATH = (
    "models/implicit_bpr_user_items.npz"
)

# True만 들어 있는 positive matrix.
POSITIVE_MATRIX_PATH = (
    "models/implicit_bpr_positive_items.npz"
)

# False만 들어 있는 explicit negative matrix.
NEGATIVE_MATRIX_PATH = (
    "models/implicit_bpr_negative_items.npz"
)

MAPPING_PATH = (
    "models/implicit_bpr_mapping.npz"
)



# ==========================================
# games metadata 로드
# ==========================================
#
# 기존 preprocessing.py의 load_games()를
# import하지 않도록 변경.
#
# 이전 ImportError:
#
# cannot import name 'load_games'
#
# 방지 목적.
# ==========================================

def load_games_meta():

    candidate_paths = [
        "data/cache/games.parquet",
        "data/cache/games_inc.parquet",
        "data/games.parquet",
        "data/raw/games.csv"
    ]


    for path in candidate_paths:

        if os.path.exists(path):

            print(
                "캐시에서 games 데이터 불러오는 중..."
            )


            if path.endswith(".parquet"):

                meta = pd.read_parquet(
                    path
                )

            else:

                meta = pd.read_csv(
                    path
                )


            return meta


    raise FileNotFoundError(
        "games metadata 파일을 찾을 수 없습니다."
    )



def resolve_name_to_appid(meta, name):
    """
    사용자가 입력한 게임 이름을 app_id로 변환.
    동명이인 게임이 있으면 사용자에게 선택하게 함.
    """

    candidates = meta[
        meta["Name"] == name
    ]


    if len(candidates) == 0:

        print(
            f"'{name}' 게임을 찾을 수 없습니다."
        )

        return None


    if len(candidates) == 1:

        return candidates[
            "app_id"
        ].iloc[0]


    print(
        f"\n'{name}'과 일치하는 게임이 여러 개 있습니다. "
        f"선택해주세요:"
    )


    candidates = candidates.reset_index(
        drop=True
    )


    for i, row in candidates.iterrows():

        print(
            f"  [{i}] "
            f"app_id: {row['app_id']} "
            f"- {row['Name']}"
        )


    while True:

        choice = input(
            "번호를 입력하세요: "
        ).strip()


        if (
            choice.isdigit()
            and int(choice) < len(candidates)
        ):

            return candidates.iloc[
                int(choice)
            ]["app_id"]


        print(
            "올바른 번호를 입력해주세요."
        )



# ==========================================
# implicit BPR
# -> 기존 MF 평가 pipeline 연결
# ==========================================
#
# 기존 run_mf_evaluation()에서는:
#
# recommender.recommend(
#     user_id=user_id,
#     app_id_list=played_app_ids,
#     top_n=10
# )
#
# 형태를 사용함.
#
#
# implicit BPR에서는:
#
# model.recommend(
#     userid=user_idx,
#     user_items=user_items[user_idx],
#     N=10
# )
#
# 형태를 사용함.
#
#
# 따라서 Adapter를 이용해
# 기존 evaluation.py를 수정하지 않고
# 동일한 pipeline을 그대로 사용.
# ==========================================

class ImplicitBPRAdapter:

    def __init__(
        self,
        model,
        user_items,
        user_ids,
        item_ids
    ):

        self.model = model

        self.user_items = user_items

        self.user_ids = np.asarray(
            user_ids
        )

        self.item_ids = np.asarray(
            item_ids
        )


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
            user_idx >= len(
                self.user_ids
            )
            or
            self.user_ids[
                user_idx
            ] != user_id
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

        item_indices, scores = (
            self.model.recommend(
                userid=int(
                    user_idx
                ),

                user_items=self.user_items[
                    user_idx
                ],

                N=top_n,

                # 이미 Train에서 본 게임 제외
                filter_already_liked_items=True
            )
        )


        # ======================================
        # implicit 내부 item index
        # -> 실제 Steam app_id
        # ======================================

        app_ids = self.item_ids[
            item_indices
        ]


        result = pd.DataFrame(
            {
                "app_id": app_ids,
                "score": scores
            }
        )


        return result



# ==========================================
# Explicit False BPR Fine-Tuning
# ==========================================
#
# implicit 라이브러리의 BPR은 sparse matrix의
# non-zero 값을 모두 positive로 취급하기 때문에
# -1을 넣어도 False를 negative로 학습하지 않는다.
#
# 따라서:
#
# 1) True만 positive로 implicit BPR을 빠르게 학습
# 2) 같은 유저의 True item(i) / False item(j)을 뽑아
#    score(u, i) > score(u, j)가 되도록 추가 업데이트
#
# 하는 방식으로 False를 실제 explicit negative로 학습한다.
#
# 기존 implicit 모델의 user/item factor를 그대로 사용하므로
# 추천/저장/evaluation pipeline은 그대로 유지된다.
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

    print(
        "\n===== Explicit False Fine-Tuning 준비 ====="
    )

    positive_items = positive_items.tocsr()
    negative_items = negative_items.tocsr()

    # 유저별 True interaction 개수
    positive_counts = np.diff(
        positive_items.indptr
    )

    # False interaction을 (user, item) pair로 변환
    negative_coo = negative_items.tocoo()

    negative_users = negative_coo.row.astype(
        np.int32,
        copy=False
    )

    negative_item_ids = negative_coo.col.astype(
        np.int32,
        copy=False
    )

    # True가 하나도 없는 유저는
    # (positive, negative) pair를 만들 수 없으므로 제외
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
            "False와 True를 동시에 가진 유저가 없어 "
            "Explicit False Fine-Tuning을 건너뜁니다."
        )

        return model

    rng = np.random.default_rng(
        random_state
    )

    # implicit BPR은 마지막 1차원을
    # item bias용으로 추가해서 사용한다.
    latent_dim = model.factors

    user_factors = model.user_factors
    item_factors = model.item_factors

    # CPU BPR 기준
    # shape = (n_users, factors + 1)
    if user_factors.shape[1] < latent_dim:
        raise ValueError(
            "BPR user factor shape가 예상과 다릅니다."
        )

    has_item_bias = (
        item_factors.shape[1]
        > latent_dim
    )

    for epoch in range(
        epochs
    ):

        print(
            f"\n===== Explicit False Epoch "
            f"{epoch + 1}/{epochs} ====="
        )

        # False interaction 순서를 섞어서 학습
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

            negative_ids = negative_item_ids[
                batch_indices
            ]

            # ----------------------------------
            # 같은 유저의 True item 하나 랜덤 샘플
            # ----------------------------------

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
                starts
                + offsets
            )

            positive_ids = positive_items.indices[
                positive_positions
            ]

            # ----------------------------------
            # 현재 factor 복사
            # ----------------------------------

            p_u = user_factors[
                users,
                :latent_dim
            ].copy()

            q_i = item_factors[
                positive_ids,
                :latent_dim
            ].copy()

            q_j = item_factors[
                negative_ids,
                :latent_dim
            ].copy()

            # ----------------------------------
            # x_uij = score(u, i) - score(u, j)
            # ----------------------------------

            score_diff = np.einsum(
                "ij,ij->i",
                p_u,
                q_i - q_j
            )

            if has_item_bias:

                b_i = item_factors[
                    positive_ids,
                    latent_dim
                ].copy()

                b_j = item_factors[
                    negative_ids,
                    latent_dim
                ].copy()

                score_diff += (
                    b_i - b_j
                )

            # sigmoid(-x)
            # = BPR gradient에서 사용하는 항
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
                    + np.exp(
                        clipped
                    )
                )
            ).astype(
                np.float32,
                copy=False
            )

            # ----------------------------------
            # BPR gradient
            # ----------------------------------

            grad_p = (
                gradient_weight[:, None]
                * (q_i - q_j)
                - regularization
                * p_u
            )

            grad_q_i = (
                gradient_weight[:, None]
                * p_u
                - regularization
                * q_i
            )

            grad_q_j = (
                -gradient_weight[:, None]
                * p_u
                - regularization
                * q_j
            )

            # 같은 user/item이 batch 안에
            # 여러 번 등장할 수 있으므로
            # np.add.at으로 업데이트 누적

            np.add.at(
                user_factors[
                    :, :latent_dim
                ],
                users,
                learning_rate
                * grad_p
            )

            np.add.at(
                item_factors[
                    :, :latent_dim
                ],
                positive_ids,
                learning_rate
                * grad_q_i
            )

            np.add.at(
                item_factors[
                    :, :latent_dim
                ],
                negative_ids,
                learning_rate
                * grad_q_j
            )

            # implicit BPR의 마지막 차원은 item bias
            if has_item_bias:

                grad_b_i = (
                    gradient_weight
                    - regularization
                    * b_i
                )

                grad_b_j = (
                    -gradient_weight
                    - regularization
                    * b_j
                )

                np.add.at(
                    item_factors[
                        :, latent_dim
                    ],
                    positive_ids,
                    learning_rate
                    * grad_b_i
                )

                np.add.at(
                    item_factors[
                        :, latent_dim
                    ],
                    negative_ids,
                    learning_rate
                    * grad_b_j
                )

            # ----------------------------------
            # 학습 상태 확인용 loss / accuracy
            # ----------------------------------

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

        # implicit BPR의 user 마지막 차원은
        # item bias를 더하기 위한 상수 1.0
        if (
            user_factors.shape[1]
            > latent_dim
        ):

            user_factors[
                :, latent_dim
            ] = 1.0

        print(
            "Explicit False Loss:",
            total_loss
            / max(
                total_samples,
                1
            )
        )

        print(
            "Explicit Pair Accuracy:",
            total_correct
            / max(
                total_samples,
                1
            )
        )

    # factor가 바뀌었으므로
    # recommend용 norm cache 초기화
    model._item_norms = None
    model._user_norms = None

    print(
        "\n===== Explicit False Fine-Tuning Complete ====="
    )

    return model



def main():

    # ==========================================
    # 1. 데이터 로드
    # ==========================================

    meta = load_games_meta()


    # 필요한 컬럼만 유지

    meta = (
        meta[
            [
                "app_id",
                "Name"
            ]
        ]
        .drop_duplicates(
            subset="app_id"
        )
    )


    # ==========================================
    # [기존 CF 모델용 데이터 로드]
    # ==========================================

    # user_history = load_recommendations()


    # ==========================================
    # [기존 User / Item CF interaction matrix]
    # ==========================================

    # interaction_matrix, user_to_idx, game_to_idx, idx_to_game = \
    #     build_interaction_matrix(user_history)


    # ==========================================
    # [기존 Item-based CF]
    # ==========================================

    # recommender = ItemBasedCFRecommender(
    #     interaction_matrix,
    #     game_to_idx,
    #     idx_to_game,
    #     k=30
    # )


    # ==========================================
    # [기존 User-based CF]
    # ==========================================

    # recommender = UserBasedCFRecommender(
    #     interaction_matrix,
    #     game_to_idx,
    #     idx_to_game,
    #     k=30
    # )


    # ==========================================
    # [기존 Content-based]
    # ==========================================

    # recommender = ContentBasedRecommender(
    #     ...
    # )


    # ==========================================
    # 2. MF Train / Test 데이터 로드
    # ==========================================

    train_df, test_df = (
        load_mf_split()
    )


    print(
        "Train:",
        train_df.shape
    )

    print(
        "Test :",
        test_df.shape
    )


    # ==========================================
    # 3. implicit BPR 파일 경로
    # ==========================================
    #
    # MODEL_PATH / MATRIX_PATH / MAPPING_PATH는
    # main.py 상단 BPR Experiment Config에서
    # 한 번에 관리.
    # ==========================================

    os.makedirs(
        "models",
        exist_ok=True
    )


    # ==========================================
    # 4. User × Item Sparse Matrix 준비
    # ==========================================
    #
    # MATRIX_PATH
    # -> True + False 전체 interaction
    # -> 추천할 때 이미 본 게임을 제외하기 위한 seen matrix
    #
    # POSITIVE_MATRIX_PATH
    # -> True만 1
    # -> Explicit False 모드에서 implicit BPR의 positive 학습용
    #
    # NEGATIVE_MATRIX_PATH
    # -> False만 1
    # -> Explicit False pairwise fine-tuning용
    # ==========================================


    # ------------------------------------------
    # 전체 Seen Matrix / Mapping 로드 또는 생성
    # ------------------------------------------

    if (
        os.path.exists(
            MATRIX_PATH
        )
        and
        os.path.exists(
            MAPPING_PATH
        )
    ):

        print(
            "\n===== implicit BPR 데이터 로드 ====="
        )

        user_items = load_npz(
            MATRIX_PATH
        )

        mapping = np.load(
            MAPPING_PATH
        )

        user_ids = mapping[
            "user_ids"
        ]

        item_ids = mapping[
            "item_ids"
        ]

        print(
            "Users:",
            len(user_ids)
        )

        print(
            "Items:",
            len(item_ids)
        )

        print(
            "Interactions:",
            user_items.nnz
        )


    # ------------------------------------------
    # Seen Matrix 없음
    # -> 기존 방식 그대로 생성
    # ------------------------------------------

    else:

        print(
            "\n===== implicit BPR Sparse Matrix 생성 ====="
        )

        user_codes, user_ids = (
            pd.factorize(
                train_df[
                    "user_id"
                ],
                sort=True
            )
        )

        item_codes, item_ids = (
            pd.factorize(
                train_df[
                    "app_id"
                ],
                sort=True
            )
        )

        user_codes = (
            user_codes.astype(
                np.int32,
                copy=False
            )
        )

        item_codes = (
            item_codes.astype(
                np.int32,
                copy=False
            )
        )

        print(
            "Users:",
            len(user_ids)
        )

        print(
            "Items:",
            len(item_ids)
        )

        print(
            "Raw Interactions:",
            len(train_df)
        )

        values = np.ones(
            len(train_df),
            dtype=np.float32
        )

        user_items = csr_matrix(
            (
                values,
                (
                    user_codes,
                    item_codes
                )
            ),
            shape=(
                len(user_ids),
                len(item_ids)
            ),
            dtype=np.float32
        )

        user_items.sum_duplicates()
        user_items.data[:] = 1.0

        print(
            "\nSparse Matrix:",
            user_items.shape
        )

        print(
            "Interactions:",
            user_items.nnz
        )

        print(
            "\nSparse Matrix 저장 중..."
        )

        save_npz(
            MATRIX_PATH,
            user_items,
            compressed=False
        )

        np.savez(
            MAPPING_PATH,
            user_ids=np.asarray(
                user_ids
            ),
            item_ids=np.asarray(
                item_ids
            )
        )

        print(
            "Sparse Matrix / Mapping 저장 완료"
        )

        del user_codes
        del item_codes
        del values

        gc.collect()


    # ------------------------------------------
    # Explicit False 모드일 때
    # True / False Matrix 별도 준비
    # ------------------------------------------

    positive_items = None
    negative_items = None

    if BPR_USE_EXPLICIT_FALSE:

        if (
            os.path.exists(
                POSITIVE_MATRIX_PATH
            )
            and
            os.path.exists(
                NEGATIVE_MATRIX_PATH
            )
        ):

            print(
                "\n===== True / False Matrix 로드 ====="
            )

            positive_items = load_npz(
                POSITIVE_MATRIX_PATH
            )

            negative_items = load_npz(
                NEGATIVE_MATRIX_PATH
            )

        else:

            print(
                "\n===== True / False Matrix 생성 ====="
            )

            # 기존 Mapping이 sort=True로 생성되어 있으므로
            # searchsorted로 실제 ID -> 내부 index 변환

            train_user_ids = train_df[
                "user_id"
            ].to_numpy()

            train_item_ids = train_df[
                "app_id"
            ].to_numpy()

            user_codes = np.searchsorted(
                user_ids,
                train_user_ids
            ).astype(
                np.int32,
                copy=False
            )

            item_codes = np.searchsorted(
                item_ids,
                train_item_ids
            ).astype(
                np.int32,
                copy=False
            )

            recommended_mask = train_df[
                "is_recommended"
            ].astype(
                bool
            ).to_numpy()

            positive_user_codes = user_codes[
                recommended_mask
            ]

            positive_item_codes = item_codes[
                recommended_mask
            ]

            negative_user_codes = user_codes[
                ~recommended_mask
            ]

            negative_item_codes = item_codes[
                ~recommended_mask
            ]

            positive_items = csr_matrix(
                (
                    np.ones(
                        len(positive_user_codes),
                        dtype=np.float32
                    ),
                    (
                        positive_user_codes,
                        positive_item_codes
                    )
                ),
                shape=user_items.shape,
                dtype=np.float32
            )

            negative_items = csr_matrix(
                (
                    np.ones(
                        len(negative_user_codes),
                        dtype=np.float32
                    ),
                    (
                        negative_user_codes,
                        negative_item_codes
                    )
                ),
                shape=user_items.shape,
                dtype=np.float32
            )

            positive_items.sum_duplicates()
            positive_items.data[:] = 1.0

            negative_items.sum_duplicates()
            negative_items.data[:] = 1.0

            print(
                "True interactions:",
                positive_items.nnz
            )

            print(
                "False interactions:",
                negative_items.nnz
            )

            print(
                "True / False Matrix 저장 중..."
            )

            save_npz(
                POSITIVE_MATRIX_PATH,
                positive_items,
                compressed=False
            )

            save_npz(
                NEGATIVE_MATRIX_PATH,
                negative_items,
                compressed=False
            )

            print(
                "True / False Matrix 저장 완료"
            )

            del train_user_ids
            del train_item_ids
            del user_codes
            del item_codes
            del recommended_mask
            del positive_user_codes
            del positive_item_codes
            del negative_user_codes
            del negative_item_codes

            gc.collect()

        print(
            "True interactions:",
            positive_items.nnz
        )

        print(
            "False interactions:",
            negative_items.nnz
        )

    # ==========================================
    # 5. implicit BPR 모델 준비
    # ==========================================


    # ------------------------------------------
    # 저장 모델 존재
    # -> 모델 로드
    # ------------------------------------------

    if os.path.exists(
        MODEL_PATH
    ):

        print(
            "\n===== 저장된 implicit BPR 모델 로드 ====="
        )


        bpr_model = (
            BayesianPersonalizedRanking.load(
                MODEL_PATH
            )
        )


    # ------------------------------------------
    # 저장 모델 없음
    # -> 전체 Train 학습
    # -> 모델 저장
    # ------------------------------------------

    else:

        print(
            "\n===== implicit BPR 전체 Train 학습 ====="
        )


        bpr_model = (
            BayesianPersonalizedRanking(

                # =================================
                # Latent Factor 수
                # =================================

                factors=BPR_FACTORS,


                # =================================
                # Learning Rate
                # =================================

                learning_rate=BPR_LEARNING_RATE,


                # =================================
                # L2 Regularization
                # =================================

                regularization=BPR_REGULARIZATION,


                # =================================
                # Iteration
                #
                # 상단 BPR_ITERATIONS 값만 수정하면
                # 학습 반복 횟수와 MODEL_PATH가
                # 함께 변경됨.
                # =================================

                iterations=BPR_ITERATIONS,


                # =================================
                # Negative Sampling 검증
                #
                # 랜덤으로 뽑은 negative가
                # 실제로 이미 본 item인지 확인.
                #
                # True:
                # 이미 본 item이면 negative로
                # 사용하지 않음.
                # =================================

                verify_negative_samples=BPR_VERIFY_NEGATIVE_SAMPLES,


                # =================================
                # CPU Thread
                #
                # 0 = 가능한 CPU core 자동 사용
                # =================================

                num_threads=BPR_NUM_THREADS,


                random_state=RANDOM_STATE
            )
        )


        print(
            "\n===== Training Start ====="
        )


        # ======================================
        # 학습 Matrix 선택
        # ======================================
        #
        # 기존 모드(False):
        # True + False 전체 interaction을 positive로 사용
        #
        # Explicit False 모드(True):
        # 1차 implicit BPR에서는 True만 positive로 사용
        # 2차 fine-tuning에서 False를 explicit negative로 사용
        # ======================================

        if BPR_USE_EXPLICIT_FALSE:

            fit_matrix = positive_items

            print(
                "학습 모드: True=Positive / False=Explicit Negative"
            )

        else:

            fit_matrix = user_items

            print(
                "학습 모드: True+False=Positive / Unseen=Negative"
            )


        bpr_model.fit(
            fit_matrix,

            # 진행률 표시
            show_progress=True
        )


        # ======================================
        # False explicit negative 추가 학습
        # ======================================

        if BPR_USE_EXPLICIT_FALSE:

            bpr_model = (
                fine_tune_with_explicit_false(
                    model=bpr_model,
                    positive_items=positive_items,
                    negative_items=negative_items,
                    epochs=BPR_EXPLICIT_FALSE_EPOCHS,
                    learning_rate=BPR_EXPLICIT_FALSE_LEARNING_RATE,
                    regularization=BPR_EXPLICIT_FALSE_REGULARIZATION,
                    batch_size=BPR_EXPLICIT_FALSE_BATCH_SIZE,
                    random_state=RANDOM_STATE
                )
            )


        print(
            "\n===== Training Complete ====="
        )


        # ======================================
        # 모델 저장
        # ======================================

        bpr_model.save(
            MODEL_PATH
        )


        print(
            "\nimplicit BPR 모델 저장 완료:"
        )


        print(
            MODEL_PATH
        )


    # ==========================================
    # 기존 MF 평가 pipeline과 연결
    # ==========================================

    recommender = (
        ImplicitBPRAdapter(
            model=bpr_model,
            user_items=user_items,
            user_ids=user_ids,
            item_ids=item_ids
        )
    )


    # ==========================================
    # [기존 Item-based 추천 입력 방식]
    # ==========================================

    """
    played_games = []

    while True:

        game = input(
            "게임 이름: "
        ).strip()

        if game == "":
            break

        played_games.append(
            game
        )


    if len(played_games) == 0:

        print(
            "최소 1개의 게임을 입력해야 합니다."
        )

        return


    played_app_ids = []

    for game_name in played_games:

        app_id = resolve_name_to_appid(
            meta,
            game_name
        )

        if app_id is not None:

            played_app_ids.append(
                app_id
            )


    if len(played_app_ids) == 0:

        print(
            "유효한 게임이 없습니다."
        )

        return
    """


    # ==========================================
    # 6. BPR 추천 테스트
    # ==========================================

    # Test 데이터에 존재하는 사용자 중
    # 랜덤으로 유저 1명 선택
    #
    # drop_duplicates()
    # -> 같은 user_id를 한 번씩만 남겨서
    #    모든 유저가 동일한 확률로 선택되게 함

    user_id = (
        test_df[
            "user_id"
        ]
        .drop_duplicates()
        .sample(
            n=1,
            random_state=RANDOM_STATE
        )
        .iloc[0]
    )


    print(
        f"\n랜덤 선택 user_id: "
        f"{user_id}"
    )


    # ==========================================
    # 해당 사용자의 Train 상호작용
    # ==========================================

    played_app_ids = (
        train_df.loc[
            train_df[
                "user_id"
            ] == user_id,
            "app_id"
        ]
        .tolist()
    )


    if len(played_app_ids) == 0:

        print(
            f"user_id {user_id}는 "
            "Train 데이터에 존재하지 않습니다."
        )

        return


    print(
        f"Train 상호작용 게임 수: "
        f"{len(played_app_ids)}"
    )


    # ==========================================
    # 7. BPR Top-10 추천
    # ==========================================

    result = (
        recommender.recommend(
            user_id=user_id,
            app_id_list=played_app_ids,
            top_n=TOP_N
        )
    )


    # 추천 AppID에 게임 이름 붙이기

    result = result.merge(
        meta[
            [
                "app_id",
                "Name"
            ]
        ],
        on="app_id",
        how="left"
    )


    print(
        "\n===== BPR 추천 결과 ====="
    )


    print(
        result[
            [
                "app_id",
                "Name",
                "score"
            ]
        ].to_string(
            index=False
        )
    )


    # ==========================================
    # [진단 0] Seen Item 제외 확인
    # ==========================================

    seen_items = set(
        played_app_ids
    )


    recommended_items = set(
        result[
            "app_id"
        ].tolist()
    )


    overlap = (
        seen_items
        & recommended_items
    )


    print(
        "\nSeen item overlap:",
        overlap
    )


    if len(overlap) == 0:

        print(
            "정상: 이미 본 게임이 추천되지 않음"
        )

    else:

        print(
            "문제: 이미 본 게임이 추천됨"
        )


    # ==========================================
    # 8. 해당 사용자의 실제 Test 정답 확인
    # ==========================================

    user_test = test_df[
        test_df[
            "user_id"
        ] == user_id
    ]


    test_result = (
        user_test[
            [
                "app_id",
                "is_recommended"
            ]
        ]
        .merge(
            meta[
                [
                    "app_id",
                    "Name"
                ]
            ],
            on="app_id",
            how="left"
        )
    )


    print(
        "\n===== 실제 Test 데이터 ====="
    )


    print(
        test_result[
            [
                "app_id",
                "Name",
                "is_recommended"
            ]
        ].to_string(
            index=False
        )
    )


    # ==========================================
    # [진단 1] 추천 게임의 학습 데이터 통계
    # ==========================================

    item_stats = (
        train_df
        .groupby(
            "app_id"
        )
        .agg(
            interaction_count=(
                "user_id",
                "size"
            ),

            true_count=(
                "is_recommended",
                "sum"
            ),

            true_ratio=(
                "is_recommended",
                "mean"
            )
        )
        .reset_index()
    )


    diagnosis_result = (
        result[
            [
                "app_id",
                "Name"
            ]
        ]
        .merge(
            item_stats,
            on="app_id",
            how="left"
        )
    )


    print(
        "\n===== 추천 게임 학습 통계 ====="
    )


    print(
        diagnosis_result.to_string(
            index=False
        )
    )


    # ==========================================
    # [기존 평가 시스템]
    # ==========================================

    """
    valid_app_ids = set(
        meta["app_id"]
    )

    user_history = user_history[
        user_history[
            "app_id"
        ].isin(
            valid_app_ids
        )
    ]
    """


    # ==========================================
    # [기존 evaluate_pipeline 방식]
    # ==========================================

    """
    eval_df, summary = evaluate_pipeline(
        recommender=recommender,
        user_history=user_history,
        lower_bound=10,
        upper_bound=78,
        n_groups=4,
        sample_per_group=100,
        top_n=10
    )
    """


    # ==========================================
    # [기존 User / Item CF 평가 방식]
    # ==========================================

    """
    eligible_users = build_user_review_groups(
        user_history,
        lower_bound=10,
        upper_bound=78
    )

    sampled_users = stratified_sample_users(
        eligible_users,
        sample_per_group=100,
        random_state=42
    )

    eval_df = run_evaluation(
        recommender,
        user_history,
        sampled_users,
        user_to_idx=user_to_idx,
        top_n=10
    )
    """


    # ==========================================
    # 9. BPR 평가
    # ==========================================

    print(
        "\n===== BPR 평가 시작 ====="
    )


    # ------------------------------------------
    # 9-1. 평가 대상 사용자 구성
    # ------------------------------------------

    eligible_users = (
        build_mf_user_review_groups(
            train_df,
            test_df,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND
        )
    )


    # ------------------------------------------
    # 9-2. 각 review group에서 100명씩
    #
    # 총 400명
    # ------------------------------------------

    sampled_users = (
        stratified_sample_users(
            eligible_users,
            sample_per_group=SAMPLE_PER_GROUP,
            random_state=RANDOM_STATE
        )
    )


    # ------------------------------------------
    # 9-3. BPR 평가 실행
    # ------------------------------------------

    eval_df = (
        run_mf_evaluation(

            recommender=recommender,

            train_df=train_df,

            test_df=test_df,

            sampled_users=sampled_users,

            top_n=TOP_N,


            # ==================================
            # 기본 평가
            # ==================================
            #
            # False
            #
            # Test의
            #
            # True + False interaction
            #
            # 전체를 relevant item으로 평가.
            #
            #
            # 이후 추가 실험:
            #
            # positive_only=True
            #
            # -> True interaction만 정답
            #
            # ==================================

            positive_only=POSITIVE_ONLY
        )
    )


    # ==========================================
    # [진단 2] 사용자별 추천 게임 반복 빈도 확인
    # ==========================================

    from collections import Counter


    # sampled user의 Train interaction 정리

    sampled_ids = set(
        sampled_users[
            "user_id"
        ]
    )


    target_train = train_df[
        train_df[
            "user_id"
        ].isin(
            sampled_ids
        )
    ]


    train_by_user = (
        target_train
        .groupby(
            "user_id"
        )[
            "app_id"
        ]
        .apply(
            list
        )
        .to_dict()
    )


    # 모든 추천 결과 수집

    recommend_counter = Counter()


    for user_id in sampled_ids:

        played_app_ids = (
            train_by_user.get(
                user_id,
                []
            )
        )


        if len(
            played_app_ids
        ) == 0:

            continue


        rec_result = (
            recommender.recommend(
                user_id=user_id,
                app_id_list=played_app_ids,
                top_n=TOP_N
            )
        )


        if (
            rec_result is None
            or
            len(
                rec_result
            ) == 0
        ):

            continue


        recommend_counter.update(
            rec_result[
                "app_id"
            ].tolist()
        )


    # 가장 많이 추천된 게임

    top_common = pd.DataFrame(
        recommend_counter.most_common(
            30
        ),

        columns=[
            "app_id",
            "recommend_count"
        ]
    )


    # 게임 이름 추가

    top_common = (
        top_common.merge(
            meta[
                [
                    "app_id",
                    "Name"
                ]
            ],

            on="app_id",
            how="left"
        )
    )


    # ==========================================
    # 위에서 이미 만든 item_stats 재사용
    # ==========================================

    top_common = (
        top_common.merge(
            item_stats,
            on="app_id",
            how="left"
        )
    )


    print(
        "\n===== 여러 사용자에게 반복 추천되는 게임 TOP 30 ====="
    )


    print(
        top_common.to_string(
            index=False
        )
    )


    # ------------------------------------------
    # 9-4. 평가 결과 출력
    # ------------------------------------------

    summary = (
        print_evaluation_report(
            eval_df,
            top_n=TOP_N
        )
    )


    # ------------------------------------------
    # 9-5. 실제 추천 개수 확인
    # ------------------------------------------

    n_rec_summary = (
        eval_df
        .groupby(
            "review_group",
            observed=True
        )[
            "n_recommended"
        ]
        .agg(
            [
                "mean",
                "min",
                "max",
                "count"
            ]
        )
    )


    print(
        "\n===== 그룹별 추천 개수 ====="
    )


    print(
        n_rec_summary
    )


    print(
        f"\n전체 n_recommended 평균: "
        f"{eval_df['n_recommended'].mean():.2f} / 10"
    )


    full_ratio = (
        eval_df[
            "n_recommended"
        ] == 10
    ).mean()


    print(
        f"10개 꽉 채워 추천된 유저 비율: "
        f"{full_ratio:.1%}"
    )



if __name__ == "__main__":

    main()