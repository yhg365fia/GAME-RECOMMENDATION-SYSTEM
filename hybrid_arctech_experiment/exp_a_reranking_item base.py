import sys
from pathlib import Path

import numpy as np
import pandas as pd

from implicit.cpu.bpr import BayesianPersonalizedRanking


# =========================================================
# Project Root
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from data_split import load_mf_split

from evaluation import (
    run_mf_evaluation,
    print_evaluation_report,
)

from models.itembase import (
    build_interaction_matrix,
    ItemBasedCFRecommender,
)


# =========================================================
# Experiment Config
# =========================================================

# Item-Based에서 먼저 뽑을 후보 수
CANDIDATE_N = 30

# 최종 추천 개수
TOP_N = 10

# Item-Based 내부 item neighbor 수
ITEM_K = 30


# =========================================================
# 저장 경로
# =========================================================

SAVED_MODEL_DIR = (
    ROOT
    / "models"
    / "saved_model"
)

RESULT_DIR = (
    SAVED_MODEL_DIR
    / "results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 평가 사용자
# =========================================================

SAMPLED_USERS_PATH = (
    RESULT_DIR
    / "hybrid_sampled_users.csv"
)


# =========================================================
# 기존 Item-Based Baseline
# =========================================================

BASELINE_EVAL_PATH = (
    RESULT_DIR
    / "hybrid_baseline_item_eval.csv"
)


# =========================================================
# 최종 BPR 모델
# =========================================================

BPR_MODEL_NAME = (
    "bpr_iter15_factors60_reg0.006_explicit_false.npz"
)


# =========================================================
# Experiment A 결과
# =========================================================

EVAL_PATH = (
    RESULT_DIR
    / "hybrid_exp_a_bpr_rerank_eval.csv"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "hybrid_exp_a_bpr_rerank_summary.csv"
)

GROUP_PATH = (
    RESULT_DIR
    / "hybrid_exp_a_bpr_rerank_group_summary.csv"
)

COMPARE_PATH = (
    RESULT_DIR
    / "hybrid_exp_a_bpr_rerank_vs_baseline.csv"
)

CANDIDATE_PATH = (
    RESULT_DIR
    / "hybrid_exp_a_candidate_diagnostics.csv"
)


# =========================================================
# 파일 자동 탐색
# =========================================================

def find_file(
    base_dir,
    filename,
):
    """
    base_dir 아래를 재귀적으로 검색해서
    filename과 정확히 일치하는 파일을 찾는다.
    """

    matches = list(
        base_dir.rglob(
            filename
        )
    )

    if len(matches) == 0:
        return None

    if len(matches) > 1:

        print(
            f"\n주의: {filename} 파일이 여러 개 발견됨"
        )

        for path in matches:
            print(" -", path)

        print(
            "첫 번째 파일을 사용합니다."
        )

    return matches[0]


# =========================================================
# BPR 파일 찾기
# =========================================================

def find_bpr_files():

    print(
        "\n===== BPR 저장 파일 탐색 ====="
    )

    # -----------------------------------------------------
    # 1. 모델
    # -----------------------------------------------------

    model_path = find_file(
        SAVED_MODEL_DIR,
        BPR_MODEL_NAME,
    )

    if model_path is None:

        raise FileNotFoundError(
            "\n최종 BPR 모델을 찾을 수 없습니다.\n"
            f"검색 위치:\n{SAVED_MODEL_DIR}\n\n"
            f"찾는 파일:\n{BPR_MODEL_NAME}"
        )


    # -----------------------------------------------------
    # 2. mapping
    # -----------------------------------------------------

    mapping_path = find_file(
        SAVED_MODEL_DIR,
        "implicit_bpr_mapping.npz",
    )


    # 예전 구조에 mapping만 models/에 남아있을 수도 있으므로
    # saved_model에 없으면 models 전체에서 한 번 더 찾는다.
    if mapping_path is None:

        mapping_path = find_file(
            ROOT / "models",
            "implicit_bpr_mapping.npz",
        )


    if mapping_path is None:

        raise FileNotFoundError(
            "\nBPR mapping 파일을 찾을 수 없습니다.\n"
            "찾는 파일:\n"
            "implicit_bpr_mapping.npz"
        )


    print(
        "BPR Model:"
    )

    print(
        model_path
    )

    print(
        "\nBPR Mapping:"
    )

    print(
        mapping_path
    )


    return (
        model_path,
        mapping_path,
    )


# =========================================================
# Experiment A Recommender
# =========================================================

class ItemCandidateBPRReranker:

    """
    Experiment A

    Item-Based CF
        ↓
    Top-30 Candidates
        ↓
    BPR score
        ↓
    Reranking
        ↓
    Final Top-10


    중요한 점:

    BPR이 전체 게임에서 새로운 후보를 가져오는 것이 아니다.

    Item-Based가 이미 뽑은 30개에 대해서만
    BPR 점수를 계산한다.

    즉 이 실험의 질문은:

    "Item-Based가 후보는 잘 찾고 있는데
     순위를 잘못 매기고 있는가?"

    이다.
    """

    def __init__(
        self,
        item_model,
        bpr_model,
        user_ids,
        item_ids,
        candidate_n=30,
    ):

        self.item_model = item_model
        self.bpr_model = bpr_model

        self.user_ids = np.asarray(
            user_ids
        )

        self.item_ids = np.asarray(
            item_ids
        )

        self.candidate_n = candidate_n


        # 후보군 자체 성능 분석용
        self.candidate_log = {}


        # mapping 진단
        self.missing_users = 0
        self.missing_items = 0


    # -----------------------------------------------------
    # 실제 ID → BPR 내부 Index
    # -----------------------------------------------------

    @staticmethod
    def find_index(
        sorted_ids,
        value,
    ):

        idx = int(
            np.searchsorted(
                sorted_ids,
                value,
            )
        )

        if idx >= len(sorted_ids):
            return None

        if sorted_ids[idx] != value:
            return None

        return idx


    # -----------------------------------------------------
    # 추천
    # -----------------------------------------------------

    def recommend(
        self,
        user_id,
        app_id_list=None,
        top_n=10,
    ):

        # =================================================
        # 1. Item-Based Top-30 후보
        # =================================================

        candidates = (
            self.item_model.recommend(
                app_id_list=app_id_list,
                top_n=self.candidate_n,
            )
        )


        if (
            candidates is None
            or len(candidates) == 0
        ):

            self.candidate_log[
                user_id
            ] = []

            return pd.DataFrame(
                columns=[
                    "app_id",
                    "bpr_score",
                    "item_candidate_rank",
                ]
            )


        candidate_app_ids = (
            candidates[
                "app_id"
            ]
            .to_numpy()
        )


        # Item-Based 원래 순위
        candidate_ranks = np.arange(
            1,
            len(candidate_app_ids) + 1,
        )


        # 후보군 성능 분석용
        self.candidate_log[
            user_id
        ] = (
            candidate_app_ids.tolist()
        )


        # =================================================
        # 2. user_id → BPR user index
        # =================================================

        user_idx = (
            self.find_index(
                self.user_ids,
                user_id,
            )
        )


        if user_idx is None:

            self.missing_users += 1

            return pd.DataFrame(
                columns=[
                    "app_id",
                    "bpr_score",
                    "item_candidate_rank",
                ]
            )


        # =================================================
        # 3. 후보 app_id → BPR item index
        # =================================================

        item_indices = np.searchsorted(
            self.item_ids,
            candidate_app_ids,
        )


        valid = (
            item_indices
            <
            len(self.item_ids)
        )


        # searchsorted는 없는 값도 위치를 반환하므로
        # 실제 값이 일치하는지 한 번 더 확인
        matched = np.zeros(
            len(candidate_app_ids),
            dtype=bool,
        )


        valid_positions = np.where(
            valid
        )[0]


        if len(valid_positions) > 0:

            matched[
                valid_positions
            ] = (

                self.item_ids[
                    item_indices[
                        valid_positions
                    ]
                ]

                ==

                candidate_app_ids[
                    valid_positions
                ]

            )


        valid = (
            valid
            &
            matched
        )


        self.missing_items += int(
            (~valid).sum()
        )


        # BPR에 존재하는 후보가 하나도 없는 경우
        if not valid.any():

            return pd.DataFrame(
                columns=[
                    "app_id",
                    "bpr_score",
                    "item_candidate_rank",
                ]
            )


        candidate_app_ids = (
            candidate_app_ids[
                valid
            ]
        )


        candidate_ranks = (
            candidate_ranks[
                valid
            ]
        )


        item_indices = (
            item_indices[
                valid
            ]
        )


        # =================================================
        # 4. BPR 점수 계산
        # =================================================
        #
        # BPR 재학습 X
        #
        # 이미 학습된 user/item latent factor로
        # 후보 30개의 점수만 계산
        #
        # score(u, i) = p_u · q_i
        # =================================================

        user_factor = (

            self.bpr_model
            .user_factors[
                user_idx
            ]

        )


        candidate_item_factors = (

            self.bpr_model
            .item_factors[
                item_indices
            ]

        )


        bpr_scores = (

            candidate_item_factors
            @
            user_factor

        )


        # =================================================
        # 5. BPR 점수로 재정렬
        # =================================================

        order = np.argsort(
            -bpr_scores,
            kind="stable",
        )


        candidate_app_ids = (
            candidate_app_ids[
                order
            ]
        )


        bpr_scores = (
            bpr_scores[
                order
            ]
        )


        candidate_ranks = (
            candidate_ranks[
                order
            ]
        )


        # =================================================
        # 6. Top-10
        # =================================================

        n_actual = min(
            top_n,
            len(candidate_app_ids),
        )


        result = pd.DataFrame(
            {
                "app_id":
                    candidate_app_ids[
                        :n_actual
                    ],

                "bpr_score":
                    bpr_scores[
                        :n_actual
                    ],

                "item_candidate_rank":
                    candidate_ranks[
                        :n_actual
                    ],
            }
        )


        return result


# =========================================================
# Candidate@30 성능 분석
# =========================================================

def build_candidate_diagnostics(
    reranker,
    test_df,
    sampled_users,
):

    """
    Item-Based Top-30 안에
    실제 Test Positive Item이 얼마나 들어있는지 확인.

    이 값은 매우 중요하다.

    Top-30에 정답이 없다면
    아무리 좋은 reranker라도
    정답을 Top-10으로 가져올 수 없다.
    """

    sampled_ids = set(
        sampled_users[
            "user_id"
        ]
    )


    # -----------------------------------------------------
    # Positive Test만 정답
    # -----------------------------------------------------

    positive_test = (

        test_df[
            (
                test_df[
                    "user_id"
                ].isin(
                    sampled_ids
                )
            )
            &
            (
                test_df[
                    "is_recommended"
                ]
                == True
            )
        ]

        .groupby(
            "user_id"
        )[
            "app_id"
        ]

        .apply(list)

        .to_dict()

    )


    rows = []


    for row in (
        sampled_users
        .itertuples(
            index=False
        )
    ):

        user_id = row.user_id


        test_ids = set(
            positive_test.get(
                user_id,
                [],
            )
        )


        if len(test_ids) == 0:
            continue


        candidate_ids = set(
            reranker
            .candidate_log
            .get(
                user_id,
                [],
            )
        )


        hits = len(
            candidate_ids
            &
            test_ids
        )


        rows.append(
            {
                "user_id":
                    user_id,

                "review_group":
                    row.review_group,

                "n_games":
                    row.n_games,

                "candidate_count":
                    len(
                        candidate_ids
                    ),

                "n_test":
                    len(
                        test_ids
                    ),

                "candidate_hits":
                    hits,

                "candidate_hit":
                    int(
                        hits > 0
                    ),

                "candidate_recall":
                    (
                        hits
                        /
                        len(test_ids)
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# =========================================================
# 전체 결과 요약
# =========================================================

def make_summary(
    eval_df,
):

    total_hits = int(
        eval_df[
            "hits"
        ].sum()
    )


    total_recommended = int(
        eval_df[
            "n_recommended"
        ].sum()
    )


    total_test = int(
        eval_df[
            "n_test"
        ].sum()
    )


    micro_precision = (

        total_hits
        /
        total_recommended

        if total_recommended > 0

        else 0.0

    )


    micro_recall = (

        total_hits
        /
        total_test

        if total_test > 0

        else 0.0

    )


    micro_f1 = (

        2
        *
        micro_precision
        *
        micro_recall
        /
        (
            micro_precision
            +
            micro_recall
        )

        if (
            micro_precision
            +
            micro_recall
        ) > 0

        else 0.0

    )


    return {

        "experiment":
            "item30_bpr_reranking",

        "candidate_n":
            CANDIDATE_N,

        "top_n":
            TOP_N,

        "n_users":
            len(eval_df),

        "precision_at_10":
            eval_df[
                "precision"
            ].mean(),

        "recall_at_10":
            eval_df[
                "recall"
            ].mean(),

        "hit_rate_at_10":
            eval_df[
                "hit"
            ].mean(),

        "ndcg_at_10":
            eval_df[
                "ndcg"
            ].mean(),

        "micro_precision_at_10":
            micro_precision,

        "micro_recall_at_10":
            micro_recall,

        "micro_f1_at_10":
            micro_f1,

        "hits":
            total_hits,

        "n_recommended":
            total_recommended,

        "n_test":
            total_test,
    }


# =========================================================
# Baseline 비교
# =========================================================

def compare_with_baseline(
    rerank_eval_df,
):

    if not BASELINE_EVAL_PATH.exists():

        print(
            "\nBaseline CSV가 없어 "
            "사용자별 비교는 생략합니다."
        )

        return None


    baseline_df = pd.read_csv(
        BASELINE_EVAL_PATH
    )


    columns = [
        "user_id",
        "precision",
        "recall",
        "hits",
        "hit",
        "ndcg",
    ]


    baseline = (

        baseline_df[
            columns
        ]

        .rename(
            columns={
                "precision":
                    "baseline_precision",

                "recall":
                    "baseline_recall",

                "hits":
                    "baseline_hits",

                "hit":
                    "baseline_hit",

                "ndcg":
                    "baseline_ndcg",
            }
        )

    )


    rerank = (

        rerank_eval_df[
            columns
        ]

        .rename(
            columns={
                "precision":
                    "rerank_precision",

                "recall":
                    "rerank_recall",

                "hits":
                    "rerank_hits",

                "hit":
                    "rerank_hit",

                "ndcg":
                    "rerank_ndcg",
            }
        )

    )


    comparison = baseline.merge(
        rerank,
        on="user_id",
        how="inner",
    )


    comparison[
        "delta_precision"
    ] = (

        comparison[
            "rerank_precision"
        ]

        -

        comparison[
            "baseline_precision"
        ]

    )


    comparison[
        "delta_recall"
    ] = (

        comparison[
            "rerank_recall"
        ]

        -

        comparison[
            "baseline_recall"
        ]

    )


    comparison[
        "delta_hits"
    ] = (

        comparison[
            "rerank_hits"
        ]

        -

        comparison[
            "baseline_hits"
        ]

    )


    comparison[
        "delta_hit"
    ] = (

        comparison[
            "rerank_hit"
        ]

        -

        comparison[
            "baseline_hit"
        ]

    )


    comparison[
        "delta_ndcg"
    ] = (

        comparison[
            "rerank_ndcg"
        ]

        -

        comparison[
            "baseline_ndcg"
        ]

    )


    return comparison


# =========================================================
# Main
# =========================================================

def main():

    print()

    print(
        "=========================================="
    )

    print(
        " Hybrid Experiment A - BPR Reranking"
    )

    print(
        " Item-Based Top-30 -> BPR -> Top-10"
    )

    print(
        "=========================================="
    )


    # =====================================================
    # 1. MF Split
    # =====================================================

    print(
        "\n===== 1. 기존 MF Train / Test 로드 ====="
    )


    train_df, test_df = (
        load_mf_split()
    )


    print(
        "Train:",
        train_df.shape,
    )

    print(
        "Test :",
        test_df.shape,
    )


    # =====================================================
    # 2. Item-Based Matrix
    # =====================================================

    print(
        "\n===== 2. Item-Based Interaction Matrix 생성 ====="
    )


    (
        interaction_matrix,
        user_to_idx,
        game_to_idx,
        idx_to_game,
    ) = build_interaction_matrix(
        train_df
    )


    print(
        "Matrix:",
        interaction_matrix.shape,
    )

    print(
        "Users:",
        interaction_matrix.shape[0],
    )

    print(
        "Games:",
        interaction_matrix.shape[1],
    )


    # =====================================================
    # 3. Item-Based Model
    # =====================================================

    print(
        "\n===== 3. Item-Based CF 생성 ====="
    )


    item_model = (
        ItemBasedCFRecommender(
            interaction_matrix=
                interaction_matrix,

            game_to_idx=
                game_to_idx,

            idx_to_game=
                idx_to_game,

            k=
                ITEM_K,
        )
    )


    # =====================================================
    # 4. BPR 저장 파일 탐색
    # =====================================================

    print(
        "\n===== 4. 최종 BPR 모델 준비 ====="
    )


    (
        bpr_model_path,
        bpr_mapping_path,
    ) = find_bpr_files()


    # =====================================================
    # 5. BPR Model Load
    # =====================================================

    print(
        "\n===== 5. BPR 모델 로드 ====="
    )


    bpr_model = (
        BayesianPersonalizedRanking.load(
            str(
                bpr_model_path
            )
        )
    )


    mapping = np.load(
        bpr_mapping_path
    )


    user_ids = (
        mapping[
            "user_ids"
        ]
    )


    item_ids = (
        mapping[
            "item_ids"
        ]
    )


    print(
        "User factors:",
        bpr_model
        .user_factors
        .shape,
    )


    print(
        "Item factors:",
        bpr_model
        .item_factors
        .shape,
    )


    print(
        "Mapping Users:",
        len(
            user_ids
        ),
    )


    print(
        "Mapping Items:",
        len(
            item_ids
        ),
    )


    # -----------------------------------------------------
    # Factor / Mapping 검증
    # -----------------------------------------------------

    if (
        bpr_model
        .user_factors
        .shape[0]

        !=

        len(
            user_ids
        )
    ):

        raise ValueError(
            "\nBPR user factor 수와 "
            "mapping user 수가 다릅니다."
        )


    if (
        bpr_model
        .item_factors
        .shape[0]

        !=

        len(
            item_ids
        )
    ):

        raise ValueError(
            "\nBPR item factor 수와 "
            "mapping item 수가 다릅니다."
        )


    # =====================================================
    # 6. 평가 사용자 400명
    # =====================================================

    print(
        "\n===== 6. 평가 사용자 준비 ====="
    )


    if not SAMPLED_USERS_PATH.exists():

        raise FileNotFoundError(
            "\n기존 Hybrid 평가 사용자 파일 없음:\n"
            f"{SAMPLED_USERS_PATH}"
        )


    sampled_users = pd.read_csv(
        SAMPLED_USERS_PATH
    )


    print(
        "평가 사용자 파일:"
    )

    print(
        SAMPLED_USERS_PATH
    )


    print(
        "\n평가 사용자 수:",
        len(
            sampled_users
        ),
    )


    print(
        "\n그룹:"
    )

    print(
        sampled_users[
            "review_group"
        ].value_counts()
    )


    # =====================================================
    # 7. Reranker
    # =====================================================

    print(
        "\n===== 7. Item -> BPR Reranker 생성 ====="
    )


    reranker = (
        ItemCandidateBPRReranker(
            item_model=
                item_model,

            bpr_model=
                bpr_model,

            user_ids=
                user_ids,

            item_ids=
                item_ids,

            candidate_n=
                CANDIDATE_N,
        )
    )


    # =====================================================
    # 8. Evaluation
    # =====================================================

    print(
        "\n===== 8. BPR Reranking 평가 시작 ====="
    )


    eval_df = (
        run_mf_evaluation(
            recommender=
                reranker,

            train_df=
                train_df,

            test_df=
                test_df,

            sampled_users=
                sampled_users,

            top_n=
                TOP_N,

            positive_only=
                True,
        )
    )


    # =====================================================
    # 9. Evaluation Report
    # =====================================================

    print(
        "\n===== 9. Reranking 평가 결과 ====="
    )


    group_summary = (
        print_evaluation_report(
            eval_df,
            top_n=TOP_N,
        )
    )


    # =====================================================
    # 10. Candidate@30 Diagnostic
    # =====================================================

    print(
        "\n===== 10. Item-Based Candidate@30 분석 ====="
    )


    candidate_df = (
        build_candidate_diagnostics(
            reranker=
                reranker,

            test_df=
                test_df,

            sampled_users=
                sampled_users,
        )
    )


    candidate_hit_rate = (
        candidate_df[
            "candidate_hit"
        ].mean()
    )


    candidate_recall = (
        candidate_df[
            "candidate_recall"
        ].mean()
    )


    candidate_hits = int(
        candidate_df[
            "candidate_hits"
        ].sum()
    )


    print(
        f"Candidate@{CANDIDATE_N} "
        f"Hit Rate: "
        f"{candidate_hit_rate:.4f}"
    )


    print(
        f"Candidate@{CANDIDATE_N} "
        f"Recall:    "
        f"{candidate_recall:.4f}"
    )


    print(
        f"Candidate@{CANDIDATE_N} "
        f"Hits:      "
        f"{candidate_hits}"
    )


    print(
        "\n===== Candidate 그룹별 결과 ====="
    )


    candidate_group = (

        candidate_df
        .groupby(
            "review_group",
            observed=True,
        )
        .agg(
            candidate_recall=(
                "candidate_recall",
                "mean",
            ),

            candidate_hit_rate=(
                "candidate_hit",
                "mean",
            ),

            candidate_hits=(
                "candidate_hits",
                "sum",
            ),

            n_users=(
                "user_id",
                "count",
            ),
        )

        .reset_index()

    )


    print(
        candidate_group.to_string(
            index=False
        )
    )


    # =====================================================
    # 11. Summary
    # =====================================================

    summary = make_summary(
        eval_df
    )


    summary[
        "candidate_hit_rate"
    ] = candidate_hit_rate


    summary[
        "candidate_recall"
    ] = candidate_recall


    summary[
        "candidate_hits"
    ] = candidate_hits


    # =====================================================
    # 12. Baseline 비교
    # =====================================================

    print(
        "\n===== 11. 기존 Item-Based Baseline 비교 ====="
    )


    comparison = compare_with_baseline(
        eval_df
    )


    if comparison is not None:

        print(
            "공통 평가 사용자:",
            len(
                comparison
            ),
        )


        print(
            "Baseline Hits:",
            int(
                comparison[
                    "baseline_hits"
                ].sum()
            ),
        )


        print(
            "Rerank Hits:",
            int(
                comparison[
                    "rerank_hits"
                ].sum()
            ),
        )


        print(
            "Δ Hits:",
            int(
                comparison[
                    "delta_hits"
                ].sum()
            ),
        )


        print(
            "평균 Δ Precision:",
            f"{comparison['delta_precision'].mean():.6f}",
        )


        print(
            "평균 Δ Recall:",
            f"{comparison['delta_recall'].mean():.6f}",
        )


        print(
            "평균 Δ NDCG:",
            f"{comparison['delta_ndcg'].mean():.6f}",
        )


        # ----------------------------------------------
        # 유저 단위 승/패
        # ----------------------------------------------

        improved_users = int(
            (
                comparison[
                    "delta_hits"
                ] > 0
            ).sum()
        )


        worsened_users = int(
            (
                comparison[
                    "delta_hits"
                ] < 0
            ).sum()
        )


        same_users = int(
            (
                comparison[
                    "delta_hits"
                ] == 0
            ).sum()
        )


        print(
            "\nHit 기준 사용자 변화:"
        )

        print(
            "개선:",
            improved_users,
        )

        print(
            "악화:",
            worsened_users,
        )

        print(
            "동일:",
            same_users,
        )


    # =====================================================
    # 13. 저장
    # =====================================================

    print(
        "\n===== 12. 결과 저장 ====="
    )


    eval_df.to_csv(
        EVAL_PATH,
        index=False,
    )


    pd.DataFrame(
        [
            summary
        ]
    ).to_csv(
        SUMMARY_PATH,
        index=False,
    )


    group_summary.to_csv(
        GROUP_PATH,
        index=False,
    )


    candidate_df.to_csv(
        CANDIDATE_PATH,
        index=False,
    )


    if comparison is not None:

        comparison.to_csv(
            COMPARE_PATH,
            index=False,
        )


    print(
        EVAL_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        GROUP_PATH
    )

    print(
        CANDIDATE_PATH
    )


    if comparison is not None:

        print(
            COMPARE_PATH
        )


    # =====================================================
    # 14. Final Summary
    # =====================================================

    print()

    print(
        "=========================================="
    )

    print(
        " Experiment A Summary"
    )

    print(
        "=========================================="
    )


    print(
        f"Candidate Generator : "
        f"Item-Based Top-{CANDIDATE_N}"
    )


    print(
        "Reranker            : "
        "BPR"
    )


    print(
        "BPR Model           : "
        "iter15 / factors60 / "
        "reg0.006 / explicit_false"
    )


    print(
        f"Final Recommendation: "
        f"Top-{TOP_N}"
    )


    print()


    print(
        f"Precision@10: "
        f"{summary['precision_at_10']:.6f}"
    )


    print(
        f"Recall@10:    "
        f"{summary['recall_at_10']:.6f}"
    )


    print(
        f"Hit Rate@10:  "
        f"{summary['hit_rate_at_10']:.6f}"
    )


    print(
        f"NDCG@10:      "
        f"{summary['ndcg_at_10']:.6f}"
    )


    print(
        f"Hits:         "
        f"{summary['hits']}"
    )


    print()


    print(
        f"Candidate@30 Recall: "
        f"{candidate_recall:.6f}"
    )


    print(
        f"Candidate@30 HR:     "
        f"{candidate_hit_rate:.6f}"
    )


    print(
        f"Candidate@30 Hits:   "
        f"{candidate_hits}"
    )


    print()


    print(
        "BPR mapping 없는 사용자:",
        reranker.missing_users,
    )


    print(
        "BPR mapping 없는 Candidate Item:",
        reranker.missing_items,
    )


    print()

    print(
        "=========================================="
    )

    print(
        " Experiment A Complete"
    )

    print(
        "=========================================="
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()