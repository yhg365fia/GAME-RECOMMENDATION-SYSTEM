# hybrid_arctech_experiment/exp_b_weighted_multi_retriever.py

import sys
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from implicit.cpu.bpr import BayesianPersonalizedRanking


# =========================================================
# Project Root
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
CURRENT_DIR = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


# =========================================================
# 기존 Case 3 코드 재사용
#
# - 400명 평가 사용자
# - 평가 subset
# - Item cache
# - User cache
# - BPR cache
# - Content cache
# - evaluation 함수
# =========================================================

import exp_c_fusion_ablation as base


# =========================================================
# Experiment Config
# =========================================================

TOP_N = 10


# ---------------------------------------------------------
# Fusion 실험에서 얻은 중요도 기반 Candidate 배분
#
# User weight sweep의 best 영역:
#
# Item    = 0.558824
# User    = 0.050000
# BPR     = 0.223529
# Content = 0.167647
#
# Candidate 총량을 100으로 고정
# ---------------------------------------------------------

EXPERIMENT_CASES = {

    # User-Based 제거
    # 나머지 3개 비율 재정규화
    #
    # Item    0.5882 -> 59
    # BPR     0.2353 -> 24
    # Content 0.1765 -> 17
    #
    "without_user": {
        "item": 59,
        "bpr": 24,
        "content": 17,
        "user": 0,
    },

    # User=0.05 best fusion 비율
    #
    # Item    0.5588 -> 56
    # BPR     0.2235 -> 22
    # Content 0.1676 -> 17
    # User    0.0500 -> 5
    #
    "with_user": {
        "item": 56,
        "bpr": 22,
        "content": 17,
        "user": 5,
    },
}


# =========================================================
# Paths
# =========================================================

RESULT_DIR = (
    base.RESULT_DIR
    / "case2_weighted_retriever"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH = (
    RESULT_DIR
    / "case2_weighted_retriever_summary.csv"
)

COMPARE_PATH = (
    RESULT_DIR
    / "case2_weighted_retriever_comparison.csv"
)


# =========================================================
# Utility
# =========================================================

def unique_preserve_order(values):

    seen = set()
    result = []

    for value in values:

        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def cache_to_dict(df):

    return (
        df
        .sort_values(
            ["user_id", "rank"],
            kind="stable",
        )
        .groupby(
            "user_id",
            sort=False,
        )["app_id"]
        .apply(list)
        .to_dict()
    )


# =========================================================
# Weighted Multi-Retriever + BPR Ranker
# =========================================================

class WeightedMultiRetrieverBPRRanker:

    """
    Candidate Stage
    ----------------

    Case A:
        Item59
        + BPR24
        + Content17

    Case B:
        Item56
        + BPR22
        + Content17
        + User5

    각각 UNION

        ↓

    모든 후보를 동일한 BPR 모델로 score

        ↓

    BPR Ranking

        ↓

    Top-10


    주의:
    BPR이 Retriever이면서 Ranker이기도 하므로
    BPR Top-10이 이미 후보군에 포함되어 있을 경우
    최종 Top-10이 Pure BPR Top-10과 동일해질 가능성이 매우 높다.

    따라서 pure_bpr_overlap도 함께 기록한다.
    """

    def __init__(
        self,
        item_cache,
        user_cache,
        bpr_cache,
        content_cache,
        bpr_model,
        bpr_user_ids,
        bpr_item_ids,
        candidate_counts,
        case_name,
    ):

        self.item = cache_to_dict(
            item_cache
        )

        self.user = cache_to_dict(
            user_cache
        )

        self.bpr = cache_to_dict(
            bpr_cache
        )

        self.content = cache_to_dict(
            content_cache
        )


        self.bpr_model = bpr_model

        self.bpr_user_ids = np.asarray(
            bpr_user_ids
        )

        self.bpr_item_ids = np.asarray(
            bpr_item_ids
        )


        self.bpr_item_to_idx = {

            app_id: idx

            for idx, app_id
            in enumerate(
                self.bpr_item_ids
            )
        }


        self.counts = (
            candidate_counts
        )

        self.case_name = (
            case_name
        )


        self.candidate_log = {}

        self.pool_logs = []

        self.missing_users = 0

        self.missing_candidates = 0

        self.call_count = 0


    # =====================================================
    # BPR User Mapping
    # =====================================================

    def find_bpr_user_index(
        self,
        user_id,
    ):

        idx = int(
            np.searchsorted(
                self.bpr_user_ids,
                user_id,
            )
        )

        if idx >= len(
            self.bpr_user_ids
        ):
            return None

        if (
            self.bpr_user_ids[idx]
            != user_id
        ):
            return None

        return idx


    # =====================================================
    # Recommend
    # =====================================================

    def recommend(
        self,
        user_id,
        app_id_list=None,
        top_n=10,
    ):

        self.call_count += 1

        if self.call_count % 50 == 0:

            print(
                f"[{self.case_name}] "
                f"{self.call_count} users 완료"
            )


        if app_id_list is None:
            app_id_list = []


        played_set = set(
            app_id_list
        )


        # =================================================
        # 1. Candidate Retrieval
        # =================================================

        item_ids = (
            self.item.get(
                user_id,
                [],
            )
            [:self.counts["item"]]
        )


        bpr_ids = (
            self.bpr.get(
                user_id,
                [],
            )
            [:self.counts["bpr"]]
        )


        content_ids = (
            self.content.get(
                user_id,
                [],
            )
            [:self.counts["content"]]
        )


        if self.counts["user"] > 0:

            user_ids = (
                self.user.get(
                    user_id,
                    [],
                )
                [:self.counts["user"]]
            )

        else:

            user_ids = []


        # =================================================
        # 혹시 모를 seen item 제거
        # =================================================

        item_ids = [

            app_id
            for app_id in item_ids

            if app_id not in played_set
        ]


        bpr_ids = [

            app_id
            for app_id in bpr_ids

            if app_id not in played_set
        ]


        content_ids = [

            app_id
            for app_id in content_ids

            if app_id not in played_set
        ]


        user_ids = [

            app_id
            for app_id in user_ids

            if app_id not in played_set
        ]


        # =================================================
        # 2. UNION
        # =================================================

        union_ids = (
            unique_preserve_order(

                item_ids
                +
                bpr_ids
                +
                content_ids
                +
                user_ids
            )
        )


        # =================================================
        # Source Rank
        # =================================================

        item_rank = {

            app_id: rank + 1

            for rank, app_id
            in enumerate(
                item_ids
            )
        }


        bpr_retrieval_rank = {

            app_id: rank + 1

            for rank, app_id
            in enumerate(
                bpr_ids
            )
        }


        content_rank = {

            app_id: rank + 1

            for rank, app_id
            in enumerate(
                content_ids
            )
        }


        user_rank = {

            app_id: rank + 1

            for rank, app_id
            in enumerate(
                user_ids
            )
        }


        # =================================================
        # Empty Candidate
        # =================================================

        if len(union_ids) == 0:

            self.candidate_log[user_id] = {

                "item": item_ids,
                "bpr": bpr_ids,
                "content": content_ids,
                "user": user_ids,

                "union": [],
                "scoreable_union": [],
                "final": [],

                "pure_bpr_top10": [],
            }

            return pd.DataFrame(
                columns=[
                    "app_id",
                    "bpr_score",
                ]
            )


        # =================================================
        # 3. BPR User
        # =================================================

        user_idx = (
            self.find_bpr_user_index(
                user_id
            )
        )


        if user_idx is None:

            self.missing_users += 1

            return pd.DataFrame(
                columns=[
                    "app_id",
                    "bpr_score",
                ]
            )


        # =================================================
        # 4. Candidate -> BPR Index
        # =================================================

        scoreable_app_ids = []

        scoreable_indices = []


        for app_id in union_ids:

            item_idx = (
                self.bpr_item_to_idx.get(
                    app_id
                )
            )

            if item_idx is None:

                self.missing_candidates += 1

                continue


            scoreable_app_ids.append(
                app_id
            )

            scoreable_indices.append(
                item_idx
            )


        if len(
            scoreable_app_ids
        ) == 0:

            return pd.DataFrame(
                columns=[
                    "app_id",
                    "bpr_score",
                ]
            )


        scoreable_app_ids = np.asarray(
            scoreable_app_ids
        )

        scoreable_indices = np.asarray(
            scoreable_indices,
            dtype=np.int64,
        )


        # =================================================
        # 5. BPR Score
        # =================================================

        user_factor = (
            self.bpr_model
            .user_factors[
                user_idx
            ]
        )


        candidate_factors = (
            self.bpr_model
            .item_factors[
                scoreable_indices
            ]
        )


        bpr_scores = (

            candidate_factors
            @
            user_factor
        )


        # =================================================
        # 6. BPR Ranking
        # =================================================

        order = np.argsort(
            -bpr_scores,
            kind="stable",
        )


        ranked_app_ids = (
            scoreable_app_ids[
                order
            ]
        )


        ranked_scores = (
            bpr_scores[
                order
            ]
        )


        n_actual = min(
            top_n,
            len(
                ranked_app_ids
            ),
        )


        final_app_ids = (
            ranked_app_ids[
                :n_actual
            ]
        )


        final_scores = (
            ranked_scores[
                :n_actual
            ]
        )


        # =================================================
        # Pure BPR Top-10
        #
        # Case3 BPR retrieval cache는 같은 BPR 모델의
        # 전체 item ranking 결과이므로 reference로 사용.
        # =================================================

        pure_bpr_top10 = (

            self.bpr
            .get(
                user_id,
                [],
            )
            [:TOP_N]
        )


        # =================================================
        # Log
        # =================================================

        self.candidate_log[
            user_id
        ] = {

            "item":
                item_ids,

            "bpr":
                bpr_ids,

            "content":
                content_ids,

            "user":
                user_ids,

            "union":
                union_ids,

            "scoreable_union":
                scoreable_app_ids.tolist(),

            "final":
                final_app_ids.tolist(),

            "pure_bpr_top10":
                pure_bpr_top10,
        }


        # =================================================
        # Candidate Pool
        # =================================================

        score_map = dict(
            zip(
                scoreable_app_ids.tolist(),
                bpr_scores.tolist(),
            )
        )


        final_rank_map = {

            app_id: rank + 1

            for rank, app_id
            in enumerate(
                final_app_ids.tolist()
            )
        }


        rows = []


        for app_id in union_ids:

            rows.append(
                {

                    "case_name":
                        self.case_name,

                    "user_id":
                        user_id,

                    "app_id":
                        app_id,


                    # -------------------------------------
                    # Candidate Source
                    # -------------------------------------

                    "from_item":
                        int(
                            app_id
                            in
                            item_rank
                        ),

                    "from_bpr":
                        int(
                            app_id
                            in
                            bpr_retrieval_rank
                        ),

                    "from_content":
                        int(
                            app_id
                            in
                            content_rank
                        ),

                    "from_user":
                        int(
                            app_id
                            in
                            user_rank
                        ),


                    # -------------------------------------
                    # Original Rank
                    # -------------------------------------

                    "item_rank":
                        item_rank.get(
                            app_id,
                            np.nan,
                        ),

                    "bpr_retrieval_rank":
                        bpr_retrieval_rank.get(
                            app_id,
                            np.nan,
                        ),

                    "content_rank":
                        content_rank.get(
                            app_id,
                            np.nan,
                        ),

                    "user_rank":
                        user_rank.get(
                            app_id,
                            np.nan,
                        ),


                    # -------------------------------------
                    # Final Ranker
                    # -------------------------------------

                    "bpr_rerank_score":
                        score_map.get(
                            app_id,
                            np.nan,
                        ),

                    "final_rank":
                        final_rank_map.get(
                            app_id,
                            np.nan,
                        ),

                    "selected_top10":
                        int(
                            app_id
                            in
                            final_rank_map
                        ),

                    "pure_bpr_top10":
                        int(
                            app_id
                            in
                            pure_bpr_top10
                        ),
                }
            )


        self.pool_logs.append(
            pd.DataFrame(
                rows
            )
        )


        return pd.DataFrame(
            {

                "app_id":
                    final_app_ids,

                "bpr_score":
                    final_scores,
            }
        )


# =========================================================
# Candidate Diagnostics
# =========================================================

def build_candidate_diagnostics(
    recommender,
    test_df,
    sampled_users,
):

    sampled_ids = set(
        sampled_users[
            "user_id"
        ]
    )


    positive_test = (

        test_df[
            test_df[
                "user_id"
            ].isin(
                sampled_ids
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


    for row in sampled_users.itertuples(
        index=False
    ):

        user_id = (
            row.user_id
        )


        truth = set(
            positive_test.get(
                user_id,
                [],
            )
        )


        if len(truth) == 0:
            continue


        log = (
            recommender
            .candidate_log
            .get(
                user_id,
                {}
            )
        )


        item_set = set(
            log.get(
                "item",
                [],
            )
        )

        bpr_set = set(
            log.get(
                "bpr",
                [],
            )
        )

        content_set = set(
            log.get(
                "content",
                [],
            )
        )

        user_set = set(
            log.get(
                "user",
                [],
            )
        )

        union_set = set(
            log.get(
                "union",
                [],
            )
        )

        scoreable_set = set(
            log.get(
                "scoreable_union",
                [],
            )
        )

        final_set = set(
            log.get(
                "final",
                [],
            )
        )

        pure_bpr_set = set(
            log.get(
                "pure_bpr_top10",
                [],
            )
        )


        n_test = len(
            truth
        )


        # =================================================
        # Hit Sets
        # =================================================

        item_hits = (
            item_set
            &
            truth
        )

        bpr_hits = (
            bpr_set
            &
            truth
        )

        content_hits = (
            content_set
            &
            truth
        )

        user_hits = (
            user_set
            &
            truth
        )

        union_hits = (
            union_set
            &
            truth
        )

        scoreable_hits = (
            scoreable_set
            &
            truth
        )


        rows.append(
            {

                "user_id":
                    user_id,

                "review_group":
                    row.review_group,

                "n_games":
                    row.n_games,

                "n_test":
                    n_test,


                # -----------------------------------------
                # Candidate Count
                # -----------------------------------------

                "item_candidate_count":
                    len(
                        item_set
                    ),

                "bpr_candidate_count":
                    len(
                        bpr_set
                    ),

                "content_candidate_count":
                    len(
                        content_set
                    ),

                "user_candidate_count":
                    len(
                        user_set
                    ),

                "union_candidate_count":
                    len(
                        union_set
                    ),

                "scoreable_union_count":
                    len(
                        scoreable_set
                    ),


                # -----------------------------------------
                # Candidate Recall
                # -----------------------------------------

                "item_candidate_recall":
                    len(
                        item_hits
                    )
                    /
                    n_test,

                "bpr_candidate_recall":
                    len(
                        bpr_hits
                    )
                    /
                    n_test,

                "content_candidate_recall":
                    len(
                        content_hits
                    )
                    /
                    n_test,

                "user_candidate_recall":
                    len(
                        user_hits
                    )
                    /
                    n_test,

                "union_candidate_recall":
                    len(
                        union_hits
                    )
                    /
                    n_test,

                "scoreable_union_recall":
                    len(
                        scoreable_hits
                    )
                    /
                    n_test,


                # -----------------------------------------
                # Candidate Hits
                # -----------------------------------------

                "item_candidate_hits":
                    len(
                        item_hits
                    ),

                "bpr_candidate_hits":
                    len(
                        bpr_hits
                    ),

                "content_candidate_hits":
                    len(
                        content_hits
                    ),

                "user_candidate_hits":
                    len(
                        user_hits
                    ),

                "union_candidate_hits":
                    len(
                        union_hits
                    ),


                # -----------------------------------------
                # Pure BPR 비교
                # -----------------------------------------

                "final_pure_bpr_overlap":
                    len(
                        final_set
                        &
                        pure_bpr_set
                    ),

                "final_exactly_pure_bpr":
                    int(
                        final_set
                        ==
                        pure_bpr_set
                        and
                        len(
                            final_set
                        )
                        ==
                        TOP_N
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# =========================================================
# Summary
# =========================================================

def make_summary(
    case_name,
    candidate_counts,
    eval_df,
    candidate_df,
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

        if total_recommended
        else 0.0
    )


    micro_recall = (

        total_hits
        /
        total_test

        if total_test
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

        "case_name":
            case_name,


        # ---------------------------------------------
        # Candidate Allocation
        # ---------------------------------------------

        "item_candidate_n":
            candidate_counts[
                "item"
            ],

        "bpr_candidate_n":
            candidate_counts[
                "bpr"
            ],

        "content_candidate_n":
            candidate_counts[
                "content"
            ],

        "user_candidate_n":
            candidate_counts[
                "user"
            ],


        # ---------------------------------------------
        # Final Metrics
        # ---------------------------------------------

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

        "hits":
            total_hits,

        "micro_precision_at_10":
            micro_precision,

        "micro_recall_at_10":
            micro_recall,

        "micro_f1_at_10":
            micro_f1,


        # ---------------------------------------------
        # Candidate Stage
        # ---------------------------------------------

        "item_candidate_recall":
            candidate_df[
                "item_candidate_recall"
            ].mean(),

        "bpr_candidate_recall":
            candidate_df[
                "bpr_candidate_recall"
            ].mean(),

        "content_candidate_recall":
            candidate_df[
                "content_candidate_recall"
            ].mean(),

        "user_candidate_recall":
            candidate_df[
                "user_candidate_recall"
            ].mean(),

        "union_candidate_recall":
            candidate_df[
                "union_candidate_recall"
            ].mean(),

        "scoreable_union_recall":
            candidate_df[
                "scoreable_union_recall"
            ].mean(),

        "avg_union_candidates":
            candidate_df[
                "union_candidate_count"
            ].mean(),


        # ---------------------------------------------
        # BPR Dominance Diagnostic
        # ---------------------------------------------

        "avg_final_pure_bpr_overlap":
            candidate_df[
                "final_pure_bpr_overlap"
            ].mean(),

        "pure_bpr_exact_user_ratio":
            candidate_df[
                "final_exactly_pure_bpr"
            ].mean(),
    }


# =========================================================
# Run One Case
# =========================================================

def run_case(
    case_name,
    candidate_counts,
    item_cache,
    user_cache,
    bpr_cache,
    content_cache,
    bpr_model,
    bpr_user_ids,
    bpr_item_ids,
    train_eval,
    test_eval,
    sampled_users,
):

    print(
        "\n"
        "=================================================="
    )

    print(
        f" Case 2 Weighted Retriever: "
        f"{case_name}"
    )

    print(
        "=================================================="
    )


    print(
        "Candidate Allocation"
    )

    print(
        f"Item    : "
        f"{candidate_counts['item']}"
    )

    print(
        f"BPR     : "
        f"{candidate_counts['bpr']}"
    )

    print(
        f"Content : "
        f"{candidate_counts['content']}"
    )

    print(
        f"User    : "
        f"{candidate_counts['user']}"
    )

    print(
        "\nUNION -> BPR Ranking -> Top10"
    )


    recommender = (
        WeightedMultiRetrieverBPRRanker(

            item_cache=
                item_cache,

            user_cache=
                user_cache,

            bpr_cache=
                bpr_cache,

            content_cache=
                content_cache,

            bpr_model=
                bpr_model,

            bpr_user_ids=
                bpr_user_ids,

            bpr_item_ids=
                bpr_item_ids,

            candidate_counts=
                candidate_counts,

            case_name=
                case_name,
        )
    )


    # =====================================================
    # Evaluation
    # =====================================================

    start = time.time()


    eval_df = (
        base.run_mf_evaluation(

            recommender=
                recommender,

            train_df=
                train_eval,

            test_df=
                test_eval,

            sampled_users=
                sampled_users,

            top_n=
                TOP_N,

            positive_only=
                True,
        )
    )


    elapsed = (
        time.time()
        -
        start
    )


    print(
        f"\n평가 시간: "
        f"{elapsed:.2f}초"
    )


    # =====================================================
    # Final Metrics
    # =====================================================

    group_summary = (
        base.print_evaluation_report(

            eval_df,

            top_n=
                TOP_N,
        )
    )


    # =====================================================
    # Candidate Diagnostics
    # =====================================================

    candidate_df = (
        build_candidate_diagnostics(

            recommender=
                recommender,

            test_df=
                test_eval,

            sampled_users=
                sampled_users,
        )
    )


    summary = (
        make_summary(

            case_name=
                case_name,

            candidate_counts=
                candidate_counts,

            eval_df=
                eval_df,

            candidate_df=
                candidate_df,
        )
    )


    summary[
        "evaluation_seconds"
    ] = elapsed


    # =====================================================
    # Console Diagnostics
    # =====================================================

    print(
        "\n===== Candidate Recall ====="
    )

    print(
        "Item:",
        f"{summary['item_candidate_recall']:.6f}"
    )

    print(
        "BPR:",
        f"{summary['bpr_candidate_recall']:.6f}"
    )

    print(
        "Content:",
        f"{summary['content_candidate_recall']:.6f}"
    )

    print(
        "User:",
        f"{summary['user_candidate_recall']:.6f}"
    )

    print(
        "UNION:",
        f"{summary['union_candidate_recall']:.6f}"
    )

    print(
        "Scoreable UNION:",
        f"{summary['scoreable_union_recall']:.6f}"
    )

    print(
        "평균 UNION 후보 수:",
        f"{summary['avg_union_candidates']:.2f}"
    )


    print(
        "\n===== Final BPR Ranking ====="
    )

    print(
        "Precision@10:",
        f"{summary['precision_at_10']:.6f}"
    )

    print(
        "Recall@10:",
        f"{summary['recall_at_10']:.6f}"
    )

    print(
        "HR@10:",
        f"{summary['hit_rate_at_10']:.6f}"
    )

    print(
        "NDCG@10:",
        f"{summary['ndcg_at_10']:.6f}"
    )

    print(
        "Hits:",
        summary[
            "hits"
        ]
    )


    print(
        "\n===== Pure BPR 영향 확인 ====="
    )

    print(
        "최종 Top10 중 평균 Pure BPR Top10 개수:",
        f"{summary['avg_final_pure_bpr_overlap']:.2f}"
    )

    print(
        "Pure BPR Top10과 완전히 같은 사용자 비율:",
        f"{summary['pure_bpr_exact_user_ratio']:.4f}"
    )


    # =====================================================
    # Candidate Pool
    # =====================================================

    if len(
        recommender.pool_logs
    ):

        pool_df = pd.concat(

            recommender.pool_logs,

            ignore_index=True,
        )

    else:

        pool_df = (
            pd.DataFrame()
        )


    # =====================================================
    # Save
    # =====================================================

    eval_path = (
        RESULT_DIR
        / f"{case_name}_eval.csv"
    )

    group_path = (
        RESULT_DIR
        / f"{case_name}_group_summary.csv"
    )

    candidate_path = (
        RESULT_DIR
        / f"{case_name}_candidate_diagnostics.csv"
    )

    pool_path = (
        RESULT_DIR
        / f"{case_name}_candidate_pool.csv"
    )


    eval_df.to_csv(
        eval_path,
        index=False,
    )

    group_summary.to_csv(
        group_path,
        index=False,
    )

    candidate_df.to_csv(
        candidate_path,
        index=False,
    )

    pool_df.to_csv(
        pool_path,
        index=False,
    )


    print(
        "\n저장:"
    )

    print(
        eval_path
    )

    print(
        group_path
    )

    print(
        candidate_path
    )

    print(
        pool_path
    )


    return (
        summary,
        eval_df,
    )


# =========================================================
# Main
# =========================================================

def main():

    total_start = (
        time.time()
    )


    print(
        "\n"
        "=================================================="
    )

    print(
        " Case 2 Weighted Multi-Retriever Experiment"
    )

    print(
        " Fusion Importance -> Candidate Allocation"
    )

    print(
        " Final Ranker = BPR"
    )

    print(
        "=================================================="
    )


    # =====================================================
    # 1. 평가 사용자
    # =====================================================

    sampled_users = (
        base.load_sampled_users()
    )


    # =====================================================
    # 2. 평가 subset
    # =====================================================

    full_split_holder = {}

    train_eval, test_eval = (
        base.load_or_create_eval_subset(

            sampled_users,

            full_split_holder,
        )
    )


    # =====================================================
    # 3. Retrieval Cache
    # =====================================================

    print(
        "\n===== Retrieval Cache ====="
    )


    cf_holder = {}


    item_cache = (
        base.build_item_cache(

            sampled_users,
            train_eval,
            full_split_holder,
            cf_holder,
        )
    )


    user_cache = (
        base.build_user_cache(

            sampled_users,
            train_eval,
            full_split_holder,
            cf_holder,
        )
    )


    bpr_cache = (
        base.build_bpr_cache(

            sampled_users,
            train_eval,
        )
    )


    _, item_ids = (
        base.load_bpr_mapping_only()
    )


    content_cache = (
        base.build_content_cache(

            sampled_users,
            train_eval,
            item_ids,
        )
    )


    base.print_cache_coverage(
        "Item",
        item_cache,
        sampled_users,
    )

    base.print_cache_coverage(
        "BPR",
        bpr_cache,
        sampled_users,
    )

    base.print_cache_coverage(
        "Content",
        content_cache,
        sampled_users,
    )

    base.print_cache_coverage(
        "User",
        user_cache,
        sampled_users,
    )


    # =====================================================
    # 4. BPR Model
    # =====================================================

    print(
        "\n===== BPR Ranker ====="
    )


    (
        model_path,
        mapping_path,
        _,
    ) = base.find_bpr_files()


    print(
        "Model:",
        model_path
    )

    print(
        "Mapping:",
        mapping_path
    )


    bpr_model = (
        BayesianPersonalizedRanking.load(
            str(
                model_path
            )
        )
    )


    mapping = np.load(
        mapping_path
    )


    bpr_user_ids = np.asarray(
        mapping[
            "user_ids"
        ]
    )

    bpr_item_ids = np.asarray(
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


    # =====================================================
    # 메모리 정리
    # =====================================================

    full_split_holder.clear()
    cf_holder.clear()

    gc.collect()


    # =====================================================
    # 5. Experiments
    # =====================================================

    summaries = []

    eval_results = {}


    for (
        case_name,
        candidate_counts,
    ) in EXPERIMENT_CASES.items():

        summary, eval_df = (
            run_case(

                case_name=
                    case_name,

                candidate_counts=
                    candidate_counts,

                item_cache=
                    item_cache,

                user_cache=
                    user_cache,

                bpr_cache=
                    bpr_cache,

                content_cache=
                    content_cache,

                bpr_model=
                    bpr_model,

                bpr_user_ids=
                    bpr_user_ids,

                bpr_item_ids=
                    bpr_item_ids,

                train_eval=
                    train_eval,

                test_eval=
                    test_eval,

                sampled_users=
                    sampled_users,
            )
        )


        summaries.append(
            summary
        )

        eval_results[
            case_name
        ] = eval_df


    # =====================================================
    # 6. Final Summary
    # =====================================================

    summary_df = (
        pd.DataFrame(
            summaries
        )
    )


    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )


    print(
        "\n"
        "=================================================="
    )

    print(
        " Final Comparison"
    )

    print(
        "=================================================="
    )


    display_columns = [

        "case_name",

        "item_candidate_n",
        "bpr_candidate_n",
        "content_candidate_n",
        "user_candidate_n",

        "avg_union_candidates",
        "union_candidate_recall",

        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",

        "hits",

        "avg_final_pure_bpr_overlap",
        "pure_bpr_exact_user_ratio",
    ]


    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )


    # =====================================================
    # 7. User 추가 효과 직접 비교
    # =====================================================

    no_user = (
        eval_results[
            "without_user"
        ]
    )


    with_user = (
        eval_results[
            "with_user"
        ]
    )


    compare_columns = [

        "user_id",
        "precision",
        "recall",
        "hits",
        "hit",
        "ndcg",
    ]


    left = (

        no_user[
            compare_columns
        ]

        .rename(
            columns={

                "precision":
                    "without_user_precision",

                "recall":
                    "without_user_recall",

                "hits":
                    "without_user_hits",

                "hit":
                    "without_user_hit",

                "ndcg":
                    "without_user_ndcg",
            }
        )
    )


    right = (

        with_user[
            compare_columns
        ]

        .rename(
            columns={

                "precision":
                    "with_user_precision",

                "recall":
                    "with_user_recall",

                "hits":
                    "with_user_hits",

                "hit":
                    "with_user_hit",

                "ndcg":
                    "with_user_ndcg",
            }
        )
    )


    comparison = (
        left.merge(

            right,

            on=
                "user_id",

            how=
                "inner",
        )
    )


    comparison[
        "delta_precision"
    ] = (

        comparison[
            "with_user_precision"
        ]

        -

        comparison[
            "without_user_precision"
        ]
    )


    comparison[
        "delta_recall"
    ] = (

        comparison[
            "with_user_recall"
        ]

        -

        comparison[
            "without_user_recall"
        ]
    )


    comparison[
        "delta_hits"
    ] = (

        comparison[
            "with_user_hits"
        ]

        -

        comparison[
            "without_user_hits"
        ]
    )


    comparison[
        "delta_ndcg"
    ] = (

        comparison[
            "with_user_ndcg"
        ]

        -

        comparison[
            "without_user_ndcg"
        ]
    )


    comparison.to_csv(
        COMPARE_PATH,
        index=False,
    )


    print(
        "\n===== User-Based 추가 효과 ====="
    )


    print(
        "Δ Precision:",
        f"{comparison['delta_precision'].mean():.6f}"
    )

    print(
        "Δ Recall:",
        f"{comparison['delta_recall'].mean():.6f}"
    )

    print(
        "Δ Hits:",
        int(
            comparison[
                "delta_hits"
            ].sum()
        )
    )

    print(
        "Δ NDCG:",
        f"{comparison['delta_ndcg'].mean():.6f}"
    )


    print(
        "\nSummary 저장:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        COMPARE_PATH
    )


    print(
        "\n전체 실행 시간:",
        f"{time.time() - total_start:.1f}초"
    )


if __name__ == "__main__":
    main()