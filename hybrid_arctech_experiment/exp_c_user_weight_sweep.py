import sys
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Project Root / 기존 Case 3 Ablation 코드 재사용
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
CURRENT_DIR = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import exp_c_fusion_ablation as base


# =========================================================
# Experiment Config
# =========================================================

# ---------------------------------------------------------
# 이번 실험에서 확인할 User-Based weight
# ---------------------------------------------------------
#
# 핵심:
# User weight만 변화시키고
# Item : BPR : Content의 상대적 비율
#
#     0.50 : 0.20 : 0.15
#
# 은 계속 유지한다.
#
# User=0.15일 때:
#     Item=0.50
#     User=0.15
#     BPR=0.20
#     Content=0.15
#
# 즉 기존 Full Case 3와 동일하다.
# ---------------------------------------------------------

USER_WEIGHT_VALUES = [
    0.000,
    0.025,
    0.050,
    0.100,
    0.150,
]

# User를 제외한 기존 상대 비율
NON_USER_BASE_WEIGHTS = {
    "item": 0.50,
    "bpr": 0.20,
    "content": 0.15,
}

NON_USER_BASE_TOTAL = sum(NON_USER_BASE_WEIGHTS.values())  # 0.85


# =========================================================
# Output Paths
# =========================================================

SWEEP_DIR = base.RESULT_DIR / "case3_user_weight_sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = SWEEP_DIR / "case3_user_weight_sweep_summary.csv"


# =========================================================
# Weight Generator
# =========================================================

def make_user_weight_sweep_weights(user_weight):
    """
    User-Based weight를 지정하고,
    나머지 Item / BPR / Content는 기존 상대 비율을 유지한다.

    예:
        User = 0

        남은 1.0을
        Item:BPR:Content = 0.50:0.20:0.15
        비율대로 재분배

        ->
        Item    = 0.588235
        BPR     = 0.235294
        Content = 0.176471


        User = 0.05

        남은 0.95를 같은 비율로 분배

        ->
        Item    ≈ 0.558824
        BPR     ≈ 0.223529
        Content ≈ 0.167647


        User = 0.15

        ->
        기존 Case 3 Full

        Item    = 0.50
        User    = 0.15
        BPR     = 0.20
        Content = 0.15
    """

    user_weight = float(user_weight)

    if not 0.0 <= user_weight < 1.0:
        raise ValueError(
            f"user_weight는 0 이상 1 미만이어야 합니다: {user_weight}"
        )

    remaining = 1.0 - user_weight

    item_weight = (
        remaining
        * NON_USER_BASE_WEIGHTS["item"]
        / NON_USER_BASE_TOTAL
    )

    bpr_weight = (
        remaining
        * NON_USER_BASE_WEIGHTS["bpr"]
        / NON_USER_BASE_TOTAL
    )

    content_weight = (
        remaining
        * NON_USER_BASE_WEIGHTS["content"]
        / NON_USER_BASE_TOTAL
    )

    weights = {
        "item": item_weight,
        "user": user_weight,
        "bpr": bpr_weight,
        "content": content_weight,
    }

    total = sum(weights.values())

    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Weight 합이 1.0이 아닙니다: {total}"
        )

    return weights


# =========================================================
# User Weight Sweep용 Rank Fusion
# =========================================================

class UserWeightSweepFusionRecommender:
    """
    기존 Cached Rank Fusion과 동일한 방식.

    차이점:
    weight가 0인 모델은

    1. Candidate union
    2. Fusion score
    3. Tie-break

    모두에서 완전히 제외한다.

    따라서 User weight=0이면
    User-Based는 추천 결과에 어떠한 영향도 주지 않는다.
    """

    MODEL_ORDER = [
        "item",
        "user",
        "bpr",
        "content",
    ]

    def __init__(
        self,
        item_cache,
        user_cache,
        bpr_cache,
        content_cache,
    ):
        self.item = base.cache_to_dict(item_cache)
        self.user = base.cache_to_dict(user_cache)
        self.bpr = base.cache_to_dict(bpr_cache)
        self.content = base.cache_to_dict(content_cache)

        self.logs = {}

    @staticmethod
    def rank_score(rank):
        return (
            base.MODEL_TOP_N - rank + 1
        ) / base.MODEL_TOP_N

    def recommend(
        self,
        user_id,
        app_id_list=None,
        top_n=10,
    ):
        # -------------------------------------------------
        # 현재 global weights
        # -------------------------------------------------

        weights = {
            "item": base.ITEM_WEIGHT,
            "user": base.USER_WEIGHT,
            "bpr": base.BPR_WEIGHT,
            "content": base.CONTENT_WEIGHT,
        }

        # weight가 실제로 존재하는 모델만 사용
        active_models = [
            model
            for model in self.MODEL_ORDER
            if weights[model] > 0
        ]

        model_lists = {
            "item": self.item.get(
                user_id, []
            )[:base.MODEL_TOP_N],

            "user": self.user.get(
                user_id, []
            )[:base.MODEL_TOP_N],

            "bpr": self.bpr.get(
                user_id, []
            )[:base.MODEL_TOP_N],

            "content": self.content.get(
                user_id, []
            )[:base.MODEL_TOP_N],
        }

        candidates = {}

        def ensure(app_id):
            if app_id not in candidates:
                candidates[app_id] = {
                    "app_id": app_id,

                    "item_rank": np.nan,
                    "user_rank": np.nan,
                    "bpr_rank": np.nan,
                    "content_rank": np.nan,

                    "item_rank_score": 0.0,
                    "user_rank_score": 0.0,
                    "bpr_rank_score": 0.0,
                    "content_rank_score": 0.0,
                }

            return candidates[app_id]

        # -------------------------------------------------
        # Candidate Union
        #
        # weight=0 모델은 후보 생성에서도 제외
        # -------------------------------------------------

        for model_name in active_models:

            rec_list = model_lists[model_name]

            for rank, app_id in enumerate(
                rec_list,
                start=1,
            ):
                row = ensure(app_id)

                row[f"{model_name}_rank"] = rank

                row[
                    f"{model_name}_rank_score"
                ] = self.rank_score(rank)

        # -------------------------------------------------
        # 후보가 하나도 없는 경우
        # -------------------------------------------------

        if not candidates:

            self.logs[user_id] = {
                "item": [],
                "user": [],
                "bpr": [],
                "content": [],
                "fusion": [],
            }

            return pd.DataFrame(
                columns=[
                    "app_id",
                    "fusion_score",
                ]
            )

        fusion_df = pd.DataFrame(
            candidates.values()
        )

        # -------------------------------------------------
        # Weighted Rank Fusion
        # -------------------------------------------------

        fusion_df["fusion_score"] = 0.0

        for model_name in active_models:

            fusion_df["fusion_score"] += (
                weights[model_name]
                * fusion_df[
                    f"{model_name}_rank_score"
                ]
            )

        # -------------------------------------------------
        # Tie-break
        #
        # 기존 Case 3의
        #
        # Item -> User -> BPR -> Content
        #
        # 순서는 유지하지만,
        # weight=0 모델은 아예 제외한다.
        #
        # 따라서 User=0이면
        #
        # Item -> BPR -> Content
        #
        # 로 동점 처리.
        # -------------------------------------------------

        tie_columns = []

        for model_name in active_models:

            tie_col = f"_{model_name}_tie"

            fusion_df[tie_col] = (
                fusion_df[
                    f"{model_name}_rank"
                ]
                .fillna(999999)
            )

            tie_columns.append(tie_col)

        sort_columns = [
            "fusion_score",
            *tie_columns,
        ]

        ascending = [
            False,
            *([True] * len(tie_columns)),
        ]

        fusion_df = fusion_df.sort_values(
            sort_columns,
            ascending=ascending,
            kind="stable",
        ).reset_index(drop=True)

        # -------------------------------------------------
        # Top-N
        # -------------------------------------------------

        result = (
            fusion_df
            .head(top_n)
            .drop(
                columns=tie_columns
            )
        )

        # -------------------------------------------------
        # Diagnostics log
        #
        # inactive 모델은 빈 리스트로 기록
        # -------------------------------------------------

        logs = {}

        for model_name in self.MODEL_ORDER:

            if model_name in active_models:
                logs[model_name] = (
                    model_lists[model_name][
                        :base.TOP_N
                    ]
                )
            else:
                logs[model_name] = []

        logs["fusion"] = (
            result["app_id"].tolist()
        )

        self.logs[user_id] = logs

        return result


# =========================================================
# One Sweep Evaluation
# =========================================================

def run_one_user_weight_case(
    user_weight,
    weights,
    item_cache,
    user_cache,
    bpr_cache,
    content_cache,
    train_eval,
    test_eval,
    sampled_users,
):
    """
    특정 User weight 하나에 대해

        Cached Retrieval
            ↓
        Weighted Rank Fusion
            ↓
        Top-10
            ↓
        Evaluation

    을 수행한다.
    """

    base.apply_global_weights(weights)

    case_name = (
        f"user_w_{user_weight:.3f}"
        .replace(".", "p")
    )

    print("\n" + "=" * 65)
    print(
        f" User Weight Sweep: "
        f"User={user_weight:.3f}"
    )
    print("=" * 65)

    print(
        f"Item    : "
        f"{base.ITEM_WEIGHT:.6f}"
    )
    print(
        f"User    : "
        f"{base.USER_WEIGHT:.6f}"
    )
    print(
        f"BPR     : "
        f"{base.BPR_WEIGHT:.6f}"
    )
    print(
        f"Content : "
        f"{base.CONTENT_WEIGHT:.6f}"
    )

    total_weight = (
        base.ITEM_WEIGHT
        + base.USER_WEIGHT
        + base.BPR_WEIGHT
        + base.CONTENT_WEIGHT
    )

    if not np.isclose(
        total_weight,
        1.0,
    ):
        raise ValueError(
            "Fusion weight 합이 "
            f"1.0이 아닙니다: "
            f"{total_weight}"
        )

    # -----------------------------------------------------
    # Fusion Model
    # -----------------------------------------------------

    fusion_model = (
        UserWeightSweepFusionRecommender(
            item_cache=item_cache,
            user_cache=user_cache,
            bpr_cache=bpr_cache,
            content_cache=content_cache,
        )
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    eval_start = time.time()

    eval_df = base.run_mf_evaluation(
        recommender=fusion_model,
        train_df=train_eval,
        test_df=test_eval,
        sampled_users=sampled_users,
        top_n=base.TOP_N,
        positive_only=True,
    )

    elapsed = (
        time.time()
        - eval_start
    )

    print(
        f"\n평가 시간: "
        f"{elapsed:.2f}초"
    )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    group_summary = (
        base.print_evaluation_report(
            eval_df,
            top_n=base.TOP_N,
        )
    )

    summary = base.make_summary(
        eval_df
    )

    summary[
        "experiment"
    ] = "case3_user_weight_sweep"

    summary[
        "case_name"
    ] = case_name

    summary[
        "item_weight"
    ] = base.ITEM_WEIGHT

    summary[
        "user_weight"
    ] = base.USER_WEIGHT

    summary[
        "bpr_weight"
    ] = base.BPR_WEIGHT

    summary[
        "content_weight"
    ] = base.CONTENT_WEIGHT

    summary[
        "evaluation_seconds"
    ] = elapsed

    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------

    diagnostic_df = (
        base.build_fusion_diagnostics(
            fusion_model=fusion_model,
            test_df=test_eval,
            sampled_users=sampled_users,
        )
    )

    if len(diagnostic_df):

        summary[
            "fusion_recovered_hits"
        ] = int(
            diagnostic_df[
                "fusion_recovered_hits"
            ].sum()
        )

        summary[
            "fusion_lost_hits"
        ] = int(
            diagnostic_df[
                "fusion_lost_hits"
            ].sum()
        )

        summary[
            "fusion_delta_hits_vs_item"
        ] = int(
            diagnostic_df[
                "fusion_delta_hits"
            ].sum()
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    eval_path = (
        SWEEP_DIR
        / f"{case_name}_eval.csv"
    )

    group_path = (
        SWEEP_DIR
        / f"{case_name}_group_summary.csv"
    )

    diagnostic_path = (
        SWEEP_DIR
        / f"{case_name}_diagnostics.csv"
    )

    eval_df.to_csv(
        eval_path,
        index=False,
    )

    group_summary.to_csv(
        group_path,
        index=False,
    )

    diagnostic_df.to_csv(
        diagnostic_path,
        index=False,
    )

    print("\n저장:")
    print(eval_path)
    print(group_path)
    print(diagnostic_path)

    print(
        "\n결과: "
        f"P@10="
        f"{summary['precision_at_10']:.6f}"
        " | "
        f"R@10="
        f"{summary['recall_at_10']:.6f}"
        " | "
        f"HR@10="
        f"{summary['hit_rate_at_10']:.6f}"
        " | "
        f"NDCG@10="
        f"{summary['ndcg_at_10']:.6f}"
        " | "
        f"Hits="
        f"{summary['hits']}"
    )

    return summary


# =========================================================
# Main
# =========================================================

def main():

    total_start = time.time()

    print(
        "\n"
        "=================================================="
    )
    print(
        " Case 3 - User-Based Weight Sweep"
    )
    print(
        " Item:BPR:Content Relative Ratio Fixed"
    )
    print(
        f" Each Top-{base.MODEL_TOP_N}"
        f" -> Fusion"
        f" -> Top-{base.TOP_N}"
    )
    print(
        "=================================================="
    )

    print(
        "\nUser Weight Values:"
    )

    for value in USER_WEIGHT_VALUES:
        print(
            f" - {value:.3f}"
        )

    print(
        "\nItem:BPR:Content의 상대 비율은 "
        "0.50:0.20:0.15로 고정합니다."
    )

    # -----------------------------------------------------
    # 1. Lazy holders
    # -----------------------------------------------------

    full_split_holder = {}
    cf_holder = {}

    # -----------------------------------------------------
    # 2. 평가 사용자
    # -----------------------------------------------------

    sampled_users = (
        base.load_sampled_users()
    )

    train_eval, test_eval = (
        base.load_or_create_eval_subset(
            sampled_users,
            full_split_holder,
        )
    )

    # -----------------------------------------------------
    # 3. Retrieval Cache 준비
    # -----------------------------------------------------

    print(
        "\n===== Retrieval Cache 준비 ====="
    )

    item_cache = (
        base.build_item_cache(
            sampled_users,
            train_eval,
            full_split_holder,
            cf_holder,
        )
    )

    base.print_cache_coverage(
        "Item",
        item_cache,
        sampled_users,
    )

    user_cache = (
        base.build_user_cache(
            sampled_users,
            train_eval,
            full_split_holder,
            cf_holder,
        )
    )

    base.print_cache_coverage(
        "User",
        user_cache,
        sampled_users,
    )

    bpr_cache = (
        base.build_bpr_cache(
            sampled_users,
            train_eval,
        )
    )

    base.print_cache_coverage(
        "BPR",
        bpr_cache,
        sampled_users,
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
        "Content",
        content_cache,
        sampled_users,
    )

    # -----------------------------------------------------
    # 4. 메모리 정리
    # -----------------------------------------------------

    full_split_holder.clear()
    cf_holder.clear()

    gc.collect()

    # -----------------------------------------------------
    # 5. User Weight Sweep
    # -----------------------------------------------------

    summaries = []

    for user_weight in USER_WEIGHT_VALUES:

        weights = (
            make_user_weight_sweep_weights(
                user_weight
            )
        )

        summary = (
            run_one_user_weight_case(
                user_weight=user_weight,
                weights=weights,
                item_cache=item_cache,
                user_cache=user_cache,
                bpr_cache=bpr_cache,
                content_cache=content_cache,
                train_eval=train_eval,
                test_eval=test_eval,
                sampled_users=sampled_users,
            )
        )

        summaries.append(
            summary
        )

    # -----------------------------------------------------
    # 6. Summary
    # -----------------------------------------------------

    summary_df = pd.DataFrame(
        summaries
    )

    # User=0 기준
    no_user_row = (
        summary_df[
            np.isclose(
                summary_df["user_weight"],
                0.0,
            )
        ]
        .iloc[0]
    )

    # 기존 Full Case 3 기준
    full_row = (
        summary_df[
            np.isclose(
                summary_df["user_weight"],
                0.15,
            )
        ]
        .iloc[0]
    )

    # -----------------------------------------------------
    # User=0 대비 Delta
    # -----------------------------------------------------

    summary_df[
        "delta_precision_vs_user0"
    ] = (
        summary_df["precision_at_10"]
        - no_user_row["precision_at_10"]
    )

    summary_df[
        "delta_recall_vs_user0"
    ] = (
        summary_df["recall_at_10"]
        - no_user_row["recall_at_10"]
    )

    summary_df[
        "delta_hit_rate_vs_user0"
    ] = (
        summary_df["hit_rate_at_10"]
        - no_user_row["hit_rate_at_10"]
    )

    summary_df[
        "delta_ndcg_vs_user0"
    ] = (
        summary_df["ndcg_at_10"]
        - no_user_row["ndcg_at_10"]
    )

    summary_df[
        "delta_hits_vs_user0"
    ] = (
        summary_df["hits"]
        - no_user_row["hits"]
    )

    # -----------------------------------------------------
    # Full(User=0.15) 대비 Delta
    # -----------------------------------------------------

    summary_df[
        "delta_precision_vs_full"
    ] = (
        summary_df["precision_at_10"]
        - full_row["precision_at_10"]
    )

    summary_df[
        "delta_recall_vs_full"
    ] = (
        summary_df["recall_at_10"]
        - full_row["recall_at_10"]
    )

    summary_df[
        "delta_hit_rate_vs_full"
    ] = (
        summary_df["hit_rate_at_10"]
        - full_row["hit_rate_at_10"]
    )

    summary_df[
        "delta_ndcg_vs_full"
    ] = (
        summary_df["ndcg_at_10"]
        - full_row["ndcg_at_10"]
    )

    summary_df[
        "delta_hits_vs_full"
    ] = (
        summary_df["hits"]
        - full_row["hits"]
    )

    # -----------------------------------------------------
    # 저장
    # -----------------------------------------------------

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # 보기 좋은 최종 비교표
    # -----------------------------------------------------

    display_columns = [
        "user_weight",
        "item_weight",
        "bpr_weight",
        "content_weight",
        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",
        "hits",
        "delta_hits_vs_user0",
        "delta_ndcg_vs_user0",
    ]

    print(
        "\n"
        "=================================================="
    )
    print(
        " User Weight Sweep Final Comparison"
    )
    print(
        "=================================================="
    )

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nSummary 저장:"
    )

    print(
        SUMMARY_PATH
    )

    # -----------------------------------------------------
    # 실제 적용된 weight 확인
    # -----------------------------------------------------

    print(
        "\n===== Weight 확인 ====="
    )

    for user_weight in USER_WEIGHT_VALUES:

        weights = (
            make_user_weight_sweep_weights(
                user_weight
            )
        )

        print(
            f"User={user_weight:.3f} | "
            f"Item={weights['item']:.4f}, "
            f"User={weights['user']:.4f}, "
            f"BPR={weights['bpr']:.4f}, "
            f"Content={weights['content']:.4f}"
        )

    print(
        f"\n전체 실행 시간: "
        f"{time.time() - total_start:.1f}초"
    )

    print(
        "Retrieval cache가 이미 존재하면 "
        "대부분의 시간은 5회의 Fusion 평가에만 사용됩니다."
    )


if __name__ == "__main__":
    main()