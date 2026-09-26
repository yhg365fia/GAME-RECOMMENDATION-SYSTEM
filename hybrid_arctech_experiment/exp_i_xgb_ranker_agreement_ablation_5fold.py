# hybrid_arctech_experiment/exp_i_xgb_ranker_agreement_ablation_5fold.py
# ============================================================
# XGBoost Ranker - Agreement Feature Ablation (User-Level 5-Fold CV)
#
# 목적
# ------------------------------------------------------------
# Full14 Context Feature Set을 기준으로,
# Agreement Feature 5개를 추가한 Full19에서
# 각 Agreement Feature를 하나씩 제거하는 Ablation을 수행한다.
#
# 이후:
# 1) 각 Agreement Feature의 제거 효과를 5-Fold 평균 NDCG@10으로 측정
# 2) "제거하면 성능이 떨어지는" Feature = 도움이 되는 Agreement 신호
# 3) 도움이 되는 Agreement Feature만 Full14에 추가한 Reduced 모델 생성
# 4) Full14 vs Full19 vs Reduced를 같은 5-Fold 기준으로 최종 비교
#
# 중요 원칙
# ------------------------------------------------------------
# - User-level 5-Fold
# - 같은 fold에서 모든 모델 비교
# - Final 400 users 사용 안 함
# - Candidate / Feature cache 재사용
# - XGB hyperparameter 고정
# - Metric = Validation NDCG@10
#
# 판단 방식
# ------------------------------------------------------------
# Full19 - AblatedModel > 0
#   -> 해당 feature를 제거했더니 성능 하락
#   -> Full19에서 도움 되는 feature
#
# Full19 - AblatedModel ~= 0
#   -> 영향 작음
#
# Full19 - AblatedModel < 0
#   -> 제거했더니 성능 상승
#   -> 잡음 / 불필요 가능성
#
# Reduced 모델은 "평균 제거 효과가 양수"인 Agreement Feature만 사용.
# ============================================================

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from xgboost import XGBRanker


# ============================================================
# Project Root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Config
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 30

# "유효하다" 판정의 최소 평균 제거 효과.
# 기본값 0.0:
# 제거했을 때 평균 NDCG가 아주 조금이라도 내려가면 retain.
#
# 너무 민감하다고 느끼면 0.0005 또는 0.001로 올릴 수 있음.
RETAIN_THRESHOLD = 0.0


# ============================================================
# Feature Sets
# ============================================================

BASE_FEATURES = [
    "item_score_norm",
    "bpr_score_norm",
    "content_score_norm",
    "user_score_norm",

    "item_rank",
    "bpr_rank",
    "content_rank",
    "user_rank",
]


SOURCE_FEATURES = [
    "retriever_count",
    "is_bpr_candidate",
    "is_content_candidate",
    "is_user_candidate",
]


FULL_14_FEATURES = [
    *BASE_FEATURES,
    *SOURCE_FEATURES,
    "item_popularity",
    "user_interaction_count",
]


AGREEMENT_FEATURES = [
    "is_multi_retriever",
    "is_bpr_content_candidate",
    "is_bpr_user_candidate",
    "is_content_user_candidate",
    "is_all_three_candidate",
]


FULL_19_FEATURES = [
    *FULL_14_FEATURES,
    *AGREEMENT_FEATURES,
]


# ============================================================
# Paths
# ============================================================

SAVED_MODEL_DIR = (
    ROOT
    / "models"
    / "saved_model"
)

RESULT_DIR = (
    SAVED_MODEL_DIR
    / "results"
)

CACHE_DIR = (
    SAVED_MODEL_DIR
    / "ltr_cache"
    / "xgb_exp1"
)

ABLATION_DIR = (
    RESULT_DIR
    / "xgb_agreement_ablation_5fold"
)


ABLATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


LTR_USERS_PATH = (
    CACHE_DIR
    / "ltr_train_users_1000.csv"
)


EXP2_FEATURE_CACHE_PATH = (
    CACHE_DIR
    / "xgb_exp2_context_features.parquet"
)


EXP3_FEATURE_CACHE_PATH = (
    CACHE_DIR
    / "xgb_exp3_agreement_features.parquet"
)


RAW_RESULTS_PATH = (
    ABLATION_DIR
    / "agreement_ablation_5fold_raw.csv"
)


MODEL_SUMMARY_PATH = (
    ABLATION_DIR
    / "agreement_ablation_5fold_summary.csv"
)


AGREEMENT_EFFECT_PATH = (
    ABLATION_DIR
    / "agreement_feature_effects.csv"
)


REDUCED_COMPARISON_PATH = (
    ABLATION_DIR
    / "reduced_model_5fold_comparison.csv"
)


FINAL_SUMMARY_PATH = (
    ABLATION_DIR
    / "final_model_selection_summary.csv"
)


# ============================================================
# XGB Best A
# ============================================================

XGB_PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",

    "n_estimators": 500,
    "learning_rate": 0.05,

    "max_depth": 4,
    "min_child_weight": 1,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    "reg_lambda": 1.0,
    "reg_alpha": 0.0,

    "random_state": RANDOM_STATE,
    "tree_method": "hist",

    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
}


# ============================================================
# Agreement Feature 생성
# ============================================================

def add_agreement_features(
    feature_df: pd.DataFrame,
):

    df = feature_df.copy()

    required = [
        "retriever_count",
        "is_bpr_candidate",
        "is_content_candidate",
        "is_user_candidate",
    ]

    missing = [
        column
        for column
        in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Agreement Feature 생성에 필요한 column이 없습니다: "
            f"{missing}"
        )


    bpr = (
        df[
            "is_bpr_candidate"
        ]
        .astype(
            np.int8
        )
    )


    content = (
        df[
            "is_content_candidate"
        ]
        .astype(
            np.int8
        )
    )


    user = (
        df[
            "is_user_candidate"
        ]
        .astype(
            np.int8
        )
    )


    df[
        "is_multi_retriever"
    ] = (
        df[
            "retriever_count"
        ]
        .ge(
            2
        )
        .astype(
            np.int8
        )
    )


    df[
        "is_bpr_content_candidate"
    ] = (
        bpr
        *
        content
    ).astype(
        np.int8
    )


    df[
        "is_bpr_user_candidate"
    ] = (
        bpr
        *
        user
    ).astype(
        np.int8
    )


    df[
        "is_content_user_candidate"
    ] = (
        content
        *
        user
    ).astype(
        np.int8
    )


    df[
        "is_all_three_candidate"
    ] = (
        bpr
        *
        content
        *
        user
    ).astype(
        np.int8
    )


    return df


# ============================================================
# Cache Load
# ============================================================

def load_feature_cache():

    if EXP3_FEATURE_CACHE_PATH.exists():

        print(
            "\n===== Exp3 Agreement Feature Cache Load ====="
        )

        print(
            EXP3_FEATURE_CACHE_PATH
        )

        feature_df = pd.read_parquet(
            EXP3_FEATURE_CACHE_PATH
        )


    elif EXP2_FEATURE_CACHE_PATH.exists():

        print(
            "\n===== Exp2 Context Feature Cache Load ====="
        )

        print(
            EXP2_FEATURE_CACHE_PATH
        )

        feature_df = pd.read_parquet(
            EXP2_FEATURE_CACHE_PATH
        )

        print(
            "\nAgreement Feature 생성"
        )

        feature_df = add_agreement_features(
            feature_df
        )

        feature_df.to_parquet(
            EXP3_FEATURE_CACHE_PATH,
            index=False,
        )

        print(
            "Exp3 cache 저장:"
        )

        print(
            EXP3_FEATURE_CACHE_PATH
        )


    else:

        raise FileNotFoundError(
            "\nFeature cache를 찾을 수 없습니다.\n"
            f"Exp2: {EXP2_FEATURE_CACHE_PATH}\n"
            f"Exp3: {EXP3_FEATURE_CACHE_PATH}"
        )


    return feature_df


# ============================================================
# Validation
# ============================================================

def validate_columns(
    feature_df: pd.DataFrame,
):

    required = {
        "user_id",
        "app_id",
        "label",
        *FULL_19_FEATURES,
    }


    missing = sorted(
        required
        -
        set(
            feature_df.columns
        )
    )


    if missing:

        raise ValueError(
            "Feature cache에 필요한 column이 없습니다: "
            f"{missing}"
        )


# ============================================================
# Ranker Data
# ============================================================

def prepare_ranker_data(
    df: pd.DataFrame,
    features: list[str],
):

    sorted_df = (
        df
        .sort_values(
            [
                "user_id",
                "app_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


    X = (
        sorted_df[
            features
        ]
        .astype(
            np.float32
        )
    )


    y = (
        sorted_df[
            "label"
        ]
        .astype(
            np.float32
        )
    )


    group = (
        sorted_df
        .groupby(
            "user_id",
            sort=False,
        )
        .size()
        .to_numpy(
            dtype=np.int32
        )
    )


    return (
        X,
        y,
        group,
    )


def create_ranker():

    return XGBRanker(
        **XGB_PARAMS
    )


# ============================================================
# One Fold / One Model
# ============================================================

def run_one_fold(
    *,
    fold_number: int,
    model_name: str,
    features: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
):

    (
        X_train,
        y_train,
        train_group,
    ) = prepare_ranker_data(
        train_df,
        features,
    )


    (
        X_valid,
        y_valid,
        valid_group,
    ) = prepare_ranker_data(
        valid_df,
        features,
    )


    model = create_ranker()


    start = time.perf_counter()


    model.fit(

        X_train,
        y_train,

        group=
            train_group,

        eval_set=[
            (
                X_valid,
                y_valid,
            )
        ],

        eval_group=[
            valid_group
        ],

        verbose=False,
    )


    elapsed = (
        time.perf_counter()
        -
        start
    )


    result = {
        "fold":
            fold_number,

        "model":
            model_name,

        "n_features":
            len(
                features
            ),

        "best_iteration":
            int(
                model.best_iteration
            ),

        "validation_ndcg_at_10":
            float(
                model.best_score
            ),

        "train_seconds":
            elapsed,
    }


    print(
        f"{model_name:<32} "
        f"| features={len(features):>2} "
        f"| iter={result['best_iteration']:>3} "
        f"| NDCG@10={result['validation_ndcg_at_10']:.9f}"
    )


    return result


# ============================================================
# Build Ablation Experiments
# ============================================================

def build_ablation_experiments():

    experiments = {
        "full14":
            FULL_14_FEATURES,

        "full19":
            FULL_19_FEATURES,
    }


    for feature in AGREEMENT_FEATURES:

        experiment_name = (
            "full19_minus_"
            +
            feature
        )


        experiments[
            experiment_name
        ] = [
            candidate_feature

            for candidate_feature
            in FULL_19_FEATURES

            if candidate_feature != feature
        ]


    return experiments


# ============================================================
# Run CV for Arbitrary Experiments
# ============================================================

def run_user_level_5fold(
    *,
    feature_df: pd.DataFrame,
    user_ids: np.ndarray,
    experiments: dict[str, list[str]],
    label: str,
):

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        f" {label}"
    )

    print(
        "=" * 90
    )


    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


    rows = []


    for fold_index, (
        train_user_idx,
        valid_user_idx,
    ) in enumerate(
        kfold.split(
            user_ids
        ),
        start=1,
    ):

        print(
            "\n"
            +
            "-" * 90
        )

        print(
            f" FOLD {fold_index}/{N_SPLITS}"
        )

        print(
            "-" * 90
        )


        train_users = set(
            user_ids[
                train_user_idx
            ]
            .tolist()
        )


        valid_users = set(
            user_ids[
                valid_user_idx
            ]
            .tolist()
        )


        train_df = (
            feature_df[
                feature_df[
                    "user_id"
                ]
                .isin(
                    train_users
                )
            ]
            .copy()
        )


        valid_df = (
            feature_df[
                feature_df[
                    "user_id"
                ]
                .isin(
                    valid_users
                )
            ]
            .copy()
        )


        for model_name, features in (
            experiments.items()
        ):

            result = run_one_fold(

                fold_number=
                    fold_index,

                model_name=
                    model_name,

                features=
                    features,

                train_df=
                    train_df,

                valid_df=
                    valid_df,
            )


            rows.append(
                result
            )


    return pd.DataFrame(
        rows
    )


# ============================================================
# Summaries
# ============================================================

def summarize_models(
    result_df: pd.DataFrame,
):

    rows = []


    for model_name, group in (
        result_df
        .groupby(
            "model",
            sort=False,
        )
    ):

        scores = (
            group[
                "validation_ndcg_at_10"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        rows.append(
            {
                "model":
                    model_name,

                "n_features":
                    int(
                        group[
                            "n_features"
                        ]
                        .iloc[0]
                    ),

                "mean_ndcg_at_10":
                    float(
                        np.mean(
                            scores
                        )
                    ),

                "std_ndcg_at_10":
                    float(
                        np.std(
                            scores,
                            ddof=1,
                        )
                    ),

                "min_ndcg_at_10":
                    float(
                        np.min(
                            scores
                        )
                    ),

                "max_ndcg_at_10":
                    float(
                        np.max(
                            scores
                        )
                    ),

                "mean_best_iteration":
                    float(
                        group[
                            "best_iteration"
                        ]
                        .mean()
                    ),

                "mean_train_seconds":
                    float(
                        group[
                            "train_seconds"
                        ]
                        .mean()
                    ),
            }
        )


    summary_df = pd.DataFrame(
        rows
    )


    summary_df = (
        summary_df
        .sort_values(
            "mean_ndcg_at_10",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    summary_df[
        "rank"
    ] = np.arange(
        1,
        len(
            summary_df
        )
        +
        1
    )


    return (
        summary_df[
            [
                "rank",
                "model",
                "n_features",
                "mean_ndcg_at_10",
                "std_ndcg_at_10",
                "min_ndcg_at_10",
                "max_ndcg_at_10",
                "mean_best_iteration",
                "mean_train_seconds",
            ]
        ]
    )


# ============================================================
# Agreement Effect
# ============================================================

def calculate_agreement_effects(
    summary_df: pd.DataFrame,
):

    full19_mean = float(
        summary_df.loc[
            summary_df[
                "model"
            ]
            ==
            "full19",
            "mean_ndcg_at_10",
        ]
        .iloc[0]
    )


    rows = []


    for feature in AGREEMENT_FEATURES:

        ablation_name = (
            "full19_minus_"
            +
            feature
        )


        ablated_mean = float(
            summary_df.loc[
                summary_df[
                    "model"
                ]
                ==
                ablation_name,
                "mean_ndcg_at_10",
            ]
            .iloc[0]
        )


        # positive:
        # Full19가 ablated보다 높음
        # -> 제거 시 성능 하락
        # -> feature가 도움
        effect = (
            full19_mean
            -
            ablated_mean
        )


        if effect > RETAIN_THRESHOLD:

            decision = "retain"

            interpretation = (
                "제거 시 평균 NDCG 하락 -> 도움"
            )

        elif effect < -RETAIN_THRESHOLD:

            decision = "remove"

            interpretation = (
                "제거 시 평균 NDCG 상승 -> 잡음 가능"
            )

        else:

            decision = "neutral"

            interpretation = (
                "평균 영향 매우 작음"
            )


        rows.append(
            {
                "feature":
                    feature,

                "full19_mean_ndcg":
                    full19_mean,

                "ablated_mean_ndcg":
                    ablated_mean,

                "effect_full19_minus_ablated":
                    effect,

                "decision":
                    decision,

                "interpretation":
                    interpretation,
            }
        )


    effect_df = pd.DataFrame(
        rows
    )


    effect_df = (
        effect_df
        .sort_values(
            "effect_full19_minus_ablated",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    return effect_df


# ============================================================
# Main
# ============================================================

def main():

    total_start = time.perf_counter()


    print(
        "=" * 90
    )

    print(
        " XGBoost Agreement Feature Ablation - User-Level 5-Fold CV"
    )

    print(
        " Baseline: Full14"
    )

    print(
        " Expanded: Full19"
    )

    print(
        " Final 400 users are NOT used"
    )

    print(
        "=" * 90
    )


    # ========================================================
    # 1. Load LTR Users
    # ========================================================

    if not LTR_USERS_PATH.exists():

        raise FileNotFoundError(
            f"LTR user file이 없습니다: {LTR_USERS_PATH}"
        )


    ltr_users_df = pd.read_csv(
        LTR_USERS_PATH
    )


    if "user_id" not in ltr_users_df.columns:

        raise ValueError(
            "LTR user file에 user_id column이 없습니다."
        )


    user_ids = (
        ltr_users_df[
            "user_id"
        ]
        .drop_duplicates()
        .to_numpy()
    )


    print(
        "\nLTR Unique Users:",
        len(
            user_ids
        ),
    )


    # ========================================================
    # 2. Load Feature Cache
    # ========================================================

    feature_df = load_feature_cache()


    print(
        "Feature DF:",
        feature_df.shape,
    )


    validate_columns(
        feature_df
    )


    ltr_user_set = set(
        user_ids.tolist()
    )


    feature_df = (
        feature_df[
            feature_df[
                "user_id"
            ]
            .isin(
                ltr_user_set
            )
        ]
        .copy()
    )


    print(
        "LTR Feature Rows:",
        len(
            feature_df
        ),
    )


    # ========================================================
    # 3. Agreement Ablation Experiments
    # ========================================================

    experiments = (
        build_ablation_experiments()
    )


    print(
        "\n===== Experiment List ====="
    )


    for name, features in (
        experiments.items()
    ):

        print(
            f"{name:<40} -> {len(features)} features"
        )


    raw_df = run_user_level_5fold(

        feature_df=
            feature_df,

        user_ids=
            user_ids,

        experiments=
            experiments,

        label=
            "AGREEMENT FEATURE ABLATION",
    )


    raw_df.to_csv(
        RAW_RESULTS_PATH,
        index=False,
    )


    # ========================================================
    # 4. Summary
    # ========================================================

    summary_df = summarize_models(
        raw_df
    )


    summary_df.to_csv(
        MODEL_SUMMARY_PATH,
        index=False,
    )


    print(
        "\n"
        +
        "=" * 90
    )

    print(
        " AGREEMENT ABLATION SUMMARY"
    )

    print(
        "=" * 90
    )


    print(
        summary_df
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 5. Agreement Feature Effects
    # ========================================================

    effect_df = calculate_agreement_effects(
        summary_df
    )


    effect_df.to_csv(
        AGREEMENT_EFFECT_PATH,
        index=False,
    )


    print(
        "\n"
        +
        "=" * 90
    )

    print(
        " AGREEMENT FEATURE EFFECT"
    )

    print(
        "=" * 90
    )


    print(
        effect_df
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 6. Build Reduced Agreement Model
    # ========================================================

    retained_agreement_features = (
        effect_df.loc[
            effect_df[
                "decision"
            ]
            ==
            "retain",
            "feature",
        ]
        .tolist()
    )


    removed_agreement_features = [
        feature

        for feature
        in AGREEMENT_FEATURES

        if feature
        not in retained_agreement_features
    ]


    reduced_features = [
        *FULL_14_FEATURES,
        *retained_agreement_features,
    ]


    print(
        "\n"
        +
        "=" * 90
    )

    print(
        " REDUCED AGREEMENT MODEL"
    )

    print(
        "=" * 90
    )


    print(
        "Retain threshold:",
        RETAIN_THRESHOLD,
    )


    print(
        "\nRetained Agreement Features:"
    )


    if retained_agreement_features:

        for feature in retained_agreement_features:

            print(
                " +",
                feature,
            )

    else:

        print(
            " (none)"
        )


    print(
        "\nRemoved Agreement Features:"
    )


    if removed_agreement_features:

        for feature in removed_agreement_features:

            print(
                " -",
                feature,
            )

    else:

        print(
            " (none)"
        )


    print(
        "\nReduced Feature Count:",
        len(
            reduced_features
        ),
    )


    # ========================================================
    # 7. Final CV Comparison
    # ========================================================

    reduced_experiments = {
        "full14":
            FULL_14_FEATURES,

        "full19":
            FULL_19_FEATURES,

        "reduced":
            reduced_features,
    }


    reduced_raw_df = run_user_level_5fold(

        feature_df=
            feature_df,

        user_ids=
            user_ids,

        experiments=
            reduced_experiments,

        label=
            "FINAL: FULL14 vs FULL19 vs REDUCED",
    )


    reduced_summary_df = summarize_models(
        reduced_raw_df
    )


    reduced_summary_df.to_csv(
        REDUCED_COMPARISON_PATH,
        index=False,
    )


    print(
        "\n"
        +
        "=" * 90
    )

    print(
        " FINAL REDUCED MODEL COMPARISON"
    )

    print(
        "=" * 90
    )


    print(
        reduced_summary_df
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 8. Final Selection
    # ========================================================

    best_row = (
        reduced_summary_df
        .iloc[0]
    )


    best_model = (
        best_row[
            "model"
        ]
    )


    best_mean = float(
        best_row[
            "mean_ndcg_at_10"
        ]
    )


    full14_mean = float(
        reduced_summary_df.loc[
            reduced_summary_df[
                "model"
            ]
            ==
            "full14",
            "mean_ndcg_at_10",
        ]
        .iloc[0]
    )


    full19_mean = float(
        reduced_summary_df.loc[
            reduced_summary_df[
                "model"
            ]
            ==
            "full19",
            "mean_ndcg_at_10",
        ]
        .iloc[0]
    )


    reduced_mean = float(
        reduced_summary_df.loc[
            reduced_summary_df[
                "model"
            ]
            ==
            "reduced",
            "mean_ndcg_at_10",
        ]
        .iloc[0]
    )


    selection_df = pd.DataFrame(
        [
            {
                "selected_model":
                    best_model,

                "selected_mean_ndcg_at_10":
                    best_mean,

                "full14_mean_ndcg_at_10":
                    full14_mean,

                "full19_mean_ndcg_at_10":
                    full19_mean,

                "reduced_mean_ndcg_at_10":
                    reduced_mean,

                "reduced_feature_count":
                    len(
                        reduced_features
                    ),

                "retained_agreement_features":
                    ",".join(
                        retained_agreement_features
                    ),

                "removed_agreement_features":
                    ",".join(
                        removed_agreement_features
                    ),

                "retain_threshold":
                    RETAIN_THRESHOLD,
            }
        ]
    )


    selection_df.to_csv(
        FINAL_SUMMARY_PATH,
        index=False,
    )


    print(
        "\n"
        +
        "=" * 90
    )

    print(
        " FINAL DECISION"
    )

    print(
        "=" * 90
    )


    print(
        f"Full14 Mean  : {full14_mean:.9f}"
    )

    print(
        f"Full19 Mean  : {full19_mean:.9f}"
    )

    print(
        f"Reduced Mean : {reduced_mean:.9f}"
    )


    print(
        "\nSelected Model:",
        best_model,
    )


    # ========================================================
    # 9. Important Warning
    # ========================================================

    print(
        "\n===== Interpretation Note ====="
    )


    print(
        "Agreement Ablation과 Reduced 모델 선택은 "
        "동일한 5-Fold CV 기준으로 수행되었습니다."
    )


    print(
        "Final 400 users는 이 스크립트에서 사용하지 않았습니다."
    )


    print(
        "Reduced가 Full14보다 높으면 Agreement 일부를 채택할 근거가 생깁니다."
    )


    print(
        "Reduced가 Full14와 비슷하거나 낮으면 "
        "더 단순한 Full14를 최종 선택하는 것이 합리적입니다."
    )


    # ========================================================
    # 10. Saved
    # ========================================================

    print(
        "\n===== Saved ====="
    )


    print(
        "Raw Ablation:"
    )

    print(
        RAW_RESULTS_PATH
    )


    print(
        "\nAblation Summary:"
    )

    print(
        MODEL_SUMMARY_PATH
    )


    print(
        "\nAgreement Effects:"
    )

    print(
        AGREEMENT_EFFECT_PATH
    )


    print(
        "\nReduced Comparison:"
    )

    print(
        REDUCED_COMPARISON_PATH
    )


    print(
        "\nFinal Selection:"
    )

    print(
        FINAL_SUMMARY_PATH
    )


    print(
        "\nTotal Runtime:"
        f" {time.perf_counter() - total_start:.1f} sec"
    )


if __name__ == "__main__":
    main()
