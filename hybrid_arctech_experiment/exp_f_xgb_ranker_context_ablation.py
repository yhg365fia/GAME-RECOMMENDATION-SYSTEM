# hybrid_arctech_experiment/exp_f_xgb_ranker_context_ablation.py
# ============================================================
# XGBoost Ranker - Context Feature Ablation
#
# 목적
# ------------------------------------------------------------
# Exp2에서 만든 14-feature cache를 그대로 재사용하여
# 신규 Context Feature가 실제 Validation 성능에 얼마나
# 기여하는지 통제된 Ablation으로 확인한다.
#
# 비교 실험
# 1) Baseline 8
# 2) Full 14
# 3) Full 14 - item_popularity
# 4) Full 14 - Source Group
#    - retriever_count
#    - is_bpr_candidate
#    - is_content_candidate
#    - is_user_candidate
# 5) Full 14 - user_interaction_count
#
# 중요 원칙
# ------------------------------------------------------------
# - Candidate / Feature cache는 동일
# - LTR User 1000명 동일
# - Train 800 / Validation 200 split 동일
# - XGB hyperparameter 동일
# - Early Stopping 동일
# - Feature Set만 변경
# - Final 400명은 실험 선택에 사용하지 않음
# - Validation NDCG@10으로 최종 Feature Set을 고른 뒤
#   선택된 모델 하나만 Final 400명에 평가
#
# MAP / Popularity / Novelty는 여기서 평가하지 않는다.
# 최종 main 통합 실행에서 확인한다.
# ============================================================

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBRanker


# ============================================================
# Project Root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from data_split import load_mf_split


# ============================================================
# Config
# ============================================================

RANDOM_STATE = 42

TOP_N = 10
VALID_RATIO = 0.20
EARLY_STOPPING_ROUNDS = 30


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


CONTEXT_FEATURES = [
    *SOURCE_FEATURES,
    "item_popularity",
    "user_interaction_count",
]


FULL_FEATURES = [
    *BASE_FEATURES,
    *CONTEXT_FEATURES,
]


EXPERIMENTS = {
    # 기준
    "baseline_8": BASE_FEATURES,

    # 현재 Exp2
    "full_14": FULL_FEATURES,

    # Ablation 1
    "minus_popularity": [
        feature
        for feature in FULL_FEATURES
        if feature != "item_popularity"
    ],

    # Ablation 2
    "minus_source_group": [
        feature
        for feature in FULL_FEATURES
        if feature not in SOURCE_FEATURES
    ],

    # Ablation 3
    "minus_user_activity": [
        feature
        for feature in FULL_FEATURES
        if feature != "user_interaction_count"
    ],
}


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

ABLATION_MODEL_DIR = (
    SAVED_MODEL_DIR
    / "xgb_context_ablation"
)

ABLATION_RESULT_DIR = (
    RESULT_DIR
    / "xgb_context_ablation"
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ABLATION_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ABLATION_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FINAL_EVAL_USERS_PATH = (
    RESULT_DIR
    / "hybrid_sampled_users.csv"
)

LTR_USERS_PATH = (
    CACHE_DIR
    / "ltr_train_users_1000.csv"
)

EXPANDED_FEATURE_CACHE_PATH = (
    CACHE_DIR
    / "xgb_exp2_context_features.parquet"
)


SUMMARY_PATH = (
    ABLATION_RESULT_DIR
    / "ablation_validation_summary.csv"
)

BEST_IMPORTANCE_PATH = (
    ABLATION_RESULT_DIR
    / "best_feature_importance.csv"
)

FINAL_RESULT_PATH = (
    ABLATION_RESULT_DIR
    / "best_model_final400_eval.csv"
)

FINAL_SUMMARY_PATH = (
    ABLATION_RESULT_DIR
    / "best_model_final400_summary.csv"
)


# ============================================================
# XGB Hyperparameter
#
# Regularization Tuning Best A 그대로 유지
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
# Utility
# ============================================================

def validate_required_files():

    required_paths = [
        EXPANDED_FEATURE_CACHE_PATH,
        LTR_USERS_PATH,
        FINAL_EVAL_USERS_PATH,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                "\n필수 파일이 없습니다:\n"
                f"{path}\n\n"
                "먼저 Exp2 Context Feature 실험을 실행해 "
                "xgb_exp2_context_features.parquet cache를 생성하세요."
            )


def validate_feature_columns(
    feature_df: pd.DataFrame,
):

    required_columns = {
        "user_id",
        "app_id",
        "label",
        *FULL_FEATURES,
    }

    missing = sorted(
        required_columns
        -
        set(
            feature_df.columns
        )
    )

    if missing:

        raise ValueError(
            "Expanded Feature Cache에 "
            f"필요한 column이 없습니다: {missing}"
        )


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

    groups = (
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
        sorted_df,
        X,
        y,
        groups,
    )


def create_ranker():

    return XGBRanker(
        **XGB_PARAMS
    )


def train_one_experiment(
    experiment_name: str,
    features: list[str],
    train_rank_df: pd.DataFrame,
    valid_rank_df: pd.DataFrame,
):

    print(
        "\n"
        +
        "=" * 78
    )

    print(
        f" EXPERIMENT: {experiment_name}"
    )

    print(
        "=" * 78
    )

    print(
        f"Feature count: {len(features)}"
    )

    for index, feature in enumerate(
        features,
        start=1,
    ):

        print(
            f"{index:02d}. {feature}"
        )


    (
        _,
        X_train,
        y_train,
        train_group,
    ) = prepare_ranker_data(
        train_rank_df,
        features,
    )


    (
        _,
        X_valid,
        y_valid,
        valid_group,
    ) = prepare_ranker_data(
        valid_rank_df,
        features,
    )


    print(
        "\n===== Dataset ====="
    )

    print(
        "Train users:",
        len(
            train_group
        ),
    )

    print(
        "Train rows :",
        len(
            X_train
        ),
    )

    print(
        "Valid users:",
        len(
            valid_group
        ),
    )

    print(
        "Valid rows :",
        len(
            X_valid
        ),
    )

    print(
        "Positive labels(train):",
        int(
            y_train.sum()
        ),
    )

    print(
        "Positive labels(valid):",
        int(
            y_valid.sum()
        ),
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


    best_iteration = int(
        model.best_iteration
    )

    best_score = float(
        model.best_score
    )


    print(
        "\n===== Result ====="
    )

    print(
        "Best iteration:",
        best_iteration,
    )

    print(
        "Best validation NDCG@10:",
        f"{best_score:.9f}",
    )

    print(
        "Train time:",
        f"{elapsed:.2f} sec",
    )


    importance_df = (
        pd.DataFrame(
            {
                "feature":
                    features,

                "importance":
                    model
                    .feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    importance_df[
        "rank"
    ] = (
        np.arange(
            1,
            len(
                importance_df
            )
            +
            1
        )
    )


    importance_df = (
        importance_df[
            [
                "rank",
                "feature",
                "importance",
            ]
        ]
    )


    print(
        "\n===== Feature Importance ====="
    )

    print(
        importance_df
        .to_string(
            index=False
        )
    )


    importance_path = (
        ABLATION_RESULT_DIR
        /
        f"{experiment_name}_importance.csv"
    )


    importance_df.to_csv(
        importance_path,
        index=False,
    )


    model_path = (
        ABLATION_MODEL_DIR
        /
        f"{experiment_name}.json"
    )


    model.save_model(
        model_path
    )


    return {
        "experiment":
            experiment_name,

        "n_features":
            len(
                features
            ),

        "features":
            features,

        "best_iteration":
            best_iteration,

        "validation_ndcg_at_10":
            best_score,

        "train_seconds":
            elapsed,

        "model":
            model,

        "model_path":
            str(
                model_path
            ),

        "importance_df":
            importance_df,
    }


# ============================================================
# Final 400 Evaluation
# ============================================================

def evaluate_final_400(
    model,
    features: list[str],
    final_df: pd.DataFrame,
    positive_test_dict: dict,
):

    rows = []


    for user_id, group in (
        final_df
        .groupby(
            "user_id",
            sort=False,
        )
    ):

        group = (
            group
            .copy()
        )


        X_user = (
            group[
                features
            ]
            .astype(
                np.float32
            )
        )


        predictions = (
            model.predict(
                X_user
            )
        )


        group[
            "xgb_score"
        ] = predictions


        ranked = (
            group
            .sort_values(
                "xgb_score",
                ascending=False,
                kind="stable",
            )
            .head(
                TOP_N
            )
        )


        recommended = (
            ranked[
                "app_id"
            ]
            .tolist()
        )


        relevant = (
            positive_test_dict.get(
                user_id,
                set(),
            )
        )


        hits = sum(
            app_id in relevant
            for app_id
            in recommended
        )


        precision = (
            hits
            /
            len(
                recommended
            )
            if recommended
            else 0.0
        )


        recall = (
            hits
            /
            len(
                relevant
            )
            if relevant
            else 0.0
        )


        hit_rate = float(
            hits > 0
        )


        dcg = 0.0

        for rank, app_id in enumerate(
            recommended,
            start=1,
        ):

            if app_id in relevant:

                dcg += (
                    1.0
                    /
                    np.log2(
                        rank + 1
                    )
                )


        ideal_hits = min(
            len(
                relevant
            ),
            TOP_N,
        )


        idcg = sum(
            1.0
            /
            np.log2(
                rank + 1
            )

            for rank
            in range(
                1,
                ideal_hits + 1,
            )
        )


        ndcg = (
            dcg
            /
            idcg

            if idcg > 0

            else 0.0
        )


        rows.append(
            {
                "user_id":
                    user_id,

                "precision":
                    precision,

                "recall":
                    recall,

                "hit_rate":
                    hit_rate,

                "ndcg":
                    ndcg,

                "hits":
                    hits,

                "n_test":
                    len(
                        relevant
                    ),

                "n_recommended":
                    len(
                        recommended
                    ),
            }
        )


    result = pd.DataFrame(
        rows
    )


    summary = {
        "n_users":
            len(
                result
            ),

        "precision_at_10":
            float(
                result[
                    "precision"
                ].mean()
            ),

        "recall_at_10":
            float(
                result[
                    "recall"
                ].mean()
            ),

        "hit_rate_at_10":
            float(
                result[
                    "hit_rate"
                ].mean()
            ),

        "ndcg_at_10":
            float(
                result[
                    "ndcg"
                ].mean()
            ),

        "hits":
            int(
                result[
                    "hits"
                ].sum()
            ),
    }


    return (
        result,
        summary,
    )


# ============================================================
# Main
# ============================================================

def main():

    total_start = time.perf_counter()


    print(
        "=" * 78
    )

    print(
        " XGBoost Context Feature Ablation"
    )

    print(
        " Candidate = BPR56 + Content39 + User5"
    )

    print(
        " Cache     = Exp2 14-feature cache reuse"
    )

    print(
        " Selection = Validation NDCG@10 only"
    )

    print(
        " Final 400 = Best experiment only"
    )

    print(
        "=" * 78
    )


    # ========================================================
    # 1. Required Files
    # ========================================================

    validate_required_files()


    # ========================================================
    # 2. Load User Lists
    # ========================================================

    ltr_users = pd.read_csv(
        LTR_USERS_PATH
    )


    final_eval_users = pd.read_csv(
        FINAL_EVAL_USERS_PATH
    )


    ltr_ids = set(
        ltr_users[
            "user_id"
        ]
        .tolist()
    )


    final_ids = set(
        final_eval_users[
            "user_id"
        ]
        .tolist()
    )


    overlap = (
        ltr_ids
        &
        final_ids
    )


    if overlap:

        raise ValueError(
            "LTR users와 Final users가 겹칩니다: "
            f"{len(overlap)}"
        )


    print(
        "\nLTR Users  :",
        len(
            ltr_ids
        ),
    )

    print(
        "Final Users:",
        len(
            final_ids
        ),
    )


    # ========================================================
    # 3. Expanded Feature Cache
    # ========================================================

    print(
        "\n===== Expanded Feature Cache ====="
    )

    print(
        EXPANDED_FEATURE_CACHE_PATH
    )


    feature_df = pd.read_parquet(
        EXPANDED_FEATURE_CACHE_PATH
    )


    print(
        "Feature DF:",
        feature_df.shape,
    )


    validate_feature_columns(
        feature_df
    )


    # ========================================================
    # 4. LTR / Final Feature Rows
    # ========================================================

    ltr_df = (
        feature_df[
            feature_df[
                "user_id"
            ]
            .isin(
                ltr_ids
            )
        ]
        .copy()
    )


    final_df = (
        feature_df[
            feature_df[
                "user_id"
            ]
            .isin(
                final_ids
            )
        ]
        .copy()
    )


    print(
        "LTR Feature Rows  :",
        len(
            ltr_df
        ),
    )

    print(
        "Final Feature Rows:",
        len(
            final_df
        ),
    )


    # ========================================================
    # 5. Same 800 / 200 Split
    # ========================================================

    unique_ltr_users = (
        ltr_users[
            "user_id"
        ]
        .to_numpy()
    )


    (
        train_users,
        valid_users,
    ) = train_test_split(

        unique_ltr_users,

        test_size=
            VALID_RATIO,

        random_state=
            RANDOM_STATE,
    )


    train_rank_df = (
        ltr_df[
            ltr_df[
                "user_id"
            ]
            .isin(
                train_users
            )
        ]
        .copy()
    )


    valid_rank_df = (
        ltr_df[
            ltr_df[
                "user_id"
            ]
            .isin(
                valid_users
            )
        ]
        .copy()
    )


    print(
        "\n===== Fixed User Split ====="
    )

    print(
        "Train users:",
        len(
            train_users
        ),
    )

    print(
        "Valid users:",
        len(
            valid_users
        ),
    )

    print(
        "Train rows :",
        len(
            train_rank_df
        ),
    )

    print(
        "Valid rows :",
        len(
            valid_rank_df
        ),
    )


    # ========================================================
    # 6. Ablation Loop
    # ========================================================

    experiment_results = []


    for experiment_name, features in (
        EXPERIMENTS.items()
    ):

        result = train_one_experiment(

            experiment_name=
                experiment_name,

            features=
                features,

            train_rank_df=
                train_rank_df,

            valid_rank_df=
                valid_rank_df,
        )


        experiment_results.append(
            result
        )


    # ========================================================
    # 7. Validation Summary
    # ========================================================

    summary_df = pd.DataFrame(
        [
            {
                "experiment":
                    result[
                        "experiment"
                    ],

                "n_features":
                    result[
                        "n_features"
                    ],

                "best_iteration":
                    result[
                        "best_iteration"
                    ],

                "validation_ndcg_at_10":
                    result[
                        "validation_ndcg_at_10"
                    ],

                "train_seconds":
                    result[
                        "train_seconds"
                    ],
            }

            for result
            in experiment_results
        ]
    )


    full14_score = float(
        summary_df.loc[
            summary_df[
                "experiment"
            ]
            ==
            "full_14",
            "validation_ndcg_at_10",
        ]
        .iloc[0]
    )


    baseline8_score = float(
        summary_df.loc[
            summary_df[
                "experiment"
            ]
            ==
            "baseline_8",
            "validation_ndcg_at_10",
        ]
        .iloc[0]
    )


    summary_df[
        "delta_vs_full14"
    ] = (
        summary_df[
            "validation_ndcg_at_10"
        ]
        -
        full14_score
    )


    summary_df[
        "delta_vs_baseline8"
    ] = (
        summary_df[
            "validation_ndcg_at_10"
        ]
        -
        baseline8_score
    )


    summary_df[
        "rank"
    ] = (
        summary_df[
            "validation_ndcg_at_10"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )


    summary_df = (
        summary_df
        .sort_values(
            [
                "validation_ndcg_at_10",
                "n_features",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )


    print(
        "\n"
        +
        "=" * 78
    )

    print(
        " VALIDATION ABLATION SUMMARY"
    )

    print(
        "=" * 78
    )


    print(
        summary_df[
            [
                "rank",
                "experiment",
                "n_features",
                "best_iteration",
                "validation_ndcg_at_10",
                "delta_vs_full14",
                "delta_vs_baseline8",
                "train_seconds",
            ]
        ]
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 8. Select Best by Validation Only
    # ========================================================

    best_experiment_name = (
        summary_df
        .iloc[0][
            "experiment"
        ]
    )


    best_result = next(
        result

        for result
        in experiment_results

        if (
            result[
                "experiment"
            ]
            ==
            best_experiment_name
        )
    )


    best_model = (
        best_result[
            "model"
        ]
    )


    best_features = (
        best_result[
            "features"
        ]
    )


    best_importance_df = (
        best_result[
            "importance_df"
        ]
    )


    best_importance_df.to_csv(
        BEST_IMPORTANCE_PATH,
        index=False,
    )


    print(
        "\n"
        +
        "=" * 78
    )

    print(
        " SELECTED BY VALIDATION"
    )

    print(
        "=" * 78
    )

    print(
        "Best Experiment:",
        best_experiment_name,
    )

    print(
        "Validation NDCG@10:",
        f"{best_result['validation_ndcg_at_10']:.9f}",
    )

    print(
        "Feature Count:",
        len(
            best_features
        ),
    )

    print(
        "Features:"
    )

    for feature in best_features:

        print(
            " -",
            feature,
        )


    # ========================================================
    # 9. Load Test only after selection
    #
    # Final 400은 여기까지 실험 선택에 전혀 사용하지 않음
    # ========================================================

    print(
        "\n===== MF Test Load for FINAL 400 ====="
    )


    _, test_df = load_mf_split()


    positive_test = (
        test_df[
            (
                test_df[
                    "user_id"
                ]
                .isin(
                    final_ids
                )
            )
            &
            (
                test_df[
                    "is_recommended"
                ]
                ==
                True
            )
        ]
    )


    positive_test_dict = (
        positive_test
        .groupby(
            "user_id"
        )[
            "app_id"
        ]
        .apply(
            lambda values:
                set(
                    values.tolist()
                )
        )
        .to_dict()
    )


    # ========================================================
    # 10. FINAL 400 - Best Model only
    # ========================================================

    print(
        "\n"
        +
        "=" * 78
    )

    print(
        " FINAL 400 USERS - SELECTED MODEL ONLY"
    )

    print(
        "=" * 78
    )


    (
        final_result_df,
        final_summary,
    ) = evaluate_final_400(

        model=
            best_model,

        features=
            best_features,

        final_df=
            final_df,

        positive_test_dict=
            positive_test_dict,
    )


    final_result_df.to_csv(
        FINAL_RESULT_PATH,
        index=False,
    )


    final_summary_df = pd.DataFrame(
        [
            {
                "experiment":
                    best_experiment_name,

                "n_features":
                    len(
                        best_features
                    ),

                "validation_ndcg_at_10":
                    best_result[
                        "validation_ndcg_at_10"
                    ],

                **final_summary,
            }
        ]
    )


    final_summary_df.to_csv(
        FINAL_SUMMARY_PATH,
        index=False,
    )


    print(
        f"Selected Experiment : "
        f"{best_experiment_name}"
    )

    print(
        f"Validation NDCG@10  : "
        f"{best_result['validation_ndcg_at_10']:.6f}"
    )

    print(
        f"Precision@10        : "
        f"{final_summary['precision_at_10']:.4f}"
    )

    print(
        f"Recall@10           : "
        f"{final_summary['recall_at_10']:.4f}"
    )

    print(
        f"HR@10               : "
        f"{final_summary['hit_rate_at_10']:.4f}"
    )

    print(
        f"NDCG@10             : "
        f"{final_summary['ndcg_at_10']:.4f}"
    )

    print(
        f"Hits                : "
        f"{final_summary['hits']}"
    )


    # ========================================================
    # 11. Interpretation Helpers
    # ========================================================

    print(
        "\n"
        +
        "=" * 78
    )

    print(
        " ABLATION INTERPRETATION"
    )

    print(
        "=" * 78
    )


    def score_of(name):

        return float(
            summary_df.loc[
                summary_df[
                    "experiment"
                ]
                ==
                name,
                "validation_ndcg_at_10",
            ]
            .iloc[0]
        )


    popularity_drop = (
        full14_score
        -
        score_of(
            "minus_popularity"
        )
    )


    source_drop = (
        full14_score
        -
        score_of(
            "minus_source_group"
        )
    )


    user_activity_drop = (
        full14_score
        -
        score_of(
            "minus_user_activity"
        )
    )


    print(
        "Full14 - (-Popularity)      : "
        f"{popularity_drop:+.6f}"
    )

    print(
        "Full14 - (-Source Group)    : "
        f"{source_drop:+.6f}"
    )

    print(
        "Full14 - (-User Activity)   : "
        f"{user_activity_drop:+.6f}"
    )


    print(
        "\n해석:"
    )

    print(
        "- 값이 +이면: 해당 Feature/Group을 제거했을 때 성능이 내려감"
    )

    print(
        "             -> Full 모델에서 도움이 된 신호"
    )

    print(
        "- 값이 0 근처면: 영향이 작음"
    )

    print(
        "- 값이 -이면: 제거했더니 성능이 오름"
    )

    print(
        "             -> 잡음/불필요 Feature 가능성"
    )


    # ========================================================
    # 12. Saved Files
    # ========================================================

    print(
        "\n===== Saved ====="
    )

    print(
        "Validation Summary:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\nBest Importance:"
    )

    print(
        BEST_IMPORTANCE_PATH
    )

    print(
        "\nFinal 400 Result:"
    )

    print(
        FINAL_RESULT_PATH
    )

    print(
        "\nFinal 400 Summary:"
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
