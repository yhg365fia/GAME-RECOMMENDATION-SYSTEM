# hybrid_arctech_experiment/exp_h_xgb_ranker_user_5fold_cv.py
# ============================================================
# XGBoost Ranker - User-Level 5-Fold Cross Validation
#
# 목적
# ------------------------------------------------------------
# Full14 Context Features와
# Full19 Agreement Features를
# 동일한 1,000명 LTR 사용자에 대해
# "사용자 단위" 5-Fold Cross Validation으로 비교한다.
#
# 중요:
# - 한 사용자의 candidate rows가 Train/Valid에 섞이지 않음
# - 각 fold에서 약 800 users Train / 200 users Validation
# - Full14 / Full19는 완전히 동일한 fold 사용
# - XGB hyperparameter 동일
# - Final 400 users는 이 실험에서 절대 사용하지 않음
#
# 판단 기준
# ------------------------------------------------------------
# 1순위: 5-Fold Mean Validation NDCG@10
# 2순위: Fold win count / paired delta 참고
# Mean이 사실상 같다면 더 단순한 Full14를 선택하는 것이 합리적
#
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


FEATURE_SETS = {
    "full14": FULL_14_FEATURES,
    "full19": FULL_19_FEATURES,
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

CV_RESULT_DIR = (
    RESULT_DIR
    / "xgb_user_5fold_cv"
)


CV_RESULT_DIR.mkdir(
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


FOLD_RESULT_PATH = (
    CV_RESULT_DIR
    / "full14_vs_full19_5fold_results.csv"
)


SUMMARY_PATH = (
    CV_RESULT_DIR
    / "full14_vs_full19_5fold_summary.csv"
)


PAIRED_RESULT_PATH = (
    CV_RESULT_DIR
    / "full14_vs_full19_paired_fold_comparison.csv"
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

    """
    Exp3 cache가 없을 경우
    Exp2의 source flag만 이용해 Agreement Feature 5개를 생성한다.
    """

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

    """
    우선 Exp3 Agreement cache를 사용한다.

    없으면 Exp2 cache를 불러와
    Agreement Feature 5개를 즉시 생성한다.
    """

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
            "\nExp3 cache가 없어 Agreement Feature를 즉시 생성합니다."
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
            f"Exp3: {EXP3_FEATURE_CACHE_PATH}\n"
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

    """
    XGBRanker group=query(user) 단위 정렬.
    같은 user의 rows가 연속되도록 보장한다.
    """

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
# One Fold / One Feature Set
# ============================================================

def run_one_fold(
    *,
    fold_number: int,
    model_name: str,
    features: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
):

    print(
        "\n"
        +
        "-" * 78
    )

    print(
        f" Fold {fold_number} | {model_name}"
    )

    print(
        "-" * 78
    )


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


    print(
        "Feature count:",
        len(
            features
        ),
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
        "Best iteration:",
        best_iteration,
    )

    print(
        "Validation NDCG@10:",
        f"{best_score:.9f}",
    )

    print(
        "Train time:",
        f"{elapsed:.2f} sec",
    )


    return {
        "fold":
            fold_number,

        "model":
            model_name,

        "n_features":
            len(
                features
            ),

        "train_users":
            len(
                train_group
            ),

        "valid_users":
            len(
                valid_group
            ),

        "train_rows":
            len(
                X_train
            ),

        "valid_rows":
            len(
                X_valid
            ),

        "positive_train":
            int(
                y_train.sum()
            ),

        "positive_valid":
            int(
                y_valid.sum()
            ),

        "best_iteration":
            best_iteration,

        "validation_ndcg_at_10":
            best_score,

        "train_seconds":
            elapsed,
    }


# ============================================================
# Main
# ============================================================

def main():

    total_start = time.perf_counter()


    print(
        "=" * 78
    )

    print(
        " XGBoost Ranker - User-Level 5-Fold Cross Validation"
    )

    print(
        " Compare: Full14 vs Full19"
    )

    print(
        " Metric : Validation NDCG@10"
    )

    print(
        " Final 400 users are NOT used"
    )

    print(
        "=" * 78
    )


    # ========================================================
    # 1. LTR Users
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


    if len(
        user_ids
    ) < N_SPLITS:

        raise ValueError(
            f"사용자 수가 {N_SPLITS}-Fold보다 적습니다."
        )


    # ========================================================
    # 2. Feature Cache
    # ========================================================

    feature_df = load_feature_cache()


    print(
        "Feature DF:",
        feature_df.shape,
    )


    validate_columns(
        feature_df
    )


    # LTR 1000명만 사용
    ltr_id_set = set(
        user_ids.tolist()
    )


    feature_df = (
        feature_df[
            feature_df[
                "user_id"
            ]
            .isin(
                ltr_id_set
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


    actual_users = set(
        feature_df[
            "user_id"
        ]
        .unique()
        .tolist()
    )


    missing_users = (
        ltr_id_set
        -
        actual_users
    )


    if missing_users:

        raise ValueError(
            "LTR user 중 Feature cache에 없는 사용자가 있습니다: "
            f"{len(missing_users)}명"
        )


    # ========================================================
    # 3. User-Level 5-Fold
    # ========================================================

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


    fold_results = []


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
            "=" * 78
        )

        print(
            f" FOLD {fold_index}/{N_SPLITS}"
        )

        print(
            "=" * 78
        )


        train_users = (
            user_ids[
                train_user_idx
            ]
        )


        valid_users = (
            user_ids[
                valid_user_idx
            ]
        )


        train_user_set = set(
            train_users.tolist()
        )


        valid_user_set = set(
            valid_users.tolist()
        )


        overlap = (
            train_user_set
            &
            valid_user_set
        )


        if overlap:

            raise RuntimeError(
                "Train / Validation user overlap 발생"
            )


        train_df = (
            feature_df[
                feature_df[
                    "user_id"
                ]
                .isin(
                    train_user_set
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
                    valid_user_set
                )
            ]
            .copy()
        )


        print(
            "Train user count:",
            len(
                train_user_set
            ),
        )

        print(
            "Valid user count:",
            len(
                valid_user_set
            ),
        )


        # ----------------------------------------------------
        # 같은 fold에서 Full14 / Full19 1:1 비교
        # ----------------------------------------------------

        for model_name, features in (
            FEATURE_SETS.items()
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


            fold_results.append(
                result
            )


    # ========================================================
    # 4. Fold Result
    # ========================================================

    result_df = pd.DataFrame(
        fold_results
    )


    result_df.to_csv(
        FOLD_RESULT_PATH,
        index=False,
    )


    print(
        "\n"
        +
        "=" * 78
    )

    print(
        " 5-FOLD RAW RESULTS"
    )

    print(
        "=" * 78
    )


    pivot = (
        result_df
        .pivot(
            index="fold",
            columns="model",
            values="validation_ndcg_at_10",
        )
    )


    pivot[
        "delta_full19_minus_full14"
    ] = (
        pivot[
            "full19"
        ]
        -
        pivot[
            "full14"
        ]
    )


    print(
        pivot
        .to_string()
    )


    # ========================================================
    # 5. Model Summary
    # ========================================================

    summary_rows = []


    for model_name in [
        "full14",
        "full19",
    ]:

        model_results = (
            result_df[
                result_df[
                    "model"
                ]
                ==
                model_name
            ]
        )


        scores = (
            model_results[
                "validation_ndcg_at_10"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        iterations = (
            model_results[
                "best_iteration"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        summary_rows.append(
            {
                "model":
                    model_name,

                "n_features":
                    int(
                        model_results[
                            "n_features"
                        ]
                        .iloc[0]
                    ),

                "mean_validation_ndcg_at_10":
                    float(
                        np.mean(
                            scores
                        )
                    ),

                "std_validation_ndcg_at_10":
                    float(
                        np.std(
                            scores,
                            ddof=1,
                        )
                    ),

                "min_validation_ndcg_at_10":
                    float(
                        np.min(
                            scores
                        )
                    ),

                "max_validation_ndcg_at_10":
                    float(
                        np.max(
                            scores
                        )
                    ),

                "mean_best_iteration":
                    float(
                        np.mean(
                            iterations
                        )
                    ),

                "mean_train_seconds":
                    float(
                        model_results[
                            "train_seconds"
                        ]
                        .mean()
                    ),
            }
        )


    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_df = (
        summary_df
        .sort_values(
            "mean_validation_ndcg_at_10",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    summary_df[
        "rank"
    ] = (
        np.arange(
            1,
            len(
                summary_df
            )
            +
            1
        )
    )


    summary_df = (
        summary_df[
            [
                "rank",
                "model",
                "n_features",
                "mean_validation_ndcg_at_10",
                "std_validation_ndcg_at_10",
                "min_validation_ndcg_at_10",
                "max_validation_ndcg_at_10",
                "mean_best_iteration",
                "mean_train_seconds",
            ]
        ]
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
        " 5-FOLD SUMMARY"
    )

    print(
        "=" * 78
    )


    print(
        summary_df
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 6. Paired Fold Comparison
    # ========================================================

    paired_df = (
        pivot
        .reset_index()
    )


    paired_df[
        "winner"
    ] = np.where(
        paired_df[
            "delta_full19_minus_full14"
        ]
        >
        0,
        "full19",
        np.where(
            paired_df[
                "delta_full19_minus_full14"
            ]
            <
            0,
            "full14",
            "tie",
        ),
    )


    paired_df.to_csv(
        PAIRED_RESULT_PATH,
        index=False,
    )


    deltas = (
        paired_df[
            "delta_full19_minus_full14"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    full14_wins = int(
        (
            deltas < 0
        )
        .sum()
    )


    full19_wins = int(
        (
            deltas > 0
        )
        .sum()
    )


    ties = int(
        (
            deltas == 0
        )
        .sum()
    )


    mean_delta = float(
        np.mean(
            deltas
        )
    )


    std_delta = float(
        np.std(
            deltas,
            ddof=1,
        )
    )


    print(
        "\n"
        +
        "=" * 78
    )

    print(
        " PAIRED FOLD COMPARISON"
    )

    print(
        "=" * 78
    )


    print(
        paired_df[
            [
                "fold",
                "full14",
                "full19",
                "delta_full19_minus_full14",
                "winner",
            ]
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\nFold wins"
    )

    print(
        "Full14:",
        full14_wins,
    )

    print(
        "Full19:",
        full19_wins,
    )

    print(
        "Tie   :",
        ties,
    )


    print(
        "\nMean paired delta "
        "(Full19 - Full14):",
        f"{mean_delta:+.9f}",
    )


    print(
        "Std paired delta:",
        f"{std_delta:.9f}",
    )


    # ========================================================
    # 7. Final Decision
    # ========================================================

    full14_mean = float(
        summary_df.loc[
            summary_df[
                "model"
            ]
            ==
            "full14",
            "mean_validation_ndcg_at_10",
        ]
        .iloc[0]
    )


    full19_mean = float(
        summary_df.loc[
            summary_df[
                "model"
            ]
            ==
            "full19",
            "mean_validation_ndcg_at_10",
        ]
        .iloc[0]
    )


    print(
        "\n"
        +
        "=" * 78
    )

    print(
        " FINAL CV DECISION"
    )

    print(
        "=" * 78
    )


    print(
        f"Full14 Mean NDCG@10: "
        f"{full14_mean:.9f}"
    )

    print(
        f"Full19 Mean NDCG@10: "
        f"{full19_mean:.9f}"
    )

    print(
        f"Delta Full19-Full14: "
        f"{full19_mean - full14_mean:+.9f}"
    )


    if full19_mean > full14_mean:

        selected = "full19"

        reason = (
            "5-Fold 평균 Validation NDCG@10이 Full19가 더 높음"
        )

    elif full14_mean > full19_mean:

        selected = "full14"

        reason = (
            "5-Fold 평균 Validation NDCG@10이 Full14가 더 높음"
        )

    else:

        selected = "full14"

        reason = (
            "평균 성능이 동일하므로 더 단순한 Full14 선택"
        )


    print(
        "\nSelected Feature Set:",
        selected,
    )

    print(
        "Reason:",
        reason,
    )


    # ========================================================
    # 8. Saved
    # ========================================================

    print(
        "\n===== Saved ====="
    )


    print(
        "Fold Results:"
    )

    print(
        FOLD_RESULT_PATH
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_PATH
    )


    print(
        "\nPaired Comparison:"
    )

    print(
        PAIRED_RESULT_PATH
    )


    print(
        "\nTotal Runtime:"
        f" {time.perf_counter() - total_start:.1f} sec"
    )


if __name__ == "__main__":
    main()
