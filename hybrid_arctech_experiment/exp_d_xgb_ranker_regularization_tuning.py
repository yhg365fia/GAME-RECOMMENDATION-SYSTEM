# hybrid_arctech_experiment/exp_d_xgb_ranker_regularization_tuning.py
#
# XGBoost Ranker 규제 튜닝
# ---------------------------------------------------------
# 전제:
# - Candidate = BPR56 + Content39 + User5
# - Features = 4 scores + 4 tie-aware ranks
# - LTR users = 1000
# - Train 800 / Validation 200
# - Final 400 users는 하이퍼파라미터 선택에 사용하지 않음
# - Early Stopping 유지
#
# 실험:
# baseline: depth=6, child=1, lambda=1, alpha=0
# A       : depth=4, child=1, lambda=1, alpha=0
# B       : depth=4, child=5, lambda=1, alpha=0
# C       : depth=4, child=5, lambda=5, alpha=0
# D       : depth=4, child=5, lambda=5, alpha=0.1
# E       : depth=3, child=5, lambda=5, alpha=0.1
#
# 선택 기준:
# validation NDCG@10 최고
#
# 최종:
# 가장 좋은 설정 1개만 Final 400 users에서 평가

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBRanker


# =========================================================
# Project Root
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from data_split import load_mf_split


# =========================================================
# Config
# =========================================================

RANDOM_STATE = 42
VALID_RATIO = 0.20
TOP_N = 10
EARLY_STOPPING_ROUNDS = 30
NO_SIGNAL_RANK = 101


FEATURES = [
    "item_score_norm",
    "bpr_score_norm",
    "content_score_norm",
    "user_score_norm",

    "item_rank",
    "bpr_rank",
    "content_rank",
    "user_rank",
]


RANK_SPECS = {
    "item_rank":
        "item_score_norm",

    "bpr_rank":
        "bpr_score_norm",

    "content_rank":
        "content_score_norm",

    "user_rank":
        "user_score_norm",
}


# =========================================================
# Paths
# =========================================================

SAVED_MODEL_DIR = ROOT / "models" / "saved_model"
RESULT_DIR = SAVED_MODEL_DIR / "results"
CACHE_DIR = SAVED_MODEL_DIR / "ltr_cache" / "xgb_exp1"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_DIR.mkdir(
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

ORIGINAL_FEATURE_CACHE_PATH = (
    CACHE_DIR
    / "xgb_exp1_features.parquet"
)

FIXED_FEATURE_CACHE_PATH = (
    CACHE_DIR
    / "xgb_exp1_features_all_rank_tie_fixed.parquet"
)

TUNING_RESULT_PATH = (
    RESULT_DIR
    / "xgb_regularization_tuning.csv"
)

BEST_MODEL_PATH = (
    SAVED_MODEL_DIR
    / "xgb_ranker_regularization_best.json"
)

FINAL_RESULT_PATH = (
    RESULT_DIR
    / "xgb_ranker_regularization_best_eval.csv"
)


# =========================================================
# Tuning Grid
# =========================================================

EXPERIMENTS = [

    {
        "name":
            "baseline",

        "max_depth":
            6,

        "min_child_weight":
            1,

        "reg_lambda":
            1.0,

        "reg_alpha":
            0.0,
    },

    {
        "name":
            "A",

        "max_depth":
            4,

        "min_child_weight":
            1,

        "reg_lambda":
            1.0,

        "reg_alpha":
            0.0,
    },

    {
        "name":
            "B",

        "max_depth":
            4,

        "min_child_weight":
            5,

        "reg_lambda":
            1.0,

        "reg_alpha":
            0.0,
    },

    {
        "name":
            "C",

        "max_depth":
            4,

        "min_child_weight":
            5,

        "reg_lambda":
            5.0,

        "reg_alpha":
            0.0,
    },

    {
        "name":
            "D",

        "max_depth":
            4,

        "min_child_weight":
            5,

        "reg_lambda":
            5.0,

        "reg_alpha":
            0.1,
    },

    {
        "name":
            "E",

        "max_depth":
            3,

        "min_child_weight":
            5,

        "reg_lambda":
            5.0,

        "reg_alpha":
            0.1,
    },
]


# =========================================================
# Rank Fix
# =========================================================

def rebuild_rank_column(
    df,
    score_col,
    rank_col,
):

    fixed = df.copy()

    new_rank = (
        fixed
        .groupby(
            "user_id",
            sort=False,
        )[score_col]
        .rank(
            method="min",
            ascending=False,
            na_option="bottom",
        )
        .fillna(
            NO_SIGNAL_RANK
        )
        .astype(
            np.int32
        )
    )

    score_nunique = (
        fixed
        .groupby(
            "user_id",
            sort=False,
        )[score_col]
        .transform(
            "nunique"
        )
    )

    no_signal_mask = (
        score_nunique <= 1
    )

    new_rank.loc[
        no_signal_mask
    ] = (
        NO_SIGNAL_RANK
    )

    fixed[
        rank_col
    ] = new_rank

    return fixed


def build_or_load_fixed_features():

    if FIXED_FEATURE_CACHE_PATH.exists():

        print(
            "\n===== Tie-Aware Feature Cache 로드 ====="
        )

        print(
            FIXED_FEATURE_CACHE_PATH
        )

        fixed_df = (
            pd.read_parquet(
                FIXED_FEATURE_CACHE_PATH
            )
        )

        print(
            "Feature DF:",
            fixed_df.shape,
        )

        return fixed_df

    if not ORIGINAL_FEATURE_CACHE_PATH.exists():

        raise FileNotFoundError(
            f"기존 feature cache 없음: "
            f"{ORIGINAL_FEATURE_CACHE_PATH}"
        )

    print(
        "\n===== 기존 Feature Cache 로드 ====="
    )

    print(
        ORIGINAL_FEATURE_CACHE_PATH
    )

    fixed_df = (
        pd.read_parquet(
            ORIGINAL_FEATURE_CACHE_PATH
        )
    )

    print(
        "Original Feature DF:",
        fixed_df.shape,
    )

    print(
        "\n===== 4개 Rank Tie Fix ====="
    )

    for rank_col, score_col in (
        RANK_SPECS.items()
    ):

        before = (
            fixed_df[
                rank_col
            ]
            .copy()
        )

        fixed_df = (
            rebuild_rank_column(
                fixed_df,
                score_col=
                    score_col,
                rank_col=
                    rank_col,
            )
        )

        changed = int(
            (
                before
                !=
                fixed_df[
                    rank_col
                ]
            )
            .sum()
        )

        print(
            f"{rank_col}: "
            f"{changed} rows 수정"
        )

    fixed_df.to_parquet(
        FIXED_FEATURE_CACHE_PATH,
        index=False,
    )

    print(
        "\nTie-Aware Feature 저장:"
    )

    print(
        FIXED_FEATURE_CACHE_PATH
    )

    return fixed_df


# =========================================================
# Ranker Data
# =========================================================

def prepare_ranker_data(
    df,
):

    df = (
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
        df[
            FEATURES
        ]
        .astype(
            np.float32
        )
    )

    y = (
        df[
            "label"
        ]
        .astype(
            np.float32
        )
    )

    groups = (
        df
        .groupby(
            "user_id",
            sort=False,
        )
        .size()
        .to_numpy()
    )

    return (
        df,
        X,
        y,
        groups,
    )


# =========================================================
# Final Evaluation
# =========================================================

def evaluate_ranker(
    model,
    eval_df,
    positive_test_dict,
):

    rows = []

    for user_id, group in (
        eval_df
        .groupby(
            "user_id",
            sort=False,
        )
    ):

        group = (
            group
            .copy()
        )

        scores = (
            model.predict(
                group[
                    FEATURES
                ]
                .astype(
                    np.float32
                )
            )
        )

        group[
            "xgb_score"
        ] = scores

        group = (
            group
            .sort_values(
                "xgb_score",
                ascending=False,
                kind="stable",
            )
        )

        recommended = (
            group
            .head(
                TOP_N
            )[
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

    result = (
        pd.DataFrame(
            rows
        )
    )

    print(
        "\n===== FINAL 400 Evaluation ====="
    )

    print(
        f"Precision@10: "
        f"{result['precision'].mean():.4f}"
    )

    print(
        f"Recall@10:    "
        f"{result['recall'].mean():.4f}"
    )

    print(
        f"HR@10:        "
        f"{result['hit_rate'].mean():.4f}"
    )

    print(
        f"NDCG@10:      "
        f"{result['ndcg'].mean():.4f}"
    )

    print(
        f"Hits:         "
        f"{result['hits'].sum()}"
    )

    return result


# =========================================================
# Main
# =========================================================

def main():

    total_start = (
        time.perf_counter()
    )

    print(
        "=" * 65
    )

    print(
        " XGBoost Ranker - Regularization Tuning"
    )

    print(
        " 4 Scores + 4 Tie-Aware Ranks"
    )

    print(
        " Early Stopping = 30"
    )

    print(
        " Model selection = Validation NDCG@10 ONLY"
    )

    print(
        "=" * 65
    )


    # =====================================================
    # 1. Feature
    # =====================================================

    feature_df = (
        build_or_load_fixed_features()
    )


    # =====================================================
    # 2. User Lists
    # =====================================================

    if not LTR_USERS_PATH.exists():

        raise FileNotFoundError(
            f"LTR user cache 없음: "
            f"{LTR_USERS_PATH}"
        )

    if not FINAL_EVAL_USERS_PATH.exists():

        raise FileNotFoundError(
            f"Final user cache 없음: "
            f"{FINAL_EVAL_USERS_PATH}"
        )


    ltr_users = (
        pd.read_csv(
            LTR_USERS_PATH
        )
    )

    final_eval_users = (
        pd.read_csv(
            FINAL_EVAL_USERS_PATH
        )
    )


    ltr_ids = set(
        ltr_users[
            "user_id"
        ]
    )

    final_ids = set(
        final_eval_users[
            "user_id"
        ]
    )


    overlap = (
        ltr_ids
        &
        final_ids
    )

    if overlap:

        raise ValueError(
            f"LTR / Final overlap: "
            f"{len(overlap)}"
        )


    # =====================================================
    # 3. Split Feature
    # =====================================================

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


    # =====================================================
    # 4. Train / Valid User Split
    # =====================================================

    unique_ltr_users = (
        ltr_users[
            "user_id"
        ]
        .to_numpy()
    )

    (
        train_users,
        valid_users,
    ) = (
        train_test_split(
            unique_ltr_users,

            test_size=
                VALID_RATIO,

            random_state=
                RANDOM_STATE,
        )
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


    (
        train_rank_df,
        X_train,
        y_train,
        train_group,
    ) = (
        prepare_ranker_data(
            train_rank_df
        )
    )


    (
        valid_rank_df,
        X_valid,
        y_valid,
        valid_group,
    ) = (
        prepare_ranker_data(
            valid_rank_df
        )
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
        "Train positives:",
        int(
            y_train.sum()
        ),
    )

    print(
        "Valid positives:",
        int(
            y_valid.sum()
        ),
    )


    # =====================================================
    # 5. Regularization Tuning
    #
    # Final 400은 여기서 절대 사용하지 않음
    # =====================================================

    tuning_rows = []

    best_model = None
    best_config = None
    best_score = -np.inf


    print(
        "\n"
        +
        "=" * 65
    )

    print(
        " REGULARIZATION EXPERIMENTS"
    )

    print(
        "=" * 65
    )


    for config in EXPERIMENTS:

        experiment_start = (
            time.perf_counter()
        )

        print(
            "\n"
            +
            "-" * 65
        )

        print(
            f"Experiment "
            f"{config['name']}"
        )

        print(
            "max_depth        =",
            config[
                "max_depth"
            ],
        )

        print(
            "min_child_weight =",
            config[
                "min_child_weight"
            ],
        )

        print(
            "reg_lambda       =",
            config[
                "reg_lambda"
            ],
        )

        print(
            "reg_alpha        =",
            config[
                "reg_alpha"
            ],
        )

        print(
            "-" * 65
        )


        model = (
            XGBRanker(

                objective=
                    "rank:ndcg",

                eval_metric=
                    "ndcg@10",

                n_estimators=
                    500,

                learning_rate=
                    0.05,

                max_depth=
                    config[
                        "max_depth"
                    ],

                min_child_weight=
                    config[
                        "min_child_weight"
                    ],

                subsample=
                    0.8,

                colsample_bytree=
                    0.8,

                reg_lambda=
                    config[
                        "reg_lambda"
                    ],

                reg_alpha=
                    config[
                        "reg_alpha"
                    ],

                random_state=
                    RANDOM_STATE,

                tree_method=
                    "hist",

                early_stopping_rounds=
                    EARLY_STOPPING_ROUNDS,
            )
        )


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

            verbose=
                False,
        )


        current_score = float(
            model.best_score
        )

        current_iteration = int(
            model.best_iteration
        )

        elapsed = (
            time.perf_counter()
            -
            experiment_start
        )


        print(
            "Best iteration:",
            current_iteration,
        )

        print(
            "Best validation NDCG@10:",
            f"{current_score:.6f}",
        )

        print(
            "실행 시간:",
            f"{elapsed:.1f}초",
        )


        tuning_rows.append(
            {
                "experiment":
                    config[
                        "name"
                    ],

                "max_depth":
                    config[
                        "max_depth"
                    ],

                "min_child_weight":
                    config[
                        "min_child_weight"
                    ],

                "reg_lambda":
                    config[
                        "reg_lambda"
                    ],

                "reg_alpha":
                    config[
                        "reg_alpha"
                    ],

                "best_iteration":
                    current_iteration,

                "validation_ndcg_10":
                    current_score,

                "runtime_sec":
                    elapsed,
            }
        )


        if (
            current_score
            >
            best_score
        ):

            best_score = (
                current_score
            )

            best_model = (
                model
            )

            best_config = (
                config.copy()
            )


    # =====================================================
    # 6. Tuning Result
    # =====================================================

    tuning_df = (
        pd.DataFrame(
            tuning_rows
        )
        .sort_values(
            "validation_ndcg_10",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    tuning_df[
        "rank"
    ] = (
        np.arange(
            1,
            len(
                tuning_df
            )
            +
            1
        )
    )


    tuning_df = (
        tuning_df[
            [
                "rank",
                "experiment",
                "max_depth",
                "min_child_weight",
                "reg_lambda",
                "reg_alpha",
                "best_iteration",
                "validation_ndcg_10",
                "runtime_sec",
            ]
        ]
    )


    print(
        "\n"
        +
        "=" * 65
    )

    print(
        " VALIDATION RESULT"
    )

    print(
        "=" * 65
    )

    print(
        tuning_df.to_string(
            index=False
        )
    )


    tuning_df.to_csv(
        TUNING_RESULT_PATH,
        index=False,
    )


    print(
        "\n튜닝 결과 저장:"
    )

    print(
        TUNING_RESULT_PATH
    )


    # =====================================================
    # 7. Best Config
    # =====================================================

    print(
        "\n"
        +
        "=" * 65
    )

    print(
        " BEST CONFIG"
    )

    print(
        "=" * 65
    )

    print(
        "Experiment:",
        best_config[
            "name"
        ],
    )

    print(
        "max_depth:",
        best_config[
            "max_depth"
        ],
    )

    print(
        "min_child_weight:",
        best_config[
            "min_child_weight"
        ],
    )

    print(
        "reg_lambda:",
        best_config[
            "reg_lambda"
        ],
    )

    print(
        "reg_alpha:",
        best_config[
            "reg_alpha"
        ],
    )

    print(
        "Validation NDCG@10:",
        f"{best_score:.6f}",
    )

    print(
        "Best iteration:",
        best_model.best_iteration,
    )


    # =====================================================
    # 8. Best Model 저장
    # =====================================================

    best_model.save_model(
        BEST_MODEL_PATH
    )


    print(
        "\nBest 모델 저장:"
    )

    print(
        BEST_MODEL_PATH
    )


    # =====================================================
    # 9. Feature Importance
    # =====================================================

    importance_df = (
        pd.DataFrame(
            {
                "feature":
                    FEATURES,

                "importance":
                    best_model
                    .feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
    )


    print(
        "\n===== Best Model Feature Importance ====="
    )

    print(
        importance_df.to_string(
            index=False
        )
    )


    # =====================================================
    # 10. 여기서 처음 Final 400 준비
    # =====================================================

    print(
        "\n"
        +
        "=" * 65
    )

    print(
        " FINAL 400 USERS"
    )

    print(
        " Best config 선택 완료 후 단 1회 평가"
    )

    print(
        "=" * 65
    )


    print(
        "\n===== MF Test 로드 ====="
    )

    _, test_df = (
        load_mf_split()
    )


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
            lambda x:
                set(
                    x.tolist()
                )
        )
        .to_dict()
    )


    # =====================================================
    # 11. Final 400 Evaluation
    # =====================================================

    final_result = (
        evaluate_ranker(

            model=
                best_model,

            eval_df=
                final_df,

            positive_test_dict=
                positive_test_dict,
        )
    )


    final_result.to_csv(
        FINAL_RESULT_PATH,
        index=False,
    )


    print(
        "\nFinal 결과 저장:"
    )

    print(
        FINAL_RESULT_PATH
    )


    # =====================================================
    # 12. Reference
    # =====================================================

    print(
        "\n"
        +
        "=" * 65
    )

    print(
        " REFERENCE"
    )

    print(
        "=" * 65
    )


    print(
        "\nItem Ranker Hybrid"
    )

    print(
        "P@10    = 0.0872"
    )

    print(
        "R@10    = 0.1110"
    )

    print(
        "HR@10   = 0.5400"
    )

    print(
        "NDCG@10 = 0.1216"
    )

    print(
        "Hits     = 348"
    )


    print(
        "\nXGB Exp1.1 Early Stopping"
    )

    print(
        "P@10    = 0.0878"
    )

    print(
        "R@10    = 0.1138"
    )

    print(
        "HR@10   = 0.5425"
    )

    print(
        "NDCG@10 = 0.1250"
    )

    print(
        "Hits     = 351"
    )


    print(
        "\nXGB Exp1.2 user_rank Fix"
    )

    print(
        "P@10    = 0.0873"
    )

    print(
        "R@10    = 0.1135"
    )

    print(
        "HR@10   = 0.5450"
    )

    print(
        "NDCG@10 = 0.1268"
    )

    print(
        "Hits     = 349"
    )


    # =====================================================
    # 13. Runtime
    # =====================================================

    print(
        "\n총 실행 시간:"
        f" {time.perf_counter() - total_start:.1f}초"
    )


if __name__ == "__main__":
    main()
