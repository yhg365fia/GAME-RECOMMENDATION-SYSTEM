# hybrid_arctech_experiment/exp_d_xgb_ranker_exp1_all_rank_tie_fix.py
#
# 목적
# ---------------------------------------------------------
# 1) 기존 Exp1의 4개 rank feature 모두 동점 처리 문제 수정
#    - item_rank
#    - bpr_rank
#    - content_rank
#    - user_rank
#
# 2) 같은 score는 반드시 같은 rank
# 3) 한 사용자 안에서 해당 모델 score가 전부 동일하면
#    rank=101 (NO_SIGNAL) 처리
# 4) 기존 10시간짜리 feature cache는 재사용하되,
#    normalized score에서 rank를 안전하게 재구성
# 5) XGBRanker Early Stopping 유지
#
# 중요:
# min-max normalization은 점수의 대소관계와 동점을 보존하므로
# 기존 *_score_norm으로 rank를 다시 계산해도 순위 관계는 보존된다.

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

# 기존 Exp1의 scorer / feature / evaluation 코드 재사용
import hybrid_arctech_experiment.exp_d_xgb_ranker_exp1 as exp1


# =========================================================
# Config
# =========================================================

RANDOM_STATE = 42

VALID_RATIO = 0.20
TOP_N = 10

EARLY_STOPPING_ROUNDS = 30

# Candidate 최대 100
# 101 = 해당 모델이 후보 간 ranking signal을 전혀 주지 못함
NO_SIGNAL_RANK = 101


# =========================================================
# Feature Spec
# =========================================================

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

# 기존 Exp1에서 생성한 원본 feature cache
ORIGINAL_FEATURE_CACHE_PATH = (
    CACHE_DIR
    / "xgb_exp1_features.parquet"
)

# 4개 rank를 모두 수정한 새 cache
FIXED_FEATURE_CACHE_PATH = (
    CACHE_DIR
    / "xgb_exp1_features_all_rank_tie_fixed.parquet"
)

MODEL_PATH = (
    SAVED_MODEL_DIR
    / "xgb_ranker_exp1_all_rank_tie_fixed.json"
)

RESULT_PATH = (
    RESULT_DIR
    / "xgb_ranker_exp1_all_rank_tie_fixed_eval.csv"
)


# =========================================================
# 1. 앞으로 Feature 새로 만들 때 사용할
#    올바른 tie-aware rank 함수
# =========================================================

def assign_rank_tie_aware(
    scores,
    no_signal_rank=NO_SIGNAL_RANK,
):

    """
    score가 높은 순으로 rank 부여.

    기존 문제:
        np.argsort(..., kind="stable") 후
        1,2,3,...을 강제로 부여하면
        score가 같은 후보도 서로 다른 rank를 받음.

    수정:
        - 같은 score -> 같은 rank
        - 모든 score가 동일 -> NO_SIGNAL_RANK(101)
        - 비유한 finite score는 rank=101

    예:
        [0.9, 0.7, 0.7, 0.2]
        -> [1, 2, 2, 4]

        [0, 0, 0, 0]
        -> [101, 101, 101, 101]
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    n = len(scores)

    result = np.full(
        n,
        no_signal_rank,
        dtype=np.int32,
    )

    if n == 0:
        return result

    finite_mask = np.isfinite(
        scores
    )

    if not finite_mask.any():
        return result

    finite_scores = scores[
        finite_mask
    ]

    # 모든 후보가 같은 점수
    # -> 모델이 후보를 구분하지 못함
    if (
        np.max(finite_scores)
        -
        np.min(finite_scores)
        <
        1e-12
    ):
        return result

    # pandas rank:
    # method="min" -> 동점은 같은 rank
    # [0.9, 0.7, 0.7, 0.2]
    # -> [1, 2, 2, 4]
    finite_ranks = (
        pd.Series(
            finite_scores
        )
        .rank(
            method="min",
            ascending=False,
        )
        .astype(
            np.int32
        )
        .to_numpy()
    )

    result[
        finite_mask
    ] = finite_ranks

    return result


# ---------------------------------------------------------
# 기존 exp1.build_feature_rows()가 내부에서
# exp1.assign_rank()를 호출하므로,
# 앞으로 이 파일에서 새 feature를 생성하더라도
# 4개 모델 모두 자동으로 tie-aware rank 사용
# ---------------------------------------------------------

exp1.assign_rank = (
    assign_rank_tie_aware
)


# =========================================================
# 2. 기존 Cache 진단
# =========================================================

def diagnose_rank_feature(
    df,
    score_col,
    rank_col,
):

    print(
        "\n"
        +
        "-" * 60
    )

    print(
        f"[{rank_col}]"
    )

    # -----------------------------------------------------
    # 모든 score가 동일한 사용자
    # -----------------------------------------------------

    score_nunique_by_user = (
        df
        .groupby(
            "user_id",
            sort=False,
        )[score_col]
        .nunique(
            dropna=False
        )
    )

    all_same_users = (
        score_nunique_by_user
        <= 1
    )

    print(
        "score가 전부 동일한 사용자:",
        int(
            all_same_users.sum()
        ),
    )

    # -----------------------------------------------------
    # 같은 score인데 rank가 여러 개인 tie group
    # -----------------------------------------------------

    tie_rank_count = (
        df
        .groupby(
            [
                "user_id",
                score_col,
            ],
            sort=False,
            dropna=False,
        )[rank_col]
        .nunique()
    )

    broken_ties = (
        tie_rank_count
        > 1
    )

    print(
        "같은 score인데 서로 다른 rank가 부여된 tie 그룹:",
        int(
            broken_ties.sum()
        ),
    )

    return {
        "all_same_users":
            int(
                all_same_users.sum()
            ),

        "broken_tie_groups":
            int(
                broken_ties.sum()
            ),
    }


def diagnose_all_ranks(
    df,
):

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        " ALL RANK DIAGNOSTIC"
    )

    print(
        "=" * 60
    )

    result = {}

    for rank_col, score_col in (
        RANK_SPECS.items()
    ):

        result[
            rank_col
        ] = diagnose_rank_feature(
            df,
            score_col,
            rank_col,
        )

    return result


# =========================================================
# 3. 한 Rank Feature 수정
# =========================================================

def rebuild_rank_column(
    df,
    score_col,
    rank_col,
):

    """
    normalized score를 이용해 사용자별 rank 재구성.

    원칙:
        1) 같은 score -> 같은 rank
        2) 모든 score 동일 -> rank=101
    """

    original_rank_col = (
        f"{rank_col}_original"
    )

    df[
        original_rank_col
    ] = (
        df[
            rank_col
        ]
        .astype(
            np.int32
        )
    )

    # -----------------------------------------------------
    # 사용자별 competition rank
    #
    # 높은 점수가 1위
    # 동점은 같은 순위
    # -----------------------------------------------------

    new_rank = (
        df
        .groupby(
            "user_id",
            sort=False,
        )[score_col]
        .rank(
            method="min",
            ascending=False,
            na_option="bottom",
        )
    )

    # 혹시 NaN이 있으면 no-signal 처리
    new_rank = (
        new_rank
        .fillna(
            NO_SIGNAL_RANK
        )
        .astype(
            np.int32
        )
    )

    # -----------------------------------------------------
    # 사용자 내 score가 전부 동일하면
    # ranking signal이 없음
    # -----------------------------------------------------

    score_nunique = (
        df
        .groupby(
            "user_id",
            sort=False,
        )[score_col]
        .transform(
            "nunique"
        )
    )

    no_signal_mask = (
        score_nunique
        <= 1
    )

    new_rank.loc[
        no_signal_mask
    ] = (
        NO_SIGNAL_RANK
    )

    df[
        rank_col
    ] = new_rank

    # -----------------------------------------------------
    # 변경량
    # -----------------------------------------------------

    changed_mask = (
        df[
            rank_col
        ]
        !=
        df[
            original_rank_col
        ]
    )

    changed_rows = int(
        changed_mask.sum()
    )

    no_signal_rows = int(
        (
            df[
                rank_col
            ]
            ==
            NO_SIGNAL_RANK
        )
        .sum()
    )

    # -----------------------------------------------------
    # 수정 후 tie 검증
    # -----------------------------------------------------

    tie_rank_count_after = (
        df
        .groupby(
            [
                "user_id",
                score_col,
            ],
            sort=False,
            dropna=False,
        )[rank_col]
        .nunique()
    )

    remaining_broken_ties = int(
        (
            tie_rank_count_after
            > 1
        )
        .sum()
    )

    print(
        f"\n[{rank_col}] FIX"
    )

    print(
        "수정된 row 수:",
        changed_rows,
    )

    print(
        "수정 비율:",
        f"{changed_rows / len(df):.2%}",
    )

    print(
        "rank=101(no-signal) row 수:",
        no_signal_rows,
    )

    print(
        "수정 후 broken tie group:",
        remaining_broken_ties,
    )

    if (
        remaining_broken_ties
        != 0
    ):

        raise RuntimeError(
            f"{rank_col} "
            f"tie 수정 실패"
        )

    return df


# =========================================================
# 4. 모든 Rank 수정
# =========================================================

def rebuild_all_rank_features(
    feature_df,
):

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        " ALL RANK TIE FIX"
    )

    print(
        "=" * 60
    )

    fixed = (
        feature_df
        .copy()
    )

    for rank_col, score_col in (
        RANK_SPECS.items()
    ):

        fixed = (
            rebuild_rank_column(
                fixed,
                score_col=
                    score_col,
                rank_col=
                    rank_col,
            )
        )

    print(
        "\n모든 rank 수정 완료"
    )

    return fixed


# =========================================================
# 5. XGB용 정렬
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
# 6. 최종 평가
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

        X_user = (
            group[
                FEATURES
            ]
            .astype(
                np.float32
            )
        )

        # Early Stopping 사용 시
        # sklearn wrapper predict는
        # best_iteration 범위를 사용
        scores = (
            model.predict(
                X_user
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

        top = (
            group
            .head(
                TOP_N
            )
        )

        recommended = (
            top[
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

        # -------------------------------------------------
        # NDCG@10
        # -------------------------------------------------

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
        "\n===== XGBoost Ranker Evaluation ====="
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
        "=" * 60
    )

    print(
        " XGBoost Exp1.3 - ALL Rank Tie Fix"
    )

    print(
        " Candidate = BPR56 + Content39 + User5"
    )

    print(
        " Features  = 4 Scores + 4 Tie-Aware Ranks"
    )

    print(
        " Early Stopping = 30"
    )

    print(
        "=" * 60
    )


    # =====================================================
    # 1. 필수 파일 검사
    # =====================================================

    required_paths = [
        ORIGINAL_FEATURE_CACHE_PATH,
        LTR_USERS_PATH,
        FINAL_EVAL_USERS_PATH,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"필수 파일 없음: "
                f"{path}"
            )


    # =====================================================
    # 2. 기존 Feature Cache
    # =====================================================

    print(
        "\n===== 기존 Exp1 Feature Cache 로드 ====="
    )

    print(
        ORIGINAL_FEATURE_CACHE_PATH
    )

    feature_df = (
        pd.read_parquet(
            ORIGINAL_FEATURE_CACHE_PATH
        )
    )

    print(
        "Feature DF:",
        feature_df.shape,
    )

    required_columns = (
        [
            "user_id",
            "app_id",
            "label",
        ]
        +
        FEATURES
    )

    missing = [
        col
        for col
        in required_columns
        if col
        not in feature_df.columns
    ]

    if missing:

        raise ValueError(
            "Feature cache에 필요한 "
            f"컬럼이 없음: {missing}"
        )


    # =====================================================
    # 3. 기존 4 Rank 문제 진단
    # =====================================================

    diagnose_all_ranks(
        feature_df
    )


    # =====================================================
    # 4. 4개 Rank 전부 수정
    # =====================================================

    fixed_df = (
        rebuild_all_rank_features(
            feature_df
        )
    )


    # =====================================================
    # 5. 수정 후 전체 검증
    # =====================================================

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        " AFTER FIX DIAGNOSTIC"
    )

    print(
        "=" * 60
    )

    diagnose_all_ranks(
        fixed_df
    )


    # =====================================================
    # 6. 수정 Feature 저장
    # =====================================================

    fixed_df.to_parquet(
        FIXED_FEATURE_CACHE_PATH,
        index=False,
    )

    print(
        "\n수정 Feature 저장:"
    )

    print(
        FIXED_FEATURE_CACHE_PATH
    )


    # =====================================================
    # 7. 사용자 목록
    # =====================================================

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

    print(
        "\nLTR Users:",
        len(
            ltr_users
        ),
    )

    print(
        "Final Evaluation Users:",
        len(
            final_eval_users
        ),
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
            "LTR / Final 사용자 겹침: "
            f"{len(overlap)}"
        )


    # =====================================================
    # 8. LTR / Final Feature 분리
    # =====================================================

    ltr_df = (
        fixed_df[
            fixed_df[
                "user_id"
            ]
            .isin(
                ltr_ids
            )
        ]
        .copy()
    )

    final_df = (
        fixed_df[
            fixed_df[
                "user_id"
            ]
            .isin(
                final_ids
            )
        ]
        .copy()
    )

    print(
        "\nLTR Feature Rows:",
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


    # =====================================================
    # 9. Test positive dictionary
    # =====================================================

    print(
        "\n===== MF Train / Test ====="
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


    all_eval_ids = (
        ltr_ids
        |
        final_ids
    )

    positive_test = (
        test_df[
            (
                test_df[
                    "user_id"
                ]
                .isin(
                    all_eval_ids
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


    # train_df는 이후 필요 없음
    del train_df


    # =====================================================
    # 10. 1000명 -> 800 Train / 200 Validation
    #
    # Exp1 / Exp1.1 / Exp1.2와 동일 split
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
    ) = prepare_ranker_data(
        train_rank_df
    )


    (
        valid_rank_df,
        X_valid,
        y_valid,
        valid_group,
    ) = prepare_ranker_data(
        valid_rank_df
    )


    print(
        "\n===== Ranker Dataset ====="
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


    # =====================================================
    # 11. XGBRanker
    #
    # Exp1.1과 동일한 hyperparameter
    # 변경점:
    # 4개의 rank feature 모두 tie-aware
    # =====================================================

    print(
        "\n===== XGBRanker 학습 ====="
    )

    model = (
        XGBRanker(

            objective=
                "rank:ndcg",

            eval_metric=
                "ndcg@10",

            n_estimators=
                300,

            learning_rate=
                0.05,

            max_depth=
                6,

            min_child_weight=
                1,

            subsample=
                0.8,

            colsample_bytree=
                0.8,

            reg_lambda=
                1.0,

            random_state=
                RANDOM_STATE,

            tree_method=
                "hist",

            early_stopping_rounds=
                EARLY_STOPPING_ROUNDS,
        )
    )


    # =====================================================
    # 12. Train
    # =====================================================

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
            10,
    )


    # =====================================================
    # 13. Early Stopping Result
    # =====================================================

    print(
        "\n===== Early Stopping Result ====="
    )

    print(
        "Best iteration:",
        model.best_iteration,
    )

    print(
        "Best validation NDCG@10:",
        model.best_score,
    )


    # =====================================================
    # 14. 저장
    # =====================================================

    model.save_model(
        MODEL_PATH
    )

    print(
        "\n모델 저장:",
        MODEL_PATH,
    )


    # =====================================================
    # 15. Feature Importance
    # =====================================================

    importance_df = (
        pd.DataFrame(
            {
                "feature":
                    FEATURES,

                "importance":
                    model.feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    print(
        "\n===== Feature Importance ====="
    )

    print(
        importance_df.to_string(
            index=False
        )
    )


    # =====================================================
    # 16. Final 400
    # =====================================================

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        " FINAL 400 USERS"
    )

    print(
        " 4개 Rank 모두 Tie-Aware"
    )

    print(
        " 이 400명은 XGB 학습에 사용하지 않음"
    )

    print(
        "=" * 60
    )


    final_result = (
        evaluate_ranker(

            model=
                model,

            eval_df=
                final_df,

            positive_test_dict=
                positive_test_dict,
        )
    )


    final_result.to_csv(
        RESULT_PATH,
        index=False,
    )


    print(
        "\n결과 저장:"
    )

    print(
        RESULT_PATH
    )


    # =====================================================
    # 17. 비교 기준
    # =====================================================

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        " COMPARISON"
    )

    print(
        "=" * 60
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
        "\nXGB Exp1"
    )

    print(
        "P@10    = 0.0828"
    )

    print(
        "R@10    = 0.1043"
    )

    print(
        "HR@10   = 0.5100"
    )

    print(
        "NDCG@10 = 0.1168"
    )

    print(
        "Hits     = 331"
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
    # 18. Runtime
    # =====================================================

    print(
        "\n총 실행 시간:"
        f" {time.perf_counter() - total_start:.1f}초"
    )


if __name__ == "__main__":
    main()
