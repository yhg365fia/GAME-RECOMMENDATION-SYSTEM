# hybrid_arctech_experiment/exp_e_xgb_ranker_context_features.py
#
# XGBoost Ranker - Context Feature Experiment
# ------------------------------------------------------------
# 기존 8개 feature:
#   1) item_score_norm
#   2) bpr_score_norm
#   3) content_score_norm
#   4) user_score_norm
#   5) item_rank
#   6) bpr_rank
#   7) content_rank
#   8) user_rank
#
# 추가 6개 feature:
#   9)  retriever_count
#   10) is_bpr_candidate
#   11) is_content_candidate
#   12) is_user_candidate
#   13) item_popularity
#   14) user_interaction_count
#
# 실험 조건:
# - Candidate = BPR56 + Content39 + User5
# - 기존 ALL-RANK tie-aware feature cache 재사용
# - source flag는 실제 각 Retriever의 retrieve() 결과로 다시 계산
# - item_popularity는 TRAIN interaction count만 사용
# - user_interaction_count도 TRAIN interaction count만 사용
# - Test 정보는 label / 최종 평가에만 사용
# - LTR 사용자 1000명 = Train 800 / Validation 200
# - Final 400명은 학습에 사용하지 않음
#
# XGB 설정:
# - 이전 규제 튜닝 Best A 그대로 유지
# - max_depth=4
# - min_child_weight=1
# - reg_lambda=1.0
# - reg_alpha=0.0
# - early_stopping_rounds=30
#
# 목적:
# 기존 Best XGB와 hyperparameter를 동일하게 두고
# "추가 6개 feature 자체가 성능을 개선하는지" 확인한다.

import sys
import gc
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
from models.itembase import build_interaction_matrix

import hybrid_arctech_experiment.exp_d_xgb_ranker_exp1 as exp1


# =========================================================
# Config
# =========================================================

RANDOM_STATE = 42

BPR_N = 56
CONTENT_N = 39
USER_N = 5

TOP_N = 10
VALID_RATIO = 0.20

EARLY_STOPPING_ROUNDS = 30


# =========================================================
# Features
# =========================================================

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

NEW_FEATURES = [
    "retriever_count",
    "is_bpr_candidate",
    "is_content_candidate",
    "is_user_candidate",
    "item_popularity",
    "user_interaction_count",
]

FEATURES = (
    BASE_FEATURES
    +
    NEW_FEATURES
)


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

# 이전 Exp1.3에서 만든 4-rank tie-aware cache
BASE_FEATURE_CACHE_PATH = (
    CACHE_DIR
    / "xgb_exp1_features_all_rank_tie_fixed.parquet"
)

# 이번 실험용 확장 feature cache
EXPANDED_FEATURE_CACHE_PATH = (
    CACHE_DIR
    / "xgb_exp2_context_features.parquet"
)

MODEL_PATH = (
    SAVED_MODEL_DIR
    / "xgb_ranker_exp2_context_features.json"
)

RESULT_PATH = (
    RESULT_DIR
    / "xgb_ranker_exp2_context_features_eval.csv"
)

IMPORTANCE_PATH = (
    RESULT_DIR
    / "xgb_ranker_exp2_context_features_importance.csv"
)


# =========================================================
# Utility
# =========================================================

def prepare_ranker_data(df):

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
# 1. Expanded Feature 생성
# =========================================================

def build_expanded_features(
    base_feature_df,
    train_df,
    all_user_ids,
):

    print(
        "\n"
        +
        "=" * 65
    )

    print(
        " 추가 Context Feature 생성"
    )

    print(
        "=" * 65
    )


    # -----------------------------------------------------
    # 1-A. 1400명 Train subset / history
    # -----------------------------------------------------

    print(
        "\n===== Train Subset ====="
    )

    train_subset = (
        train_df[
            train_df[
                "user_id"
            ]
            .isin(
                all_user_ids
            )
        ]
        .copy()
    )

    print(
        "Train subset:",
        train_subset.shape,
    )


    histories = (
        train_subset
        .groupby(
            "user_id"
        )[
            "app_id"
        ]
        .apply(list)
        .to_dict()
    )


    # -----------------------------------------------------
    # 1-B. user_interaction_count
    #
    # 각 유저의 Train interaction 수
    # Test 사용하지 않음
    # -----------------------------------------------------

    print(
        "\n===== User Interaction Count ====="
    )

    user_interaction_count = (
        train_subset
        .groupby(
            "user_id"
        )
        .size()
        .to_dict()
    )

    print(
        "사용자 count 계산 완료:",
        len(
            user_interaction_count
        ),
    )


    # -----------------------------------------------------
    # 1-C. item_popularity
    #
    # 전체 Train에서 app_id별 interaction 수
    # Test 사용하지 않음
    # -----------------------------------------------------

    print(
        "\n===== Item Popularity ====="
    )

    item_popularity = (
        train_df[
            "app_id"
        ]
        .value_counts(
            sort=False
        )
        .to_dict()
    )

    print(
        "아이템 popularity 계산 완료:",
        len(
            item_popularity
        ),
    )


    # -----------------------------------------------------
    # 1-D. Global Interaction Matrix
    #
    # 정확한 User-Based top5 source flag를 재현하기 위해 필요
    # -----------------------------------------------------

    print(
        "\n===== Interaction Matrix ====="
    )

    (
        interaction_matrix,
        user_to_idx,
        game_to_idx,
        idx_to_game,
    ) = (
        build_interaction_matrix(
            train_df
        )
    )

    print(
        "Matrix:",
        interaction_matrix.shape,
    )


    # -----------------------------------------------------
    # 1-E. Retriever 준비
    #
    # ItemCF는 필요 없음.
    # 기존 8개 Item score/rank는 cache를 그대로 사용함.
    # -----------------------------------------------------

    print(
        "\n===== Retriever 준비 ====="
    )

    bpr = (
        exp1.BPRScorer()
    )

    content = (
        exp1.ContentScorer(
            item_ids=
                bpr.item_ids,

            train_subset=
                train_subset,
        )
    )

    user_cf = (
        exp1.UserCFScorer(

            interaction_matrix=
                interaction_matrix,

            user_to_idx=
                user_to_idx,

            game_to_idx=
                game_to_idx,

            idx_to_game=
                idx_to_game,

            histories=
                histories,
        )
    )


    # -----------------------------------------------------
    # 1-F. 실제 Retriever 결과를 이용해 source flag 생성
    # -----------------------------------------------------

    print(
        "\n===== Retriever Source Feature 생성 ====="
    )

    source_rows = []

    grouped_features = {
        user_id:
            group[
                [
                    "user_id",
                    "app_id",
                ]
            ]
            .copy()

        for user_id, group
        in base_feature_df.groupby(
            "user_id",
            sort=False,
        )
    }


    total_users = len(
        all_user_ids
    )

    uncovered_total = 0


    for count, user_id in enumerate(
        all_user_ids,
        start=1,
    ):

        user_candidates_df = (
            grouped_features.get(
                user_id
            )
        )

        if user_candidates_df is None:
            continue


        # ---------------------------------------------
        # 실제 Candidate Retriever 결과
        # ---------------------------------------------

        bpr_candidates = set(
            bpr.retrieve(
                user_id,
                BPR_N,
            )
        )

        content_candidates = set(
            content.retrieve(
                user_id,
                CONTENT_N,
            )
        )

        user_candidates = set(
            user_cf.retrieve(
                user_id,
                USER_N,
            )
        )


        user_interactions = int(
            user_interaction_count.get(
                user_id,
                0,
            )
        )


        for app_id in (
            user_candidates_df[
                "app_id"
            ]
            .tolist()
        ):

            is_bpr = int(
                app_id
                in
                bpr_candidates
            )

            is_content = int(
                app_id
                in
                content_candidates
            )

            is_user = int(
                app_id
                in
                user_candidates
            )

            retriever_count = (
                is_bpr
                +
                is_content
                +
                is_user
            )


            if retriever_count == 0:

                uncovered_total += 1


            source_rows.append(
                {
                    "user_id":
                        user_id,

                    "app_id":
                        app_id,

                    "retriever_count":
                        retriever_count,

                    "is_bpr_candidate":
                        is_bpr,

                    "is_content_candidate":
                        is_content,

                    "is_user_candidate":
                        is_user,

                    "item_popularity":
                        int(
                            item_popularity.get(
                                app_id,
                                0,
                            )
                        ),

                    "user_interaction_count":
                        user_interactions,
                }
            )


        if (
            count % 50 == 0
            or
            count == total_users
        ):

            print(
                f"Source Feature: "
                f"{count}/{total_users}"
            )


    source_df = (
        pd.DataFrame(
            source_rows
        )
    )


    print(
        "\nSource Feature DF:",
        source_df.shape,
    )

    print(
        "Retriever count=0 후보:",
        uncovered_total,
    )


    # -----------------------------------------------------
    # 정상이라면 candidate는 반드시
    # BPR / Content / User 중 적어도 하나에서 왔어야 함.
    # -----------------------------------------------------

    if uncovered_total > 0:

        print(
            "\n[WARNING]"
        )

        print(
            "기존 feature cache의 candidate와 "
            "현재 Retriever 결과가 일부 다릅니다."
        )

        print(
            "모델/캐시 변경 또는 동점 후보 선택 차이가 "
            "있는지 확인하세요."
        )


    # -----------------------------------------------------
    # 1-G. 기존 8 Feature + 새 6 Feature merge
    # -----------------------------------------------------

    expanded_df = (
        base_feature_df
        .merge(
            source_df,

            on=[
                "user_id",
                "app_id",
            ],

            how="left",

            validate="one_to_one",
        )
    )


    # source가 누락되면 silent하게 학습하지 않음
    missing_new = (
        expanded_df[
            NEW_FEATURES
        ]
        .isna()
        .any(
            axis=1
        )
    )


    if (
        missing_new.any()
    ):

        missing_count = int(
            missing_new.sum()
        )

        raise ValueError(
            "새 feature merge 실패: "
            f"{missing_count} rows"
        )


    # dtype 정리
    expanded_df[
        [
            "retriever_count",
            "is_bpr_candidate",
            "is_content_candidate",
            "is_user_candidate",
            "user_interaction_count",
        ]
    ] = (
        expanded_df[
            [
                "retriever_count",
                "is_bpr_candidate",
                "is_content_candidate",
                "is_user_candidate",
                "user_interaction_count",
            ]
        ]
        .astype(
            np.int16
        )
    )


    expanded_df[
        "item_popularity"
    ] = (
        expanded_df[
            "item_popularity"
        ]
        .astype(
            np.int32
        )
    )


    # -----------------------------------------------------
    # 1-H. Sanity Check
    # -----------------------------------------------------

    print(
        "\n===== New Feature Sanity Check ====="
    )

    print(
        expanded_df[
            NEW_FEATURES
        ]
        .describe()
        .T
        .to_string()
    )


    print(
        "\nRetriever Count 분포:"
    )

    print(
        expanded_df[
            "retriever_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    print(
        "\nSource Flag 평균:"
    )

    for col in [
        "is_bpr_candidate",
        "is_content_candidate",
        "is_user_candidate",
    ]:

        print(
            f"{col}: "
            f"{expanded_df[col].mean():.4f}"
        )


    # -----------------------------------------------------
    # 메모리 정리
    # -----------------------------------------------------

    del interaction_matrix
    del user_to_idx
    del game_to_idx
    del idx_to_game
    del bpr
    del content
    del user_cf

    gc.collect()


    return expanded_df


# =========================================================
# 2. Final Evaluation
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
        "\n===== XGBoost Final Evaluation ====="
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
        "=" * 70
    )

    print(
        " XGBoost Ranker - Context Feature Experiment"
    )

    print(
        " Candidate = BPR56 + Content39 + User5"
    )

    print(
        " Base = 4 Scores + 4 Tie-Aware Ranks"
    )

    print(
        " New  = Retriever Source + Popularity + User Activity"
    )

    print(
        " XGB  = Best A (depth=4)"
    )

    print(
        "=" * 70
    )


    # =====================================================
    # 1. 필수 파일
    # =====================================================

    required_paths = [
        BASE_FEATURE_CACHE_PATH,
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
    # 2. User Lists
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


    all_user_ids = (
        ltr_users[
            "user_id"
        ]
        .tolist()
        +
        final_eval_users[
            "user_id"
        ]
        .tolist()
    )


    print(
        "\nLTR Users:",
        len(
            ltr_users
        ),
    )


    print(
        "Final Users:",
        len(
            final_eval_users
        ),
    )


    # =====================================================
    # 3. Base Feature Cache
    # =====================================================

    print(
        "\n===== Base Tie-Aware Feature Cache ====="
    )


    base_feature_df = (
        pd.read_parquet(
            BASE_FEATURE_CACHE_PATH
        )
    )


    print(
        "Base Feature DF:",
        base_feature_df.shape,
    )


    required_base_cols = (
        [
            "user_id",
            "app_id",
            "label",
        ]
        +
        BASE_FEATURES
    )


    missing_base = [
        col

        for col
        in required_base_cols

        if col
        not in base_feature_df.columns
    ]


    if missing_base:

        raise ValueError(
            "Base feature cache에 "
            f"필요한 column 없음: {missing_base}"
        )


    # =====================================================
    # 4. Train / Test
    #
    # Expanded cache가 이미 있으면
    # source feature 재생성은 생략
    # =====================================================

    train_df = None
    test_df = None


    if (
        EXPANDED_FEATURE_CACHE_PATH.exists()
    ):

        print(
            "\n===== Expanded Feature Cache 로드 ====="
        )

        print(
            EXPANDED_FEATURE_CACHE_PATH
        )


        feature_df = (
            pd.read_parquet(
                EXPANDED_FEATURE_CACHE_PATH
            )
        )


        print(
            "Expanded Feature DF:",
            feature_df.shape,
        )


    else:

        print(
            "\n===== MF Train / Test ====="
        )


        (
            train_df,
            test_df,
        ) = (
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


        feature_df = (
            build_expanded_features(

                base_feature_df=
                    base_feature_df,

                train_df=
                    train_df,

                all_user_ids=
                    all_user_ids,
            )
        )


        feature_df.to_parquet(
            EXPANDED_FEATURE_CACHE_PATH,
            index=False,
        )


        print(
            "\nExpanded Feature 저장:"
        )


        print(
            EXPANDED_FEATURE_CACHE_PATH
        )


    # =====================================================
    # 5. 최종 Feature 검증
    # =====================================================

    required_cols = (
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
        in required_cols

        if col
        not in feature_df.columns
    ]


    if missing:

        raise ValueError(
            "Expanded feature에 "
            f"필요한 column 없음: {missing}"
        )


    print(
        "\n===== Feature Set ====="
    )


    for i, feature in enumerate(
        FEATURES,
        start=1,
    ):

        print(
            f"{i:02d}. "
            f"{feature}"
        )


    # =====================================================
    # 6. LTR / Final Split
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
    # 7. LTR 1000 -> 800 / 200
    #
    # 기존 실험과 동일 split
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
    # 8. XGBRanker
    #
    # 이전 Regularization Best A를 그대로 사용
    # feature 효과만 공정하게 비교
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
                500,

            learning_rate=
                0.05,

            max_depth=
                4,

            min_child_weight=
                1,

            subsample=
                0.8,

            colsample_bytree=
                0.8,

            reg_lambda=
                1.0,

            reg_alpha=
                0.0,

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
            10,
    )


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
    # 9. Save Model
    # =====================================================

    model.save_model(
        MODEL_PATH
    )


    print(
        "\n모델 저장:"
    )


    print(
        MODEL_PATH
    )


    # =====================================================
    # 10. Feature Importance
    # =====================================================

    importance_df = (
        pd.DataFrame(
            {
                "feature":
                    FEATURES,

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
        importance_df.to_string(
            index=False
        )
    )


    importance_df.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )


    # =====================================================
    # 11. Final 400 평가 준비
    #
    # Test를 아직 안 불렀다면 여기서 로드
    # =====================================================

    if test_df is None:

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
    # 12. Final 400 Evaluation
    # =====================================================

    print(
        "\n"
        +
        "=" * 70
    )


    print(
        " FINAL 400 USERS"
    )


    print(
        " Context Feature Experiment"
    )


    print(
        "=" * 70
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
        "\nFinal 결과 저장:"
    )


    print(
        RESULT_PATH
    )


    # =====================================================
    # 13. Previous Best Comparison
    # =====================================================

    print(
        "\n"
        +
        "=" * 70
    )


    print(
        " PREVIOUS BEST REFERENCE"
    )


    print(
        "=" * 70
    )


    print(
        "\nXGB Best A - 기존 8 Features"
    )


    print(
        "P@10    = 0.0890"
    )


    print(
        "R@10    = 0.1141"
    )


    print(
        "HR@10   = 0.5475"
    )


    print(
        "NDCG@10 = 0.1294"
    )


    print(
        "Hits     = 356"
    )


    # =====================================================
    # 14. Runtime
    # =====================================================

    print(
        "\n총 실행 시간:"
        f" {time.perf_counter() - total_start:.1f}초"
    )


if __name__ == "__main__":
    main()
