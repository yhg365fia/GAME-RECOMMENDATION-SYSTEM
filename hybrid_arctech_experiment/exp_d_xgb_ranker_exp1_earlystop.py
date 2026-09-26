# hybrid_arctech_experiment/exp_d_xgb_ranker_exp1_earlystop.py

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


# 기존 Exp1 함수/설정 재사용
import hybrid_arctech_experiment.exp_d_xgb_ranker_exp1 as exp1

from data_split import load_mf_split


# =========================================================
# Config
# =========================================================

RANDOM_STATE = 42
VALID_RATIO = 0.20
EARLY_STOPPING_ROUNDS = 30


# =========================================================
# Paths
# =========================================================

SAVED_MODEL_DIR = ROOT / "models" / "saved_model"
RESULT_DIR = SAVED_MODEL_DIR / "results"
CACHE_DIR = SAVED_MODEL_DIR / "ltr_cache" / "xgb_exp1"

FINAL_EVAL_USERS_PATH = RESULT_DIR / "hybrid_sampled_users.csv"
LTR_USERS_PATH = CACHE_DIR / "ltr_train_users_1000.csv"
FEATURE_CACHE_PATH = CACHE_DIR / "xgb_exp1_features.parquet"

MODEL_PATH = SAVED_MODEL_DIR / "xgb_ranker_exp1_earlystop.json"
RESULT_PATH = RESULT_DIR / "xgb_ranker_exp1_earlystop_eval.csv"


# =========================================================
# Main
# =========================================================

def main():

    start = time.perf_counter()

    print("=" * 60)
    print(" XGBoost Ranker - Experiment 1.1")
    print(" Early Stopping Only")
    print(" Candidate = BPR56 + Content39 + User5")
    print(" Features  = 4 Scores + 4 Ranks")
    print(f" Early stopping rounds = {EARLY_STOPPING_ROUNDS}")
    print("=" * 60)


    # -----------------------------------------------------
    # 1. 기존 cache 확인
    # -----------------------------------------------------

    for path in [
        FINAL_EVAL_USERS_PATH,
        LTR_USERS_PATH,
        FEATURE_CACHE_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"필수 파일 없음: {path}"
            )


    # -----------------------------------------------------
    # 2. Train/Test 로드
    # -----------------------------------------------------

    print("\n===== MF Train / Test =====")

    train_df, test_df = load_mf_split()

    print("Train:", train_df.shape)
    print("Test :", test_df.shape)


    # -----------------------------------------------------
    # 3. 기존 사용자 목록
    # -----------------------------------------------------

    ltr_users = pd.read_csv(
        LTR_USERS_PATH
    )

    final_eval_users = pd.read_csv(
        FINAL_EVAL_USERS_PATH
    )

    print("\nLTR Users:", len(ltr_users))
    print(
        "Final Evaluation Users:",
        len(final_eval_users),
    )


    # -----------------------------------------------------
    # 4. 기존 Feature Cache 로드
    # -----------------------------------------------------

    print(
        "\n===== 기존 Exp1 Feature Cache 로드 ====="
    )

    print(FEATURE_CACHE_PATH)

    feature_df = pd.read_parquet(
        FEATURE_CACHE_PATH
    )

    print(
        "Feature DF:",
        feature_df.shape,
    )


    # -----------------------------------------------------
    # 5. LTR / Final 분리
    # -----------------------------------------------------

    ltr_ids = set(
        ltr_users["user_id"]
    )

    final_ids = set(
        final_eval_users["user_id"]
    )

    overlap = ltr_ids & final_ids

    if overlap:
        raise ValueError(
            f"LTR / Final 사용자 겹침: "
            f"{len(overlap)}"
        )

    ltr_df = (
        feature_df[
            feature_df["user_id"]
            .isin(ltr_ids)
        ]
        .copy()
    )

    final_df = (
        feature_df[
            feature_df["user_id"]
            .isin(final_ids)
        ]
        .copy()
    )

    print(
        "LTR Feature Rows:",
        len(ltr_df),
    )

    print(
        "Final Feature Rows:",
        len(final_df),
    )


    # -----------------------------------------------------
    # 6. Positive Test Dictionary
    # -----------------------------------------------------

    all_eval_ids = (
        ltr_ids
        |
        final_ids
    )

    positive_test = (
        test_df[
            (
                test_df["user_id"]
                .isin(all_eval_ids)
            )
            &
            (
                test_df["is_recommended"]
                == True
            )
        ]
    )

    positive_test_dict = (
        positive_test
        .groupby("user_id")["app_id"]
        .apply(
            lambda x: set(
                x.tolist()
            )
        )
        .to_dict()
    )


    # -----------------------------------------------------
    # 7. 기존과 동일한 User Split
    #    1000 -> Train 800 / Valid 200
    # -----------------------------------------------------

    unique_ltr_users = (
        ltr_users["user_id"]
        .to_numpy()
    )

    train_users, valid_users = (
        train_test_split(
            unique_ltr_users,
            test_size=VALID_RATIO,
            random_state=RANDOM_STATE,
        )
    )

    train_rank_df = (
        ltr_df[
            ltr_df["user_id"]
            .isin(train_users)
        ]
        .copy()
    )

    valid_rank_df = (
        ltr_df[
            ltr_df["user_id"]
            .isin(valid_users)
        ]
        .copy()
    )

    (
        train_rank_df,
        X_train,
        y_train,
        train_group,
    ) = exp1.prepare_ranker_data(
        train_rank_df
    )

    (
        valid_rank_df,
        X_valid,
        y_valid,
        valid_group,
    ) = exp1.prepare_ranker_data(
        valid_rank_df
    )

    print(
        "\n===== Ranker Dataset ====="
    )

    print(
        "Train users:",
        len(train_group),
    )

    print(
        "Train rows :",
        len(X_train),
    )

    print(
        "Valid users:",
        len(valid_group),
    )

    print(
        "Valid rows :",
        len(X_valid),
    )

    print(
        "Positive labels(train):",
        int(y_train.sum()),
    )

    print(
        "Positive labels(valid):",
        int(y_valid.sum()),
    )


    # -----------------------------------------------------
    # 8. XGBRanker
    #
    # 기존 Exp1과 동일
    # 단 Early Stopping만 추가
    # -----------------------------------------------------

    print(
        "\n===== XGBRanker 학습 ====="
    )

    model = XGBRanker(

        objective="rank:ndcg",
        eval_metric="ndcg@10",

        n_estimators=300,

        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,

        subsample=0.8,
        colsample_bytree=0.8,

        reg_lambda=1.0,

        random_state=RANDOM_STATE,

        tree_method="hist",

        # ★ 이번 실험의 유일한 핵심 변경
        early_stopping_rounds=
            EARLY_STOPPING_ROUNDS,
    )


    # -----------------------------------------------------
    # 9. 학습
    # -----------------------------------------------------

    model.fit(

        X_train,
        y_train,

        group=train_group,

        eval_set=[
            (
                X_valid,
                y_valid,
            )
        ],

        eval_group=[
            valid_group
        ],

        verbose=10,
    )


    # -----------------------------------------------------
    # 10. Early Stopping 결과
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # 11. 모델 저장
    # -----------------------------------------------------

    model.save_model(
        MODEL_PATH
    )

    print(
        "\n모델 저장:",
        MODEL_PATH,
    )


    # -----------------------------------------------------
    # 12. Feature Importance
    # -----------------------------------------------------

    importance_df = (
        pd.DataFrame(
            {
                "feature":
                    exp1.FEATURES,

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


    # -----------------------------------------------------
    # 13. Final 400 평가
    #
    # predict()는 Early Stopping 사용 시
    # best_iteration 기준으로 예측
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print(" FINAL 400 USERS")
    print(" 이 400명은 XGB 학습에 사용하지 않음")
    print("=" * 60)

    final_result = (
        exp1.evaluate_ranker(

            model=model,

            eval_df=final_df,

            positive_test_dict=
                positive_test_dict,
        )
    )


    # -----------------------------------------------------
    # 14. 결과 저장
    # -----------------------------------------------------

    final_result.to_csv(
        RESULT_PATH,
        index=False,
    )

    print(
        "\n결과 저장:",
        RESULT_PATH,
    )


    # -----------------------------------------------------
    # 15. 비교 기준
    # -----------------------------------------------------

    print(
        "\n===== 기존 결과 비교 ====="
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
        "\nXGBoost Exp1"
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
        "\n총 실행 시간:"
        f" {time.perf_counter() - start:.1f}초"
    )


if __name__ == "__main__":
    main()
