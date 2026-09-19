import sys
from pathlib import Path

# ==========================================
# 0. Project Root 설정
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# 1. 기존 프로젝트 코드 import
# ==========================================

from data_split import load_mf_split

from models.itembase import (
    build_interaction_matrix,
    ItemBasedCFRecommender,
)

from evaluation import (
    build_mf_user_review_groups,
    stratified_sample_users,
    run_mf_evaluation,
    print_evaluation_report,
)


# ==========================================
# 2. 실험 설정
# ==========================================

ITEM_K = 30
TOP_N = 10

LOWER_BOUND = 10
UPPER_BOUND = 78

SAMPLE_PER_GROUP = 100
RANDOM_STATE = 42

# Hybrid 실험에서는
# True interaction만 Test 정답으로 통일
POSITIVE_ONLY = True


# ==========================================
# 3. 결과 저장 경로
# ==========================================

RESULT_DIR = (
    PROJECT_ROOT
    / "models"
    / "saved_model"
    / "results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVAL_PATH = (
    RESULT_DIR
    / "hybrid_baseline_item_eval.csv"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "hybrid_baseline_item_summary.csv"
)

SAMPLED_USERS_PATH = (
    RESULT_DIR
    / "hybrid_sampled_users.csv"
)


# ==========================================
# 4. Item-Based Adapter
# ==========================================
#
# run_mf_evaluation()은
#
# recommender.recommend(
#     user_id=...,
#     app_id_list=...,
#     top_n=...
# )
#
# 형태를 사용함.
#
# 기존 ItemBasedCFRecommender는
#
# recommend(
#     app_id_list,
#     top_n,
#     exclude_user_idx=None
# )
#
# 이므로 Adapter로 연결.
#
# Item-Based는 user_id 자체가 필요 없으므로
# 그냥 무시한다.
# ==========================================

class ItemBasedMFAdapter:

    def __init__(self, model):
        self.model = model

    def recommend(
        self,
        user_id=None,
        app_id_list=None,
        top_n=10
    ):
        return self.model.recommend(
            app_id_list=app_id_list,
            top_n=top_n
        )


# ==========================================
# 5. Main
# ==========================================

def main():

    print(
        "\n"
        "==========================================\n"
        " Hybrid Experiment - Item-Based Baseline\n"
        "=========================================="
    )


    # ======================================
    # 5-1. 이미 저장된 MF Split 로드
    # ======================================
    #
    # 여기서는 split을 새로 하지 않는다.
    #
    # data/split/
    # ├─ mf_train.parquet
    # └─ mf_test.parquet
    #
    # 기존 파일을 그대로 불러옴.
    # ======================================

    print(
        "\n===== 1. 기존 MF Train / Test 로드 ====="
    )

    train_df, test_df = load_mf_split()

    print(
        "Train:",
        train_df.shape
    )

    print(
        "Test :",
        test_df.shape
    )


    # ======================================
    # 5-2. Train 데이터로 Item-Based Matrix 생성
    # ======================================
    #
    # Test는 모델 생성에 절대 사용하지 않음.
    # ======================================

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
        interaction_matrix.shape
    )

    print(
        "Users:",
        len(user_to_idx)
    )

    print(
        "Games:",
        len(game_to_idx)
    )


    # ======================================
    # 5-3. 기존 Item-Based 모델 생성
    # ======================================

    print(
        "\n===== 3. Item-Based CF 생성 ====="
    )

    item_model = ItemBasedCFRecommender(
        interaction_matrix=interaction_matrix,
        game_to_idx=game_to_idx,
        idx_to_game=idx_to_game,
        k=ITEM_K
    )

    recommender = ItemBasedMFAdapter(
        item_model
    )


    # ======================================
    # 5-4. 평가 사용자 준비
    # ======================================
    #
    # 데이터 split이 아님.
    #
    # 이미 만들어진 Train/Test 중
    # 평가할 400명을 선택하는 과정.
    #
    # 첫 실행 때만 저장하고,
    # 이후 Hybrid A/B/C에서는
    # 이 CSV를 그대로 재사용한다.
    # ======================================

    print(
        "\n===== 4. 평가 사용자 준비 ====="
    )

    if SAMPLED_USERS_PATH.exists():

        print(
            "기존 Hybrid 평가 사용자 로드:"
        )

        print(
            SAMPLED_USERS_PATH
        )

        import pandas as pd

        sampled_users = pd.read_csv(
            SAMPLED_USERS_PATH
        )

    else:

        eligible_users = (
            build_mf_user_review_groups(
                train_df,
                test_df,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND
            )
        )

        sampled_users = (
            stratified_sample_users(
                eligible_users,
                sample_per_group=SAMPLE_PER_GROUP,
                random_state=RANDOM_STATE
            )
        )

        sampled_users.to_csv(
            SAMPLED_USERS_PATH,
            index=False
        )

        print(
            "\nHybrid 평가 사용자 저장 완료:"
        )

        print(
            SAMPLED_USERS_PATH
        )


    print(
        "\n평가 사용자 수:",
        len(sampled_users)
    )


    # ======================================
    # 5-5. Item-Based Baseline 평가
    # ======================================

    print(
        "\n===== 5. Item-Based Baseline 평가 ====="
    )

    eval_df = run_mf_evaluation(
        recommender=recommender,
        train_df=train_df,
        test_df=test_df,
        sampled_users=sampled_users,
        top_n=TOP_N,
        positive_only=POSITIVE_ONLY
    )


    # ======================================
    # 5-6. 평가 결과 출력
    # ======================================

    print(
        "\n===== 6. 평가 결과 ====="
    )

    summary = print_evaluation_report(
        eval_df,
        top_n=TOP_N
    )


    # ======================================
    # 5-7. 추천 개수 확인
    # ======================================

    n_rec_summary = (
        eval_df
        .groupby(
            "review_group",
            observed=True
        )["n_recommended"]
        .agg(
            [
                "mean",
                "min",
                "max",
                "count"
            ]
        )
    )

    print(
        "\n===== 그룹별 추천 개수 ====="
    )

    print(
        n_rec_summary
    )


    print(
        f"\n전체 n_recommended 평균: "
        f"{eval_df['n_recommended'].mean():.2f} / {TOP_N}"
    )


    full_ratio = (
        eval_df["n_recommended"]
        == TOP_N
    ).mean()


    print(
        f"{TOP_N}개 꽉 채워 추천된 유저 비율: "
        f"{full_ratio:.1%}"
    )


    # ======================================
    # 5-8. 결과 저장
    # ======================================

    eval_df.to_csv(
        EVAL_PATH,
        index=False
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )


    print(
        "\n===== 결과 저장 완료 ====="
    )

    print(
        EVAL_PATH
    )

    print(
        SUMMARY_PATH
    )


    # ======================================
    # 5-9. 핵심 지표 한 번 더 출력
    # ======================================

    print(
        "\n===== Item-Based Hybrid Baseline ====="
    )

    print(
        f"Precision@{TOP_N}: "
        f"{eval_df['precision'].mean():.6f}"
    )

    print(
        f"Recall@{TOP_N}:    "
        f"{eval_df['recall'].mean():.6f}"
    )

    print(
        f"Hit Rate@{TOP_N}:  "
        f"{eval_df['hit'].mean():.6f}"
    )

    print(
        f"NDCG@{TOP_N}:      "
        f"{eval_df['ndcg'].mean():.6f}"
    )

    print(
        f"Hits:              "
        f"{eval_df['hits'].sum()}"
    )


    print(
        "\n=========================================="
    )

    print(
        " Item-Based Baseline Complete"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()