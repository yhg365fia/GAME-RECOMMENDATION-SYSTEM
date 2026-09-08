from models.content_base import ContentBasedRecommender
from models.itembase import ItemBasedCFRecommender
from models.userbase import build_interaction_matrix, UserBasedCFRecommender
from models.Funk_SVD import FunkSVDRecommender

from preprocessing import (
    load_games,
    load_recommendations,
    load_train,
    preprocess
)

from data_split import load_mf_split

from evaluation import (
    evaluate_pipeline,

    # Funk SVD 평가용
    build_mf_user_review_groups,
    stratified_sample_users,
    run_mf_evaluation,
    print_evaluation_report
)

import pandas as pd
import os


print(os.getcwd())


def resolve_name_to_appid(meta, name):
    """
    사용자가 입력한 게임 이름을 app_id로 변환.
    동명이인 게임이 있으면 사용자에게 선택하게 함.
    """

    candidates = meta[
        meta["Name"] == name
    ]

    if len(candidates) == 0:
        print(
            f"'{name}' 게임을 찾을 수 없습니다."
        )
        return None

    if len(candidates) == 1:
        return candidates[
            "app_id"
        ].iloc[0]

    print(
        f"\n'{name}'과 일치하는 게임이 여러 개 있습니다. "
        f"선택해주세요:"
    )

    candidates = candidates.reset_index(
        drop=True
    )

    for i, row in candidates.iterrows():

        print(
            f"  [{i}] "
            f"app_id: {row['app_id']} "
            f"- {row['Name']}"
        )

    while True:

        choice = input(
            "번호를 입력하세요: "
        ).strip()

        if (
            choice.isdigit()
            and int(choice) < len(candidates)
        ):

            return candidates.iloc[
                int(choice)
            ]["app_id"]

        print(
            "올바른 번호를 입력해주세요."
        )



def main():

    # ==========================================
    # 1. 데이터 로드
    # ==========================================

    meta = load_games()


    # ==========================================
    # [기존 CF 모델용 데이터 로드]
    # ==========================================

    # user_history = load_recommendations()


    # ==========================================
    # [기존 User / Item CF interaction matrix]
    # ==========================================

    # interaction_matrix, user_to_idx, game_to_idx, idx_to_game = \
    #     build_interaction_matrix(user_history)


    # ==========================================
    # [기존 Item-based CF]
    # ==========================================

    # recommender = ItemBasedCFRecommender(
    #     interaction_matrix,
    #     game_to_idx,
    #     idx_to_game,
    #     k=30
    # )


    # ==========================================
    # [기존 User-based CF]
    # ==========================================

    # recommender = UserBasedCFRecommender(
    #     interaction_matrix,
    #     game_to_idx,
    #     idx_to_game,
    #     k=30
    # )


    # ==========================================
    # [기존 Content-based]
    # ==========================================

    # recommender = ContentBasedRecommender(
    #     ...
    # )


    # ==========================================
    # 2. MF Train / Test 데이터 로드
    # ==========================================

    train_df, test_df = load_mf_split()

    print(
        "Train:",
        train_df.shape
    )

    print(
        "Test :",
        test_df.shape
    )


    # ==========================================
    # 3. Funk SVD 모델 준비
    # ==========================================

    recommender = FunkSVDRecommender(
        n_factors=40,
        n_epochs=5,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
        model_path="models/funk_svd_model.pkl"
    )


    # 저장 모델 존재
    # -> 모델 로드
    #
    # 저장 모델 없음
    # -> 전체 Train 학습
    # -> 모델 저장

    recommender.fit(
        train_df
    )


    # ==========================================
    # [기존 Item-based 추천 입력 방식]
    # ==========================================

    """
    played_games = []

    while True:

        game = input(
            "게임 이름: "
        ).strip()

        if game == "":
            break

        played_games.append(
            game
        )


    if len(played_games) == 0:

        print(
            "최소 1개의 게임을 입력해야 합니다."
        )

        return


    played_app_ids = []

    for game_name in played_games:

        app_id = resolve_name_to_appid(
            meta,
            game_name
        )

        if app_id is not None:

            played_app_ids.append(
                app_id
            )


    if len(played_app_ids) == 0:

        print(
            "유효한 게임이 없습니다."
        )

        return


    result = recommender.recommend(
        app_id_list=played_app_ids,
        top_n=10
    )


    result = result.merge(
        meta[
            ["app_id", "Name"]
        ],
        on="app_id",
        how="left"
    )


    print(
        "\n===== 추천 결과 ====="
    )

    print(
        result[
            ["app_id", "Name"]
        ].to_string(
            index=False
        )
    )
    """


    # ==========================================
    # 4. Funk SVD 추천 테스트
    # ==========================================

    # Test 데이터에 존재하는 사용자 중
    # 랜덤으로 유저 1명 선택
    #
    # drop_duplicates()
    # -> 같은 user_id를 한 번씩만 남겨서
    #    모든 유저가 동일한 확률로 선택되게 함

    user_id = (
        test_df[
            "user_id"
        ]
        .drop_duplicates()
        .sample(
            n=1
        )
        .iloc[0]
    )


    print(
        f"\n랜덤 선택 user_id: "
        f"{user_id}"
    )


    # ==========================================
    # 해당 사용자의 Train 상호작용
    # ==========================================

    played_app_ids = (
        train_df.loc[
            train_df[
                "user_id"
            ] == user_id,
            "app_id"
        ]
        .tolist()
    )


    if len(played_app_ids) == 0:

        print(
            f"user_id {user_id}는 "
            "Train 데이터에 존재하지 않습니다."
        )

        return


    print(
        f"Train 상호작용 게임 수: "
        f"{len(played_app_ids)}"
    )


    # ==========================================
    # 5. Funk SVD Top-10 추천
    # ==========================================

    result = recommender.recommend(
        user_id=user_id,
        app_id_list=played_app_ids,
        top_n=10
    )


    # 추천 AppID에 게임 이름 붙이기

    result = result.merge(
        meta[
            [
                "app_id",
                "Name"
            ]
        ],
        on="app_id",
        how="left"
    )


    print(
        "\n===== Funk SVD 추천 결과 ====="
    )


    print(
        result[
            [
                "app_id",
                "Name"
            ]
        ].to_string(
            index=False
        )
    )


    # ==========================================
    # 6. 해당 사용자의 실제 Test 정답 확인
    # ==========================================

    user_test = test_df[
        test_df[
            "user_id"
        ] == user_id
    ]


    test_result = (
        user_test[
            [
                "app_id",
                "is_recommended"
            ]
        ]
        .merge(
            meta[
                [
                    "app_id",
                    "Name"
                ]
            ],
            on="app_id",
            how="left"
        )
    )


    print(
        "\n===== 실제 Test 데이터 ====="
    )


    print(
        test_result[
            [
                "app_id",
                "Name",
                "is_recommended"
            ]
        ].to_string(
            index=False
        )
    )


    # ==========================================
    # [기존 평가 시스템]
    # ==========================================

    """
    valid_app_ids = set(
        meta["app_id"]
    )

    user_history = user_history[
        user_history[
            "app_id"
        ].isin(
            valid_app_ids
        )
    ]
    """


    # ==========================================
    # [기존 evaluate_pipeline 방식]
    # ==========================================

    """
    from evaluation import evaluate_pipeline

    eval_df, summary = evaluate_pipeline(
        recommender=recommender,
        user_history=user_history,
        lower_bound=10,
        upper_bound=78,
        n_groups=4,
        sample_per_group=100,
        top_n=10
    )
    """


    # ==========================================
    # [기존 User / Item CF 평가 방식]
    # ==========================================

    """
    from evaluation import (
        build_user_review_groups,
        stratified_sample_users,
        run_evaluation,
        print_evaluation_report
    )

    eligible_users = build_user_review_groups(
        user_history,
        lower_bound=10,
        upper_bound=78
    )

    sampled_users = stratified_sample_users(
        eligible_users,
        sample_per_group=100,
        random_state=42
    )

    eval_df = run_evaluation(
        recommender,
        user_history,
        sampled_users,
        user_to_idx=user_to_idx,
        top_n=10
    )

    summary = print_evaluation_report(
        eval_df,
        top_n=10
    )


    n_rec_summary = (
        eval_df
        .groupby(
            "review_group",
            observed=True
        )[
            "n_recommended"
        ]
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
        n_rec_summary
    )


    print(
        f"\n전체 n_recommended 평균: "
        f"{eval_df['n_recommended'].mean():.2f} / 10"
    )


    full_ratio = (
        eval_df[
            "n_recommended"
        ] == 10
    ).mean()


    print(
        f"10개 꽉 채워 추천된 유저 비율: "
        f"{full_ratio:.1%}"
    )
    """


    # ==========================================
    # 7. Funk SVD 평가
    # ==========================================

    print(
        "\n===== Funk SVD 평가 시작 ====="
    )


    # ------------------------------------------
    # 7-1. 평가 대상 사용자 구성
    # ------------------------------------------

    eligible_users = (
        build_mf_user_review_groups(
            train_df,
            test_df,
            lower_bound=10,
            upper_bound=78
        )
    )


    # ------------------------------------------
    # 7-2. 각 review group에서 100명씩
    # ------------------------------------------

    sampled_users = (
        stratified_sample_users(
            eligible_users,
            sample_per_group=100,
            random_state=42
        )
    )


    # ------------------------------------------
    # 7-3. Funk SVD 평가 실행
    # ------------------------------------------

    eval_df = run_mf_evaluation(
        recommender=recommender,
        train_df=train_df,
        test_df=test_df,
        sampled_users=sampled_users,
        top_n=10,
        positive_only=True
    )


    # ------------------------------------------
    # 7-4. 평가 결과 출력
    # ------------------------------------------

    summary = print_evaluation_report(
        eval_df,
        top_n=10
    )


    # ------------------------------------------
    # 7-5. 실제 추천 개수 확인
    # ------------------------------------------

    n_rec_summary = (
        eval_df
        .groupby(
            "review_group",
            observed=True
        )[
            "n_recommended"
        ]
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
        f"{eval_df['n_recommended'].mean():.2f} / 10"
    )


    full_ratio = (
        eval_df[
            "n_recommended"
        ] == 10
    ).mean()


    print(
        f"10개 꽉 채워 추천된 유저 비율: "
        f"{full_ratio:.1%}"
    )



if __name__ == "__main__":
    main()