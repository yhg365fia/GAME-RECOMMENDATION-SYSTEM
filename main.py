from models.content_base import ContentBasedRecommender
from models.itembase import ItemBasedCFRecommender
from models.userbase import build_interaction_matrix, UserBasedCFRecommender
from preprocessing import load_games, load_recommendations, load_train, preprocess
from evaluation import evaluate_pipeline
import pandas as pd

import os

print(os.getcwd())

def resolve_name_to_appid(meta, name):
    
    """사용자가 입력한 게임 '이름'을 app_id로 변환.
    동명이인 게임이 있으면 사용자에게 선택하게 함.1"""
    
    candidates = meta[meta["Name"] == name]

    if len(candidates) == 0:
        print(f"'{name}' 게임을 찾을 수 없습니다.")
        return None

    if len(candidates) == 1:
        return candidates["app_id"].iloc[0]

    # 이름이 중복되는 경우 -> 사용자에게 AppID 선택하게 함
    print(f"\n'{name}'과 일치하는 게임이 여러 개 있습니다. 선택해주세요:")
    candidates = candidates.reset_index(drop=True)
    for i, row in candidates.iterrows():
        print(f"  [{i}] app_id: {row['app_id']} - {row['Name']}")

    while True:
        choice = input("번호를 입력하세요: ").strip()
        if choice.isdigit() and int(choice) < len(candidates):
            return candidates.iloc[int(choice)]["app_id"]
        print("올바른 번호를 입력해주세요.")


def main():

    meta = load_games()
    user_history = load_recommendations()

    interaction_matrix, user_to_idx, game_to_idx, idx_to_game = \
        build_interaction_matrix(user_history)

    recommender = ItemBasedCFRecommender(
        interaction_matrix,
        game_to_idx,
        idx_to_game,
        k=30
    )

    played_games = []

    while True:
        game = input("게임 이름: ").strip()

        if game == "":
            break

        played_games.append(game)

    if len(played_games) == 0:
        print("최소 1개의 게임을 입력해야 합니다.")
        return

    played_app_ids = []

    for game_name in played_games:
        app_id = resolve_name_to_appid(meta, game_name)

        if app_id is not None:
            played_app_ids.append(app_id)

    if len(played_app_ids) == 0:
        print("유효한 게임이 없습니다.")
        return

    result = recommender.recommend(
    app_id_list=played_app_ids,
    top_n=10
)

# 추천 AppID에 게임 이름 붙이기
    result = result.merge(
        meta[["app_id", "Name"]],
        on="app_id",
        how="left"
    )

    print("\n===== 추천 결과 =====")
    print(result[["app_id", "Name"]].to_string(index=False))


    # =========================================
    # 평가 시스템 적용
    # =========================================
    ###valid_app_ids = set(meta["app_id"])

    #user_history = user_history[
    #    user_history["app_id"].isin(valid_app_ids)
    #]

    # --- [기존 evaluate_pipeline 방식 - 시그니처 불일치로 미사용, 참고용 보관] ---
    """
    from evaluation import evaluate_pipeline

    eval_df, summary = evaluate_pipeline(
        recommender=recommender,
        user_history=user_history,   # user_id, app_id 컬럼 포함
        lower_bound=10,
        upper_bound=78,
        n_groups=4,
        sample_per_group=100,
        top_n=10
        )
    """
    # --- [현재 사용하는 평가 방식] ---
    from evaluation import build_user_review_groups, stratified_sample_users, run_evaluation, print_evaluation_report
    
    eligible_users = build_user_review_groups(user_history, lower_bound=10, upper_bound=78)
    sampled_users = stratified_sample_users(eligible_users, sample_per_group=100, random_state=42)

    eval_df = run_evaluation(recommender, user_history, sampled_users, user_to_idx=user_to_idx, top_n=10)
    summary = print_evaluation_report(eval_df, top_n=10)

    # n_recommended가 그룹별로 평균 몇 개였는지 확인
    n_rec_summary = eval_df.groupby("review_group", observed=True)["n_recommended"].agg(
        ["mean", "min", "max", "count"]
    )
    print(n_rec_summary)

    # 전체 평균도 같이
    print(f"\n전체 n_recommended 평균: {eval_df['n_recommended'].mean():.2f} / 10")

    # 10개 다 채워진 유저 비율도 확인
    full_ratio = (eval_df["n_recommended"] == 10).mean()
    print(f"10개 꽉 채워 추천된 유저 비율: {full_ratio:.1%}")
        
if __name__ == "__main__":
    main()