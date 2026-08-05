from models.content_base import ContentBasedRecommender
from models.userbase import build_interaction_matrix, UserBasedCFRecommender
from preprocessing import load_games, load_recommendations, load_train, preprocess
from evaluation import evaluate_pipeline
import pandas as pd

import os

print(os.getcwd())
"""
def resolve_name_to_appid(meta, name):
    
    사용자가 입력한 게임 '이름'을 app_id로 변환.
    동명이인 게임이 있으면 사용자에게 선택하게 함.
    
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

"""
def main():

    
    # =========================================
    # Recommender 생성부
    # 아래 두 블록 중 하나만 활성화해서 사용
    # =========================================

    # --- [Content-based] ---
    # recommender = ContentBasedRecommender(...)   # 기존에 쓰던 생성 코드로 교체

    # --- [User-based CF] ---
    from models.userbase import build_interaction_matrix, UserBasedCFRecommender
    user_history = load_recommendations()
    
    interaction_matrix, user_to_idx, game_to_idx, idx_to_game = build_interaction_matrix(user_history)
    recommender = UserBasedCFRecommender(interaction_matrix, game_to_idx, idx_to_game, k=30)

    # =========================================
    # 게임 입력받아 단건 추천 (기존 로직 그대로)
    # =========================================
    """
    played_games = []
    while True:
        game = input("게임 이름: ").strip()
        if game == "":
            break
        played_games.append(game)

    if len(played_games) == 0:
        print("최소 1개의 게임을 입력해야 합니다.")
        return

    # 이름 -> AppID 변환 (여기서 중복 처리까지 완료됨)
    played_app_ids = []
    for game_name in played_games:
        app_id = resolve_name_to_appid(meta, game_name)
        if app_id is not None:
            played_app_ids.append(app_id)

    if len(played_app_ids) == 0:
        print("유효한 게임이 없습니다.")
        return

    # recommend()는 오직 AppID 리스트만 받음
    result = recommender.recommend(app_id_list=played_app_ids, top_n=10)
    print(result)
    """

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

    eval_df = run_evaluation(recommender, user_history, sampled_users, top_n=10)
    summary = print_evaluation_report(eval_df, top_n=10)


if __name__ == "__main__":
    main()