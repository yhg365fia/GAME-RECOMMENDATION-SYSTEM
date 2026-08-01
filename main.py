from models.content_base import ContentBasedRecommender
from preprocessing import load_games, load_recommendations, load_train, preprocess
from evaluation import evaluate_pipeline
import pandas as pd

import os

print(os.getcwd())

def resolve_name_to_appid(meta, name):
    """
    사용자가 입력한 게임 '이름'을 app_id로 변환.
    동명이인 게임이 있으면 사용자에게 선택하게 함.
    """
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

    games = load_games()
    meta = preprocess(games)   # 게임 단위로 dedup 된 meta (AppID 기준)

    recommender = ContentBasedRecommender()
    recommender.fit(meta)

    print("Steam Game Recommendation System")
    print("플레이했던 게임을 하나씩 입력하세요.")
    print("(입력을 끝내려면 그냥 Enter를 누르세요.)")

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
    
    #평가 시스템 적용

    user_history = load_recommendations()
    valid_app_ids = set(meta["app_id"])

    user_history = user_history[
    user_history["app_id"].isin(valid_app_ids)
]
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
if __name__ == "__main__":
    main()