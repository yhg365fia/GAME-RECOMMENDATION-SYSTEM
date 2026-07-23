from models.content_base import ContentBasedRecommender
from preprocessing import load_games, preprocess


def main():

    # 1. 데이터 불러오기
    games = load_games()

    # 2. 데이터 전처리
    meta = preprocess(games)

    # 3. 추천 모델 생성
    recommender = ContentBasedRecommender()

    # 4. 모델 학습
    recommender.fit(meta)

    # 5. 사용자 입력
    print("Steam Game Recommendation System")
    print("1. Content-Based Recommendation System")
    game_name = input("추천 기준 게임 입력: ")

    # 6. 추천 실행
    result = recommender.recommend(
        game_name=game_name,
        top_n=10
    )

    # 7. 결과 출력
    print(result)


if __name__ == "__main__":
    main()