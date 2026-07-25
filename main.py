from models.content_base import ContentBasedRecommender
from preprocessing import load_games, load_train, preprocess


def main():

    # 1. 데이터 불러오기
    train_games = load_train()

    # 2. 데이터 전처리
    meta = preprocess(train_games)

    # 3. 추천 모델 생성
    recommender = ContentBasedRecommender()

    # 4. 모델 학습
    recommender.fit(meta)

    # 5. 사용자 입력
    # 5. 사용자 입력
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

    # 6. 추천 실행
    result = recommender.recommend(
        game_list=played_games,
        top_n=10
    )

    # 7. 결과 출력
    print(result)


if __name__ == "__main__":
    main()