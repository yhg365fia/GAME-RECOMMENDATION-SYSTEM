from surprise import Dataset, Reader, SVD


def train_funk_svd(
    train_df,
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
):
    """
    Funk SVD 학습

    train_df:
        user_id
        app_id
        is_recommended
    """

    ratings = train_df[
        ["user_id", "app_id", "is_recommended"]
    ].copy()

    # True / False -> 1 / 0
    ratings["is_recommended"] = (
        ratings["is_recommended"]
        .astype(int)
    )

    # rating 범위 설정
    reader = Reader(
        rating_scale=(0, 1)
    )

    # Surprise 데이터셋으로 변환
    data = Dataset.load_from_df(
        ratings[
            ["user_id", "app_id", "is_recommended"]
        ],
        reader
    )

    # train 전체를 학습 데이터로 사용
    trainset = data.build_full_trainset()

    # Funk SVD 모델
    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state
    )

    print("Funk SVD 학습 시작...")

    model.fit(trainset)

    print("Funk SVD 학습 완료")

    return model


def predict_score(
    model,
    user_id,
    app_id
):
    """
    특정 user-item의 예측 선호 점수
    """

    prediction = model.predict(
        user_id,
        app_id
    )

    return prediction.est


def recommend_funk_svd(
    model,
    user_id,
    candidate_items,
    top_n=10
):
    """
    후보 아이템 중 예측 점수가 높은 Top-N 추천
    """

    predictions = []

    for app_id in candidate_items:

        score = model.predict(
            user_id,
            app_id
        ).est

        predictions.append(
            (app_id, score)
        )

    # 높은 점수 순으로 정렬
    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # app_id만 반환
    recommendations = [
        app_id
        for app_id, score
        in predictions[:top_n]
    ]

    return recommendations