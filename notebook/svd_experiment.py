# Surprise 라이브러리에서 필요한 클래스와 메서드 불러오기
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

from preprocessing import load_recommendations


# ==============================
# 1. 데이터 로드
# ==============================
ratings = load_recommendations()

# Surprise는
# user / item / rating
# 3개의 컬럼만 사용
ratings = ratings[
    ["user_id", "app_id", "is_recommended"]
]

# 우선 실험용으로 100,000개 샘플링
ratings = ratings.sample(
    n=100_000,
    random_state=42
)

# is_recommended
# False -> 0
# True  -> 1
ratings["is_recommended"] = (
    ratings["is_recommended"]
    .astype(int)
)

print(ratings.head())
print(ratings.shape)


# ==============================
# 2. Surprise Dataset 생성
# ==============================

# 현재 rating 값이 0 또는 1이므로
# rating_scale도 (0, 1)
reader = Reader(
    rating_scale=(0, 1)
)

data = Dataset.load_from_df(
    ratings[
        ["user_id", "app_id", "is_recommended"]
    ],
    reader
)


# ==============================
# 3. SVD 모델 생성
# ==============================

svd = SVD(
    random_state=42
)


# ==============================
# 4. Cross Validation
# ==============================

results = cross_validate(
    svd,
    data,
    measures=["RMSE", "MAE"],
    cv=5,
    verbose=True
)


# ==============================
# 5. 평균 성능 출력
# ==============================

print("\n===== SVD Evaluation =====")

print(
    "Mean RMSE:",
    results["test_rmse"].mean()
)

print(
    "Mean MAE:",
    results["test_mae"].mean()
)