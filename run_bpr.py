import pandas as pd

from models.bpr import BPR


# ==========================================
# 1. 기존 MF Split 직접 로드
# ==========================================

train_df = pd.read_parquet(
    "data/split/mf_train.parquet"
)

test_df = pd.read_parquet(
    "data/split/mf_test.parquet"
)

print("Train:", train_df.shape)
print("Test :", test_df.shape)


# ==========================================
# 2. 데이터 확인
# ==========================================

print("\n===== Train 데이터 =====")

print(
    "사용자 수:",
    train_df["user_id"].nunique()
)

print(
    "게임 수:",
    train_df["app_id"].nunique()
)

print(
    "\nis_recommended:"
)

print(
    train_df["is_recommended"]
    .value_counts()
)


# ==========================================
# 3. 전체 Item Universe
# ==========================================

all_items = (
    train_df["app_id"]
    .dropna()
    .unique()
)

print(
    "\n전체 Item Universe:",
    len(all_items)
)


# ==========================================
# 4. 테스트용 1000명
# ==========================================

sample_users = (
    train_df["user_id"]
    .drop_duplicates()
    .sample(
        n=1000,
        random_state=42
    )
    .to_numpy()
)


train_small = train_df[
    train_df["user_id"]
    .isin(sample_users)
].copy()


print(
    "\n===== Small Sample ====="
)

print(
    "Users:",
    train_small["user_id"].nunique()
)

print(
    "Interactions:",
    len(train_small)
)


# ==========================================
# 5. BPR
# ==========================================

model = BPR(
    n_factors=40,
    n_epochs=5,
    learning_rate=0.05,
    reg=0.001,
    negative_mode="unseen",
    random_state=42
)


# ==========================================
# 6. 학습
# ==========================================

model.fit(
    train_small,
    all_items
)


# ==========================================
# 7. 추천 테스트
# ==========================================

test_user = sample_users[0]

recommendations = model.recommend(
    test_user,
    k=10
)


print(
    f"\n===== User {test_user} 추천 ====="
)

for rank, (app_id, score) in enumerate(
    recommendations,
    start=1
):
    print(
        f"{rank:2d}. "
        f"app_id={app_id} "
        f"score={score:.4f}"
    )


# ==========================================
# 8. Seen Item 검사
# ==========================================

seen_items = set(
    train_small.loc[
        train_small["user_id"] == test_user,
        "app_id"
    ]
)

recommended_items = {
    app_id
    for app_id, score
    in recommendations
}

overlap = (
    seen_items
    & recommended_items
)

print(
    "\nSeen item overlap:",
    overlap
)

if not overlap:
    print(
        "정상: 이미 본 게임이 추천되지 않음"
    )
else:
    print(
        "문제: seen item이 추천됨"
    )