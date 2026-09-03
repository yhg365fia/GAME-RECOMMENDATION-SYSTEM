from data_split import load_mf_split
from models.Funk_SVD import train_funk_svd


# ==========================================
# 1. Split 데이터 로드
# ==========================================

train_df, test_df = load_mf_split()

print("Train:", train_df.shape)
print("Test :", test_df.shape)


# ==========================================
# 2. 우선 작은 샘플로 정상 작동 확인
# ==========================================

train_sample = train_df.sample(
    n=100_000,
    random_state=42
)


# ==========================================
# 3. Funk SVD 학습
# ==========================================

model = train_funk_svd(
    train_sample
)



print(train_sample["user_id"].nunique())
print(train_sample["app_id"].nunique())
print(train_sample["is_recommended"].value_counts())