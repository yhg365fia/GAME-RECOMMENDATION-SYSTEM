from data_split import load_mf_split
from models.Funk_SVD import train_funk_svd


# ==========================================
# 1. Split 데이터 로드
# ==========================================

train_df, test_df = load_mf_split()

print("Train:", train_df.shape)
print("Test :", test_df.shape)


# ==========================================
# 2. 전체 Train 데이터 확인
# ==========================================

print("\n===== Train 데이터 정보 =====")
print("사용자 수:", train_df["user_id"].nunique())
print("게임 수:", train_df["app_id"].nunique())
print("\nis_recommended 분포:")
print(train_df["is_recommended"].value_counts())


# ==========================================
# 3. 전체 Train 데이터로 Funk SVD 학습
# ==========================================

model = train_funk_svd(
    train_df
)

print("\n전체 Train 데이터 Funk SVD 학습 완료")