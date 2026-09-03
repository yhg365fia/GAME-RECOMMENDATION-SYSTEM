import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from preprocessing import load_recommendations


# ==========================================
# 저장 경로
# ==========================================

SPLIT_DIR = "data/split"

MF_TRAIN_PATH = os.path.join(
    SPLIT_DIR,
    "mf_train.parquet"
)

MF_TEST_PATH = os.path.join(
    SPLIT_DIR,
    "mf_test.parquet"
)


# ==========================================
# 평가 대상 사용자 찾기
# ==========================================

def get_eligible_users(
    user_history,
    lower_bound=10,
    upper_bound=78
):
    user_counts = (
        user_history
        .groupby("user_id")
        .size()
        .reset_index(name="n_games")
    )

    eligible_users = user_counts[
        (user_counts["n_games"] >= lower_bound) &
        (user_counts["n_games"] <= upper_bound)
    ].copy()

    print(
        f"평가 대상 유저 수: "
        f"{len(eligible_users):,}"
    )

    print(
        eligible_users["n_games"].describe()
    )

    return eligible_users


# ==========================================
# MF Train / Test Split
# ==========================================

def make_mf_split(
    user_history,
    eligible_users,
    test_size=0.3,
    random_state=42
):
    """
    평가 대상 사용자(10~78 interactions):
        기존 방식과 동일하게 사용자별 70:30 split

    그 외 사용자:
        전부 Train

    최적화:
        사용자마다 train_test_split을 실행하지 않고
        interaction 개수별로 한 번씩만 실행
    """

    eligible_ids = set(
        eligible_users["user_id"]
    )

    # ------------------------------------------
    # 1. 70:30으로 나눌 사용자만 선택
    # ------------------------------------------

    target_history = user_history[
        user_history["user_id"].isin(eligible_ids)
    ].copy()

    print(
        f"\nSplit 대상 interaction: "
        f"{len(target_history):,}"
    )


    # ------------------------------------------
    # 2. 각 interaction이
    #    해당 사용자에서 몇 번째인지 계산
    # ------------------------------------------

    target_history["_position"] = (
        target_history
        .groupby("user_id")
        .cumcount()
    )

    target_history["_group_size"] = (
        target_history
        .groupby("user_id")["user_id"]
        .transform("size")
    )


    # ------------------------------------------
    # 3. interaction 개수별 test 위치 계산
    #
    # 예:
    # 10개 가진 사용자들은
    # train_test_split을 딱 한 번 실행
    #
    # 같은 random_state + 같은 데이터 개수면
    # 기존 코드와 같은 위치가 test로 선택됨
    # ------------------------------------------

    unique_sizes = np.sort(
        target_history["_group_size"].unique()
    )

    max_size = int(unique_sizes.max())

    # (group_size, position)을
    # 하나의 고유 숫자로 만들기 위한 base
    key_base = max_size + 1

    test_keys = []

    for n_games in unique_sizes:

        positions = np.arange(
            n_games
        )

        _, test_positions = train_test_split(
            positions,
            test_size=test_size,
            random_state=random_state
        )

        keys = (
            n_games * key_base
            + test_positions
        )

        test_keys.extend(keys)


    test_keys = np.asarray(
        test_keys,
        dtype=np.int32
    )


    # ------------------------------------------
    # 4. 전체 interaction의 key 생성
    # ------------------------------------------

    row_keys = (
        target_history["_group_size"]
        .to_numpy()
        * key_base
        +
        target_history["_position"]
        .to_numpy()
    )


    # ------------------------------------------
    # 5. 어떤 interaction이 test인지 판단
    # ------------------------------------------

    is_test = np.isin(
        row_keys,
        test_keys
    )


    # ------------------------------------------
    # 6. Test로 숨겨둘 원본 index
    # ------------------------------------------

    test_indices = (
        target_history
        .index[is_test]
    )


    # 임시 DataFrame은 이제 필요 없음
    del target_history
    del row_keys
    del is_test


    # ------------------------------------------
    # 7. 최종 Train / Test
    #
    # Test:
    # 평가 대상 사용자의 30%
    #
    # Train:
    # 그 나머지 전체 데이터
    # ------------------------------------------

    test_df = (
        user_history
        .loc[test_indices]
        .copy()
        .reset_index(drop=True)
    )

    train_df = (
        user_history
        .drop(index=test_indices)
        .reset_index(drop=True)
    )


    print("\n===== MF Split Result =====")

    print(
        f"전체 데이터 : "
        f"{len(user_history):,}"
    )

    print(
        f"Train      : "
        f"{len(train_df):,}"
    )

    print(
        f"Test       : "
        f"{len(test_df):,}"
    )

    print(
        f"Train + Test: "
        f"{len(train_df) + len(test_df):,}"
    )

    return train_df, test_df


# ==========================================
# 저장
# ==========================================

def save_mf_split(
    train_df,
    test_df
):
    os.makedirs(
        SPLIT_DIR,
        exist_ok=True
    )

    print("\nTrain 저장 중...")

    train_df.to_parquet(
        MF_TRAIN_PATH,
        index=False
    )

    print(
        f"Train 저장 완료: "
        f"{MF_TRAIN_PATH}"
    )


    print("\nTest 저장 중...")

    test_df.to_parquet(
        MF_TEST_PATH,
        index=False
    )

    print(
        f"Test 저장 완료: "
        f"{MF_TEST_PATH}"
    )


# ==========================================
# 불러오기
# ==========================================

def load_mf_split():

    print("MF split 불러오는 중...")

    train_df = pd.read_parquet(
        MF_TRAIN_PATH
    )

    test_df = pd.read_parquet(
        MF_TEST_PATH
    )

    print(
        f"Train: {train_df.shape}"
    )

    print(
        f"Test : {test_df.shape}"
    )

    return train_df, test_df


# ==========================================
# 직접 실행
# ==========================================

if __name__ == "__main__":

    # 전체 interaction 로드
    user_history = load_recommendations()

    # 10~78 interaction 사용자 전체
    eligible_users = get_eligible_users(
        user_history,
        lower_bound=10,
        upper_bound=78
    )

    # Split
    train_df, test_df = make_mf_split(
        user_history=user_history,
        eligible_users=eligible_users,
        test_size=0.3,
        random_state=42
    )

    # 저장
    save_mf_split(
        train_df,
        test_df
    )