import pandas as pd


import os
import pandas as pd

CACHE_DIR = "data/cache"


def load_games():

    cache_path = os.path.join(CACHE_DIR, "games.parquet")

    # 캐시가 있으면 캐시에서 바로 불러오기
    if os.path.exists(cache_path):
        print("캐시에서 games 데이터 불러오는 중...")
        return pd.read_parquet(cache_path)

    print("원본 CSV에서 games 데이터 읽는 중...")

    columns = [
        'app_id', 'Name', 'Release date', 'Estimated owners', 'Peak CCU',
        'Required age', 'Price', 'Discount', 'DLC count', 'About the game',
        'Supported languages', 'Full audio languages', 'Reviews', 'Header image',
        'Website', 'Support url', 'Support email', 'Windows', 'Mac', 'Linux',
        'Metacritic score', 'Metacritic url', 'User score', 'Positive', 'Negative',
        'Score rank', 'Achievements', 'Recommendations', 'Notes',
        'Average playtime forever', 'Average playtime two weeks',
        'Median playtime forever', 'Median playtime two weeks',
        'Developers', 'Publishers', 'Categories', 'Genres', 'Tags',
        'Screenshots', 'Movies'
    ]

    games = pd.read_csv(
        "data/games_inc.csv",
        header=0,
        names=columns
    )

    # 캐시 저장
    os.makedirs(CACHE_DIR, exist_ok=True)
    games.to_parquet(cache_path)
    print(f"캐시 저장 완료: {cache_path}")

    return games
import pandas as pd

# train.csv / test.csv에서 실제로 필요한 컬럼만 지정
USE_COLS = [
    "user_id",
    "app_id",          # AppID와 중복이므로 이것만 사용
    "Name",
    "Genres",
    "Tags",
    "Categories",
    "About the game",
    "hours",
    "is_recommended",
]

import os
import pandas as pd

USE_COLS = [
    "user_id",
    "app_id",
    "Name",
    "Genres",
    "Tags",
    "Categories",
    "About the game",
    "hours",
    "is_recommended",
]

DTYPES = {
    "user_id": "int32",
    "app_id": "int32",
    "Name": "string",
    "Genres": "string",
    "Tags": "string",
    "Categories": "string",
    "About the game": "string",
    "hours": "float32",
    "is_recommended": "boolean",
}

CACHE_DIR = "data/cache"


def _load_with_cache(csv_path: str, cache_path: str) -> pd.DataFrame:
    # 캐시 파일이 있으면 그걸 읽음 (훨씬 빠름)
    if os.path.exists(cache_path):
        print(f"캐시에서 불러오는 중... ({cache_path})")
        return pd.read_parquet(cache_path)

    # 캐시가 없으면 원본 CSV 읽고 캐시로 저장
    print(f"원본 CSV 읽는 중... ({csv_path})")
    df = pd.read_csv(
        csv_path,
        usecols=lambda c: c in USE_COLS,
        dtype=DTYPES,
    )

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"캐시 저장 완료 ({cache_path})")

    return df



def load_recommendations():
    """
    recommendations.csv 로딩 (Parquet 캐싱 적용)
    """
    cache_path = os.path.join(CACHE_DIR, "recommendations.parquet")

    if os.path.exists(cache_path):
        print("캐시에서 recommendations 데이터 불러오는 중...")
        return pd.read_parquet(cache_path)

    print("원본 CSV에서 recommendations 데이터 읽는 중...")
    user_history = pd.read_csv("data/recommendations.csv")

    os.makedirs(CACHE_DIR, exist_ok=True)
    user_history.to_parquet(cache_path)
    print(f"캐시 저장 완료: {cache_path}")

    return user_history

def load_train():
    print("Loading train data...")
    return _load_with_cache(
        "data/split/train.csv",
        os.path.join(CACHE_DIR, "train.parquet"),
    )


def load_test():
    print("Loading test data...")
    return _load_with_cache(
        "data/split/test.csv",
        os.path.join(CACHE_DIR, "test.parquet"),
    )
def preprocess(gamesgenres):

    # 필요한 컬럼만 선택
    meta = gamesgenres[
        ["app_id", "Name", "Genres", "Tags", "Categories", "About the game"]
    ].copy()

    # 이름 없는 게임 제거
    meta = meta.dropna(subset=["Name"])

    # Genres, Tags, Categories, About the game 중
    # 최소 3개 이상 값이 있는 데이터만 사용
    meta = meta.dropna(
        subset=["Genres", "Tags", "Categories", "About the game"],
        thresh=4
    )


    # Name 앞뒤 공백 제거 (dedup 정확도를 위해 먼저 정리)
    meta["Name"] = meta["Name"].str.strip()

    # Name 기준 중복 제거 (동일 이름의 게임 중 첫 번째만 유지)
    meta = meta.drop_duplicates(subset="Name", keep="first")
    meta = meta.reset_index(drop=True)

    # 결측치 처리
    text_columns = ["Genres", "Tags", "Categories", "About the game"]
    for col in text_columns:
        meta[col] = meta[col].fillna("").astype(str)

    # Combined Features 생성
    meta["combined_features"] = (
        meta["Genres"].str.replace(",", " ")
        + " "
        + meta["Tags"].str.replace(",", " ")
        + " "
        + meta["Categories"].str.replace(",", " ")
    )

    return meta