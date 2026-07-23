import pandas as pd


def load_games():

    columns = [
        'AppID', 'Name', 'Release date', 'Estimated owners', 'Peak CCU',
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

    return games

def preprocess(gamesgenres):

    # 필요한 컬럼만 선택
    meta = gamesgenres[
        ["AppID", "Name", "Genres", "Tags", "Categories", "About the game"]
    ].copy()

    # 이름 없는 게임 제거
    meta = meta.dropna(subset=["Name"])

    # Genres, Tags, Categories, About the game 중
    # 최소 3개 이상 값이 있는 데이터만 사용
    meta = meta.dropna(
        subset=["Genres", "Tags", "Categories", "About the game"],
        thresh=4
    )

    # 결측치 처리
    text_columns = [
        "Genres",
        "Tags",
        "Categories",
        "About the game"
    ]

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
    meta = meta.reset_index(drop=True)

    return meta