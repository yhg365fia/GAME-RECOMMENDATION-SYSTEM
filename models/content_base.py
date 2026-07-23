from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class ContentBasedRecommender:

    def __init__(self):
        self.meta = None
        self.tfidf = None
        self.tfidf_matrix = None

    def fit(self, meta):
        self.meta = meta

        self.tfidf = TfidfVectorizer(
            stop_words="english",
            lowercase=True
        )

        self.tfidf_matrix = self.tfidf.fit_transform(
            meta["combined_features"]
        )
    def recommend(self, game_name, appid=None, top_n=10):

        # 같은 이름의 게임 찾기
        candidates = self.meta[self.meta["Name"] == game_name]

        if len(candidates) == 0:
            print(f"'{game_name}' 게임을 찾을 수 없습니다.")
            return

        if len(candidates) > 1 and appid is None:
            print("같은 이름의 게임이 여러 개 있습니다.")
            print(candidates[["AppID", "Name"]])
            print("\n원하는 AppID를 입력하여 다시 실행하세요.")
            return

        if appid is not None:
            candidates = candidates[candidates["AppID"] == appid]

            if len(candidates) == 0:
                print("해당 AppID를 찾을 수 없습니다.")
                return

        # 추천 기준 게임 인덱스
        idx = candidates.index[0]

        # 유사도 계산
        sim_scores = cosine_similarity(
            self.tfidf_matrix[idx],
            self.tfidf_matrix
        ).flatten()

        # 유사도 높은 순 정렬
        sim_indices = sim_scores.argsort()[::-1]

        result = []
        used_names = set()

        for i in sim_indices:

            # 자기 자신 제외
            if i == idx:
                continue

            name = self.meta.iloc[i]["Name"]

            # 이미 추가한 게임이면 건너뜀
            if name in used_names:
                continue

            used_names.add(name)

            result.append({
                "AppID": self.meta.iloc[i]["AppID"],
                "Name": name,
                "Similarity": sim_scores[i]
            })

            # top_n개 채우면 종료
            if len(result) == top_n:
                break

        return pd.DataFrame(result)