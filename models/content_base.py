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
            self.meta["combined_features"]
        )

        # 게임 이름 -> 인덱스 딕셔너리
        self.game_to_idx = (
            self.meta.reset_index()
                    .set_index("Name")["index"]
                    .to_dict()
    )
    def recommend(self, game_list, top_n=10):

        similarity_list = []
        played_indices = []

        # 입력한 게임마다 유사도 계산
        for game_name in game_list:

            # 같은 이름의 게임 찾기
            candidates = self.meta[self.meta["Name"] == game_name]

            if len(candidates) == 0:
                print(f"'{game_name}' 게임을 찾을 수 없습니다.")
                continue

            # 같은 이름의 게임이 여러 개면 첫 번째 사용
            idx = candidates.index[0]

            played_indices.append(idx)

            # 해당 게임과 모든 게임의 코사인 유사도
            sim_scores = cosine_similarity(
                self.tfidf_matrix[idx],
                self.tfidf_matrix
            ).flatten()

            similarity_list.append(sim_scores)

        # 입력한 게임을 하나도 찾지 못한 경우
        if len(similarity_list) == 0:
            print("입력한 게임을 찾을 수 없습니다.")
            return

        # 각 게임의 유사도를 평균
        mean_scores = np.mean(similarity_list, axis=0)

        # 유사도 높은 순 정렬
        sim_indices = mean_scores.argsort()[::-1]

        result = []
        used_names = set()

        for i in sim_indices:

            # 이미 입력한 게임 제외
            if i in played_indices:
                continue

            name = self.meta.iloc[i]["Name"]

            # 같은 이름의 게임 중복 제거
            if name in used_names:
                continue

            used_names.add(name)

            result.append({
                "AppID": self.meta.iloc[i]["AppID"],
                "Name": name,
                "Similarity": mean_scores[i]
            })

            if len(result) == top_n:
                break

        return pd.DataFrame(result)