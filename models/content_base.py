import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


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

        # 게임 app_id -> 인덱스 딕셔너리
        self.appid_to_idx = (
            self.meta.reset_index()
                    .set_index("app_id")["index"]
                    .to_dict()
        )

    def recommend(self, app_id_list, top_n=10):
        """
        app_id_list: 사용자가 플레이한 게임들의 app_id 리스트
        """

        similarity_list = []
        played_indices = []

        # 입력한 게임마다 유사도 계산
        for app_id in app_id_list:

            if app_id not in self.appid_to_idx:
                print(f"app_id '{app_id}' 게임을 찾을 수 없습니다.")
                continue

            idx = self.appid_to_idx[app_id]
            played_indices.append(idx)

            sim_scores = cosine_similarity(
                self.tfidf_matrix[idx],
                self.tfidf_matrix
            ).flatten()

            similarity_list.append(sim_scores)

        if len(similarity_list) == 0:
            print("입력한 게임을 찾을 수 없습니다.")
            return

        mean_scores = np.mean(similarity_list, axis=0)
        sim_indices = mean_scores.argsort()[::-1]

        result = []
        used_appids = set()

        for i in sim_indices:

            # 이미 입력한 게임 제외
            if i in played_indices:
                continue

            app_id = self.meta.iloc[i]["app_id"]


            result.append({
                "app_id": app_id,
                "Name": self.meta.iloc[i]["Name"],
                "Similarity": mean_scores[i]
            })

            if len(result) == top_n:
                break

        return pd.DataFrame(result)