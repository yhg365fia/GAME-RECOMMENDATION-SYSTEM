import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_interaction_matrix(user_history):
    """
    user_history(user_id, app_id, is_recommended)로부터
    User x Item sparse interaction matrix를 만든다.
    is_recommended: True -> +1, False -> -1, 상호작용 없음 -> 0
    """
    unique_users = user_history["user_id"].unique()
    unique_games = user_history["app_id"].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    game_to_idx = {aid: idx for idx, aid in enumerate(unique_games)}
    idx_to_game = {idx: aid for aid, idx in game_to_idx.items()}

    scores = user_history["is_recommended"].map({True: 1, False: -1}).values
    row_idx = user_history["user_id"].map(user_to_idx).values
    col_idx = user_history["app_id"].map(game_to_idx).values

    interaction_matrix = csr_matrix(
        (scores, (row_idx, col_idx)),
        shape=(len(unique_users), len(unique_games)),
    )

    return interaction_matrix, user_to_idx, game_to_idx, idx_to_game


class ItemBasedCFRecommender:
    """
    Item-based Collaborative Filtering 추천기.
    evaluation.py의 run_evaluation()이 기대하는
    recommend(app_id_list, top_n) 인터페이스를 따른다.
    """

    def __init__(self, interaction_matrix, game_to_idx, idx_to_game, k=30):
        self.interaction_matrix = interaction_matrix
        self.game_to_idx = game_to_idx
        self.idx_to_game = idx_to_game
        self.k = k
        self.n_games = interaction_matrix.shape[1]
        
        # User x Item -> Item x User
        self.item_matrix = interaction_matrix.T.tocsr()

    def recommend(self, app_id_list, top_n=10, exclude_user_idx=None):
        
        # 1. train 게임 목록으로만 쿼리 벡터 생성 (test 누수 방지)
        col_idx = [self.game_to_idx[a] for a in app_id_list if a in self.game_to_idx]
        if len(col_idx) == 0:
            return pd.DataFrame(columns=["app_id"])

        # 2. train에 있는 item들의 Item-User 벡터 가져오기
        source_items = self.item_matrix[col_idx]
        
        # 2. 전체 유저와 코사인 유사도 계산 (dense 변환 방지)
        sims = cosine_similarity(source_items, self.item_matrix, dense_output=False)
        
        for row_idx, item_idx in enumerate(col_idx):
            sims[row_idx, item_idx] = 0

        sims.eliminate_zeros()

        # 3. top-K 이웃 추출
        topk_sims_list = []
        topk_indices_list = []

        for row_idx in range(sims.shape[0]):
            row = sims.getrow(row_idx)

            positive_mask = row.data > 0
            positive_data = row.data[positive_mask]
            positive_indices = row.indices[positive_mask]

            if len(positive_data) == 0:
                continue

            k = min(self.k, len(positive_data))

            top_k_pos = np.argpartition(positive_data, -k)[-k:]

            topk_sims_list.append(positive_data[top_k_pos])
            topk_indices_list.append(positive_indices[top_k_pos])

        predicted_scores = np.zeros(self.item_matrix.shape[0])

        for topk_indices, topk_sims in zip(topk_indices_list, topk_sims_list):
            predicted_scores[topk_indices] += topk_sims

        # 5. 이미 상호작용한 게임(train에 있던 것)은 추천 후보에서 제외
        predicted_scores[col_idx] = -np.inf

        # 6. 근거 없는(0점 이하) 게임은 후보에서 제외 - 억지로 채우지 않음
        valid_idx = np.where(predicted_scores > 0)[0]
        if len(valid_idx) == 0:
            return pd.DataFrame(columns=["app_id"])

        # 7. 유효 후보 중 점수 높은 top_n개를 뽑아 순위대로 정렬
        n_actual = min(top_n, len(valid_idx))
        top_n_idx = valid_idx[np.argpartition(predicted_scores[valid_idx], -n_actual)[-n_actual:]]
        top_n_idx = top_n_idx[np.argsort(-predicted_scores[top_n_idx])]

        recommended_app_ids = [self.idx_to_game[i] for i in top_n_idx]

        return pd.DataFrame({"app_id": recommended_app_ids})

        