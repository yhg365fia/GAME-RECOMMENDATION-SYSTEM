# hybrid_arctech_experiment/exp_d_xgb_ranker_exp1.py

import sys
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from implicit.cpu.bpr import BayesianPersonalizedRanking
from xgboost import XGBRanker


# =========================================================
# Project Root
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_split import load_mf_split
from models.itembase import build_interaction_matrix

import hybrid_arctech_experiment.exp_c_fusion_ablation as base


# =========================================================
# Config
# =========================================================

RANDOM_STATE = 42

# ---------------------------------------------------------
# Candidate Retriever
# ---------------------------------------------------------

BPR_N = 56
CONTENT_N = 39
USER_N = 5

CANDIDATE_SIZE = 100
TOP_N = 10

# ---------------------------------------------------------
# CF
# ---------------------------------------------------------

ITEM_K = 30
USER_K = 30

# ---------------------------------------------------------
# LTR Users
# ---------------------------------------------------------

LTR_USER_COUNT = 1000

# 1000명 중
# 800 = XGB train
# 200 = validation
VALID_RATIO = 0.20


# =========================================================
# Experiment 1 Features
# =========================================================

FEATURES = [
    "item_score_norm",
    "bpr_score_norm",
    "content_score_norm",
    "user_score_norm",

    "item_rank",
    "bpr_rank",
    "content_rank",
    "user_rank",
]


# =========================================================
# Paths
# =========================================================

SAVED_MODEL_DIR = ROOT / "models" / "saved_model"
RESULT_DIR = SAVED_MODEL_DIR / "results"
CACHE_DIR = SAVED_MODEL_DIR / "ltr_cache" / "xgb_exp1"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FINAL_EVAL_USERS_PATH = (
    RESULT_DIR
    / "hybrid_sampled_users.csv"
)

LTR_USERS_PATH = (
    CACHE_DIR
    / "ltr_train_users_1000.csv"
)

FEATURE_CACHE_PATH = (
    CACHE_DIR
    / "xgb_exp1_features.parquet"
)

MODEL_PATH = (
    SAVED_MODEL_DIR
    / "xgb_ranker_exp1.json"
)

RESULT_PATH = (
    RESULT_DIR
    / "xgb_ranker_exp1_eval.csv"
)


# =========================================================
# Utility
# =========================================================

def find_sorted_index(sorted_values, value):

    idx = int(
        np.searchsorted(
            sorted_values,
            value,
        )
    )

    if idx >= len(sorted_values):
        return None

    if sorted_values[idx] != value:
        return None

    return idx


def unique_preserve_order(values):

    seen = set()
    result = []

    for value in values:

        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def minmax_normalize(values):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    finite = np.isfinite(values)

    result = np.zeros(
        len(values),
        dtype=np.float32,
    )

    if not finite.any():
        return result

    valid = values[finite]

    min_v = valid.min()
    max_v = valid.max()

    if max_v - min_v < 1e-12:
        return result

    result[finite] = (
        values[finite] - min_v
    ) / (
        max_v - min_v
    )

    return result


def assign_rank(scores):

    """
    score 높은 순으로
    1, 2, 3, ... rank 부여
    """

    scores = np.asarray(scores)

    order = np.argsort(
        -scores,
        kind="stable",
    )

    ranks = np.empty(
        len(scores),
        dtype=np.int32,
    )

    ranks[order] = (
        np.arange(
            len(scores)
        )
        + 1
    )

    return ranks


# =========================================================
# 1. LTR 학습 사용자 1000명 추출
# =========================================================

def select_ltr_users(
    train_df,
    test_df,
    final_eval_users,
):

    if LTR_USERS_PATH.exists():

        print(
            "\nLTR 사용자 cache 로드:"
        )
        print(LTR_USERS_PATH)

        return pd.read_csv(
            LTR_USERS_PATH
        )

    print(
        "\n===== LTR 학습 사용자 1000명 선택 ====="
    )

    final_eval_ids = set(
        final_eval_users[
            "user_id"
        ].tolist()
    )

    # -----------------------------------------------------
    # Test에 positive interaction이 있는 사용자만
    # -----------------------------------------------------

    positive_test_users = (
        test_df.loc[
            test_df["is_recommended"] == True,
            "user_id",
        ]
        .drop_duplicates()
    )

    positive_test_users = (
        positive_test_users[
            ~positive_test_users.isin(
                final_eval_ids
            )
        ]
    )

    positive_set = set(
        positive_test_users.tolist()
    )

    # -----------------------------------------------------
    # Train interaction count
    # -----------------------------------------------------

    print(
        "Train interaction count 계산..."
    )

    counts = (
        train_df[
            train_df["user_id"].isin(
                positive_set
            )
        ]
        .groupby("user_id")
        .size()
        .rename("n_games")
        .reset_index()
    )

    # 기존 평가와 동일한 sparsity 구간
    def review_group(n):

        if 10 <= n <= 15:
            return "10-15개"

        if 16 <= n <= 25:
            return "16-25개"

        if 26 <= n <= 45:
            return "26-45개"

        if 46 <= n <= 78:
            return "46-78개"

        return None

    counts["review_group"] = (
        counts["n_games"]
        .map(review_group)
    )

    counts = (
        counts[
            counts["review_group"]
            .notna()
        ]
        .copy()
    )

    # -----------------------------------------------------
    # 250 × 4 = 1000
    # -----------------------------------------------------

    sampled = []

    for group_name in [
        "10-15개",
        "16-25개",
        "26-45개",
        "46-78개",
    ]:

        group_df = counts[
            counts["review_group"]
            == group_name
        ]

        if len(group_df) < 250:

            raise ValueError(
                f"{group_name} 사용자 부족: "
                f"{len(group_df)}"
            )

        part = group_df.sample(
            n=250,
            random_state=RANDOM_STATE,
        )

        sampled.append(part)

    sampled = pd.concat(
        sampled,
        ignore_index=True,
    )

    sampled.to_csv(
        LTR_USERS_PATH,
        index=False,
    )

    print(
        sampled["review_group"]
        .value_counts()
    )

    print(
        "저장:",
        LTR_USERS_PATH,
    )

    return sampled


# =========================================================
# 2. BPR
# =========================================================

class BPRScorer:

    def __init__(self):

        (
            model_path,
            mapping_path,
            user_items_path,
        ) = base.find_bpr_files()

        print(
            "\n===== BPR 로드 ====="
        )

        print(
            "Model:",
            model_path,
        )

        self.model = (
            BayesianPersonalizedRanking
            .load(
                str(model_path)
            )
        )

        mapping = np.load(
            mapping_path
        )

        self.user_ids = np.asarray(
            mapping["user_ids"]
        )

        self.item_ids = np.asarray(
            mapping["item_ids"]
        )

        self.user_items = (
            load_npz(
                user_items_path
            ).tocsr()
        )

    def user_index(
        self,
        user_id,
    ):

        return find_sorted_index(
            self.user_ids,
            user_id,
        )

    def item_indices(
        self,
        app_ids,
    ):

        app_ids = np.asarray(
            app_ids
        )

        idx = np.searchsorted(
            self.item_ids,
            app_ids,
        )

        valid = (
            idx
            <
            len(self.item_ids)
        )

        valid_positions = np.where(
            valid
        )[0]

        matched = np.zeros(
            len(app_ids),
            dtype=bool,
        )

        if len(valid_positions):

            matched[
                valid_positions
            ] = (
                self.item_ids[
                    idx[
                        valid_positions
                    ]
                ]
                ==
                app_ids[
                    valid_positions
                ]
            )

        result = np.full(
            len(app_ids),
            -1,
            dtype=np.int64,
        )

        result[matched] = (
            idx[matched]
        )

        return result

    def retrieve(
        self,
        user_id,
        n=BPR_N,
    ):

        user_idx = self.user_index(
            user_id
        )

        if user_idx is None:
            return []

        item_idx, scores = (
            self.model.recommend(
                userid=int(user_idx),

                user_items=(
                    self.user_items[
                        user_idx
                    ]
                ),

                N=n,

                filter_already_liked_items=True,
            )
        )

        return (
            self.item_ids[
                item_idx
            ]
            .tolist()
        )

    def score(
        self,
        user_id,
        candidate_ids,
    ):

        user_idx = self.user_index(
            user_id
        )

        result = np.zeros(
            len(candidate_ids),
            dtype=np.float32,
        )

        if user_idx is None:
            return result

        item_idx = self.item_indices(
            candidate_ids
        )

        valid = item_idx >= 0

        if not valid.any():
            return result

        user_factor = (
            self.model
            .user_factors[
                user_idx
            ]
        )

        item_factors = (
            self.model
            .item_factors[
                item_idx[valid]
            ]
        )

        result[valid] = (
            item_factors
            @
            user_factor
        )

        return result


# =========================================================
# 3. Content-Based
# =========================================================

class ContentScorer:

    def __init__(
        self,
        item_ids,
        train_subset,
    ):

        print(
            "\n===== Content Scorer 준비 ====="
        )

        (
            self.matrix,
            self.app_ids,
        ) = (
            base.build_or_load_content_matrix(
                item_ids
            )
        )

        self.matrix = (
            self.matrix.tocsr()
        )

        self.app_to_idx = {
            app_id: i
            for i, app_id
            in enumerate(
                self.app_ids
            )
        }

        positive_train = (
            train_subset[
                train_subset[
                    "is_recommended"
                ]
                ==
                True
            ]
        )

        self.positive_history = (
            positive_train
            .groupby("user_id")[
                "app_id"
            ]
            .apply(list)
            .to_dict()
        )

        self.seen_history = (
            train_subset
            .groupby("user_id")[
                "app_id"
            ]
            .apply(list)
            .to_dict()
        )

        self.profile_cache = {}

    def get_profile(
        self,
        user_id,
    ):

        if user_id in self.profile_cache:
            return self.profile_cache[
                user_id
            ]

        source_idx = [
            self.app_to_idx[
                app_id
            ]
            for app_id
            in self.positive_history.get(
                user_id,
                [],
            )
            if app_id
            in self.app_to_idx
        ]

        if not source_idx:

            profile = sp.csr_matrix(
                (
                    1,
                    self.matrix.shape[1],
                ),
                dtype=np.float32,
            )

        else:

            profile = sp.csr_matrix(
                self.matrix[
                    source_idx
                ].sum(
                    axis=0
                ),
                dtype=np.float32,
            )

            profile = normalize(
                profile,
                norm="l2",
                axis=1,
            )

        self.profile_cache[
            user_id
        ] = profile

        return profile

    def retrieve(
        self,
        user_id,
        n=CONTENT_N,
    ):

        profile = self.get_profile(
            user_id
        )

        if profile.nnz == 0:
            return []

        scores = (
            profile
            @
            self.matrix.T
        ).tocsr()

        row = scores.getrow(0)

        indices = row.indices
        values = row.data

        if len(indices) == 0:
            return []

        seen = set(
            self.seen_history.get(
                user_id,
                [],
            )
        )

        keep = np.fromiter(
            (
                self.app_ids[idx]
                not in seen
                for idx
                in indices
            ),
            dtype=bool,
            count=len(indices),
        )

        indices = indices[keep]
        values = values[keep]

        positive = values > 0

        indices = indices[positive]
        values = values[positive]

        if len(indices) == 0:
            return []

        n_actual = min(
            n,
            len(indices),
        )

        local = np.argpartition(
            -values,
            n_actual - 1,
        )[:n_actual]

        top_idx = indices[local]
        top_scores = values[local]

        order = np.argsort(
            -top_scores,
            kind="stable",
        )

        top_idx = top_idx[
            order
        ]

        return (
            self.app_ids[
                top_idx
            ]
            .tolist()
        )

    def score(
        self,
        user_id,
        candidate_ids,
    ):

        profile = self.get_profile(
            user_id
        )

        result = np.zeros(
            len(candidate_ids),
            dtype=np.float32,
        )

        if profile.nnz == 0:
            return result

        valid_positions = []
        valid_indices = []

        for pos, app_id in enumerate(
            candidate_ids
        ):

            idx = self.app_to_idx.get(
                app_id
            )

            if idx is None:
                continue

            valid_positions.append(
                pos
            )

            valid_indices.append(
                idx
            )

        if not valid_indices:
            return result

        scores = (
            profile
            @
            self.matrix[
                valid_indices
            ].T
        )

        scores = np.asarray(
            scores.toarray()
        ).ravel()

        result[
            valid_positions
        ] = scores

        return result


# =========================================================
# 4. User-Based CF
# =========================================================

class UserCFScorer:

    def __init__(
        self,
        interaction_matrix,
        user_to_idx,
        game_to_idx,
        idx_to_game,
        histories,
    ):

        print(
            "\n===== User-Based Scorer 준비 ====="
        )

        self.matrix = (
            interaction_matrix.tocsr()
        )

        self.user_to_idx = (
            user_to_idx
        )

        self.game_to_idx = (
            game_to_idx
        )

        self.idx_to_game = (
            idx_to_game
        )

        self.histories = histories

        # 각 train user vector norm
        self.user_norms = np.sqrt(
            np.asarray(
                self.matrix
                .multiply(
                    self.matrix
                )
                .sum(
                    axis=1
                )
            )
            .ravel()
        )

        self.score_cache = {}

    def calculate_scores(
        self,
        user_id,
    ):

        if user_id in self.score_cache:

            return self.score_cache[
                user_id
            ]

        app_ids = (
            self.histories.get(
                user_id,
                [],
            )
        )

        col_idx = [
            self.game_to_idx[
                app_id
            ]
            for app_id
            in app_ids
            if app_id
            in self.game_to_idx
        ]

        predicted_scores = np.zeros(
            self.matrix.shape[1],
            dtype=np.float32,
        )

        if not col_idx:

            self.score_cache[
                user_id
            ] = predicted_scores

            return predicted_scores

        # -------------------------------------------------
        # 기존 User-Based와 동일:
        # query는 history app을 1로 둔 binary vector
        # -------------------------------------------------

        query = sp.csr_matrix(
            (
                np.ones(
                    len(col_idx),
                    dtype=np.float32,
                ),
                (
                    np.zeros(
                        len(col_idx),
                        dtype=np.int32,
                    ),
                    col_idx,
                ),
            ),
            shape=(
                1,
                self.matrix.shape[1],
            ),
        )

        query_norm = np.sqrt(
            len(col_idx)
        )

        # sparse dot
        dots = (
            query
            @
            self.matrix.T
        ).tocsr()

        sims = dots.copy()

        if sims.nnz:

            denominator = (
                query_norm
                *
                self.user_norms[
                    sims.indices
                ]
            )

            valid = denominator > 0

            sims.data[valid] /= (
                denominator[valid]
            )

            sims.data[
                ~valid
            ] = 0

        # 자기 자신 제외
        user_idx = (
            self.user_to_idx
            .get(
                user_id
            )
        )

        if user_idx is not None:

            sims[
                0,
                user_idx,
            ] = 0

            sims.eliminate_zeros()

        if sims.nnz == 0:

            self.score_cache[
                user_id
            ] = predicted_scores

            return predicted_scores

        # 기존 User-Based:
        # similarity >0만 필터링하지 않고
        # 가장 큰 k개 사용
        k = min(
            USER_K,
            len(sims.data),
        )

        top_pos = np.argpartition(
            sims.data,
            -k,
        )[-k:]

        neighbor_sims = (
            sims.data[
                top_pos
            ]
        )

        neighbor_indices = (
            sims.indices[
                top_pos
            ]
        )

        neighbor_matrix = (
            self.matrix[
                neighbor_indices
            ]
        )

        weighted_sum = np.asarray(
            neighbor_sims
            @
            neighbor_matrix
        ).ravel()

        sim_sum = np.abs(
            neighbor_sims
        ).sum()

        if sim_sum > 0:

            predicted_scores = (
                weighted_sum
                /
                sim_sum
            ).astype(
                np.float32,
                copy=False,
            )

        else:

            predicted_scores = (
                weighted_sum.astype(
                    np.float32,
                    copy=False,
                )
            )

        # 이미 본 게임 제외
        predicted_scores[
            col_idx
        ] = -np.inf

        self.score_cache[
            user_id
        ] = predicted_scores

        return predicted_scores

    def retrieve(
        self,
        user_id,
        n=USER_N,
    ):

        scores = self.calculate_scores(
            user_id
        )

        valid_idx = np.where(
            np.isfinite(scores)
            &
            (scores > 0)
        )[0]

        if len(valid_idx) == 0:
            return []

        n_actual = min(
            n,
            len(valid_idx),
        )

        local = np.argpartition(
            -scores[
                valid_idx
            ],
            n_actual - 1,
        )[:n_actual]

        selected = valid_idx[
            local
        ]

        order = np.argsort(
            -scores[
                selected
            ],
            kind="stable",
        )

        selected = selected[
            order
        ]

        return [
            self.idx_to_game[
                idx
            ]
            for idx
            in selected
        ]

    def score(
        self,
        user_id,
        candidate_ids,
    ):

        scores = self.calculate_scores(
            user_id
        )

        result = np.zeros(
            len(candidate_ids),
            dtype=np.float32,
        )

        for pos, app_id in enumerate(
            candidate_ids
        ):

            idx = (
                self.game_to_idx
                .get(
                    app_id
                )
            )

            if idx is None:
                continue

            value = scores[
                idx
            ]

            if np.isfinite(
                value
            ):
                result[pos] = value

        return result


# =========================================================
# 5. Item-Based CF
# =========================================================

class ItemCFScorer:

    def __init__(
        self,
        interaction_matrix,
        game_to_idx,
        idx_to_game,
        histories,
    ):

        print(
            "\n===== Item-Based Scorer 준비 ====="
        )

        self.game_to_idx = (
            game_to_idx
        )

        self.idx_to_game = (
            idx_to_game
        )

        self.histories = histories

        # Item × User
        item_matrix = (
            interaction_matrix
            .T
            .tocsr()
            .astype(
                np.float32
            )
        )

        print(
            "Item matrix normalize..."
        )

        self.item_matrix = (
            normalize(
                item_matrix,
                norm="l2",
                axis=1,
                copy=False,
            )
        )

        # source item별 Top30 neighbor cache
        self.neighbor_cache = {}

    def get_neighbors(
        self,
        item_idx,
    ):

        if item_idx in self.neighbor_cache:

            return self.neighbor_cache[
                item_idx
            ]

        row = (
            self.item_matrix[
                item_idx
            ]
            @
            self.item_matrix.T
        ).tocsr()

        # self similarity 제거
        if row.nnz:

            self_mask = (
                row.indices
                ==
                item_idx
            )

            row.data[
                self_mask
            ] = 0

            row.eliminate_zeros()

        if row.nnz == 0:

            result = (
                np.array(
                    [],
                    dtype=np.int64,
                ),
                np.array(
                    [],
                    dtype=np.float32,
                ),
            )

            self.neighbor_cache[
                item_idx
            ] = result

            return result

        # 기존 Item-Based와 동일:
        # positive similarity만
        positive_mask = (
            row.data > 0
        )

        indices = (
            row.indices[
                positive_mask
            ]
        )

        values = (
            row.data[
                positive_mask
            ]
        )

        if len(indices) == 0:

            result = (
                np.array(
                    [],
                    dtype=np.int64,
                ),
                np.array(
                    [],
                    dtype=np.float32,
                ),
            )

            self.neighbor_cache[
                item_idx
            ] = result

            return result

        k = min(
            ITEM_K,
            len(values),
        )

        top_pos = np.argpartition(
            values,
            -k,
        )[-k:]

        indices = indices[
            top_pos
        ]

        values = values[
            top_pos
        ]

        result = (
            indices,
            values,
        )

        self.neighbor_cache[
            item_idx
        ] = result

        return result

    def calculate_scores(
        self,
        user_id,
    ):

        app_ids = self.histories.get(
            user_id,
            [],
        )

        source_idx = [
            self.game_to_idx[
                app_id
            ]
            for app_id
            in app_ids
            if app_id
            in self.game_to_idx
        ]

        predicted_scores = np.zeros(
            self.item_matrix.shape[0],
            dtype=np.float32,
        )

        if not source_idx:
            return predicted_scores

        for item_idx in source_idx:

            neighbor_idx, sims = (
                self.get_neighbors(
                    item_idx
                )
            )

            if len(neighbor_idx):

                predicted_scores[
                    neighbor_idx
                ] += sims

        # 이미 본 게임 제외
        predicted_scores[
            source_idx
        ] = -np.inf

        return predicted_scores

    def score(
        self,
        user_id,
        candidate_ids,
    ):

        scores = self.calculate_scores(
            user_id
        )

        result = np.zeros(
            len(candidate_ids),
            dtype=np.float32,
        )

        for pos, app_id in enumerate(
            candidate_ids
        ):

            idx = (
                self.game_to_idx
                .get(
                    app_id
                )
            )

            if idx is None:
                continue

            value = scores[
                idx
            ]

            if np.isfinite(
                value
            ):
                result[pos] = value

        return result


# =========================================================
# 6. Candidate + 8 Feature 생성
# =========================================================

def build_feature_rows(
    users,
    positive_test_dict,
    bpr,
    content,
    user_cf,
    item_cf,
):

    rows = []

    print(
        "\n===== XGBoost Feature 생성 ====="
    )

    for count, user_id in enumerate(
        users,
        start=1,
    ):

        # -------------------------------------------------
        # Candidate Retriever
        # -------------------------------------------------

        bpr_candidates = (
            bpr.retrieve(
                user_id,
                BPR_N,
            )
        )

        content_candidates = (
            content.retrieve(
                user_id,
                CONTENT_N,
            )
        )

        user_candidates = (
            user_cf.retrieve(
                user_id,
                USER_N,
            )
        )

        candidates = (
            unique_preserve_order(
                bpr_candidates
                +
                content_candidates
                +
                user_candidates
            )
        )

        candidates = (
            candidates[
                :CANDIDATE_SIZE
            ]
        )

        if not candidates:
            continue

        # -------------------------------------------------
        # 모든 Candidate에 대해 4개 모델 score
        # -------------------------------------------------

        bpr_score = bpr.score(
            user_id,
            candidates,
        )

        content_score = (
            content.score(
                user_id,
                candidates,
            )
        )

        user_score = (
            user_cf.score(
                user_id,
                candidates,
            )
        )

        item_score = (
            item_cf.score(
                user_id,
                candidates,
            )
        )

        # -------------------------------------------------
        # score normalization
        # 사용자 후보군 내부에서 0~1
        # -------------------------------------------------

        item_score_norm = (
            minmax_normalize(
                item_score
            )
        )

        bpr_score_norm = (
            minmax_normalize(
                bpr_score
            )
        )

        content_score_norm = (
            minmax_normalize(
                content_score
            )
        )

        user_score_norm = (
            minmax_normalize(
                user_score
            )
        )

        # -------------------------------------------------
        # 같은 Candidate UNION을
        # 각 모델 score로 다시 정렬해서 rank 생성
        # -------------------------------------------------

        item_rank = assign_rank(
            item_score
        )

        bpr_rank = assign_rank(
            bpr_score
        )

        content_rank = assign_rank(
            content_score
        )

        user_rank = assign_rank(
            user_score
        )

        relevant = positive_test_dict.get(
            user_id,
            set(),
        )

        for i, app_id in enumerate(
            candidates
        ):

            label = int(
                app_id
                in relevant
            )

            rows.append(
                {
                    "user_id": user_id,
                    "app_id": app_id,

                    "item_score_norm":
                        float(
                            item_score_norm[i]
                        ),

                    "bpr_score_norm":
                        float(
                            bpr_score_norm[i]
                        ),

                    "content_score_norm":
                        float(
                            content_score_norm[i]
                        ),

                    "user_score_norm":
                        float(
                            user_score_norm[i]
                        ),

                    "item_rank":
                        int(
                            item_rank[i]
                        ),

                    "bpr_rank":
                        int(
                            bpr_rank[i]
                        ),

                    "content_rank":
                        int(
                            content_rank[i]
                        ),

                    "user_rank":
                        int(
                            user_rank[i]
                        ),

                    "label":
                        label,
                }
            )

        if (
            count % 50 == 0
            or
            count == len(users)
        ):

            print(
                f"Feature 생성: "
                f"{count}/{len(users)}"
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# 7. XGB용 정렬
# =========================================================

def prepare_ranker_data(
    df,
):

    df = (
        df
        .sort_values(
            [
                "user_id",
                "app_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    X = df[
        FEATURES
    ].astype(
        np.float32
    )

    y = (
        df["label"]
        .astype(
            np.float32
        )
    )

    groups = (
        df
        .groupby(
            "user_id",
            sort=False,
        )
        .size()
        .to_numpy()
    )

    return (
        df,
        X,
        y,
        groups,
    )


# =========================================================
# 8. Evaluation
# =========================================================

def evaluate_ranker(
    model,
    eval_df,
    positive_test_dict,
):

    rows = []

    for user_id, group in (
        eval_df
        .groupby(
            "user_id",
            sort=False,
        )
    ):

        group = group.copy()

        scores = model.predict(
            group[
                FEATURES
            ]
            .astype(
                np.float32
            )
        )

        group[
            "xgb_score"
        ] = scores

        group = (
            group
            .sort_values(
                "xgb_score",
                ascending=False,
                kind="stable",
            )
        )

        top = (
            group
            .head(
                TOP_N
            )
        )

        recommended = (
            top[
                "app_id"
            ]
            .tolist()
        )

        relevant = (
            positive_test_dict.get(
                user_id,
                set(),
            )
        )

        hits = sum(
            app_id in relevant
            for app_id
            in recommended
        )

        precision = (
            hits
            /
            len(recommended)
            if recommended
            else 0.0
        )

        recall = (
            hits
            /
            len(relevant)
            if relevant
            else 0.0
        )

        hit_rate = float(
            hits > 0
        )

        # NDCG@10
        dcg = 0.0

        for rank, app_id in enumerate(
            recommended,
            start=1,
        ):

            if app_id in relevant:

                dcg += (
                    1.0
                    /
                    np.log2(
                        rank + 1
                    )
                )

        ideal_hits = min(
            len(relevant),
            TOP_N,
        )

        idcg = sum(
            1.0
            /
            np.log2(
                rank + 1
            )
            for rank
            in range(
                1,
                ideal_hits + 1,
            )
        )

        ndcg = (
            dcg / idcg
            if idcg > 0
            else 0.0
        )

        rows.append(
            {
                "user_id":
                    user_id,

                "precision":
                    precision,

                "recall":
                    recall,

                "hit_rate":
                    hit_rate,

                "ndcg":
                    ndcg,

                "hits":
                    hits,

                "n_test":
                    len(
                        relevant
                    ),

                "n_recommended":
                    len(
                        recommended
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        "\n===== XGBoost Ranker Evaluation ====="
    )

    print(
        f"Precision@10: "
        f"{result['precision'].mean():.4f}"
    )

    print(
        f"Recall@10:    "
        f"{result['recall'].mean():.4f}"
    )

    print(
        f"HR@10:        "
        f"{result['hit_rate'].mean():.4f}"
    )

    print(
        f"NDCG@10:      "
        f"{result['ndcg'].mean():.4f}"
    )

    print(
        f"Hits:         "
        f"{result['hits'].sum()}"
    )

    return result


# =========================================================
# Main
# =========================================================

def main():

    total_start = (
        time.perf_counter()
    )

    print(
        "=" * 60
    )

    print(
        " XGBoost Ranker - Experiment 1"
    )

    print(
        " Candidate = BPR56 + Content39 + User5"
    )

    print(
        " Features  = 4 Scores + 4 Ranks"
    )

    print(
        "=" * 60
    )

    # =====================================================
    # 1. Global MF Split
    # =====================================================

    print(
        "\n===== MF Train / Test ====="
    )

    train_df, test_df = (
        load_mf_split()
    )

    print(
        "Train:",
        train_df.shape,
    )

    print(
        "Test :",
        test_df.shape,
    )

    # =====================================================
    # 2. 기존 최종 평가 400명
    # =====================================================

    final_eval_users = (
        pd.read_csv(
            FINAL_EVAL_USERS_PATH
        )
    )

    print(
        "\nFinal Evaluation Users:",
        len(
            final_eval_users
        ),
    )

    # =====================================================
    # 3. 별도 LTR 학습 사용자 1000명
    # =====================================================

    ltr_users = select_ltr_users(
        train_df,
        test_df,
        final_eval_users,
    )

    # 반드시 겹치면 안 됨
    overlap = (
        set(
            ltr_users[
                "user_id"
            ]
        )
        &
        set(
            final_eval_users[
                "user_id"
            ]
        )
    )

    if overlap:

        raise ValueError(
            f"LTR 사용자와 최종 평가 사용자 "
            f"겹침: {len(overlap)}"
        )

    # =====================================================
    # 4. 총 1400명의 history/test만 추출
    # =====================================================

    all_user_ids = (
        ltr_users[
            "user_id"
        ].tolist()
        +
        final_eval_users[
            "user_id"
        ].tolist()
    )

    train_subset = (
        train_df[
            train_df[
                "user_id"
            ]
            .isin(
                all_user_ids
            )
        ]
        .copy()
    )

    test_subset = (
        test_df[
            test_df[
                "user_id"
            ]
            .isin(
                all_user_ids
            )
        ]
        .copy()
    )

    histories = (
        train_subset
        .groupby(
            "user_id"
        )[
            "app_id"
        ]
        .apply(list)
        .to_dict()
    )

    positive_test = (
        test_subset[
            test_subset[
                "is_recommended"
            ]
            ==
            True
        ]
    )

    positive_test_dict = (
        positive_test
        .groupby(
            "user_id"
        )[
            "app_id"
        ]
        .apply(
            lambda x: set(
                x.tolist()
            )
        )
        .to_dict()
    )

    # =====================================================
    # 5. Global Interaction Matrix
    # =====================================================

    print(
        "\n===== Interaction Matrix ====="
    )

    (
        interaction_matrix,
        user_to_idx,
        game_to_idx,
        idx_to_game,
    ) = build_interaction_matrix(
        train_df
    )

    print(
        "Matrix:",
        interaction_matrix.shape,
    )

    # =====================================================
    # 6. Scorers
    # =====================================================

    bpr = BPRScorer()

    content = ContentScorer(
        item_ids=bpr.item_ids,
        train_subset=train_subset,
    )

    user_cf = UserCFScorer(
        interaction_matrix=
            interaction_matrix,

        user_to_idx=
            user_to_idx,

        game_to_idx=
            game_to_idx,

        idx_to_game=
            idx_to_game,

        histories=
            histories,
    )

    item_cf = ItemCFScorer(
        interaction_matrix=
            interaction_matrix,

        game_to_idx=
            game_to_idx,

        idx_to_game=
            idx_to_game,

        histories=
            histories,
    )

    # =====================================================
    # 7. Feature DataFrame
    # =====================================================

    if FEATURE_CACHE_PATH.exists():

        print(
            "\nFeature cache 로드:"
        )

        print(
            FEATURE_CACHE_PATH
        )

        feature_df = (
            pd.read_parquet(
                FEATURE_CACHE_PATH
            )
        )

    else:

        feature_df = (
            build_feature_rows(
                users=all_user_ids,

                positive_test_dict=
                    positive_test_dict,

                bpr=bpr,

                content=content,

                user_cf=user_cf,

                item_cf=item_cf,
            )
        )

        feature_df.to_parquet(
            FEATURE_CACHE_PATH,
            index=False,
        )

        print(
            "\nFeature 저장:"
        )

        print(
            FEATURE_CACHE_PATH
        )

    print(
        "\nFeature DF:",
        feature_df.shape,
    )

    print(
        feature_df[
            FEATURES
            +
            ["label"]
        ]
        .head()
    )

    # =====================================================
    # 8. LTR 1000명 / Final 400명 분리
    # =====================================================

    ltr_ids = set(
        ltr_users[
            "user_id"
        ]
    )

    final_ids = set(
        final_eval_users[
            "user_id"
        ]
    )

    ltr_df = (
        feature_df[
            feature_df[
                "user_id"
            ]
            .isin(
                ltr_ids
            )
        ]
        .copy()
    )

    final_df = (
        feature_df[
            feature_df[
                "user_id"
            ]
            .isin(
                final_ids
            )
        ]
        .copy()
    )

    # =====================================================
    # 9. 1000명 -> Train 800 / Validation 200
    #
    # 반드시 USER 기준 split
    # row 기준 split 금지
    # =====================================================

    unique_ltr_users = (
        ltr_users[
            "user_id"
        ]
        .to_numpy()
    )

    (
        train_users,
        valid_users,
    ) = train_test_split(
        unique_ltr_users,

        test_size=
            VALID_RATIO,

        random_state=
            RANDOM_STATE,
    )

    train_rank_df = (
        ltr_df[
            ltr_df[
                "user_id"
            ]
            .isin(
                train_users
            )
        ]
        .copy()
    )

    valid_rank_df = (
        ltr_df[
            ltr_df[
                "user_id"
            ]
            .isin(
                valid_users
            )
        ]
        .copy()
    )

    (
        train_rank_df,
        X_train,
        y_train,
        train_group,
    ) = prepare_ranker_data(
        train_rank_df
    )

    (
        valid_rank_df,
        X_valid,
        y_valid,
        valid_group,
    ) = prepare_ranker_data(
        valid_rank_df
    )

    print(
        "\n===== Ranker Dataset ====="
    )

    print(
        "Train users:",
        len(train_group),
    )

    print(
        "Train rows :",
        len(X_train),
    )

    print(
        "Valid users:",
        len(valid_group),
    )

    print(
        "Valid rows :",
        len(X_valid),
    )

    print(
        "Positive labels(train):",
        int(
            y_train.sum()
        ),
    )

    print(
        "Positive labels(valid):",
        int(
            y_valid.sum()
        ),
    )

    # =====================================================
    # 10. XGBoost Ranker
    # =====================================================

    print(
        "\n===== XGBRanker 학습 ====="
    )

    model = XGBRanker(

        objective=
            "rank:ndcg",

        eval_metric=
            "ndcg@10",

        n_estimators=
            300,

        learning_rate=
            0.05,

        max_depth=
            6,

        min_child_weight=
            1,

        subsample=
            0.8,

        colsample_bytree=
            0.8,

        reg_lambda=
            1.0,

        random_state=
            RANDOM_STATE,

        tree_method=
            "hist",
    )

    model.fit(

        X_train,
        y_train,

        group=
            train_group,

        eval_set=[
            (
                X_valid,
                y_valid,
            )
        ],

        eval_group=[
            valid_group
        ],

        verbose=20,
    )

    model.save_model(
        MODEL_PATH
    )

    print(
        "\n모델 저장:",
        MODEL_PATH,
    )

    # =====================================================
    # 11. Feature Importance
    # =====================================================

    importance_df = (
        pd.DataFrame(
            {
                "feature":
                    FEATURES,

                "importance":
                    model.feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    print(
        "\n===== Feature Importance ====="
    )

    print(
        importance_df.to_string(
            index=False
        )
    )

    # =====================================================
    # 12. 기존 400명 FINAL 평가
    # =====================================================

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        " FINAL 400 USERS"
    )

    print(
        " 이 400명은 XGB 학습에 사용하지 않음"
    )

    print(
        "=" * 60
    )

    final_result = (
        evaluate_ranker(
            model=model,

            eval_df=
                final_df,

            positive_test_dict=
                positive_test_dict,
        )
    )

    final_result.to_csv(
        RESULT_PATH,
        index=False,
    )

    print(
        "\n결과 저장:",
        RESULT_PATH,
    )

    print(
        "\n비교 기준"
    )

    print(
        "현재 Item Ranker Hybrid"
    )

    print(
        "P@10   = 0.0872"
    )

    print(
        "R@10   = 0.1110"
    )

    print(
        "HR@10  = 0.5400"
    )

    print(
        "NDCG@10= 0.1216"
    )

    print(
        "\n총 실행 시간:"
        f" {time.perf_counter() - total_start:.1f}초"
    )


if __name__ == "__main__":
    main()