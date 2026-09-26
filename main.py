# main.py
# ============================================================
# Steam Game Recommendation System - Final Runtime
#
# Final model: E_HR
#
# Candidate Retrieval
#   Item-CF   46
#   BPR       20
#   Content   15
#   User-CF   20
#   -> Candidate Union (max 101)
#
# Feature
#   Full15 = Full14 + is_item_candidate
#
# Ranker
#   XGBoost LambdaMART
#   xgb_ranker_final_e_hr.json
#
# New User
#   Played-game BPR item embedding mean
#   -> user-vector initialization
#   -> user-only BPR fine-tuning
#
# Output
#   Top-10 Recommendation
#   + Mean Log Popularity@10
#   + Novelty@10
#
# NOTE
# - Runtime recommendation only.
# - Quantitative / qualitative evaluation are separated.
# ============================================================

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from implicit.cpu.bpr import BayesianPersonalizedRanking

try:
    from xgboost import XGBRanker
except ImportError:
    XGBRanker = None

from data_split import load_mf_split
from preprocessing import load_games


# ============================================================
# Project / Paths
# ============================================================

ROOT = Path(__file__).resolve().parent
SAVED_MODEL_DIR = ROOT / "models" / "saved_model"
# Runtime에서 임의의 신규 사용자를 처리하기 위해 필요한 전역 cache
RUNTIME_CACHE_DIR = SAVED_MODEL_DIR / "runtime_cache"

# 기존 Case 3 Content cache를 가능하면 재사용
CASE3_CACHE_DIR = SAVED_MODEL_DIR / "case3_cache"

RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

TOP_N = 10
CANDIDATE_SIZE = 101

# ------------------------------------------------------------
# Candidate Retriever - 현재 반확정 구조
# ------------------------------------------------------------

# Final E_HR candidate quota
ITEM_N = 46
BPR_N = 20
CONTENT_N = 15
USER_N = 20

# ------------------------------------------------------------
# CF
# ------------------------------------------------------------

ITEM_K = 30
USER_K = 30

# ------------------------------------------------------------
# BPR saved model
# 현재 사용 중인 대표 모델명.
# 이후 BPR tuning 결과가 바뀌면 여기만 수정.
# ------------------------------------------------------------

BPR_MODEL_NAME = (
    "bpr_iter15_factors60_reg0.006_explicit_false.npz"
)

# ------------------------------------------------------------
# New User BPR Fine-Tuning
#
# 0 = item embedding 평균만 사용
# 1~5 = 평균 초기값에서 user vector만 추가 개인화
# 우선 3 epoch으로 두고 이후 0/1/3/5 실험 가능
# ------------------------------------------------------------

NEW_USER_FT_EPOCHS = 3
NEW_USER_FT_LEARNING_RATE = 0.01
NEW_USER_FT_REGULARIZATION = 0.001

# ------------------------------------------------------------
# Final Ranker
# ------------------------------------------------------------

RANKER_MODE = "xgb"

FINAL_MODEL_NAME = "E_HR"

XGB_MODEL_PATH = (
    SAVED_MODEL_DIR
    / "xgb_ranker_final_e_hr.json"
)

# E_HR = Full15
# Full14 + is_item_candidate
# PopAffinity는 최종 Ablation에서 제거.
XGB_FEATURES = [
    "item_score_norm",
    "bpr_score_norm",
    "content_score_norm",
    "user_score_norm",

    "item_rank",
    "bpr_rank",
    "content_rank",
    "user_rank",

    "retriever_count",
    "is_bpr_candidate",
    "is_content_candidate",
    "is_user_candidate",

    "item_popularity",
    "user_interaction_count",

    "is_item_candidate",
]


# ============================================================
# Runtime Cache Paths
# ============================================================

CF_MATRIX_PATH = (
    RUNTIME_CACHE_DIR
    / "cf_signed_interaction_matrix.npz"
)

CF_USER_IDS_PATH = (
    RUNTIME_CACHE_DIR
    / "cf_user_ids.npy"
)

CF_ITEM_IDS_PATH = (
    RUNTIME_CACHE_DIR
    / "cf_item_ids.npy"
)

CONTENT_TFIDF_PATH = (
    CASE3_CACHE_DIR
    / "case3_content_tfidf.npz"
)

CONTENT_APP_IDS_PATH = (
    CASE3_CACHE_DIR
    / "case3_content_app_ids.npy"
)



# ============================================================
# Generic Utils
# ============================================================

def find_file(
    base_dir: Path,
    filename: str,
) -> Path | None:

    matches = list(
        base_dir.rglob(filename)
    )

    if not matches:
        return None

    if len(matches) > 1:
        print(
            f"[주의] {filename}이 여러 개 발견되어 "
            f"첫 번째 파일을 사용합니다."
        )

        for path in matches:
            print(" -", path)

    return matches[0]


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

    result = np.zeros(
        len(values),
        dtype=np.float32,
    )

    finite = np.isfinite(values)

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
        np.arange(len(scores))
        + 1
    )

    return ranks


def top_indices(
    scores,
    n,
):

    scores = np.asarray(scores)

    valid_idx = np.where(
        np.isfinite(scores)
    )[0]

    if len(valid_idx) == 0:
        return np.array(
            [],
            dtype=np.int64,
        )

    n_actual = min(
        n,
        len(valid_idx),
    )

    if n_actual == len(valid_idx):

        selected = valid_idx

    else:

        local = np.argpartition(
            -scores[valid_idx],
            n_actual - 1,
        )[:n_actual]

        selected = valid_idx[
            local
        ]

    order = np.argsort(
        -scores[selected],
        kind="stable",
    )

    return selected[order]




# ============================================================
# Evaluation / Recommendation Metrics
# ============================================================

def precision_at_k(
    recommended,
    relevant,
    k=TOP_N,
):

    recommended = list(
        recommended
    )[:k]

    relevant = set(
        relevant
    )

    if k <= 0:
        return 0.0

    hits = sum(
        app_id in relevant
        for app_id in recommended
    )

    return (
        hits
        /
        k
    )


def recall_at_k(
    recommended,
    relevant,
    k=TOP_N,
):

    recommended = list(
        recommended
    )[:k]

    relevant = set(
        relevant
    )

    if not relevant:
        return 0.0

    hits = sum(
        app_id in relevant
        for app_id in recommended
    )

    return (
        hits
        /
        len(
            relevant
        )
    )


def hit_rate_at_k(
    recommended,
    relevant,
    k=TOP_N,
):

    recommended = list(
        recommended
    )[:k]

    relevant = set(
        relevant
    )

    return float(
        any(
            app_id in relevant
            for app_id
            in recommended
        )
    )


def ndcg_at_k(
    recommended,
    relevant,
    k=TOP_N,
):

    recommended = list(
        recommended
    )[:k]

    relevant = set(
        relevant
    )

    if not relevant:
        return 0.0

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
        len(
            relevant
        ),
        k,
    )

    if ideal_hits == 0:
        return 0.0

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

    return (
        dcg
        /
        idcg
    )


def average_precision_at_k(
    recommended,
    relevant,
    k=TOP_N,
):

    """
    AP@K:
      relevant item이 나온 rank에서의 Precision@rank를 누적하고
      min(|relevant|, K)로 나눈다.

    MAP@K는 사용자별 AP@K의 평균.
    """

    recommended = list(
        recommended
    )[:k]

    relevant = set(
        relevant
    )

    if not relevant:
        return 0.0

    hit_count = 0
    precision_sum = 0.0

    for rank, app_id in enumerate(
        recommended,
        start=1,
    ):

        if app_id in relevant:

            hit_count += 1

            precision_sum += (
                hit_count
                /
                rank
            )

    denominator = min(
        len(
            relevant
        ),
        k,
    )

    if denominator == 0:
        return 0.0

    return (
        precision_sum
        /
        denominator
    )


def build_item_popularity(
    train_df,
):

    """
    Train interaction count 기반 popularity.

    interaction_count(item)
        = Train에서 해당 item과 상호작용한 총 횟수

    total_interactions
        = Train 전체 interaction 수
    """

    popularity = (
        train_df
        .groupby(
            "app_id"
        )
        .size()
        .astype(
            np.int64
        )
        .to_dict()
    )

    total_interactions = int(
        len(
            train_df
        )
    )

    return (
        popularity,
        total_interactions,
    )


def recommendation_popularity_metrics(
    recommended,
    item_popularity,
    total_interactions,
    k=TOP_N,
):

    """
    Mean Log Popularity@K
        = mean(log(1 + interaction_count))

    Novelty@K
        = mean(-log2(interaction_count / total_interactions))

    추천 후보는 학습 item에서 나오므로 일반적으로 count > 0.
    안전을 위해 count <= 0 item은 Novelty 계산에서 제외한다.
    """

    recommended = list(
        recommended
    )[:k]

    if not recommended:

        return {
            "mean_log_popularity_at_k": np.nan,
            "novelty_at_k": np.nan,
        }

    counts = np.asarray(
        [
            item_popularity.get(
                app_id,
                0,
            )
            for app_id
            in recommended
        ],
        dtype=np.float64,
    )

    mean_log_popularity = float(
        np.mean(
            np.log1p(
                counts
            )
        )
    )

    valid = (
        counts
        >
        0
    )

    if (
        total_interactions <= 0
        or
        not valid.any()
    ):

        novelty = np.nan

    else:

        novelty = float(
            np.mean(
                -np.log2(
                    counts[
                        valid
                    ]
                    /
                    float(
                        total_interactions
                    )
                )
            )
        )

    return {
        "mean_log_popularity_at_k": (
            mean_log_popularity
        ),
        "novelty_at_k": novelty,
    }


# ============================================================
# Games Metadata
# ============================================================

def load_games_meta():

    meta = load_games()

    if (
        "app_id" not in meta.columns
        and
        "AppID" in meta.columns
    ):
        meta = meta.rename(
            columns={
                "AppID": "app_id"
            }
        )

    if "app_id" not in meta.columns:

        raise ValueError(
            "games metadata에 app_id/AppID 컬럼이 없습니다."
        )

    if "Name" not in meta.columns:

        raise ValueError(
            "games metadata에 Name 컬럼이 없습니다."
        )

    return (
        meta
        .drop_duplicates(
            "app_id"
        )
        .reset_index(
            drop=True
        )
    )


def resolve_user_input_to_app_ids(
    meta,
    raw_values,
):

    """
    입력값은 게임 이름 또는 app_id 둘 다 허용.

    예:
        ["Portal 2", "730", "Hades"]
    """

    app_id_set = set(
        meta["app_id"]
        .astype(str)
        .tolist()
    )

    name_lookup = {}

    for _, row in meta[
        [
            "app_id",
            "Name",
        ]
    ].dropna().iterrows():

        key = (
            str(row["Name"])
            .strip()
            .lower()
        )

        name_lookup.setdefault(
            key,
            []
        ).append(
            row["app_id"]
        )

    resolved = []

    for raw in raw_values:

        value = str(raw).strip()

        if not value:
            continue

        # app_id 직접 입력
        if value in app_id_set:

            matched = meta.loc[
                meta["app_id"]
                .astype(str)
                ==
                value,
                "app_id",
            ]

            if len(matched):

                resolved.append(
                    matched.iloc[0]
                )

            continue

        # 게임 이름 입력
        key = value.lower()

        candidates = name_lookup.get(
            key,
            [],
        )

        if not candidates:

            print(
                f"[입력 제외] '{value}'를 metadata에서 찾지 못했습니다."
            )

            continue

        # 이름 중복 시 첫 번째 사용.
        # 이후 필요하면 선택 UI로 교체.
        if len(candidates) > 1:

            print(
                f"[주의] '{value}'와 같은 이름이 여러 개 있어 "
                f"첫 번째 app_id를 사용합니다: "
                f"{candidates[0]}"
            )

        resolved.append(
            candidates[0]
        )

    return unique_preserve_order(
        resolved
    )


# ============================================================
# BPR Model / Mapping
# ============================================================

def load_bpr_bundle():

    model_path = find_file(
        SAVED_MODEL_DIR,
        BPR_MODEL_NAME,
    )

    if model_path is None:

        raise FileNotFoundError(
            "BPR 모델을 찾을 수 없습니다.\n"
            f"검색 위치: {SAVED_MODEL_DIR}\n"
            f"파일명: {BPR_MODEL_NAME}"
        )

    mapping_path = find_file(
        SAVED_MODEL_DIR,
        "implicit_bpr_mapping.npz",
    )

    if mapping_path is None:

        mapping_path = find_file(
            ROOT / "models",
            "implicit_bpr_mapping.npz",
        )

    if mapping_path is None:

        raise FileNotFoundError(
            "implicit_bpr_mapping.npz를 찾을 수 없습니다."
        )

    print(
        "\n===== BPR Load ====="
    )

    print(
        "Model  :",
        model_path,
    )

    print(
        "Mapping:",
        mapping_path,
    )

    model = (
        BayesianPersonalizedRanking
        .load(
            str(model_path)
        )
    )

    mapping = np.load(
        mapping_path
    )

    user_ids = np.asarray(
        mapping["user_ids"]
    )

    item_ids = np.asarray(
        mapping["item_ids"]
    )

    if (
        model.item_factors.shape[0]
        != len(item_ids)
    ):

        raise ValueError(
            "BPR item_factors와 mapping의 item 수가 다릅니다."
        )

    return (
        model,
        user_ids,
        item_ids,
    )


# ============================================================
# Runtime CF Matrix
# ============================================================

def load_or_build_cf_matrix(
    train_df,
):

    """
    Item-CF / User-CF용 signed matrix.

    True  -> +1
    False -> -1

    최초 1회 생성 후 runtime_cache에 저장한다.
    """

    cache_exists = all(
        path.exists()
        for path in [
            CF_MATRIX_PATH,
            CF_USER_IDS_PATH,
            CF_ITEM_IDS_PATH,
        ]
    )

    if cache_exists:

        print(
            "\n===== CF Runtime Cache Load ====="
        )

        interaction_matrix = (
            load_npz(
                CF_MATRIX_PATH
            )
            .tocsr()
        )

        user_ids = np.load(
            CF_USER_IDS_PATH
        )

        item_ids = np.load(
            CF_ITEM_IDS_PATH
        )

        print(
            "Matrix:",
            interaction_matrix.shape,
        )

        return (
            interaction_matrix,
            user_ids,
            item_ids,
        )

    print(
        "\n===== CF Runtime Cache Build ====="
    )

    user_ids = np.sort(
        train_df["user_id"]
        .unique()
    )

    item_ids = np.sort(
        train_df["app_id"]
        .unique()
    )

    user_codes = np.searchsorted(
        user_ids,
        train_df["user_id"]
        .to_numpy()
    ).astype(
        np.int32,
        copy=False,
    )

    item_codes = np.searchsorted(
        item_ids,
        train_df["app_id"]
        .to_numpy()
    ).astype(
        np.int32,
        copy=False,
    )

    values = (
        train_df[
            "is_recommended"
        ]
        .map(
            {
                True: 1.0,
                False: -1.0,
            }
        )
        .to_numpy(
            dtype=np.float32,
            copy=False,
        )
    )

    interaction_matrix = sp.csr_matrix(
        (
            values,
            (
                user_codes,
                item_codes,
            ),
        ),
        shape=(
            len(user_ids),
            len(item_ids),
        ),
        dtype=np.float32,
    )

    interaction_matrix.sum_duplicates()

    save_npz(
        CF_MATRIX_PATH,
        interaction_matrix,
        compressed=False,
    )

    np.save(
        CF_USER_IDS_PATH,
        user_ids,
    )

    np.save(
        CF_ITEM_IDS_PATH,
        item_ids,
    )

    print(
        "Saved:",
        CF_MATRIX_PATH,
    )

    del user_codes
    del item_codes
    del values

    gc.collect()

    return (
        interaction_matrix,
        user_ids,
        item_ids,
    )


# ============================================================
# New User BPR
# ============================================================

class NewUserBPRScorer:

    """
    신규 사용자:
        플레이 item embedding 평균
        -> 초기 user vector
        -> item factor는 고정
        -> user vector만 BPR update
    """

    def __init__(
        self,
        model,
        item_ids,
        random_state=42,
    ):

        self.model = model

        self.item_ids = np.asarray(
            item_ids
        )

        self.random_state = (
            random_state
        )

        self.latent_dim = (
            model.factors
        )

        self.item_factors = np.asarray(
            model.item_factors
        )

        self.factor_width = (
            self.item_factors
            .shape[1]
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

        result = np.full(
            len(app_ids),
            -1,
            dtype=np.int64,
        )

        valid = (
            idx
            <
            len(self.item_ids)
        )

        positions = np.where(
            valid
        )[0]

        if len(positions):

            matched = (
                self.item_ids[
                    idx[positions]
                ]
                ==
                app_ids[
                    positions
                ]
            )

            real_positions = (
                positions[
                    matched
                ]
            )

            result[
                real_positions
            ] = (
                idx[
                    real_positions
                ]
            )

        return result

    def initialize_user_vector(
        self,
        played_app_ids,
    ):

        item_idx = self.item_indices(
            played_app_ids
        )

        item_idx = item_idx[
            item_idx >= 0
        ]

        if len(item_idx) == 0:

            raise ValueError(
                "입력한 게임 중 BPR item mapping에 존재하는 게임이 없습니다."
            )

        user_vector = np.zeros(
            self.factor_width,
            dtype=np.float32,
        )

        # latent factor 평균
        user_vector[
            :self.latent_dim
        ] = (
            self.item_factors[
                item_idx,
                :self.latent_dim,
            ]
            .mean(
                axis=0
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

        # implicit BPR에서 추가 차원이 bias 계산용 상수인 경우
        if (
            self.factor_width
            >
            self.latent_dim
        ):

            user_vector[
                self.latent_dim
            ] = 1.0

        return (
            user_vector,
            item_idx,
        )

    def fine_tune_user_vector(
        self,
        user_vector,
        positive_item_idx,
        epochs=NEW_USER_FT_EPOCHS,
        learning_rate=NEW_USER_FT_LEARNING_RATE,
        regularization=NEW_USER_FT_REGULARIZATION,
    ):

        if epochs <= 0:

            return user_vector

        rng = np.random.default_rng(
            self.random_state
        )

        positive_set = set(
            int(x)
            for x in positive_item_idx
        )

        latent_user = (
            user_vector[
                :self.latent_dim
            ]
        )

        n_items = len(
            self.item_ids
        )

        for _ in range(
            epochs
        ):

            order = rng.permutation(
                positive_item_idx
            )

            for positive_idx in order:

                # unseen negative sampling
                negative_idx = int(
                    rng.integers(
                        0,
                        n_items,
                    )
                )

                while (
                    negative_idx
                    in positive_set
                ):

                    negative_idx = int(
                        rng.integers(
                            0,
                            n_items,
                        )
                    )

                q_i = self.item_factors[
                    positive_idx,
                    :self.latent_dim,
                ]

                q_j = self.item_factors[
                    negative_idx,
                    :self.latent_dim,
                ]

                score_diff = float(
                    latent_user
                    @
                    (q_i - q_j)
                )

                # item bias까지 존재하면 score 차이에 포함
                if (
                    self.factor_width
                    >
                    self.latent_dim
                ):

                    score_diff += float(
                        self.item_factors[
                            positive_idx,
                            self.latent_dim,
                        ]
                        -
                        self.item_factors[
                            negative_idx,
                            self.latent_dim,
                        ]
                    )

                clipped = np.clip(
                    score_diff,
                    -35.0,
                    35.0,
                )

                gradient_weight = (
                    1.0
                    /
                    (
                        1.0
                        +
                        np.exp(
                            clipped
                        )
                    )
                )

                grad_user = (
                    gradient_weight
                    *
                    (q_i - q_j)
                    -
                    regularization
                    *
                    latent_user
                )

                latent_user += (
                    learning_rate
                    *
                    grad_user
                )

        user_vector[
            :self.latent_dim
        ] = latent_user

        if (
            self.factor_width
            >
            self.latent_dim
        ):

            user_vector[
                self.latent_dim
            ] = 1.0

        return user_vector

    def build_user_vector(
        self,
        played_app_ids,
    ):

        (
            user_vector,
            positive_item_idx,
        ) = (
            self.initialize_user_vector(
                played_app_ids
            )
        )

        user_vector = (
            self.fine_tune_user_vector(
                user_vector=user_vector,
                positive_item_idx=positive_item_idx,
            )
        )

        return (
            user_vector,
            positive_item_idx,
        )

    def score_all(
        self,
        user_vector,
        played_item_idx,
    ):

        scores = (
            self.item_factors
            @
            user_vector
        ).astype(
            np.float32,
            copy=False,
        )

        scores[
            played_item_idx
        ] = -np.inf

        return scores

    def retrieve(
        self,
        user_vector,
        played_item_idx,
        n=BPR_N,
    ):

        scores = self.score_all(
            user_vector,
            played_item_idx,
        )

        idx = top_indices(
            scores,
            n,
        )

        return (
            self.item_ids[
                idx
            ]
            .tolist()
        )

    def score(
        self,
        user_vector,
        candidate_ids,
    ):

        candidate_idx = (
            self.item_indices(
                candidate_ids
            )
        )

        result = np.zeros(
            len(candidate_ids),
            dtype=np.float32,
        )

        valid = (
            candidate_idx
            >=
            0
        )

        if valid.any():

            result[
                valid
            ] = (
                self.item_factors[
                    candidate_idx[
                        valid
                    ]
                ]
                @
                user_vector
            )

        return result


# ============================================================
# Content Runtime
# ============================================================

class ContentRuntimeScorer:

    def __init__(
        self,
        meta,
        bpr_item_ids,
    ):

        self.meta = meta

        self.app_ids = np.asarray(
            bpr_item_ids
        )

        self.app_to_idx = {
            app_id: idx
            for idx, app_id
            in enumerate(
                self.app_ids
            )
        }

        self.matrix = (
            self._load_or_build_matrix()
        )

    def _load_or_build_matrix(
        self,
    ):

        if (
            CONTENT_TFIDF_PATH.exists()
            and
            CONTENT_APP_IDS_PATH.exists()
        ):

            cached_app_ids = np.load(
                CONTENT_APP_IDS_PATH
            )

            if np.array_equal(
                cached_app_ids,
                self.app_ids,
            ):

                print(
                    "\n===== Content TF-IDF Cache Load ====="
                )

                return (
                    load_npz(
                        CONTENT_TFIDF_PATH
                    )
                    .tocsr()
                )

        print(
            "\n===== Content TF-IDF Build ====="
        )

        valid_set = set(
            self.app_ids.tolist()
        )

        aligned = pd.DataFrame(
            {
                "app_id": self.app_ids,
                "_order": np.arange(
                    len(
                        self.app_ids
                    )
                ),
            }
        )

        target_meta = (
            aligned
            .merge(
                self.meta[
                    self.meta[
                        "app_id"
                    ].isin(
                        valid_set
                    )
                ],
                on="app_id",
                how="left",
            )
            .sort_values(
                "_order"
            )
        )

        candidate_features = [
            "Genres",
            "Tags",
            "Categories",
            "Developers",
            "Publishers",
        ]

        feature_columns = [
            column
            for column
            in candidate_features
            if column
            in target_meta.columns
        ]

        if not feature_columns:

            raise ValueError(
                "Content feature가 없습니다. "
                "Genres / Tags / Categories / Developers / Publishers 확인 필요"
            )

        combined = pd.Series(
            "",
            index=target_meta.index,
            dtype="object",
        )

        for column in feature_columns:

            combined = (
                combined
                + " "
                + target_meta[
                    column
                ]
                .fillna("")
                .astype(str)
            )

        vectorizer = TfidfVectorizer(
            lowercase=True,
            min_df=2,
            dtype=np.float32,
            norm="l2",
        )

        matrix = (
            vectorizer
            .fit_transform(
                combined
                .str
                .lower()
            )
            .tocsr()
        )

        CONTENT_TFIDF_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_npz(
            CONTENT_TFIDF_PATH,
            matrix,
        )

        np.save(
            CONTENT_APP_IDS_PATH,
            self.app_ids,
        )

        return matrix

    def profile(
        self,
        played_app_ids,
    ):

        source_idx = [
            self.app_to_idx[
                app_id
            ]
            for app_id
            in played_app_ids
            if app_id
            in self.app_to_idx
        ]

        if not source_idx:

            return sp.csr_matrix(
                (
                    1,
                    self.matrix.shape[1],
                ),
                dtype=np.float32,
            )

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

        return profile

    def score_all(
        self,
        played_app_ids,
    ):

        profile = self.profile(
            played_app_ids
        )

        scores = np.zeros(
            len(self.app_ids),
            dtype=np.float32,
        )

        if profile.nnz == 0:

            return scores

        sparse_scores = (
            profile
            @
            self.matrix.T
        ).tocsr()

        row = sparse_scores.getrow(
            0
        )

        scores[
            row.indices
        ] = row.data

        seen_idx = [
            self.app_to_idx[
                app_id
            ]
            for app_id
            in played_app_ids
            if app_id
            in self.app_to_idx
        ]

        if seen_idx:

            scores[
                seen_idx
            ] = -np.inf

        return scores

    def retrieve(
        self,
        played_app_ids,
        n=CONTENT_N,
    ):

        scores = self.score_all(
            played_app_ids
        )

        valid = np.where(
            np.isfinite(
                scores
            )
            &
            (
                scores
                >
                0
            )
        )[0]

        if len(valid) == 0:
            return []

        n_actual = min(
            n,
            len(valid),
        )

        if n_actual < len(valid):

            local = np.argpartition(
                -scores[valid],
                n_actual - 1,
            )[:n_actual]

            selected = valid[
                local
            ]

        else:

            selected = valid

        selected = selected[
            np.argsort(
                -scores[selected],
                kind="stable",
            )
        ]

        return (
            self.app_ids[
                selected
            ]
            .tolist()
        )

    def score(
        self,
        played_app_ids,
        candidate_ids,
    ):

        profile = self.profile(
            played_app_ids
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

        values = (
            profile
            @
            self.matrix[
                valid_indices
            ].T
        )

        result[
            valid_positions
        ] = np.asarray(
            values.toarray()
        ).ravel()

        return result


# ============================================================
# User-CF Runtime
# ============================================================

class UserCFRuntimeScorer:

    def __init__(
        self,
        interaction_matrix,
        item_ids,
        k=USER_K,
    ):

        self.matrix = (
            interaction_matrix
            .tocsr()
            .astype(
                np.float32
            )
        )

        self.item_ids = np.asarray(
            item_ids
        )

        self.item_to_idx = {
            app_id: idx
            for idx, app_id
            in enumerate(
                self.item_ids
            )
        }

        self.k = k

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

    def calculate_scores(
        self,
        played_app_ids,
    ):

        source_idx = [
            self.item_to_idx[
                app_id
            ]
            for app_id
            in played_app_ids
            if app_id
            in self.item_to_idx
        ]

        result = np.zeros(
            self.matrix.shape[1],
            dtype=np.float32,
        )

        if not source_idx:
            return result

        query = sp.csr_matrix(
            (
                np.ones(
                    len(source_idx),
                    dtype=np.float32,
                ),
                (
                    np.zeros(
                        len(source_idx),
                        dtype=np.int32,
                    ),
                    source_idx,
                ),
            ),
            shape=(
                1,
                self.matrix.shape[1],
            ),
        )

        query_norm = np.sqrt(
            len(source_idx)
        )

        sims = (
            query
            @
            self.matrix.T
        ).tocsr()

        if sims.nnz == 0:
            return result

        denominator = (
            query_norm
            *
            self.user_norms[
                sims.indices
            ]
        )

        valid = (
            denominator
            >
            0
        )

        sims.data[
            valid
        ] /= (
            denominator[
                valid
            ]
        )

        sims.data[
            ~valid
        ] = 0

        sims.eliminate_zeros()

        if sims.nnz == 0:
            return result

        k = min(
            self.k,
            len(
                sims.data
            ),
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

            result = (
                weighted_sum
                /
                sim_sum
            ).astype(
                np.float32,
                copy=False,
            )

        else:

            result = (
                weighted_sum
                .astype(
                    np.float32,
                    copy=False,
                )
            )

        result[
            source_idx
        ] = -np.inf

        return result

    def retrieve(
        self,
        played_app_ids,
        n=USER_N,
    ):

        scores = (
            self.calculate_scores(
                played_app_ids
            )
        )

        valid_idx = np.where(
            np.isfinite(
                scores
            )
            &
            (
                scores
                >
                0
            )
        )[0]

        if len(valid_idx) == 0:
            return []

        n_actual = min(
            n,
            len(
                valid_idx
            ),
        )

        if n_actual < len(valid_idx):

            local = np.argpartition(
                -scores[
                    valid_idx
                ],
                n_actual - 1,
            )[:n_actual]

            selected = valid_idx[
                local
            ]

        else:

            selected = valid_idx

        selected = selected[
            np.argsort(
                -scores[
                    selected
                ],
                kind="stable",
            )
        ]

        return (
            self.item_ids[
                selected
            ]
            .tolist()
        )

    def score(
        self,
        played_app_ids,
        candidate_ids,
    ):

        scores = (
            self.calculate_scores(
                played_app_ids
            )
        )

        result = np.zeros(
            len(candidate_ids),
            dtype=np.float32,
        )

        for pos, app_id in enumerate(
            candidate_ids
        ):

            idx = self.item_to_idx.get(
                app_id
            )

            if idx is None:
                continue

            value = scores[
                idx
            ]

            if np.isfinite(
                value
            ):

                result[
                    pos
                ] = value

        return result


# ============================================================
# Item-CF Runtime
# ============================================================

class ItemCFRuntimeScorer:

    def __init__(
        self,
        interaction_matrix,
        item_ids,
        k=ITEM_K,
    ):

        self.item_ids = np.asarray(
            item_ids
        )

        self.item_to_idx = {
            app_id: idx
            for idx, app_id
            in enumerate(
                self.item_ids
            )
        }

        self.k = k

        item_matrix = (
            interaction_matrix
            .T
            .tocsr()
            .astype(
                np.float32
            )
        )

        print(
            "\n===== Item Matrix Normalize ====="
        )

        self.item_matrix = normalize(
            item_matrix,
            norm="l2",
            axis=1,
            copy=False,
        )

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

        positive = (
            row.data
            >
            0
        )

        indices = row.indices[
            positive
        ]

        values = row.data[
            positive
        ]

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
            self.k,
            len(values),
        )

        top_pos = np.argpartition(
            values,
            -k,
        )[-k:]

        result = (
            indices[
                top_pos
            ],
            values[
                top_pos
            ],
        )

        self.neighbor_cache[
            item_idx
        ] = result

        return result

    def calculate_scores(
        self,
        played_app_ids,
    ):

        source_idx = [
            self.item_to_idx[
                app_id
            ]
            for app_id
            in played_app_ids
            if app_id
            in self.item_to_idx
        ]

        scores = np.zeros(
            self.item_matrix.shape[0],
            dtype=np.float32,
        )

        if not source_idx:
            return scores

        for item_idx in source_idx:

            (
                neighbor_idx,
                similarities,
            ) = self.get_neighbors(
                item_idx
            )

            if len(
                neighbor_idx
            ):

                scores[
                    neighbor_idx
                ] += (
                    similarities
                )

        scores[
            source_idx
        ] = -np.inf

        return scores

    def score(
        self,
        played_app_ids,
        candidate_ids,
    ):

        scores = (
            self.calculate_scores(
                played_app_ids
            )
        )

        result = np.zeros(
            len(candidate_ids),
            dtype=np.float32,
        )

        for pos, app_id in enumerate(
            candidate_ids
        ):

            idx = self.item_to_idx.get(
                app_id
            )

            if idx is None:
                continue

            value = scores[
                idx
            ]

            if np.isfinite(
                value
            ):

                result[
                    pos
                ] = value

        return result

    def retrieve(
        self,
        played_app_ids,
        n=ITEM_N,
    ):

        scores = self.calculate_scores(
            played_app_ids
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

        if n_actual < len(valid_idx):

            local = np.argpartition(
                -scores[valid_idx],
                n_actual - 1,
            )[:n_actual]

            idx = valid_idx[local]

        else:

            idx = valid_idx

        idx = idx[
            np.argsort(
                -scores[idx],
                kind="stable",
            )
        ]

        return (
            self.item_ids[idx]
            .tolist()
        )


# ============================================================
# Hybrid Runtime Engine
# ============================================================

class SteamHybridRecommender:

    def __init__(
        self,
        train_df,
        meta,
        bpr_model,
        bpr_item_ids,
        cf_matrix,
        cf_item_ids,
    ):

        self.train_df = train_df
        self.meta = meta

        (
            self.item_popularity,
            self.total_interactions,
        ) = build_item_popularity(
            train_df
        )

        self.bpr = NewUserBPRScorer(
            model=bpr_model,
            item_ids=bpr_item_ids,
            random_state=RANDOM_STATE,
        )

        self.content = (
            ContentRuntimeScorer(
                meta=meta,
                bpr_item_ids=bpr_item_ids,
            )
        )

        self.user_cf = (
            UserCFRuntimeScorer(
                interaction_matrix=cf_matrix,
                item_ids=cf_item_ids,
                k=USER_K,
            )
        )

        self.item_cf = (
            ItemCFRuntimeScorer(
                interaction_matrix=cf_matrix,
                item_ids=cf_item_ids,
                k=ITEM_K,
            )
        )

        self.xgb_model = None

        if (
            RANKER_MODE
            ==
            "xgb"
        ):

            self.xgb_model = (
                self._load_xgb_model()
            )

    def _load_xgb_model(
        self,
    ):

        if XGBRanker is None:

            raise ImportError(
                "xgboost가 설치되어 있지 않습니다. "
                "pip install xgboost"
            )

        if not XGB_MODEL_PATH.exists():

            raise FileNotFoundError(
                "XGBoost ranker 모델을 찾을 수 없습니다: "
                f"{XGB_MODEL_PATH}"
            )

        model = XGBRanker()

        model.load_model(
            str(
                XGB_MODEL_PATH
            )
        )

        print(
            "\n===== XGBRanker Load ====="
        )

        print(
            XGB_MODEL_PATH
        )

        return model

    def retrieve_candidates(
        self,
        played_app_ids,
        new_user_vector,
        bpr_played_idx,
    ):

        # ----------------------------------------------------
        # Final E_HR = 4-Retriever
        # I46 + B20 + C15 + U20
        # ----------------------------------------------------

        item_candidates = (
            self.item_cf.retrieve(
                played_app_ids=played_app_ids,
                n=ITEM_N,
            )
        )

        bpr_candidates = (
            self.bpr.retrieve(
                user_vector=new_user_vector,
                played_item_idx=bpr_played_idx,
                n=BPR_N,
            )
        )

        content_candidates = (
            self.content.retrieve(
                played_app_ids=played_app_ids,
                n=CONTENT_N,
            )
        )

        user_candidates = (
            self.user_cf.retrieve(
                played_app_ids=played_app_ids,
                n=USER_N,
            )
        )

        source_info = {
            "item": set(item_candidates),
            "bpr": set(bpr_candidates),
            "content": set(content_candidates),
            "user": set(user_candidates),
        }

        candidates = unique_preserve_order(
            item_candidates
            +
            bpr_candidates
            +
            content_candidates
            +
            user_candidates
        )

        played_set = set(
            played_app_ids
        )

        candidates = [
            app_id
            for app_id
            in candidates
            if app_id not in played_set
        ]

        return (
            candidates[:CANDIDATE_SIZE],
            source_info,
        )

    def build_features(
        self,
        played_app_ids,
        candidates,
        new_user_vector,
        source_info,
    ):

        bpr_score = (
            self.bpr.score(
                user_vector=new_user_vector,
                candidate_ids=candidates,
            )
        )

        content_score = (
            self.content.score(
                played_app_ids=played_app_ids,
                candidate_ids=candidates,
            )
        )

        user_score = (
            self.user_cf.score(
                played_app_ids=played_app_ids,
                candidate_ids=candidates,
            )
        )

        item_score = (
            self.item_cf.score(
                played_app_ids=played_app_ids,
                candidate_ids=candidates,
            )
        )

        feature_df = pd.DataFrame(
            {
                "app_id": candidates,

                "item_score": item_score,
                "bpr_score": bpr_score,
                "content_score": content_score,
                "user_score": user_score,
            }
        )

        # ----------------------------------------------------
        # Base 8
        # Candidate set 기준으로 normalize / rank 재계산
        # ----------------------------------------------------

        feature_df[
            "item_score_norm"
        ] = minmax_normalize(
            item_score
        )

        feature_df[
            "bpr_score_norm"
        ] = minmax_normalize(
            bpr_score
        )

        feature_df[
            "content_score_norm"
        ] = minmax_normalize(
            content_score
        )

        feature_df[
            "user_score_norm"
        ] = minmax_normalize(
            user_score
        )

        feature_df[
            "item_rank"
        ] = assign_rank(
            item_score
        )

        feature_df[
            "bpr_rank"
        ] = assign_rank(
            bpr_score
        )

        feature_df[
            "content_rank"
        ] = assign_rank(
            content_score
        )

        feature_df[
            "user_rank"
        ] = assign_rank(
            user_score
        )

        # ----------------------------------------------------
        # Retriever source signals
        #
        # Full14에는 is_item_candidate 자체는 안 들어가지만
        # retriever_count 계산에는 Item도 포함.
        # ----------------------------------------------------

        item_set = source_info[
            "item"
        ]

        bpr_set = source_info[
            "bpr"
        ]

        content_set = source_info[
            "content"
        ]

        user_set = source_info[
            "user"
        ]

        is_item_candidate = np.asarray(
            [
                int(app_id in item_set)
                for app_id in candidates
            ],
            dtype=np.int8,
        )

        is_bpr_candidate = np.asarray(
            [
                int(app_id in bpr_set)
                for app_id in candidates
            ],
            dtype=np.int8,
        )

        is_content_candidate = np.asarray(
            [
                int(app_id in content_set)
                for app_id in candidates
            ],
            dtype=np.int8,
        )

        is_user_candidate = np.asarray(
            [
                int(app_id in user_set)
                for app_id in candidates
            ],
            dtype=np.int8,
        )

        # diagnostic only
        feature_df[
            "is_item_candidate"
        ] = is_item_candidate

        feature_df[
            "is_bpr_candidate"
        ] = is_bpr_candidate

        feature_df[
            "is_content_candidate"
        ] = is_content_candidate

        feature_df[
            "is_user_candidate"
        ] = is_user_candidate

        feature_df[
            "retriever_count"
        ] = (
            is_item_candidate
            +
            is_bpr_candidate
            +
            is_content_candidate
            +
            is_user_candidate
        ).astype(
            np.int8
        )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        feature_df[
            "item_popularity"
        ] = np.asarray(
            [
                self.item_popularity.get(
                    app_id,
                    0,
                )
                for app_id in candidates
            ],
            dtype=np.float32,
        )

        feature_df[
            "user_interaction_count"
        ] = np.full(
            len(candidates),
            len(played_app_ids),
            dtype=np.float32,
        )

        return feature_df

    def rank(
        self,
        feature_df,
        top_n=TOP_N,
    ):

        if len(
            feature_df
        ) == 0:

            return pd.DataFrame()

        if (
            RANKER_MODE
            ==
            "item"
        ):

            ranked = (
                feature_df
                .sort_values(
                    [
                        "item_score",
                        "bpr_score",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                    kind="stable",
                )
                .head(
                    top_n
                )
                .copy()
            )

            ranked[
                "final_score"
            ] = ranked[
                "item_score"
            ]

            ranked[
                "ranker"
            ] = "item"

            return ranked

        if (
            RANKER_MODE
            ==
            "xgb"
        ):

            X = feature_df[
                XGB_FEATURES
            ]

            predictions = (
                self.xgb_model
                .predict(
                    X
                )
            )

            ranked = (
                feature_df
                .assign(
                    final_score=predictions,
                    ranker="xgb",
                )
                .sort_values(
                    "final_score",
                    ascending=False,
                    kind="stable",
                )
                .head(
                    top_n
                )
                .copy()
            )

            return ranked

        raise ValueError(
            f"지원하지 않는 RANKER_MODE: {RANKER_MODE}"
        )

    def recommend(
        self,
        played_app_ids,
        top_n=TOP_N,
        verbose=True,
    ):

        played_app_ids = (
            unique_preserve_order(
                played_app_ids
            )
        )

        if len(
            played_app_ids
        ) == 0:

            raise ValueError(
                "플레이한 게임을 하나 이상 입력해야 합니다."
            )

        if verbose:

            print(
                "\n============================================================"
            )

            print(
                " Final E_HR Recommendation"
            )

            print(
                "============================================================"
            )

            print(
                "Played games:",
                len(
                    played_app_ids
                ),
            )

            print(
                "New-user BPR FT epochs:",
                NEW_USER_FT_EPOCHS,
            )

            print(
                "Candidate:",
                (
                    f"Item {ITEM_N} + BPR {BPR_N} + "
                    f"Content {CONTENT_N} + User {USER_N}"
                ),
            )

            print(
                "Ranker:",
                f"{RANKER_MODE} ({FINAL_MODEL_NAME})",
            )

        # ----------------------------------------------------
        # 1. 신규 사용자 BPR representation
        # ----------------------------------------------------

        (
            new_user_vector,
            bpr_played_idx,
        ) = (
            self.bpr
            .build_user_vector(
                played_app_ids
            )
        )

        # ----------------------------------------------------
        # 2. Candidate Retrieval
        # ----------------------------------------------------

        (
            candidates,
            source_info,
        ) = (
            self.retrieve_candidates(
                played_app_ids=played_app_ids,
                new_user_vector=new_user_vector,
                bpr_played_idx=bpr_played_idx,
            )
        )

        if verbose:

            print(
                "Candidate union:",
                len(
                    candidates
                ),
            )

        if not candidates:

            return pd.DataFrame()

        # ----------------------------------------------------
        # 3. Feature
        # ----------------------------------------------------

        feature_df = (
            self.build_features(
                played_app_ids=played_app_ids,
                candidates=candidates,
                new_user_vector=new_user_vector,
                source_info=source_info,
            )
        )

        # ----------------------------------------------------
        # 4. Rank -> Top-N
        # ----------------------------------------------------

        result = (
            self.rank(
                feature_df,
                top_n=top_n,
            )
        )

        # ----------------------------------------------------
        # 5. Metadata
        # ----------------------------------------------------

        metadata_columns = [
            column
            for column
            in [
                "app_id",
                "Name",
                "Genres",
                "Tags",
                "Categories",
            ]
            if column
            in self.meta.columns
        ]

        result = (
            result
            .merge(
                self.meta[
                    metadata_columns
                ],
                on="app_id",
                how="left",
            )
        )

        result.insert(
            0,
            "recommend_rank",
            np.arange(
                1,
                len(result) + 1,
            ),
        )

        # ----------------------------------------------------
        # 6. Popularity / Novelty
        # ----------------------------------------------------

        result[
            "interaction_count"
        ] = (
            result[
                "app_id"
            ]
            .map(
                self.item_popularity
            )
            .fillna(
                0
            )
            .astype(
                np.int64
            )
        )

        result[
            "log_popularity"
        ] = np.log1p(
            result[
                "interaction_count"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        counts = (
            result[
                "interaction_count"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        novelty = np.full(
            len(result),
            np.nan,
            dtype=np.float64,
        )

        valid = (
            counts
            >
            0
        )

        if (
            self.total_interactions > 0
            and
            valid.any()
        ):

            novelty[
                valid
            ] = (
                -np.log2(
                    counts[
                        valid
                    ]
                    /
                    float(
                        self.total_interactions
                    )
                )
            )

        result[
            "novelty"
        ] = novelty

        return (
            result
            .reset_index(
                drop=True
            )
        )


# ============================================================
# Engine Loader
# ============================================================

def build_engine():

    print(
        "============================================================"
    )

    print(
        " Steam Hybrid Recommender - Runtime"
    )

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # 1. Data
    # --------------------------------------------------------

    meta = load_games_meta()

    train_df, _ = (
        load_mf_split()
    )

    print(
        "Train:",
        train_df.shape,
    )

    print(
        "Final XGB:",
        XGB_MODEL_PATH,
    )

    # --------------------------------------------------------
    # 2. BPR
    # --------------------------------------------------------

    (
        bpr_model,
        _,
        bpr_item_ids,
    ) = load_bpr_bundle()

    # --------------------------------------------------------
    # 3. CF matrix
    # --------------------------------------------------------

    (
        cf_matrix,
        _,
        cf_item_ids,
    ) = (
        load_or_build_cf_matrix(
            train_df
        )
    )

    # --------------------------------------------------------
    # 4. Runtime engine
    # --------------------------------------------------------

    engine = (
        SteamHybridRecommender(
            train_df=train_df,
            meta=meta,
            bpr_model=bpr_model,
            bpr_item_ids=bpr_item_ids,
            cf_matrix=cf_matrix,
            cf_item_ids=cf_item_ids,
        )
    )

    return (
        engine,
        meta,
    )


# ============================================================
# Console Main
# ============================================================

def main():

    (
        engine,
        meta,
    ) = build_engine()

    print(
        "\n플레이한 게임 이름 또는 app_id를 쉼표(,)로 입력하세요."
    )

    print(
        "예: PUBG: BATTLEGROUNDS, Red Dead Redemption 2, Stray"
    )

    raw = input(
        "\nPlayed Games > "
    ).strip()

    raw_values = [
        value.strip()
        for value
        in raw.split(",")
        if value.strip()
    ]

    played_app_ids = (
        resolve_user_input_to_app_ids(
            meta,
            raw_values,
        )
    )

    if not played_app_ids:

        print(
            "\n유효한 게임 입력이 없습니다."
        )

        return

    played_names = (
        meta[
            meta[
                "app_id"
            ].isin(
                played_app_ids
            )
        ][
            [
                "app_id",
                "Name",
            ]
        ]
        .drop_duplicates(
            "app_id"
        )
    )

    print(
        "\n===== 입력 게임 ====="
    )

    print(
        played_names
        .to_string(
            index=False
        )
    )

    result = (
        engine.recommend(
            played_app_ids=played_app_ids,
            top_n=TOP_N,
        )
    )

    if len(result) == 0:

        print(
            "\n추천 결과가 없습니다."
        )

        return

    print(
        "\n===== 최종 Top-10 추천 ====="
    )

    display_columns = [
        column
        for column
        in [
            "recommend_rank",
            "app_id",
            "Name",
            "final_score",

            "item_score",
            "bpr_score",
            "content_score",
            "user_score",

            "retriever_count",

            "is_item_candidate",
            "is_bpr_candidate",
            "is_content_candidate",
            "is_user_candidate",

            "interaction_count",
            "log_popularity",
            "novelty",

            "ranker",
        ]
        if column
        in result.columns
    ]

    print(
        result[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    popularity_metrics = (
        recommendation_popularity_metrics(
            recommended=
                result[
                    "app_id"
                ].tolist(),

            item_popularity=
                engine.item_popularity,

            total_interactions=
                engine.total_interactions,

            k=TOP_N,
        )
    )

    print(
        "\n===== 추천 성향 지표 ====="
    )

    print(
        f"Mean Log Popularity@{TOP_N}: "
        f"{popularity_metrics['mean_log_popularity_at_k']:.4f}"
    )

    print(
        f"Novelty@{TOP_N}            : "
        f"{popularity_metrics['novelty_at_k']:.4f}"
    )

    print(
        f"MAP@{TOP_N}                : "
        "N/A - 신규 사용자는 held-out 정답이 없으므로 "
        "정량평가 스크립트에서 계산"
    )


if __name__ == "__main__":
    main()
