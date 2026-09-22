# hybrid_arctech_experiment/exp_b_item_ranker_ratio_sweep.py

import sys
import time
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import save_npz, load_npz
from sklearn.preprocessing import normalize


# =========================================================
# Project Root
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
CURRENT_DIR = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


from data_split import load_mf_split
from models.itembase import build_interaction_matrix

import exp_c_fusion_ablation as base


# =========================================================
# Config
# =========================================================

TOP_N = 10
ITEM_K = 30


# ---------------------------------------------------------
# Item similarity batch size
#
# 클수록 빠르지만 RAM 사용량 증가.
#
# 128:
# 속도 / 안정성 균형
#
# RAM 충분하면 256도 가능.
# ---------------------------------------------------------

ITEM_PREWARM_BATCH_SIZE = 128


# =========================================================
# 7 Experiments
#
# 실험 조건은 기존 코드와 완전히 동일.
# =========================================================

EXPERIMENTS = {

    "baseline_b59_c41": {

        "bpr_n": 59,
        "content_n": 41,
        "user_n": 0,

        "family": "baseline",
        "user_enabled": False,
    },


    "bpr70_content30": {

        "bpr_n": 70,
        "content_n": 30,
        "user_n": 0,

        "family": "bpr70_content30",
        "user_enabled": False,
    },


    "bpr50_content50": {

        "bpr_n": 50,
        "content_n": 50,
        "user_n": 0,

        "family": "bpr50_content50",
        "user_enabled": False,
    },


    "bpr40_content60": {

        "bpr_n": 40,
        "content_n": 60,
        "user_n": 0,

        "family": "bpr40_content60",
        "user_enabled": False,
    },


    "baseline_b56_c39_u5": {

        "bpr_n": 56,
        "content_n": 39,
        "user_n": 5,

        "family": "baseline",
        "user_enabled": True,
    },


    "bpr67_content28_user5": {

        "bpr_n": 67,
        "content_n": 28,
        "user_n": 5,

        "family": "bpr70_content30",
        "user_enabled": True,
    },


    "bpr48_content47_user5": {

        "bpr_n": 48,
        "content_n": 47,
        "user_n": 5,

        "family": "bpr50_content50",
        "user_enabled": True,
    },
}


# =========================================================
# Result Paths
# =========================================================

RESULT_DIR = (
    base.RESULT_DIR
    / "case2_item_ranker_ratio_sweep"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUMMARY_PATH = (
    RESULT_DIR
    / "item_ranker_ratio_sweep_summary.csv"
)


USER_COMPARE_PATH = (
    RESULT_DIR
    / "item_ranker_user_effect_comparison.csv"
)


# =========================================================
# Persistent Item Ranker Cache
# =========================================================

CACHE_DIR = (
    base.SAVED_MODEL_DIR
    / "case2_cache"
    / "item_ranker_ratio_sweep"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# app_id -> item index
# ---------------------------------------------------------

ITEM_MAPPING_PATH = (
    CACHE_DIR
    / "item_mapping.pkl"
)


# ---------------------------------------------------------
# normalized Item x User matrix
#
# 첫 실행에서만 생성.
# user score cache 완성 후에는
# 보통 다시 읽을 필요도 없음.
# ---------------------------------------------------------

ITEM_MATRIX_PATH = (
    CACHE_DIR
    / "normalized_item_matrix.npz"
)


# ---------------------------------------------------------
# source item -> Top30 neighbors
# ---------------------------------------------------------

NEIGHBOR_CACHE_PATH = (
    CACHE_DIR
    / "item_neighbor_cache.pkl"
)


# ---------------------------------------------------------
# 400명 user_id -> Item score map
#
# 이게 완성되면 다음 실행부터는
# Item matrix 자체가 필요 없음.
# ---------------------------------------------------------

USER_SCORE_CACHE_PATH = (
    CACHE_DIR
    / "item_user_score_cache.pkl"
)


# =========================================================
# Utility
# =========================================================

def unique_preserve_order(values):

    seen = set()
    result = []

    for value in values:

        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


# =========================================================
# Retrieval Cache -> Dictionary
#
# 기존 코드는 Case마다 이 groupby를 반복.
#
# 최적화:
# 시작할 때 딱 한 번만 변환.
# =========================================================

def cache_to_dict(df):

    return (
        df
        .sort_values(
            ["user_id", "rank"],
            kind="stable",
        )
        .groupby(
            "user_id",
            sort=False,
        )["app_id"]
        .apply(list)
        .to_dict()
    )


# =========================================================
# Optimized Item-Based Ranker
#
# 알고리즘 자체는 기존과 동일:
#
# source item
#     ↓
# cosine similarity
#     ↓
# positive similarity만
#     ↓
# source별 Top-K=30
#     ↓
# similarity sum
#
#
# 변경점은 계산 순서/캐싱뿐.
# =========================================================

class OptimizedItemBasedRanker:

    def __init__(
        self,
        game_to_idx,
        n_games,
        item_matrix=None,
        k=30,
    ):

        self.game_to_idx = (
            game_to_idx
        )

        self.n_games = int(
            n_games
        )

        self.k = k


        # -------------------------------------------------
        # 이미 L2 normalize된 Item x User matrix
        # -------------------------------------------------

        self.item_matrix = (
            item_matrix
        )


        # transpose를 매 batch마다 새로 만들지 않음

        self.item_matrix_t = (

            item_matrix.T

            if item_matrix
            is not None

            else None
        )


        # -------------------------------------------------
        # source item ->
        #
        # (
        #   neighbor_indices,
        #   neighbor_similarity
        # )
        # -------------------------------------------------

        self.neighbor_cache = {}


        # -------------------------------------------------
        # user_id ->
        #
        # {
        #   item_idx: item_based_score
        # }
        # -------------------------------------------------

        self.user_score_cache = {}


        self.user_score_cache_hit = 0
        self.user_score_cache_miss = 0


    # =====================================================
    # Neighbor Cache Load
    # =====================================================

    def load_neighbor_cache(
        self,
        path,
    ):

        if not path.exists():
            return False


        print(
            "\nItem Neighbor Cache 로드:"
        )

        print(path)


        try:

            with open(
                path,
                "rb",
            ) as f:

                payload = pickle.load(
                    f
                )


            if (
                payload.get(
                    "n_games"
                )
                !=
                self.n_games
            ):

                print(
                    "n_games 불일치 → cache 무시"
                )

                return False


            if (
                payload.get(
                    "k"
                )
                !=
                self.k
            ):

                print(
                    "ITEM_K 불일치 → cache 무시"
                )

                return False


            self.neighbor_cache = (
                payload[
                    "neighbor_cache"
                ]
            )


            print(
                "Neighbor Cache source item:",
                len(
                    self.neighbor_cache
                ),
            )


            return True


        except Exception as e:

            print(
                "Neighbor cache load 실패:",
                e,
            )

            return False


    # =====================================================
    # Neighbor Cache Save
    # =====================================================

    def save_neighbor_cache(
        self,
        path,
    ):

        payload = {

            "n_games":
                self.n_games,

            "k":
                self.k,

            "neighbor_cache":
                self.neighbor_cache,
        }


        print(
            "\nNeighbor Cache 저장:"
        )

        print(path)


        with open(
            path,
            "wb",
        ) as f:

            pickle.dump(
                payload,
                f,
                protocol=
                    pickle.HIGHEST_PROTOCOL,
            )


    # =====================================================
    # User Score Cache Load
    # =====================================================

    def load_user_score_cache(
        self,
        path,
        sampled_users,
    ):

        if not path.exists():
            return False


        print(
            "\nItem User Score Cache 로드:"
        )

        print(path)


        try:

            with open(
                path,
                "rb",
            ) as f:

                payload = pickle.load(
                    f
                )


            if (
                payload.get(
                    "n_games"
                )
                !=
                self.n_games
            ):

                print(
                    "n_games 불일치 → score cache 무시"
                )

                return False


            if (
                payload.get(
                    "k"
                )
                !=
                self.k
            ):

                print(
                    "ITEM_K 불일치 → score cache 무시"
                )

                return False


            score_cache = (
                payload[
                    "user_score_cache"
                ]
            )


            required_users = set(
                sampled_users[
                    "user_id"
                ].tolist()
            )


            cached_users = set(
                score_cache.keys()
            )


            if not required_users.issubset(
                cached_users
            ):

                print(
                    "400명 중 일부 score 없음 "
                    "→ cache 전체 재사용 안 함"
                )

                return False


            self.user_score_cache = (
                score_cache
            )


            print(
                "Cached users:",
                len(
                    self.user_score_cache
                ),
            )


            return True


        except Exception as e:

            print(
                "User score cache load 실패:",
                e,
            )

            return False


    # =====================================================
    # User Score Cache Save
    # =====================================================

    def save_user_score_cache(
        self,
        path,
    ):

        payload = {

            "n_games":
                self.n_games,

            "k":
                self.k,

            "user_score_cache":
                self.user_score_cache,
        }


        print(
            "\nItem User Score Cache 저장:"
        )

        print(path)


        with open(
            path,
            "wb",
        ) as f:

            pickle.dump(
                payload,
                f,
                protocol=
                    pickle.HIGHEST_PROTOCOL,
            )


        print(
            "Cached Users:",
            len(
                self.user_score_cache
            ),
        )


    # =====================================================
    # Exact Top-K Neighbor 계산
    #
    # 기존 로직 그대로.
    #
    # 차이:
    #
    # 사용자 1명씩 계산하지 않고
    # 여러 source item을 batch 계산.
    # =====================================================

    def _compute_neighbor_batch(
        self,
        source_indices,
    ):

        if len(source_indices) == 0:
            return


        if self.item_matrix is None:

            raise RuntimeError(
                "Item matrix가 없는데 "
                "새로운 neighbor 계산이 필요합니다."
            )


        source_matrix = (
            self.item_matrix[
                source_indices
            ]
        )


        # normalized dot product
        # = cosine similarity

        sims = (
            source_matrix
            @
            self.item_matrix_t
        ).tocsr()


        for (
            local_row,
            item_idx,
        ) in enumerate(
            source_indices
        ):

            row = sims.getrow(
                local_row
            )


            row_indices = (
                row.indices
            )

            row_data = (
                row.data
            )


            # ---------------------------------------------
            # 기존 조건 그대로:
            #
            # self 제외
            # positive similarity만
            # ---------------------------------------------

            valid = (

                (row_indices != item_idx)

                &

                (row_data > 0)
            )


            positive_indices = (
                row_indices[
                    valid
                ]
            )

            positive_data = (
                row_data[
                    valid
                ]
            )


            if len(
                positive_data
            ) == 0:

                self.neighbor_cache[
                    int(
                        item_idx
                    )
                ] = (

                    np.array(
                        [],
                        dtype=np.int64,
                    ),

                    np.array(
                        [],
                        dtype=np.float64,
                    ),
                )

                continue


            k = min(
                self.k,
                len(
                    positive_data
                ),
            )


            # ---------------------------------------------
            # 기존 np.argpartition 유지
            # ---------------------------------------------

            top_pos = (
                np.argpartition(
                    positive_data,
                    -k,
                )[-k:]
            )


            self.neighbor_cache[
                int(
                    item_idx
                )
            ] = (

                positive_indices[
                    top_pos
                ].copy(),

                positive_data[
                    top_pos
                ].copy(),
            )


        # chunk similarity matrix 즉시 제거

        del sims
        del source_matrix


    # =====================================================
    # 400명의 source item을 먼저 모아서
    # chunk 단위로 한번에 neighbor 계산
    # =====================================================

    def prewarm_neighbors(
        self,
        app_ids,
        batch_size=128,
    ):

        if self.item_matrix is None:
            return


        source_indices = [

            self.game_to_idx[
                app_id
            ]

            for app_id in app_ids

            if app_id
            in
            self.game_to_idx
        ]


        source_indices = list(
            dict.fromkeys(
                source_indices
            )
        )


        missing = [

            int(idx)

            for idx in source_indices

            if int(idx)
            not in
            self.neighbor_cache
        ]


        print(
            "\n===== Item Neighbor Prewarm ====="
        )


        print(
            "전체 source item:",
            len(
                source_indices
            ),
        )


        print(
            "이미 cache:",
            len(
                source_indices
            )
            -
            len(
                missing
            ),
        )


        print(
            "새로 계산:",
            len(
                missing
            ),
        )


        if len(missing) == 0:

            print(
                "추가 계산 필요 없음"
            )

            return


        start = (
            time.perf_counter()
        )


        total_batches = (

            len(missing)
            +
            batch_size
            -
            1

        ) // batch_size


        for batch_idx, start_idx in enumerate(
            range(
                0,
                len(missing),
                batch_size,
            ),
            start=1,
        ):

            batch = (
                missing[
                    start_idx:
                    start_idx
                    +
                    batch_size
                ]
            )


            self._compute_neighbor_batch(
                batch
            )


            if (
                batch_idx % 5 == 0

                or

                batch_idx
                ==
                total_batches
            ):

                print(
                    f"Neighbor batch "
                    f"{batch_idx}/"
                    f"{total_batches} 완료"
                )


            # 큰 temporary sparse matrix 정리

            gc.collect()


        elapsed = (
            time.perf_counter()
            -
            start
        )


        print(
            "Neighbor Prewarm 완료:",
            f"{elapsed:.1f}초"
        )


    # =====================================================
    # User Item Scores
    #
    # 알고리즘은 기존 predict_scores와 동일.
    # =====================================================

    def get_user_scores(
        self,
        user_id,
        app_id_list,
    ):

        if (
            user_id
            in
            self.user_score_cache
        ):

            self.user_score_cache_hit += 1

            return (
                self.user_score_cache[
                    user_id
                ]
            )


        self.user_score_cache_miss += 1


        source_indices = [

            self.game_to_idx[
                app_id
            ]

            for app_id in app_id_list

            if app_id
            in
            self.game_to_idx
        ]


        if len(
            source_indices
        ) == 0:

            self.user_score_cache[
                user_id
            ] = {}

            return {}


        # 혹시 prewarm에서 빠진 source가 있으면
        # 여기서 exact 계산

        missing = [

            int(idx)

            for idx in source_indices

            if int(idx)
            not in
            self.neighbor_cache
        ]


        if len(
            missing
        ) > 0:

            self._compute_neighbor_batch(
                missing
            )


        # ---------------------------------------------
        # 기존 알고리즘과 동일한 dense score array
        #
        # n_games = 37,567이라 매우 작음.
        # ---------------------------------------------

        scores = np.zeros(
            self.n_games,
            dtype=np.float64,
        )


        for item_idx in source_indices:

            (
                neighbor_indices,
                neighbor_scores,
            ) = (
                self.neighbor_cache[
                    int(
                        item_idx
                    )
                ]
            )


            scores[
                neighbor_indices
            ] += neighbor_scores


        # 이미 interaction한 item 제거

        scores[
            source_indices
        ] = -np.inf


        valid_indices = np.where(

            np.isfinite(
                scores
            )

            &

            (
                scores > 0
            )

        )[0]


        # ---------------------------------------------
        # positive score만 sparse dictionary로 보관
        # ---------------------------------------------

        score_map = {

            int(idx):
                float(
                    scores[
                        idx
                    ]
                )

            for idx in valid_indices
        }


        self.user_score_cache[
            user_id
        ] = (
            score_map
        )


        return (
            score_map
        )


    # =====================================================
    # Candidate Ranking
    # =====================================================

    def rank_candidates(
        self,
        user_id,
        app_id_list,
        candidate_ids,
        top_n=10,
    ):

        score_map = (
            self.get_user_scores(

                user_id=
                    user_id,

                app_id_list=
                    app_id_list,
            )
        )


        if len(
            score_map
        ) == 0:

            return (
                [],
                [],
                [],
            )


        scoreable_ids = []
        scoreable_scores = []


        for app_id in candidate_ids:

            item_idx = (
                self.game_to_idx
                .get(
                    app_id
                )
            )


            if item_idx is None:
                continue


            score = (
                score_map
                .get(
                    int(
                        item_idx
                    )
                )
            )


            if score is None:
                continue


            scoreable_ids.append(
                app_id
            )

            scoreable_scores.append(
                score
            )


        if len(
            scoreable_ids
        ) == 0:

            return (
                [],
                [],
                [],
            )


        scoreable_ids = np.asarray(
            scoreable_ids
        )

        scoreable_scores = np.asarray(
            scoreable_scores,
            dtype=np.float64,
        )


        order = np.argsort(
            -scoreable_scores,
            kind="stable",
        )


        ranked_ids = (
            scoreable_ids[
                order
            ]
        )


        ranked_scores = (
            scoreable_scores[
                order
            ]
        )


        return (

            ranked_ids[
                :top_n
            ].tolist(),

            ranked_scores[
                :top_n
            ].tolist(),

            scoreable_ids.tolist(),
        )


    # =====================================================
    # Heavy matrix 해제
    # =====================================================

    def release_matrix(
        self,
    ):

        self.item_matrix = None
        self.item_matrix_t = None

        gc.collect()


# =========================================================
# Item Mapping
# =========================================================

def save_item_mapping(
    game_to_idx,
    n_games,
):

    payload = {

        "game_to_idx":
            game_to_idx,

        "n_games":
            int(
                n_games
            ),

        "k":
            ITEM_K,
    }


    with open(
        ITEM_MAPPING_PATH,
        "wb",
    ) as f:

        pickle.dump(
            payload,
            f,
            protocol=
                pickle.HIGHEST_PROTOCOL,
        )


def load_item_mapping():

    if not ITEM_MAPPING_PATH.exists():
        return None


    with open(
        ITEM_MAPPING_PATH,
        "rb",
    ) as f:

        payload = pickle.load(
            f
        )


    if (
        payload.get(
            "k"
        )
        !=
        ITEM_K
    ):

        return None


    return payload


# =========================================================
# Item Ranker Build / Load
#
# 3단계 cache hierarchy:
#
# 1. User Score Cache
#       ↓
#    있으면 matrix조차 필요 없음
#
# 2. Neighbor Cache + normalized matrix
#
# 3. 아무것도 없으면
#    37M Train에서 새로 생성
# =========================================================

def prepare_item_ranker(
    sampled_users,
    train_eval,
):

    print(
        "\n===== Item Ranker 준비 ====="
    )


    mapping_payload = (
        load_item_mapping()
    )


    # =====================================================
    # FAST PATH
    #
    # 이미 400명 Item score가 저장되어 있으면
    # Full Train / Matrix 로드 자체를 생략.
    # =====================================================

    if (
        mapping_payload
        is not None

        and

        USER_SCORE_CACHE_PATH.exists()
    ):

        ranker = (
            OptimizedItemBasedRanker(

                game_to_idx=
                    mapping_payload[
                        "game_to_idx"
                    ],

                n_games=
                    mapping_payload[
                        "n_games"
                    ],

                item_matrix=
                    None,

                k=
                    ITEM_K,
            )
        )


        loaded = (
            ranker
            .load_user_score_cache(

                USER_SCORE_CACHE_PATH,

                sampled_users,
            )
        )


        if loaded:

            print(
                "\nFAST PATH 사용"
            )

            print(
                "37M Train load 생략"
            )

            print(
                "Interaction Matrix 생성 생략"
            )

            print(
                "Item cosine 계산 생략"
            )

            return (
                ranker,
                True,
            )


    # =====================================================
    # Matrix Load
    # =====================================================

    item_matrix = None


    if (
        mapping_payload
        is not None

        and

        ITEM_MATRIX_PATH.exists()
    ):

        print(
            "\nNormalized Item Matrix cache 로드:"
        )

        print(
            ITEM_MATRIX_PATH
        )


        matrix_start = (
            time.perf_counter()
        )


        item_matrix = load_npz(
            ITEM_MATRIX_PATH
        ).tocsr()


        print(
            "Matrix load:",
            f"{time.perf_counter() - matrix_start:.1f}초"
        )


        game_to_idx = (
            mapping_payload[
                "game_to_idx"
            ]
        )

        n_games = (
            mapping_payload[
                "n_games"
            ]
        )


    # =====================================================
    # Matrix 최초 생성
    # =====================================================

    else:

        print(
            "\nItem Matrix cache 없음"
        )

        print(
            "Full Train에서 최초 생성"
        )


        matrix_start = (
            time.perf_counter()
        )


        full_train, _ = (
            load_mf_split()
        )


        (
            interaction_matrix,
            _,
            game_to_idx,
            _,
        ) = (
            build_interaction_matrix(
                full_train
            )
        )


        print(
            "Interaction Matrix:",
            interaction_matrix.shape,
        )


        # User x Item
        # ->
        # Item x User

        item_matrix = (
            interaction_matrix
            .T
            .tocsr()
            .astype(
                np.float64,
                copy=False,
            )
        )


        print(
            "Item matrix L2 normalize..."
        )


        item_matrix = (
            normalize(
                item_matrix,
                norm="l2",
                axis=1,
                copy=False,
            )
            .tocsr()
        )


        n_games = (
            item_matrix
            .shape[0]
        )


        print(
            "Item matrix 완료:",
            item_matrix.shape,
        )


        # ---------------------------------------------
        # Cache 저장
        #
        # compressed=False:
        # 용량은 커지지만 저장/로드가 훨씬 빠름.
        # saved_model은 Git 제외.
        # ---------------------------------------------

        print(
            "\nNormalized Item Matrix 저장:"
        )

        print(
            ITEM_MATRIX_PATH
        )


        save_npz(
            ITEM_MATRIX_PATH,
            item_matrix,
            compressed=False,
        )


        save_item_mapping(
            game_to_idx=
                game_to_idx,

            n_games=
                n_games,
        )


        del interaction_matrix
        del full_train

        gc.collect()


        print(
            "Matrix 준비 총 시간:",
            f"{time.perf_counter() - matrix_start:.1f}초"
        )


    # =====================================================
    # Ranker
    # =====================================================

    ranker = (
        OptimizedItemBasedRanker(

            game_to_idx=
                game_to_idx,

            n_games=
                n_games,

            item_matrix=
                item_matrix,

            k=
                ITEM_K,
        )
    )


    # =====================================================
    # Existing Neighbor Cache
    # =====================================================

    ranker.load_neighbor_cache(
        NEIGHBOR_CACHE_PATH
    )


    # =====================================================
    # 모든 평가 history item을 미리 계산
    #
    # extra item을 계산해도 결과에는 영향 없음.
    # 단지 cache만 미리 만들어두는 것.
    # =====================================================

    source_app_ids = (
        train_eval[
            "app_id"
        ]
        .drop_duplicates()
        .tolist()
    )


    ranker.prewarm_neighbors(

        app_ids=
            source_app_ids,

        batch_size=
            ITEM_PREWARM_BATCH_SIZE,
    )


    # =====================================================
    # Persistent Neighbor Cache
    # =====================================================

    ranker.save_neighbor_cache(
        NEIGHBOR_CACHE_PATH
    )


    return (
        ranker,
        False,
    )


# =========================================================
# Experiment Recommender
#
# Retrieval dictionaries를 이미 만들어서 넘김.
#
# Case마다 DataFrame sort/groupby 하지 않음.
# =========================================================

class RatioSweepRecommender:

    def __init__(
        self,
        case_name,
        config,
        bpr_dict,
        content_dict,
        user_dict,
        item_ranker,
    ):

        self.case_name = (
            case_name
        )

        self.config = (
            config
        )


        self.bpr = bpr_dict
        self.content = content_dict
        self.user = user_dict


        self.item_ranker = (
            item_ranker
        )


        self.logs = {}

        self.call_count = 0


    # =====================================================
    # Recommend
    # =====================================================

    def recommend(
        self,
        user_id,
        app_id_list=None,
        top_n=10,
    ):

        self.call_count += 1


        if (
            self.call_count
            % 100
            == 0
        ):

            print(
                f"[{self.case_name}] "
                f"{self.call_count}/400"
            )


        if app_id_list is None:

            app_id_list = []


        played_set = set(
            app_id_list
        )


        # =================================================
        # Candidate Retrieval
        # =================================================

        bpr_ids = (

            self.bpr
            .get(
                user_id,
                [],
            )
            [
                :self.config[
                    "bpr_n"
                ]
            ]
        )


        content_ids = (

            self.content
            .get(
                user_id,
                [],
            )
            [
                :self.config[
                    "content_n"
                ]
            ]
        )


        user_ids = (

            self.user
            .get(
                user_id,
                [],
            )
            [
                :self.config[
                    "user_n"
                ]
            ]
        )


        # =================================================
        # Seen Item 제거
        # =================================================

        bpr_ids = [

            app_id

            for app_id in bpr_ids

            if app_id
            not in
            played_set
        ]


        content_ids = [

            app_id

            for app_id in content_ids

            if app_id
            not in
            played_set
        ]


        user_ids = [

            app_id

            for app_id in user_ids

            if app_id
            not in
            played_set
        ]


        # =================================================
        # UNION
        # =================================================

        union_ids = (
            unique_preserve_order(

                bpr_ids

                +

                content_ids

                +

                user_ids
            )
        )


        # =================================================
        # Empty
        # =================================================

        if len(union_ids) == 0:

            self.logs[
                user_id
            ] = {

                "bpr":
                    bpr_ids,

                "content":
                    content_ids,

                "user":
                    user_ids,

                "union":
                    [],

                "scoreable":
                    [],

                "final":
                    [],
            }


            return pd.DataFrame(
                columns=[
                    "app_id",
                    "item_score",
                ]
            )


        # =================================================
        # Item-Based Ranking
        # =================================================

        (
            final_ids,
            final_scores,
            scoreable_ids,
        ) = (
            self.item_ranker
            .rank_candidates(

                user_id=
                    user_id,

                app_id_list=
                    app_id_list,

                candidate_ids=
                    union_ids,

                top_n=
                    top_n,
            )
        )


        # =================================================
        # Log
        # =================================================

        self.logs[
            user_id
        ] = {

            "bpr":
                bpr_ids,

            "content":
                content_ids,

            "user":
                user_ids,

            "union":
                union_ids,

            "scoreable":
                scoreable_ids,

            "final":
                final_ids,
        }


        return pd.DataFrame(
            {

                "app_id":
                    final_ids,

                "item_score":
                    final_scores,
            }
        )


# =========================================================
# Candidate Diagnostics
# =========================================================

def build_diagnostics(
    recommender,
    test_df,
    sampled_users,
):

    sampled_ids = set(
        sampled_users[
            "user_id"
        ]
    )


    positive_test = (

        test_df[
            (
                test_df[
                    "user_id"
                ]
                .isin(
                    sampled_ids
                )
            )

            &

            (
                test_df[
                    "is_recommended"
                ]
                == True
            )
        ]

        .groupby(
            "user_id"
        )[
            "app_id"
        ]

        .apply(list)

        .to_dict()
    )


    rows = []


    for row in (
        sampled_users
        .itertuples(
            index=False
        )
    ):

        user_id = (
            row.user_id
        )


        relevant = set(
            positive_test.get(
                user_id,
                [],
            )
        )


        if len(
            relevant
        ) == 0:

            continue


        log = (
            recommender
            .logs
            .get(
                user_id,
                {}
            )
        )


        bpr_set = set(
            log.get(
                "bpr",
                [],
            )
        )


        content_set = set(
            log.get(
                "content",
                [],
            )
        )


        user_set = set(
            log.get(
                "user",
                [],
            )
        )


        union_set = set(
            log.get(
                "union",
                [],
            )
        )


        scoreable_set = set(
            log.get(
                "scoreable",
                [],
            )
        )


        final_set = set(
            log.get(
                "final",
                [],
            )
        )


        n_test = len(
            relevant
        )


        bpr_hits = (
            bpr_set
            &
            relevant
        )


        content_hits = (
            content_set
            &
            relevant
        )


        user_hits = (
            user_set
            &
            relevant
        )


        union_hits = (
            union_set
            &
            relevant
        )


        scoreable_hits = (
            scoreable_set
            &
            relevant
        )


        rows.append(
            {

                "user_id":
                    user_id,

                "review_group":
                    row.review_group,

                "n_games":
                    row.n_games,

                "n_test":
                    n_test,


                # Candidate Count

                "bpr_candidate_count":
                    len(
                        bpr_set
                    ),

                "content_candidate_count":
                    len(
                        content_set
                    ),

                "user_candidate_count":
                    len(
                        user_set
                    ),

                "union_candidate_count":
                    len(
                        union_set
                    ),

                "scoreable_candidate_count":
                    len(
                        scoreable_set
                    ),


                # Candidate Recall

                "bpr_candidate_recall":
                    len(
                        bpr_hits
                    )
                    /
                    n_test,

                "content_candidate_recall":
                    len(
                        content_hits
                    )
                    /
                    n_test,

                "user_candidate_recall":
                    len(
                        user_hits
                    )
                    /
                    n_test,

                "union_candidate_recall":
                    len(
                        union_hits
                    )
                    /
                    n_test,

                "scoreable_candidate_recall":
                    len(
                        scoreable_hits
                    )
                    /
                    n_test,


                # Candidate Hits

                "bpr_candidate_hits":
                    len(
                        bpr_hits
                    ),

                "content_candidate_hits":
                    len(
                        content_hits
                    ),

                "user_candidate_hits":
                    len(
                        user_hits
                    ),

                "union_candidate_hits":
                    len(
                        union_hits
                    ),


                # User unique positive

                "user_unique_hits":
                    len(

                        user_hits

                        -

                        bpr_set

                        -

                        content_set
                    ),


                # Final

                "final_hits":
                    len(
                        final_set
                        &
                        relevant
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# =========================================================
# Summary
# =========================================================

def make_summary(
    case_name,
    config,
    eval_df,
    diagnostics,
):

    total_hits = int(
        eval_df[
            "hits"
        ].sum()
    )


    total_recommended = int(
        eval_df[
            "n_recommended"
        ].sum()
    )


    total_test = int(
        eval_df[
            "n_test"
        ].sum()
    )


    micro_precision = (

        total_hits
        /
        total_recommended

        if total_recommended > 0

        else 0.0
    )


    micro_recall = (

        total_hits
        /
        total_test

        if total_test > 0

        else 0.0
    )


    micro_f1 = (

        2
        *
        micro_precision
        *
        micro_recall
        /
        (
            micro_precision
            +
            micro_recall
        )

        if (
            micro_precision
            +
            micro_recall
        ) > 0

        else 0.0
    )


    return {

        "case_name":
            case_name,

        "family":
            config[
                "family"
            ],

        "user_enabled":
            config[
                "user_enabled"
            ],


        "bpr_candidate_n":
            config[
                "bpr_n"
            ],

        "content_candidate_n":
            config[
                "content_n"
            ],

        "user_candidate_n":
            config[
                "user_n"
            ],


        # Final

        "precision_at_10":
            eval_df[
                "precision"
            ].mean(),

        "recall_at_10":
            eval_df[
                "recall"
            ].mean(),

        "hit_rate_at_10":
            eval_df[
                "hit"
            ].mean(),

        "ndcg_at_10":
            eval_df[
                "ndcg"
            ].mean(),

        "hits":
            total_hits,


        "micro_precision":
            micro_precision,

        "micro_recall":
            micro_recall,

        "micro_f1":
            micro_f1,


        # Candidate

        "avg_union_candidates":
            diagnostics[
                "union_candidate_count"
            ].mean(),

        "bpr_candidate_recall":
            diagnostics[
                "bpr_candidate_recall"
            ].mean(),

        "content_candidate_recall":
            diagnostics[
                "content_candidate_recall"
            ].mean(),

        "user_candidate_recall":
            diagnostics[
                "user_candidate_recall"
            ].mean(),

        "union_candidate_recall":
            diagnostics[
                "union_candidate_recall"
            ].mean(),

        "scoreable_candidate_recall":
            diagnostics[
                "scoreable_candidate_recall"
            ].mean(),

        "user_unique_hits":
            int(
                diagnostics[
                    "user_unique_hits"
                ].sum()
            ),
    }


# =========================================================
# Run Case
# =========================================================

def run_case(
    case_name,
    config,
    bpr_dict,
    content_dict,
    user_dict,
    item_ranker,
    train_eval,
    test_eval,
    sampled_users,
):

    print(
        "\n"
        "=================================================="
    )

    print(
        case_name
    )

    print(
        "=================================================="
    )


    print(
        "Ranker  : Item-Based"
    )

    print(
        "BPR     :",
        config[
            "bpr_n"
        ],
    )

    print(
        "Content :",
        config[
            "content_n"
        ],
    )

    print(
        "User    :",
        config[
            "user_n"
        ],
    )


    recommender = (
        RatioSweepRecommender(

            case_name=
                case_name,

            config=
                config,

            bpr_dict=
                bpr_dict,

            content_dict=
                content_dict,

            user_dict=
                user_dict,

            item_ranker=
                item_ranker,
        )
    )


    start = (
        time.perf_counter()
    )


    eval_df = (
        base.run_mf_evaluation(

            recommender=
                recommender,

            train_df=
                train_eval,

            test_df=
                test_eval,

            sampled_users=
                sampled_users,

            top_n=
                TOP_N,

            positive_only=
                True,
        )
    )


    elapsed = (
        time.perf_counter()
        -
        start
    )


    print(
        "\n평가 시간:",
        f"{elapsed:.2f}초"
    )


    group_summary = (
        base.print_evaluation_report(

            eval_df,

            top_n=
                TOP_N,
        )
    )


    diagnostics = (
        build_diagnostics(

            recommender=
                recommender,

            test_df=
                test_eval,

            sampled_users=
                sampled_users,
        )
    )


    summary = (
        make_summary(

            case_name=
                case_name,

            config=
                config,

            eval_df=
                eval_df,

            diagnostics=
                diagnostics,
        )
    )


    summary[
        "evaluation_seconds"
    ] = elapsed


    # =====================================================
    # Save
    # =====================================================

    eval_path = (
        RESULT_DIR
        /
        f"{case_name}_eval.csv"
    )


    group_path = (
        RESULT_DIR
        /
        f"{case_name}_group.csv"
    )


    diagnostic_path = (
        RESULT_DIR
        /
        f"{case_name}_diagnostics.csv"
    )


    eval_df.to_csv(
        eval_path,
        index=False,
    )


    group_summary.to_csv(
        group_path,
        index=False,
    )


    diagnostics.to_csv(
        diagnostic_path,
        index=False,
    )


    # =====================================================
    # Console
    # =====================================================

    print(
        "\n===== Candidate ====="
    )

    print(
        "UNION Recall:",
        f"{summary['union_candidate_recall']:.6f}"
    )

    print(
        "Item Scoreable Recall:",
        f"{summary['scoreable_candidate_recall']:.6f}"
    )

    print(
        "User Unique Hits:",
        summary[
            "user_unique_hits"
        ]
    )


    print(
        "\n===== Final ====="
    )

    print(
        "P@10:",
        f"{summary['precision_at_10']:.6f}"
    )

    print(
        "R@10:",
        f"{summary['recall_at_10']:.6f}"
    )

    print(
        "HR@10:",
        f"{summary['hit_rate_at_10']:.6f}"
    )

    print(
        "NDCG@10:",
        f"{summary['ndcg_at_10']:.6f}"
    )

    print(
        "Hits:",
        summary[
            "hits"
        ]
    )


    return (
        summary,
        eval_df,
    )


# =========================================================
# User Effect
# =========================================================

def build_user_effect_comparison(
    summary_df,
):

    pairs = [

        (
            "baseline_b59_c41",
            "baseline_b56_c39_u5",
            "Baseline",
        ),

        (
            "bpr70_content30",
            "bpr67_content28_user5",
            "70:30",
        ),

        (
            "bpr50_content50",
            "bpr48_content47_user5",
            "50:50",
        ),
    ]


    lookup = (
        summary_df
        .set_index(
            "case_name"
        )
    )


    rows = []


    for (
        base_case,
        user_case,
        label,
    ) in pairs:

        a = (
            lookup.loc[
                base_case
            ]
        )

        b = (
            lookup.loc[
                user_case
            ]
        )


        rows.append(
            {

                "ratio_family":
                    label,

                "delta_candidate_recall":
                    (
                        b[
                            "union_candidate_recall"
                        ]
                        -
                        a[
                            "union_candidate_recall"
                        ]
                    ),

                "delta_precision":
                    (
                        b[
                            "precision_at_10"
                        ]
                        -
                        a[
                            "precision_at_10"
                        ]
                    ),

                "delta_recall":
                    (
                        b[
                            "recall_at_10"
                        ]
                        -
                        a[
                            "recall_at_10"
                        ]
                    ),

                "delta_hr":
                    (
                        b[
                            "hit_rate_at_10"
                        ]
                        -
                        a[
                            "hit_rate_at_10"
                        ]
                    ),

                "delta_ndcg":
                    (
                        b[
                            "ndcg_at_10"
                        ]
                        -
                        a[
                            "ndcg_at_10"
                        ]
                    ),

                "delta_hits":
                    (
                        b[
                            "hits"
                        ]
                        -
                        a[
                            "hits"
                        ]
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# =========================================================
# Main
# =========================================================

def main():

    total_start = (
        time.perf_counter()
    )


    print(
        "\n"
        "=================================================="
    )

    print(
        " Optimized Case 2 Item-Ranker Ratio Sweep"
    )

    print(
        " 7 Experiments"
    )

    print(
        "=================================================="
    )


    # =====================================================
    # 1. Users
    # =====================================================

    sampled_users = (
        base.load_sampled_users()
    )


    # =====================================================
    # 2. Evaluation Subset
    # =====================================================

    holder = {}


    train_eval, test_eval = (
        base.load_or_create_eval_subset(

            sampled_users,

            holder,
        )
    )


    # =====================================================
    # 3. Retrieval Caches
    # =====================================================

    print(
        "\n===== Retrieval Cache ====="
    )


    cf_holder = {}


    user_cache = (
        base.build_user_cache(

            sampled_users,
            train_eval,
            holder,
            cf_holder,
        )
    )


    bpr_cache = (
        base.build_bpr_cache(

            sampled_users,
            train_eval,
        )
    )


    (
        _,
        bpr_item_ids,
    ) = (
        base.load_bpr_mapping_only()
    )


    content_cache = (
        base.build_content_cache(

            sampled_users,
            train_eval,
            bpr_item_ids,
        )
    )


    base.print_cache_coverage(
        "BPR",
        bpr_cache,
        sampled_users,
    )

    base.print_cache_coverage(
        "Content",
        content_cache,
        sampled_users,
    )

    base.print_cache_coverage(
        "User",
        user_cache,
        sampled_users,
    )


    # =====================================================
    # 4. Cache DataFrame -> dict
    #
    # 딱 한 번만.
    # =====================================================

    print(
        "\nRetrieval cache dictionary 변환..."
    )


    dict_start = (
        time.perf_counter()
    )


    bpr_dict = (
        cache_to_dict(
            bpr_cache
        )
    )


    content_dict = (
        cache_to_dict(
            content_cache
        )
    )


    user_dict = (
        cache_to_dict(
            user_cache
        )
    )


    print(
        "Dictionary 변환 완료:",
        f"{time.perf_counter() - dict_start:.2f}초"
    )


    # =====================================================
    # 5. Item Ranker
    # =====================================================

    item_ranker, fast_path = (
        prepare_item_ranker(

            sampled_users=
                sampled_users,

            train_eval=
                train_eval,
        )
    )


    holder.clear()
    cf_holder.clear()

    gc.collect()


    # =====================================================
    # 6. 7 Experiments
    # =====================================================

    summaries = []


    for case_idx, (
        case_name,
        config,
    ) in enumerate(
        EXPERIMENTS.items(),
        start=1,
    ):

        print(
            "\n"
            f"######## CASE "
            f"{case_idx}/7 "
            f"########"
        )


        summary, _ = (
            run_case(

                case_name=
                    case_name,

                config=
                    config,

                bpr_dict=
                    bpr_dict,

                content_dict=
                    content_dict,

                user_dict=
                    user_dict,

                item_ranker=
                    item_ranker,

                train_eval=
                    train_eval,

                test_eval=
                    test_eval,

                sampled_users=
                    sampled_users,
            )
        )


        summaries.append(
            summary
        )


        # -------------------------------------------------
        # 첫 Case에서 400명 Item score cache 완성.
        #
        # 이후 Case 2~7은 lookup만 함.
        # -------------------------------------------------

        if (
            case_idx == 1

            and

            not fast_path
        ):

            item_ranker.save_user_score_cache(
                USER_SCORE_CACHE_PATH
            )


            # 이후 matrix 필요 없음

            item_ranker.release_matrix()


            print(
                "\nItem matrix 메모리 해제 완료"
            )


    # =====================================================
    # 7. Summary
    # =====================================================

    summary_df = (
        pd.DataFrame(
            summaries
        )
    )


    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )


    display_columns = [

        "case_name",

        "bpr_candidate_n",
        "content_candidate_n",
        "user_candidate_n",

        "avg_union_candidates",

        "union_candidate_recall",
        "scoreable_candidate_recall",

        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",

        "hits",
        "user_unique_hits",

        "evaluation_seconds",
    ]


    print(
        "\n"
        "=================================================="
    )

    print(
        " Final Comparison"
    )

    print(
        "=================================================="
    )


    print(
        summary_df[
            display_columns
        ]
        .to_string(
            index=False
        )
    )


    # =====================================================
    # 8. User Effect
    # =====================================================

    user_compare = (
        build_user_effect_comparison(
            summary_df
        )
    )


    user_compare.to_csv(
        USER_COMPARE_PATH,
        index=False,
    )


    print(
        "\n"
        "=================================================="
    )

    print(
        " User Addition Effect"
    )

    print(
        "=================================================="
    )


    print(
        user_compare
        .to_string(
            index=False
        )
    )


    # =====================================================
    # 9. Best
    # =====================================================

    print(
        "\n"
        "=================================================="
    )

    print(
        " Best Cases"
    )

    print(
        "=================================================="
    )


    metrics = {

        "Precision":
            "precision_at_10",

        "Recall":
            "recall_at_10",

        "HR":
            "hit_rate_at_10",

        "NDCG":
            "ndcg_at_10",

        "Hits":
            "hits",

        "Candidate Recall":
            "union_candidate_recall",
    }


    for (
        label,
        column,
    ) in metrics.items():

        idx = (
            summary_df[
                column
            ]
            .idxmax()
        )


        row = (
            summary_df
            .loc[
                idx
            ]
        )


        print(
            f"{label:<18} "
            f"{row['case_name']} "
            f"({row[column]:.6f})"
        )


    # =====================================================
    # Cache Stats
    # =====================================================

    print(
        "\n===== Item Score Cache ====="
    )


    print(
        "Hit:",
        item_ranker
        .user_score_cache_hit
    )


    print(
        "Miss:",
        item_ranker
        .user_score_cache_miss
    )


    # =====================================================
    # Runtime
    # =====================================================

    elapsed = (
        time.perf_counter()
        -
        total_start
    )


    print(
        "\n전체 실행 시간:",
        f"{elapsed / 60:.2f}분"
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_PATH
    )


    print(
        USER_COMPARE_PATH
    )


    print(
        "\nCache directory:"
    )

    print(
        CACHE_DIR
    )


if __name__ == "__main__":
    main()