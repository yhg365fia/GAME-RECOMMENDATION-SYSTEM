import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import load_npz, csr_matrix
from sklearn.preprocessing import normalize

from implicit.cpu.bpr import BayesianPersonalizedRanking


# =========================================================
# Project Root
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from data_split import load_mf_split

from evaluation import (
    run_mf_evaluation,
    print_evaluation_report,
)

from models.itembase import build_interaction_matrix

from models.content_base import ContentBasedRecommender

from models.bpr import ImplicitBPRAdapter


# =========================================================
# Experiment Config
# =========================================================

# 각 Retriever 후보 수
ITEM_CANDIDATE_N = 30
BPR_CANDIDATE_N = 30
CONTENT_CANDIDATE_N = 30

# 최종 추천 수
TOP_N = 10

# 기존 Item-Based 설정
ITEM_K = 30


# =========================================================
# Path
# =========================================================

SAVED_MODEL_DIR = (
    ROOT
    / "models"
    / "saved_model"
)

RESULT_DIR = (
    SAVED_MODEL_DIR
    / "results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# 기존 400명 평가 사용자
SAMPLED_USERS_PATH = (
    RESULT_DIR
    / "hybrid_sampled_users.csv"
)


# Case 1 결과
CASE1_EVAL_PATH = (
    RESULT_DIR
    / "hybrid_exp_a_bpr_rerank_eval.csv"
)


# =========================================================
# BPR Files
# =========================================================

BPR_MODEL_NAME = (
    "bpr_iter15_factors60_reg0.006_explicit_false.npz"
)

BPR_MAPPING_NAME = (
    "implicit_bpr_mapping.npz"
)

BPR_USER_ITEMS_NAMES = [
    "implicit_bpr_user_items.npz",
    "bpr_user_items.npz",
]


# =========================================================
# Result Files
# =========================================================

EVAL_PATH = (
    RESULT_DIR
    / "hybrid_exp_b_multi_retriever_eval.csv"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "hybrid_exp_b_multi_retriever_summary.csv"
)

GROUP_PATH = (
    RESULT_DIR
    / "hybrid_exp_b_multi_retriever_group_summary.csv"
)

CANDIDATE_PATH = (
    RESULT_DIR
    / "hybrid_exp_b_candidate_diagnostics.csv"
)

POOL_PATH = (
    RESULT_DIR
    / "hybrid_exp_b_candidate_pool.csv"
)

COMPARE_PATH = (
    RESULT_DIR
    / "hybrid_exp_b_vs_exp_a.csv"
)


# =========================================================
# File Search
# =========================================================

def find_file(base_dir, filename):

    matches = list(
        base_dir.rglob(filename)
    )

    if len(matches) == 0:
        return None

    if len(matches) > 1:

        print(
            f"\n주의: {filename} 여러 개 발견"
        )

        for path in matches:
            print(" -", path)

        print(
            "첫 번째 파일을 사용합니다."
        )

    return matches[0]


def find_first_file(
    base_dir,
    filenames,
):

    for filename in filenames:

        path = find_file(
            base_dir,
            filename,
        )

        if path is not None:
            return path

    return None


# =========================================================
# BPR File Load
# =========================================================

def find_bpr_files():

    print(
        "\n===== BPR 저장 파일 탐색 ====="
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model_path = find_file(
        SAVED_MODEL_DIR,
        BPR_MODEL_NAME,
    )

    if model_path is None:

        raise FileNotFoundError(
            "\nBPR 모델 없음:\n"
            f"{BPR_MODEL_NAME}"
        )


    # -----------------------------------------------------
    # Mapping
    # -----------------------------------------------------

    mapping_path = find_file(
        SAVED_MODEL_DIR,
        BPR_MAPPING_NAME,
    )

    if mapping_path is None:

        mapping_path = find_file(
            ROOT / "models",
            BPR_MAPPING_NAME,
        )

    if mapping_path is None:

        raise FileNotFoundError(
            "\nBPR Mapping 없음:\n"
            f"{BPR_MAPPING_NAME}"
        )


    # -----------------------------------------------------
    # User Items
    # -----------------------------------------------------

    user_items_path = find_first_file(
        SAVED_MODEL_DIR,
        BPR_USER_ITEMS_NAMES,
    )

    if user_items_path is None:

        user_items_path = find_first_file(
            ROOT / "models",
            BPR_USER_ITEMS_NAMES,
        )

    if user_items_path is None:

        raise FileNotFoundError(
            "\nBPR 후보 생성에 필요한 "
            "user_items matrix가 없습니다.\n"
            "찾은 파일명 후보:\n"
            + "\n".join(
                BPR_USER_ITEMS_NAMES
            )
        )


    print(
        "\nBPR Model:"
    )
    print(model_path)

    print(
        "\nBPR Mapping:"
    )
    print(mapping_path)

    print(
        "\nBPR User Items:"
    )
    print(user_items_path)


    return (
        model_path,
        mapping_path,
        user_items_path,
    )


# =========================================================
# Content Metadata
#
# 기존 Content-Based 전처리:
#
# Genres
# + Tags
# + Categories
#
# -> combined_features
#
# 기존 모델 조건을 그대로 유지한다.
# =========================================================

def load_content_meta():

    candidate_paths = [

        ROOT
        / "data"
        / "cache"
        / "games.parquet",

        ROOT
        / "data"
        / "cache"
        / "games_inc.parquet",

        ROOT
        / "data"
        / "games.parquet",

        ROOT
        / "data"
        / "games_inc.parquet",
    ]


    selected_path = None

    for path in candidate_paths:

        if path.exists():

            selected_path = path
            break


    if selected_path is None:

        raise FileNotFoundError(
            "\ngames metadata 파일을 찾을 수 없습니다."
        )


    print(
        "\nContent metadata:"
    )

    print(
        selected_path
    )


    # =====================================================
    # 1. Metadata Load
    # =====================================================

    meta = pd.read_parquet(
        selected_path
    )


    print(
        "원본 Metadata:",
        meta.shape
    )


    # =====================================================
    # 2. Column 이름 호환
    # =====================================================

    rename_map = {}


    if (
        "AppID" in meta.columns
        and
        "app_id" not in meta.columns
    ):

        rename_map[
            "AppID"
        ] = "app_id"


    if (
        "genres" in meta.columns
        and
        "Genres" not in meta.columns
    ):

        rename_map[
            "genres"
        ] = "Genres"


    if (
        "tags" in meta.columns
        and
        "Tags" not in meta.columns
    ):

        rename_map[
            "tags"
        ] = "Tags"


    if (
        "categories" in meta.columns
        and
        "Categories" not in meta.columns
    ):

        rename_map[
            "categories"
        ] = "Categories"


    if (
        "name" in meta.columns
        and
        "Name" not in meta.columns
    ):

        rename_map[
            "name"
        ] = "Name"


    if rename_map:

        meta = meta.rename(
            columns=rename_map
        )


    # =====================================================
    # 3. 필수 컬럼 확인
    # =====================================================

    required_columns = [

        "app_id",
        "Name",
        "Genres",
        "Tags",
        "Categories",
    ]


    missing = [

        col

        for col in required_columns

        if col not in meta.columns
    ]


    if missing:

        print(
            "\n현재 games.parquet 컬럼:"
        )

        print(
            list(
                meta.columns
            )
        )


        raise ValueError(
            "\nContent-Based에 필요한 컬럼이 없습니다:\n"
            f"{missing}"
        )


    # =====================================================
    # 4. 필요한 컬럼만 선택
    # =====================================================

    selected_columns = [

        "app_id",
        "Name",
        "Genres",
        "Tags",
        "Categories",
    ]


    if "About the game" in meta.columns:

        selected_columns.append(
            "About the game"
        )


    meta = meta[
        selected_columns
    ].copy()


    # =====================================================
    # 5. 결측값 처리
    # =====================================================

    meta = meta.dropna(
        subset=[
            "Name"
        ]
    )


    text_columns = [

        "Genres",
        "Tags",
        "Categories",
    ]


    if "About the game" in meta.columns:

        text_columns.append(
            "About the game"
        )


    meta[
        text_columns
    ] = (

        meta[
            text_columns
        ]

        .fillna("")
    )


    # =====================================================
    # 6. 기존 combined_features 생성
    #
    # Genres + Tags + Categories
    # =====================================================

    meta[
        "combined_features"
    ] = (

        meta[
            "Genres"
        ]
        .astype(str)
        .str.replace(
            ",",
            " ",
            regex=False,
        )

        +

        " "

        +

        meta[
            "Tags"
        ]
        .astype(str)
        .str.replace(
            ",",
            " ",
            regex=False,
        )

        +

        " "

        +

        meta[
            "Categories"
        ]
        .astype(str)
        .str.replace(
            ",",
            " ",
            regex=False,
        )
    )


    # =====================================================
    # 7. app_id 중복 제거
    # =====================================================

    meta = (

        meta

        .drop_duplicates(
            subset=[
                "app_id"
            ]
        )

        .reset_index(
            drop=True
        )
    )


    # =====================================================
    # 8. Content Model용 컬럼만 반환
    # =====================================================

    meta = meta[
        [
            "app_id",
            "Name",
            "combined_features",
        ]
    ]


    print(
        "Content preprocessing 완료:",
        meta.shape
    )


    print(
        "combined_features 생성 완료"
    )


    return meta


# =========================================================
# Utility
# =========================================================

def unique_preserve_order(
    values,
):

    seen = set()

    result = []


    for value in values:

        if value in seen:
            continue

        seen.add(
            value
        )

        result.append(
            value
        )


    return result


# =========================================================
# Optimized Item-Based
#
# 기존:
#
# cosine_similarity(
#     source_items,
#     all_items
# )
#
# 를 사용자마다 반복
#
#
# 최적화:
#
# Item vector normalization 1회
# +
# source item별 Top-K neighbor cache
#
#
# 모델 조건:
#
# cosine similarity 동일
# positive similarity만 사용
# k=30 동일
# 합산 방식 동일
# =========================================================

class CachedItemBasedCFRecommender:

    def __init__(
        self,
        interaction_matrix,
        game_to_idx,
        idx_to_game,
        k=30,
    ):

        self.game_to_idx = (
            game_to_idx
        )

        self.idx_to_game = (
            idx_to_game
        )

        self.k = k


        # User x Item
        # ->
        # Item x User

        item_matrix = (
            interaction_matrix
            .T
            .tocsr()
        )


        print(
            "\nItem matrix L2 normalization..."
        )


        # sklearn cosine_similarity와 같은
        # L2 normalization + dot product 구조

        item_matrix = (
            item_matrix
            .astype(
                np.float64,
                copy=False,
            )
        )


        self.item_matrix = normalize(
            item_matrix,
            norm="l2",
            axis=1,
            copy=False,
        ).tocsr()


        print(
            "Item normalization 완료"
        )


        self.n_games = (
            self.item_matrix.shape[0]
        )


        # source item index
        # ->
        # (top-k neighbor index, similarity)

        self.neighbor_cache = {}


        self.cache_hit_count = 0

        self.cache_miss_count = 0


    # =====================================================
    # Missing Neighbor 계산
    # =====================================================

    def _cache_missing_neighbors(
        self,
        source_indices,
    ):

        unique_indices = list(
            dict.fromkeys(
                source_indices
            )
        )


        missing_indices = [

            idx

            for idx in unique_indices

            if idx not in self.neighbor_cache

        ]


        self.cache_hit_count += (

            len(
                unique_indices
            )

            -

            len(
                missing_indices
            )
        )


        if len(
            missing_indices
        ) == 0:

            return


        self.cache_miss_count += len(
            missing_indices
        )


        # -------------------------------------------------
        # normalized item dot normalized item
        # = cosine similarity
        # -------------------------------------------------

        source_matrix = (

            self.item_matrix[
                missing_indices
            ]

        )


        sims = (

            source_matrix

            @

            self.item_matrix.T

        ).tocsr()


        # =================================================
        # Source Item별 Top-K Neighbor
        # =================================================

        for local_row, item_idx in enumerate(
            missing_indices
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


            # self 제외
            # positive similarity만 사용

            valid_mask = (

                (row_indices != item_idx)

                &

                (row_data > 0)

            )


            positive_indices = (
                row_indices[
                    valid_mask
                ]
            )


            positive_data = (
                row_data[
                    valid_mask
                ]
            )


            if len(
                positive_data
            ) == 0:

                self.neighbor_cache[
                    item_idx
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


            top_k_pos = np.argpartition(
                positive_data,
                -k,
            )[-k:]


            top_indices = (
                positive_indices[
                    top_k_pos
                ]
            )


            top_sims = (
                positive_data[
                    top_k_pos
                ]
            )


            self.neighbor_cache[
                item_idx
            ] = (

                top_indices.copy(),

                top_sims.copy(),
            )


    # =====================================================
    # Recommend
    # =====================================================

    def recommend(
        self,
        app_id_list,
        top_n=10,
        exclude_user_idx=None,
    ):

        # -------------------------------------------------
        # Train에 있는 게임만
        # -------------------------------------------------

        col_idx = [

            self.game_to_idx[
                app_id
            ]

            for app_id in app_id_list

            if app_id
            in
            self.game_to_idx

        ]


        if len(
            col_idx
        ) == 0:

            return pd.DataFrame(
                columns=[
                    "app_id"
                ]
            )


        # -------------------------------------------------
        # 필요한 Source Item만 최초 1회 계산
        # -------------------------------------------------

        self._cache_missing_neighbors(
            col_idx
        )


        predicted_scores = np.zeros(
            self.n_games,
            dtype=np.float64,
        )


        # -------------------------------------------------
        # 기존 Item-Based:
        #
        # source item별 Top-K similarity 합산
        # -------------------------------------------------

        for item_idx in col_idx:

            (
                neighbor_indices,
                neighbor_sims,
            ) = self.neighbor_cache[
                item_idx
            ]


            predicted_scores[
                neighbor_indices
            ] += neighbor_sims


        # -------------------------------------------------
        # 이미 상호작용한 게임 제외
        # -------------------------------------------------

        predicted_scores[
            col_idx
        ] = -np.inf


        # -------------------------------------------------
        # score > 0만 후보
        # -------------------------------------------------

        valid_idx = np.where(
            predicted_scores > 0
        )[0]


        if len(
            valid_idx
        ) == 0:

            return pd.DataFrame(
                columns=[
                    "app_id"
                ]
            )


        n_actual = min(
            top_n,
            len(
                valid_idx
            ),
        )


        top_n_idx = valid_idx[
            np.argpartition(
                predicted_scores[
                    valid_idx
                ],
                -n_actual,
            )[-n_actual:]
        ]


        top_n_idx = top_n_idx[
            np.argsort(
                -predicted_scores[
                    top_n_idx
                ]
            )
        ]


        recommended_app_ids = [

            self.idx_to_game[
                idx
            ]

            for idx in top_n_idx

        ]


        return pd.DataFrame(
            {
                "app_id":
                    recommended_app_ids
            }
        )


# =========================================================
# Optimized Content-Based
#
# 기존:
#
# 각 플레이 게임마다
# cosine_similarity(game, all_games)
#
# 계산 후 평균
#
#
# 최적화:
#
# TF-IDF vectors 평균
# ->
# 전체 TF-IDF와 sparse dot product 1회
#
#
# TfidfVectorizer 기본 norm="l2"이므로:
#
# mean(cos(x_i, y))
#
# =
#
# mean(x_i) dot y
#
# =========================================================

class FastContentBasedRecommender(
    ContentBasedRecommender
):

    def recommend(
        self,
        app_id_list,
        top_n=10,
    ):

        played_indices = []


        for app_id in app_id_list:

            if (
                app_id
                not in
                self.appid_to_idx
            ):

                continue


            played_indices.append(
                self.appid_to_idx[
                    app_id
                ]
            )


        if len(
            played_indices
        ) == 0:

            return pd.DataFrame(
                columns=[
                    "app_id",
                    "Name",
                    "Similarity",
                ]
            )


        # -------------------------------------------------
        # 사용자가 플레이한 게임들의 TF-IDF
        # -------------------------------------------------

        source_matrix = (

            self.tfidf_matrix[
                played_indices
            ]

        )


        # -------------------------------------------------
        # 평균 TF-IDF profile
        # -------------------------------------------------

        profile = csr_matrix(
            source_matrix.sum(
                axis=0
            )
        )


        profile = profile.multiply(
            1.0
            /
            len(
                played_indices
            )
        )


        # -------------------------------------------------
        # 기존 mean cosine similarity와 동일
        # -------------------------------------------------

        score_sparse = (

            profile

            @

            self.tfidf_matrix.T

        )


        mean_scores = (

            score_sparse

            .toarray()

            .ravel()
        )


        # 기존 코드와 동일하게 전체 정렬
        sim_indices = (
            mean_scores
            .argsort()[::-1]
        )


        played_index_set = set(
            played_indices
        )


        result = []


        for idx in sim_indices:

            if idx in played_index_set:
                continue


            app_id = (

                self.meta
                .iloc[
                    idx
                ][
                    "app_id"
                ]

            )


            result.append(
                {

                    "app_id":
                        app_id,

                    "Name":
                        self.meta
                        .iloc[
                            idx
                        ][
                            "Name"
                        ],

                    "Similarity":
                        mean_scores[
                            idx
                        ],
                }
            )


            if len(
                result
            ) == top_n:

                break


        return pd.DataFrame(
            result
        )


# =========================================================
# Multi Retriever
# =========================================================

class MultiRetrieverBPRReranker:

    """
    Case 2

    Item-Based Top-30 -------┐
                             │
    BPR Top-30 --------------┼── UNION
                             │
    Content Top-30 ----------┘
                              ↓
                        BPR Rerank
                              ↓
                           Top-10
    """


    def __init__(
        self,
        item_model,
        bpr_retriever,
        content_model,
        bpr_model,
        user_ids,
        item_ids,
    ):

        self.item_model = (
            item_model
        )

        self.bpr_retriever = (
            bpr_retriever
        )

        self.content_model = (
            content_model
        )

        self.bpr_model = (
            bpr_model
        )


        self.user_ids = np.asarray(
            user_ids
        )

        self.item_ids = np.asarray(
            item_ids
        )


        # 평가 과정에서 생성된 후보 저장
        self.candidate_log = {}

        self.pool_logs = []


        self.missing_users = 0

        self.missing_candidate_items = 0


    # =====================================================
    # ID -> Index
    # =====================================================

    @staticmethod
    def find_index(
        sorted_ids,
        value,
    ):

        idx = int(
            np.searchsorted(
                sorted_ids,
                value,
            )
        )


        if idx >= len(
            sorted_ids
        ):

            return None


        if (
            sorted_ids[
                idx
            ]
            != value
        ):

            return None


        return idx


    # =====================================================
    # Recommend
    # =====================================================

    def recommend(
        self,
        user_id,
        app_id_list=None,
        top_n=10,
    ):

        if app_id_list is None:
            app_id_list = []


        played_set = set(
            app_id_list
        )


        # =================================================
        # 1. Item-Based Top-30
        # =================================================

        item_result = (
            self.item_model.recommend(
                app_id_list=
                    app_id_list,

                top_n=
                    ITEM_CANDIDATE_N,
            )
        )


        if (
            item_result is None
            or
            len(
                item_result
            ) == 0
        ):

            item_ids = []

        else:

            item_ids = (

                item_result[
                    "app_id"
                ]

                .tolist()
            )


        item_ids = unique_preserve_order(
            [
                app_id
                for app_id in item_ids
                if app_id not in played_set
            ]
        )


        # =================================================
        # 2. BPR Top-30
        # =================================================

        bpr_result = (
            self.bpr_retriever.recommend(
                user_id=
                    user_id,

                app_id_list=
                    app_id_list,

                top_n=
                    BPR_CANDIDATE_N,
            )
        )


        if (
            bpr_result is None
            or
            len(
                bpr_result
            ) == 0
        ):

            bpr_ids = []

        else:

            bpr_ids = (

                bpr_result[
                    "app_id"
                ]

                .tolist()
            )


        bpr_ids = unique_preserve_order(
            [
                app_id
                for app_id in bpr_ids
                if app_id not in played_set
            ]
        )


        # =================================================
        # 3. Content Top-30
        # =================================================

        content_result = (
            self.content_model.recommend(
                app_id_list=
                    app_id_list,

                top_n=
                    CONTENT_CANDIDATE_N,
            )
        )


        if (
            content_result is None
            or
            len(
                content_result
            ) == 0
        ):

            content_ids = []

        else:

            content_ids = (

                content_result[
                    "app_id"
                ]

                .tolist()
            )


        content_ids = unique_preserve_order(
            [
                app_id
                for app_id in content_ids
                if app_id not in played_set
            ]
        )


        # =================================================
        # 4. UNION
        # =================================================

        union_ids = unique_preserve_order(

            item_ids
            +
            bpr_ids
            +
            content_ids

        )


        # =================================================
        # Retriever별 rank
        # =================================================

        item_rank = {

            app_id:
                rank + 1

            for rank, app_id
            in enumerate(
                item_ids
            )
        }


        bpr_rank = {

            app_id:
                rank + 1

            for rank, app_id
            in enumerate(
                bpr_ids
            )
        }


        content_rank = {

            app_id:
                rank + 1

            for rank, app_id
            in enumerate(
                content_ids
            )
        }


        # =================================================
        # Retriever Score
        # =================================================

        bpr_retrieval_score = {}


        if (
            bpr_result is not None
            and
            len(
                bpr_result
            ) > 0
            and
            "score"
            in bpr_result.columns
        ):

            bpr_retrieval_score = dict(
                zip(
                    bpr_result[
                        "app_id"
                    ],

                    bpr_result[
                        "score"
                    ],
                )
            )


        content_similarity = {}


        if (
            content_result is not None
            and
            len(
                content_result
            ) > 0
            and
            "Similarity"
            in content_result.columns
        ):

            content_similarity = dict(
                zip(
                    content_result[
                        "app_id"
                    ],

                    content_result[
                        "Similarity"
                    ],
                )
            )


        # =================================================
        # UNION Empty
        # =================================================

        if len(
            union_ids
        ) == 0:

            self.candidate_log[
                user_id
            ] = {

                "item":
                    item_ids,

                "bpr":
                    bpr_ids,

                "content":
                    content_ids,

                "union":
                    [],

                "scoreable_union":
                    [],

                "final":
                    [],
            }


            return pd.DataFrame(
                columns=[
                    "app_id",
                    "bpr_score",
                ]
            )


        # =================================================
        # 5. user_id -> BPR user index
        # =================================================

        user_idx = self.find_index(
            self.user_ids,
            user_id,
        )


        if user_idx is None:

            self.missing_users += 1


            self.candidate_log[
                user_id
            ] = {

                "item":
                    item_ids,

                "bpr":
                    bpr_ids,

                "content":
                    content_ids,

                "union":
                    union_ids,

                "scoreable_union":
                    [],

                "final":
                    [],
            }


            return pd.DataFrame(
                columns=[
                    "app_id",
                    "bpr_score",
                ]
            )


        # =================================================
        # 6. Candidate -> BPR item index
        # =================================================

        union_array = np.asarray(
            union_ids
        )


        item_indices = np.searchsorted(
            self.item_ids,
            union_array,
        )


        valid = (

            item_indices
            <
            len(
                self.item_ids
            )
        )


        matched = np.zeros(
            len(
                union_array
            ),
            dtype=bool,
        )


        valid_positions = np.where(
            valid
        )[0]


        if len(
            valid_positions
        ) > 0:

            matched[
                valid_positions
            ] = (

                self.item_ids[
                    item_indices[
                        valid_positions
                    ]
                ]

                ==

                union_array[
                    valid_positions
                ]

            )


        valid = (
            valid
            &
            matched
        )


        self.missing_candidate_items += int(
            (
                ~valid
            ).sum()
        )


        scoreable_app_ids = (
            union_array[
                valid
            ]
        )


        scoreable_item_indices = (
            item_indices[
                valid
            ]
        )


        # =================================================
        # 후보가 전부 BPR mapping 밖
        # =================================================

        if len(
            scoreable_app_ids
        ) == 0:

            self.candidate_log[
                user_id
            ] = {

                "item":
                    item_ids,

                "bpr":
                    bpr_ids,

                "content":
                    content_ids,

                "union":
                    union_ids,

                "scoreable_union":
                    [],

                "final":
                    [],
            }


            return pd.DataFrame(
                columns=[
                    "app_id",
                    "bpr_score",
                ]
            )


        # =================================================
        # 7. BPR Reranking
        #
        # UNION에 들어온 후보만 점수 계산
        # =================================================

        user_factor = (

            self.bpr_model
            .user_factors[
                user_idx
            ]

        )


        candidate_factors = (

            self.bpr_model
            .item_factors[
                scoreable_item_indices
            ]

        )


        bpr_scores = (

            candidate_factors

            @

            user_factor
        )


        # =================================================
        # 8. BPR Score 정렬
        # =================================================

        order = np.argsort(
            -bpr_scores,
            kind="stable",
        )


        ranked_app_ids = (

            scoreable_app_ids[
                order
            ]

        )


        ranked_scores = (

            bpr_scores[
                order
            ]

        )


        # =================================================
        # 9. Final Top-10
        # =================================================

        n_actual = min(
            top_n,
            len(
                ranked_app_ids
            ),
        )


        final_app_ids = (

            ranked_app_ids[
                :n_actual
            ]

        )


        final_scores = (

            ranked_scores[
                :n_actual
            ]

        )


        # =================================================
        # 10. Candidate Cache
        # =================================================

        self.candidate_log[
            user_id
        ] = {

            "item":
                item_ids,

            "bpr":
                bpr_ids,

            "content":
                content_ids,

            "union":
                union_ids,

            "scoreable_union":
                scoreable_app_ids.tolist(),

            "final":
                final_app_ids.tolist(),
        }


        # =================================================
        # 11. Candidate Pool 상세 저장
        # =================================================

        rerank_score_map = dict(
            zip(
                scoreable_app_ids.tolist(),
                bpr_scores.tolist(),
            )
        )


        final_rank_map = {

            app_id:
                rank + 1

            for rank, app_id
            in enumerate(
                final_app_ids.tolist()
            )
        }


        pool_rows = []


        for app_id in union_ids:

            from_item = (
                app_id
                in
                item_rank
            )

            from_bpr = (
                app_id
                in
                bpr_rank
            )

            from_content = (
                app_id
                in
                content_rank
            )


            pool_rows.append(
                {

                    "user_id":
                        user_id,

                    "app_id":
                        app_id,

                    "from_item":
                        int(
                            from_item
                        ),

                    "from_bpr":
                        int(
                            from_bpr
                        ),

                    "from_content":
                        int(
                            from_content
                        ),

                    "retriever_count":
                        (
                            int(
                                from_item
                            )
                            +
                            int(
                                from_bpr
                            )
                            +
                            int(
                                from_content
                            )
                        ),

                    "item_rank":
                        item_rank.get(
                            app_id,
                            np.nan,
                        ),

                    "bpr_rank":
                        bpr_rank.get(
                            app_id,
                            np.nan,
                        ),

                    "content_rank":
                        content_rank.get(
                            app_id,
                            np.nan,
                        ),

                    "bpr_retrieval_score":
                        bpr_retrieval_score.get(
                            app_id,
                            np.nan,
                        ),

                    "content_similarity":
                        content_similarity.get(
                            app_id,
                            np.nan,
                        ),

                    "bpr_rerank_score":
                        rerank_score_map.get(
                            app_id,
                            np.nan,
                        ),

                    "final_rank":
                        final_rank_map.get(
                            app_id,
                            np.nan,
                        ),

                    "selected_top10":
                        int(
                            app_id
                            in
                            final_rank_map
                        ),
                }
            )


        self.pool_logs.append(
            pd.DataFrame(
                pool_rows
            )
        )


        # =================================================
        # 12. 평가용 최종 추천
        # =================================================

        return pd.DataFrame(
            {

                "app_id":
                    final_app_ids,

                "bpr_score":
                    final_scores,
            }
        )


# =========================================================
# Candidate Diagnostics
# =========================================================

def build_candidate_diagnostics(
    recommender,
    test_df,
    sampled_users,
):

    sampled_ids = set(
        sampled_users[
            "user_id"
        ]
    )


    # Positive Test Item만 정답
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

        .apply(
            list
        )

        .to_dict()
    )


    rows = []


    for row in sampled_users.itertuples(
        index=False
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
            .candidate_log
            .get(
                user_id,
                {}
            )
        )


        item_set = set(
            log.get(
                "item",
                [],
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


        union_set = set(
            log.get(
                "union",
                [],
            )
        )


        scoreable_set = set(
            log.get(
                "scoreable_union",
                [],
            )
        )


        item_hits = (
            item_set
            &
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


        n_test = len(
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


                # =========================================
                # Candidate Count
                # =========================================

                "item_candidate_count":
                    len(
                        item_set
                    ),

                "bpr_candidate_count":
                    len(
                        bpr_set
                    ),

                "content_candidate_count":
                    len(
                        content_set
                    ),

                "union_candidate_count":
                    len(
                        union_set
                    ),

                "scoreable_union_count":
                    len(
                        scoreable_set
                    ),


                # =========================================
                # Hits
                # =========================================

                "item_candidate_hits":
                    len(
                        item_hits
                    ),

                "bpr_candidate_hits":
                    len(
                        bpr_hits
                    ),

                "content_candidate_hits":
                    len(
                        content_hits
                    ),

                "union_candidate_hits":
                    len(
                        union_hits
                    ),


                # =========================================
                # Recall
                # =========================================

                "item_candidate_recall":
                    len(
                        item_hits
                    )
                    /
                    n_test,

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

                "union_candidate_recall":
                    len(
                        union_hits
                    )
                    /
                    n_test,

                "scoreable_union_recall":
                    len(
                        scoreable_hits
                    )
                    /
                    n_test,


                # =========================================
                # Candidate Hit 여부
                # =========================================

                "item_candidate_hit":
                    int(
                        len(
                            item_hits
                        ) > 0
                    ),

                "bpr_candidate_hit":
                    int(
                        len(
                            bpr_hits
                        ) > 0
                    ),

                "content_candidate_hit":
                    int(
                        len(
                            content_hits
                        ) > 0
                    ),

                "union_candidate_hit":
                    int(
                        len(
                            union_hits
                        ) > 0
                    ),


                # =========================================
                # Item 대비 추가 정답
                # =========================================

                "bpr_added_hits":
                    len(
                        bpr_hits
                        -
                        item_set
                    ),

                "content_added_hits":
                    len(
                        content_hits
                        -
                        item_set
                    ),

                "recovered_hits_vs_item":
                    len(
                        union_hits
                        -
                        item_hits
                    ),


                # =========================================
                # Unique Hits
                # =========================================

                "item_unique_hits":
                    len(
                        item_hits
                        -
                        bpr_set
                        -
                        content_set
                    ),

                "bpr_unique_hits":
                    len(
                        bpr_hits
                        -
                        item_set
                        -
                        content_set
                    ),

                "content_unique_hits":
                    len(
                        content_hits
                        -
                        item_set
                        -
                        bpr_set
                    ),


                # =========================================
                # Retriever Overlap
                # =========================================

                "item_bpr_overlap":
                    len(
                        item_set
                        &
                        bpr_set
                    ),

                "item_content_overlap":
                    len(
                        item_set
                        &
                        content_set
                    ),

                "bpr_content_overlap":
                    len(
                        bpr_set
                        &
                        content_set
                    ),

                "all_three_overlap":
                    len(
                        item_set
                        &
                        bpr_set
                        &
                        content_set
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
    eval_df,
    candidate_df,
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

        "experiment":
            "case2_multi_retriever_bpr_reranking",

        "item_candidate_n":
            ITEM_CANDIDATE_N,

        "bpr_candidate_n":
            BPR_CANDIDATE_N,

        "content_candidate_n":
            CONTENT_CANDIDATE_N,

        "top_n":
            TOP_N,

        "n_users":
            len(
                eval_df
            ),


        # Final Top-10
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


        "micro_precision_at_10":
            micro_precision,

        "micro_recall_at_10":
            micro_recall,

        "micro_f1_at_10":
            micro_f1,

        "hits":
            total_hits,


        # Candidate
        "item_candidate_recall":
            candidate_df[
                "item_candidate_recall"
            ].mean(),

        "bpr_candidate_recall":
            candidate_df[
                "bpr_candidate_recall"
            ].mean(),

        "content_candidate_recall":
            candidate_df[
                "content_candidate_recall"
            ].mean(),

        "union_candidate_recall":
            candidate_df[
                "union_candidate_recall"
            ].mean(),

        "scoreable_union_recall":
            candidate_df[
                "scoreable_union_recall"
            ].mean(),

        "avg_union_candidates":
            candidate_df[
                "union_candidate_count"
            ].mean(),

        "avg_scoreable_union_candidates":
            candidate_df[
                "scoreable_union_count"
            ].mean(),

        "recovered_hits_vs_item":
            int(
                candidate_df[
                    "recovered_hits_vs_item"
                ].sum()
            ),

        "bpr_unique_hits":
            int(
                candidate_df[
                    "bpr_unique_hits"
                ].sum()
            ),

        "content_unique_hits":
            int(
                candidate_df[
                    "content_unique_hits"
                ].sum()
            ),
    }


# =========================================================
# Case 1 Compare
# =========================================================

def compare_with_case1(
    case2_df,
):

    if not CASE1_EVAL_PATH.exists():

        print(
            "\nCase 1 결과 없음 → 비교 생략"
        )

        return None


    case1_df = pd.read_csv(
        CASE1_EVAL_PATH
    )


    columns = [

        "user_id",
        "precision",
        "recall",
        "hits",
        "hit",
        "ndcg",
    ]


    case1 = (

        case1_df[
            columns
        ]

        .rename(
            columns={

                "precision":
                    "case1_precision",

                "recall":
                    "case1_recall",

                "hits":
                    "case1_hits",

                "hit":
                    "case1_hit",

                "ndcg":
                    "case1_ndcg",
            }
        )
    )


    case2 = (

        case2_df[
            columns
        ]

        .rename(
            columns={

                "precision":
                    "case2_precision",

                "recall":
                    "case2_recall",

                "hits":
                    "case2_hits",

                "hit":
                    "case2_hit",

                "ndcg":
                    "case2_ndcg",
            }
        )
    )


    result = case1.merge(
        case2,
        on="user_id",
        how="inner",
    )


    result[
        "delta_precision"
    ] = (

        result[
            "case2_precision"
        ]

        -

        result[
            "case1_precision"
        ]
    )


    result[
        "delta_recall"
    ] = (

        result[
            "case2_recall"
        ]

        -

        result[
            "case1_recall"
        ]
    )


    result[
        "delta_hits"
    ] = (

        result[
            "case2_hits"
        ]

        -

        result[
            "case1_hits"
        ]
    )


    result[
        "delta_hit"
    ] = (

        result[
            "case2_hit"
        ]

        -

        result[
            "case1_hit"
        ]
    )


    result[
        "delta_ndcg"
    ] = (

        result[
            "case2_ndcg"
        ]

        -

        result[
            "case1_ndcg"
        ]
    )


    return result


# =========================================================
# Main
# =========================================================

def main():

    print()

    print(
        "=========================================="
    )

    print(
        " Hybrid Experiment B"
    )

    print(
        " Optimized Multi-Retriever"
    )

    print(
        " Item30 + BPR30 + Content30"
    )

    print(
        " UNION -> BPR -> Top10"
    )

    print(
        "=========================================="
    )


    # =====================================================
    # 1. Train / Test
    # =====================================================

    print(
        "\n===== 1. MF Train / Test 로드 ====="
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
    # 2. Interaction Matrix
    # =====================================================

    print(
        "\n===== 2. Interaction Matrix 생성 ====="
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
    # 3. Optimized Item-Based
    # =====================================================

    print(
        "\n===== 3. Cached Item-Based 생성 ====="
    )


    item_model = (
        CachedItemBasedCFRecommender(
            interaction_matrix=
                interaction_matrix,

            game_to_idx=
                game_to_idx,

            idx_to_game=
                idx_to_game,

            k=
                ITEM_K,
        )
    )


    # ItemBased 내부에 필요한
    # Item x User matrix는 이미 존재
    del interaction_matrix

    gc.collect()


    # =====================================================
    # 4. BPR
    # =====================================================

    print(
        "\n===== 4. BPR 로드 ====="
    )


    (
        model_path,
        mapping_path,
        user_items_path,
    ) = find_bpr_files()


    bpr_model = (
        BayesianPersonalizedRanking.load(
            str(
                model_path
            )
        )
    )


    mapping = np.load(
        mapping_path
    )


    user_ids = (
        mapping[
            "user_ids"
        ]
    )


    item_ids = (
        mapping[
            "item_ids"
        ]
    )


    bpr_user_items = load_npz(
        user_items_path
    )


    print(
        "User factors:",
        bpr_model
        .user_factors
        .shape,
    )


    print(
        "Item factors:",
        bpr_model
        .item_factors
        .shape,
    )


    print(
        "User Items:",
        bpr_user_items.shape,
    )


    # =====================================================
    # BPR Shape 검증
    # =====================================================

    if (
        bpr_model
        .user_factors
        .shape[0]

        !=

        len(
            user_ids
        )
    ):

        raise ValueError(
            "BPR user_factors와 user_ids 수가 다릅니다."
        )


    if (
        bpr_model
        .item_factors
        .shape[0]

        !=

        len(
            item_ids
        )
    ):

        raise ValueError(
            "BPR item_factors와 item_ids 수가 다릅니다."
        )


    if (
        bpr_user_items.shape[0]

        !=

        len(
            user_ids
        )
    ):

        raise ValueError(
            "BPR user_items와 user_ids 수가 다릅니다."
        )


    # =====================================================
    # 5. BPR Retriever
    # =====================================================

    print(
        "\n===== 5. BPR Retriever 생성 ====="
    )


    bpr_retriever = (
        ImplicitBPRAdapter(
            model=
                bpr_model,

            user_items=
                bpr_user_items,

            user_ids=
                user_ids,

            item_ids=
                item_ids,
        )
    )


    # =====================================================
    # 6. Content-Based
    # =====================================================

    print(
        "\n===== 6. Optimized Content-Based 준비 ====="
    )


    content_meta = (
        load_content_meta()
    )


    content_model = (
        FastContentBasedRecommender()
    )


    print(
        "\nTF-IDF fit 시작..."
    )


    content_model.fit(
        content_meta
    )


    print(
        "TF-IDF fit 완료"
    )


    print(
        "TF-IDF Matrix:",
        content_model
        .tfidf_matrix
        .shape,
    )


    # =====================================================
    # 7. 평가 사용자
    # =====================================================

    print(
        "\n===== 7. 기존 평가 사용자 준비 ====="
    )


    if not SAMPLED_USERS_PATH.exists():

        raise FileNotFoundError(
            "\n기존 400명 평가 사용자 파일 없음:\n"
            f"{SAMPLED_USERS_PATH}"
        )


    sampled_users = pd.read_csv(
        SAMPLED_USERS_PATH
    )


    print(
        "평가 사용자:",
        len(
            sampled_users
        )
    )


    print(
        "\nReview Group:"
    )


    print(
        sampled_users[
            "review_group"
        ]
        .value_counts()
    )


    # =====================================================
    # 8. Multi-Retriever
    # =====================================================

    print(
        "\n===== 8. Multi-Retriever 생성 ====="
    )


    recommender = (
        MultiRetrieverBPRReranker(
            item_model=
                item_model,

            bpr_retriever=
                bpr_retriever,

            content_model=
                content_model,

            bpr_model=
                bpr_model,

            user_ids=
                user_ids,

            item_ids=
                item_ids,
        )
    )


    # =====================================================
    # 9. Evaluation
    # =====================================================

    print(
        "\n===== 9. Case 2 평가 시작 ====="
    )


    eval_df = (
        run_mf_evaluation(
            recommender=
                recommender,

            train_df=
                train_df,

            test_df=
                test_df,

            sampled_users=
                sampled_users,

            top_n=
                TOP_N,

            positive_only=
                True,
        )
    )


    # =====================================================
    # 10. Final Top-10 Report
    # =====================================================

    print(
        "\n===== 10. 최종 Top-10 평가 ====="
    )


    group_summary = (
        print_evaluation_report(
            eval_df,
            top_n=
                TOP_N,
        )
    )


    # =====================================================
    # 11. Candidate Diagnostic
    # =====================================================

    print(
        "\n===== 11. Candidate Pool 분석 ====="
    )


    candidate_df = (
        build_candidate_diagnostics(
            recommender=
                recommender,

            test_df=
                test_df,

            sampled_users=
                sampled_users,
        )
    )


    # =====================================================
    # Candidate Count
    # =====================================================

    print(
        "\n===== 평균 후보 개수 ====="
    )


    print(
        "Item:",
        f"{candidate_df['item_candidate_count'].mean():.2f}"
    )


    print(
        "BPR:",
        f"{candidate_df['bpr_candidate_count'].mean():.2f}"
    )


    print(
        "Content:",
        f"{candidate_df['content_candidate_count'].mean():.2f}"
    )


    print(
        "UNION:",
        f"{candidate_df['union_candidate_count'].mean():.2f}"
    )


    print(
        "BPR Score 가능 UNION:",
        f"{candidate_df['scoreable_union_count'].mean():.2f}"
    )


    # =====================================================
    # Candidate Recall
    # =====================================================

    print(
        "\n===== Candidate Recall ====="
    )


    print(
        "Item Top-30:",
        f"{candidate_df['item_candidate_recall'].mean():.6f}"
    )


    print(
        "BPR Top-30:",
        f"{candidate_df['bpr_candidate_recall'].mean():.6f}"
    )


    print(
        "Content Top-30:",
        f"{candidate_df['content_candidate_recall'].mean():.6f}"
    )


    print(
        "UNION:",
        f"{candidate_df['union_candidate_recall'].mean():.6f}"
    )


    print(
        "Scoreable UNION:",
        f"{candidate_df['scoreable_union_recall'].mean():.6f}"
    )


    # =====================================================
    # Candidate HitRate
    # =====================================================

    print(
        "\n===== Candidate Hit Rate ====="
    )


    print(
        "Item:",
        f"{candidate_df['item_candidate_hit'].mean():.6f}"
    )


    print(
        "BPR:",
        f"{candidate_df['bpr_candidate_hit'].mean():.6f}"
    )


    print(
        "Content:",
        f"{candidate_df['content_candidate_hit'].mean():.6f}"
    )


    print(
        "UNION:",
        f"{candidate_df['union_candidate_hit'].mean():.6f}"
    )


    # =====================================================
    # Item 대비 새로운 Hit
    # =====================================================

    print(
        "\n===== Item-Based가 놓친 정답 복구 ====="
    )


    print(
        "BPR 추가 Hit:",
        int(
            candidate_df[
                "bpr_added_hits"
            ].sum()
        )
    )


    print(
        "Content 추가 Hit:",
        int(
            candidate_df[
                "content_added_hits"
            ].sum()
        )
    )


    print(
        "UNION 전체 복구 Hit:",
        int(
            candidate_df[
                "recovered_hits_vs_item"
            ].sum()
        )
    )


    # =====================================================
    # Unique Hit
    # =====================================================

    print(
        "\n===== Retriever Unique Hit ====="
    )


    print(
        "Item Only:",
        int(
            candidate_df[
                "item_unique_hits"
            ].sum()
        )
    )


    print(
        "BPR Only:",
        int(
            candidate_df[
                "bpr_unique_hits"
            ].sum()
        )
    )


    print(
        "Content Only:",
        int(
            candidate_df[
                "content_unique_hits"
            ].sum()
        )
    )


    # =====================================================
    # Overlap
    # =====================================================

    print(
        "\n===== Retriever 평균 Overlap ====="
    )


    print(
        "Item ∩ BPR:",
        f"{candidate_df['item_bpr_overlap'].mean():.2f}"
    )


    print(
        "Item ∩ Content:",
        f"{candidate_df['item_content_overlap'].mean():.2f}"
    )


    print(
        "BPR ∩ Content:",
        f"{candidate_df['bpr_content_overlap'].mean():.2f}"
    )


    print(
        "All Three:",
        f"{candidate_df['all_three_overlap'].mean():.2f}"
    )


    # =====================================================
    # Item Cache Statistics
    # =====================================================

    print(
        "\n===== Item-Based Cache ====="
    )


    print(
        "Cache 재사용:",
        item_model.cache_hit_count
    )


    print(
        "새로 계산한 Source Item:",
        item_model.cache_miss_count
    )


    print(
        "최종 Cached Item:",
        len(
            item_model.neighbor_cache
        )
    )


    # =====================================================
    # 12. Candidate Group Summary
    # =====================================================

    candidate_group = (

        candidate_df

        .groupby(
            "review_group",
            observed=True,
        )

        .agg(

            item_recall=(
                "item_candidate_recall",
                "mean",
            ),

            bpr_recall=(
                "bpr_candidate_recall",
                "mean",
            ),

            content_recall=(
                "content_candidate_recall",
                "mean",
            ),

            union_recall=(
                "union_candidate_recall",
                "mean",
            ),

            avg_union_candidates=(
                "union_candidate_count",
                "mean",
            ),

            recovered_hits=(
                "recovered_hits_vs_item",
                "sum",
            ),

            n_users=(
                "user_id",
                "count",
            ),
        )

        .reset_index()
    )


    print(
        "\n===== Review Group별 Candidate 결과 ====="
    )


    print(
        candidate_group.to_string(
            index=False
        )
    )


    # =====================================================
    # 13. Summary
    # =====================================================

    summary = (
        make_summary(
            eval_df=
                eval_df,

            candidate_df=
                candidate_df,
        )
    )


    # =====================================================
    # 14. Case 1 Compare
    # =====================================================

    comparison = (
        compare_with_case1(
            eval_df
        )
    )


    if comparison is not None:

        print(
            "\n===== Case 1 vs Case 2 ====="
        )


        print(
            "공통 사용자:",
            len(
                comparison
            )
        )


        print(
            "Case 1 Hits:",
            int(
                comparison[
                    "case1_hits"
                ].sum()
            )
        )


        print(
            "Case 2 Hits:",
            int(
                comparison[
                    "case2_hits"
                ].sum()
            )
        )


        print(
            "Δ Hits:",
            int(
                comparison[
                    "delta_hits"
                ].sum()
            )
        )


        print(
            "평균 Δ Precision:",
            f"{comparison['delta_precision'].mean():.6f}"
        )


        print(
            "평균 Δ Recall:",
            f"{comparison['delta_recall'].mean():.6f}"
        )


        print(
            "평균 Δ NDCG:",
            f"{comparison['delta_ndcg'].mean():.6f}"
        )


        improved_users = int(
            (
                comparison[
                    "delta_hits"
                ] > 0
            ).sum()
        )


        worsened_users = int(
            (
                comparison[
                    "delta_hits"
                ] < 0
            ).sum()
        )


        same_users = int(
            (
                comparison[
                    "delta_hits"
                ] == 0
            ).sum()
        )


        print(
            "\nHit 변화 사용자"
        )


        print(
            "개선:",
            improved_users
        )


        print(
            "악화:",
            worsened_users
        )


        print(
            "동일:",
            same_users
        )


    # =====================================================
    # 15. Candidate Pool DataFrame
    # =====================================================

    if len(
        recommender.pool_logs
    ) > 0:

        candidate_pool_df = (
            pd.concat(
                recommender.pool_logs,
                ignore_index=True,
            )
        )

    else:

        candidate_pool_df = (
            pd.DataFrame()
        )


    # =====================================================
    # 16. Save
    # =====================================================

    print(
        "\n===== 12. 결과 저장 ====="
    )


    eval_df.to_csv(
        EVAL_PATH,
        index=False,
    )


    pd.DataFrame(
        [
            summary
        ]
    ).to_csv(
        SUMMARY_PATH,
        index=False,
    )


    group_summary.to_csv(
        GROUP_PATH,
        index=False,
    )


    candidate_df.to_csv(
        CANDIDATE_PATH,
        index=False,
    )


    candidate_pool_df.to_csv(
        POOL_PATH,
        index=False,
    )


    if comparison is not None:

        comparison.to_csv(
            COMPARE_PATH,
            index=False,
        )


    print(EVAL_PATH)
    print(SUMMARY_PATH)
    print(GROUP_PATH)
    print(CANDIDATE_PATH)
    print(POOL_PATH)


    if comparison is not None:
        print(COMPARE_PATH)


    # =====================================================
    # Final Summary
    # =====================================================

    print()

    print(
        "=========================================="
    )

    print(
        " Case 2 Complete"
    )

    print(
        "=========================================="
    )


    print(
        "\n===== Candidate Stage ====="
    )


    print(
        "Item Recall:",
        f"{summary['item_candidate_recall']:.6f}"
    )


    print(
        "BPR Recall:",
        f"{summary['bpr_candidate_recall']:.6f}"
    )


    print(
        "Content Recall:",
        f"{summary['content_candidate_recall']:.6f}"
    )


    print(
        "UNION Recall:",
        f"{summary['union_candidate_recall']:.6f}"
    )


    print(
        "Scoreable UNION Recall:",
        f"{summary['scoreable_union_recall']:.6f}"
    )


    print(
        "평균 UNION 후보:",
        f"{summary['avg_union_candidates']:.2f}"
    )


    print(
        "Item 대비 복구 Hits:",
        summary[
            "recovered_hits_vs_item"
        ]
    )


    print(
        "\n===== Final Ranking Stage ====="
    )


    print(
        "Precision@10:",
        f"{summary['precision_at_10']:.6f}"
    )


    print(
        "Recall@10:",
        f"{summary['recall_at_10']:.6f}"
    )


    print(
        "HitRate@10:",
        f"{summary['hit_rate_at_10']:.6f}"
    )


    print(
        "NDCG@10:",
        f"{summary['ndcg_at_10']:.6f}"
    )


    print(
        "Micro Precision:",
        f"{summary['micro_precision_at_10']:.6f}"
    )


    print(
        "Micro Recall:",
        f"{summary['micro_recall_at_10']:.6f}"
    )


    print(
        "Micro F1:",
        f"{summary['micro_f1_at_10']:.6f}"
    )


    print(
        "Hits:",
        summary[
            "hits"
        ]
    )


    print(
        "\n===== Mapping Diagnostic ====="
    )


    print(
        "BPR mapping 없는 사용자:",
        recommender.missing_users
    )


    print(
        "BPR mapping 없는 Candidate:",
        recommender.missing_candidate_items
    )


    print(
        "\n===== Item Cache ====="
    )


    print(
        "Cache Hits:",
        item_model.cache_hit_count
    )


    print(
        "Cache Miss:",
        item_model.cache_miss_count
    )


    print()

    print(
        "=========================================="
    )

    print(
        " Experiment B Complete"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()