# hybrid_arctech_experiment/exp_k2_xgb_4retriever_ratio_size_sweep.py
# ============================================================
# Experiment K2 - 4-Retriever Ratio x Candidate Size Joint Sweep
#
# Candidate Retriever
# ------------------------------------------------------------
# Previous Fusion weight -> candidate quota
#
# Item-CF   0.50 -> Top 50
# BPR       0.20 -> Top 20
# Content   0.15 -> Top 15
# User-CF   0.15 -> Top 15
#
# Candidate Union <= 100
#
# Ranker
# ------------------------------------------------------------
# XGBoost Ranker
# Final Full14 Feature Set
#
# IMPORTANT
# ------------------------------------------------------------
# 기존 Full14의 feature "개수"는 유지하지만,
# 새 4-Retriever 구조에 맞춰 retriever_count 의미를 갱신한다.
#
# retriever_count
#   = Item + BPR + Content + User 중
#     해당 후보를 올린 Retriever 수
#
# is_item_candidate를 별도 15번째 XGB feature로 추가하지 않는다.
# Item 신호는 이미 item_score_norm / item_rank로 강하게 들어가며,
# 이번 실험의 목적은 "확정한 Full14 구조를 유지한 채"
# candidate distribution 변화에 맞춰 XGB만 재학습하는 것이다.
#
# Workflow
# ------------------------------------------------------------
# 1. 기존 BPR / Content / User-CF / Item-CF 로드
# 2. LTR 1000 + Final 400 사용자에 대해
#    새 4-Retriever candidate + Full14 feature cache 생성
# 3. LTR 1000 -> Train 800 / Valid 200
# 4. 고정 Best-A XGB params + Early Stopping
# 5. Validation best_iteration 확정
# 6. LTR 1000 전체로 production XGB 재학습
# 7. Final 400 end-to-end 평가
# 8. 최종 모델 / cache / 결과 저장
#
# 기존 추천 모델(BPR/CF/Content)은 재학습하지 않는다.
# ============================================================

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from xgboost import XGBRanker


# ============================================================
# Project Root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# main.py에 구현된 runtime scorer를 그대로 재사용한다.
import main as runtime_main


# ============================================================
# Config
# ============================================================

RANDOM_STATE = 42

TOP_N = 10

# ------------------------------------------------------------
# Ratio x Candidate Size Joint Sweep
#
# 기준 quota=100에서 크게 줄이지 않는다.
#   90  : 아주 소폭 감소
#   100 : 현재 baseline
#   110 : 소폭 증가
#   125 : 중간 증가
#   150 : 큰 증가
#
# 각 BASE_RATIO는 100을 기준으로 한 비율이며,
# candidate size에 맞춰 largest-remainder 방식으로
# 정수 quota를 자동 배분한다.
#
# 예:
#   50/20/15/15 × size 150
#   -> Item75 / BPR30 / Content23 / User22 = 150
#
# Final 400은 절대 사용하지 않는다.
# ------------------------------------------------------------

CANDIDATE_SIZES = [
    90,
    100,
    110,
    125,
    150,
]


BASE_RATIOS = [
    {
        "name": "baseline_50_20_15_15",
        "item": 50,
        "bpr": 20,
        "content": 15,
        "user": 15,
    },
    {
        "name": "item55_user10",
        "item": 55,
        "bpr": 20,
        "content": 15,
        "user": 10,
    },
    {
        "name": "item60_content10_user10",
        "item": 60,
        "bpr": 20,
        "content": 10,
        "user": 10,
    },
    {
        "name": "bpr25_user10",
        "item": 50,
        "bpr": 25,
        "content": 15,
        "user": 10,
    },
    {
        "name": "bpr30_item45",
        "item": 45,
        "bpr": 30,
        "content": 15,
        "user": 10,
    },
    {
        "name": "content20_user10",
        "item": 50,
        "bpr": 20,
        "content": 20,
        "user": 10,
    },
    {
        "name": "balanced_45_25_20_10",
        "item": 45,
        "bpr": 25,
        "content": 20,
        "user": 10,
    },
    {
        "name": "item55_bpr25",
        "item": 55,
        "bpr": 25,
        "content": 10,
        "user": 10,
    },
    {
        "name": "item45_bpr25_user15",
        "item": 45,
        "bpr": 25,
        "content": 15,
        "user": 15,
    },
    {
        "name": "diverse_40_30_20_10",
        "item": 40,
        "bpr": 30,
        "content": 20,
        "user": 10,
    },
    {
        "name": "user20",
        "item": 45,
        "bpr": 20,
        "content": 15,
        "user": 20,
    },
    {
        "name": "content25",
        "item": 45,
        "bpr": 20,
        "content": 25,
        "user": 10,
    },
]


SOURCE_KEYS = [
    "item",
    "bpr",
    "content",
    "user",
]


def scale_ratio_to_size(
    base_ratio,
    candidate_size,
):

    """
    합계 100인 base ratio를 candidate_size에 맞춰
    largest remainder 방식으로 정수 quota로 변환한다.
    """

    raw = {
        key:
            (
                candidate_size
                *
                base_ratio[
                    key
                ]
                /
                100.0
            )

        for key
        in SOURCE_KEYS
    }


    quotas = {
        key:
            int(
                np.floor(
                    raw[
                        key
                    ]
                )
            )

        for key
        in SOURCE_KEYS
    }


    remaining = (
        int(
            candidate_size
        )
        -
        sum(
            quotas.values()
        )
    )


    remainder_order = sorted(
        SOURCE_KEYS,
        key=lambda key: (
            -(
                raw[
                    key
                ]
                -
                quotas[
                    key
                ]
            ),
            SOURCE_KEYS.index(
                key
            ),
        ),
    )


    for key in remainder_order[
        :remaining
    ]:

        quotas[
            key
        ] += 1


    return quotas


EXPERIMENT_CONFIGS = []


for candidate_size in CANDIDATE_SIZES:

    for base_ratio in BASE_RATIOS:

        quotas = scale_ratio_to_size(
            base_ratio,
            candidate_size,
        )


        EXPERIMENT_CONFIGS.append(
            {
                "name":
                    (
                        f"{base_ratio['name']}"
                        f"_size{candidate_size}"
                    ),

                "base_ratio_name":
                    base_ratio[
                        "name"
                    ],

                "candidate_size":
                    int(
                        candidate_size
                    ),

                "item":
                    int(
                        quotas[
                            "item"
                        ]
                    ),

                "bpr":
                    int(
                        quotas[
                            "bpr"
                        ]
                    ),

                "content":
                    int(
                        quotas[
                            "content"
                        ]
                    ),

                "user":
                    int(
                        quotas[
                            "user"
                        ]
                    ),
            }
        )


# Superset는 60개 config에서 source별 최대 quota를 사용한다.
ITEM_N = max(
    config[
        "item"
    ]
    for config
    in EXPERIMENT_CONFIGS
)

BPR_N = max(
    config[
        "bpr"
    ]
    for config
    in EXPERIMENT_CONFIGS
)

CONTENT_N = max(
    config[
        "content"
    ]
    for config
    in EXPERIMENT_CONFIGS
)

USER_N = max(
    config[
        "user"
    ]
    for config
    in EXPERIMENT_CONFIGS
)


CANDIDATE_SIZE = (
    ITEM_N
    +
    BPR_N
    +
    CONTENT_N
    +
    USER_N
)


CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 30

CHECKPOINT_EVERY = 50

# Batch optimization
USER_CF_BATCH_SIZE = 8
ITEM_CF_PRECOMPUTE_BATCH_SIZE = 64
PROGRESS_EVERY_BATCH = True


# ============================================================
# Final Full14 Features
# ============================================================

FULL14_FEATURES = [
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
]


# ============================================================
# Paths
# ============================================================

SAVED_MODEL_DIR = (
    ROOT
    / "models"
    / "saved_model"
)

RESULT_DIR = (
    SAVED_MODEL_DIR
    / "results"
    / "xgb_4retriever_ratio_size_sweep"
)

CACHE_DIR = (
    SAVED_MODEL_DIR
    / "ltr_cache"
    / "xgb_4retriever_ratio_size_sweep"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# 기존 프로젝트에서 사용 중인 고정 user files
LTR_USERS_PATH = (
    SAVED_MODEL_DIR
    / "ltr_cache"
    / "xgb_exp1"
    / "ltr_train_users_1000.csv"
)

FINAL_USERS_PATH = (
    SAVED_MODEL_DIR
    / "results"
    / "hybrid_sampled_users.csv"
)


FEATURE_CACHE_PATH = (
    CACHE_DIR
    / "ratio_size_sweep_superset_features.parquet"
)

CHECKPOINT_PATH = (
    CACHE_DIR
    / "ratio_size_sweep_superset_checkpoint.parquet"
)

PROCESSED_USERS_PATH = (
    CACHE_DIR
    / "ratio_size_sweep_processed_users.npy"
)


VALID_MODEL_PATH = (
    SAVED_MODEL_DIR
    / "xgb_ranker_4retriever_full14_validation.json"
)

FINAL_MODEL_PATH = (
    SAVED_MODEL_DIR
    / "xgb_ranker_final_4retriever_full14.json"
)


FEATURE_IMPORTANCE_PATH = (
    RESULT_DIR
    / "feature_importance.csv"
)

VALIDATION_SUMMARY_PATH = (
    RESULT_DIR
    / "validation_summary.csv"
)

FINAL400_USER_RESULT_PATH = (
    RESULT_DIR
    / "final400_eval.csv"
)

FINAL400_SUMMARY_PATH = (
    RESULT_DIR
    / "final400_summary.csv"
)

MODEL_METADATA_PATH = (
    RESULT_DIR
    / "model_metadata.json"
)


# ============================================================
# Fixed XGB Best-A Params
# ============================================================

XGB_BASE_PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",

    "learning_rate": 0.05,

    "max_depth": 4,
    "min_child_weight": 1,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    "reg_lambda": 1.0,
    "reg_alpha": 0.0,

    "random_state": RANDOM_STATE,
    "tree_method": "hist",
}


# ============================================================
# Generic Helpers
# ============================================================

def validate_candidate_quota():

    for base_ratio in BASE_RATIOS:

        ratio_total = sum(
            base_ratio[
                key
            ]
            for key
            in SOURCE_KEYS
        )


        if ratio_total != 100:

            raise ValueError(
                "Base ratio 합이 100이 아닙니다: "
                f"{base_ratio['name']} -> {ratio_total}"
            )


    for config in EXPERIMENT_CONFIGS:

        quota_total = (
            config[
                "item"
            ]
            +
            config[
                "bpr"
            ]
            +
            config[
                "content"
            ]
            +
            config[
                "user"
            ]
        )


        if quota_total != config[
            "candidate_size"
        ]:

            raise ValueError(
                "Scaled quota 합이 candidate_size와 다릅니다: "
                f"{config['name']} -> "
                f"{quota_total} != "
                f"{config['candidate_size']}"
            )


    print(
        "\n===== Joint Sweep Config Validation ====="
    )

    print(
        "Base ratios       :",
        len(
            BASE_RATIOS
        ),
    )

    print(
        "Candidate sizes   :",
        CANDIDATE_SIZES,
    )

    print(
        "Total configs     :",
        len(
            EXPERIMENT_CONFIGS
        ),
    )

    print(
        "Superset max      :",
        (
            ITEM_N,
            BPR_N,
            CONTENT_N,
            USER_N,
        ),
        "sum=",
        CANDIDATE_SIZE,
    )


def load_user_ids(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"User file이 없습니다: {path}"
        )


    df = pd.read_csv(
        path
    )


    if "user_id" not in df.columns:

        raise ValueError(
            f"user_id column이 없습니다: {path}"
        )


    return (
        df[
            "user_id"
        ]
        .drop_duplicates()
        .to_numpy()
    )


def top_positive_ids(
    scores,
    item_ids,
    n,
):

    """
    Item / Content / User-CF retrieve와 동일하게
    finite & score > 0 인 item 중 Top-N.
    """

    scores = np.asarray(
        scores
    )

    item_ids = np.asarray(
        item_ids
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
        len(
            valid
        ),
    )


    if n_actual < len(
        valid
    ):

        local = np.argpartition(
            -scores[
                valid
            ],
            n_actual - 1,
        )[
            :n_actual
        ]

        selected = valid[
            local
        ]

    else:

        selected = valid


    selected = selected[
        np.argsort(
            -scores[
                selected
            ],
            kind="stable",
        )
    ]


    return (
        item_ids[
            selected
        ]
        .tolist()
    )


def top_bpr_ids(
    scores,
    item_ids,
    n,
):

    """
    main.py NewUserBPRScorer.retrieve와 동일:
    finite score 중 Top-N.
    """

    idx = runtime_main.top_indices(
        scores,
        n,
    )


    return (
        np.asarray(
            item_ids
        )[
            idx
        ]
        .tolist()
    )


def gather_scores(
    candidate_ids,
    all_scores,
    item_to_idx,
):

    """
    scorer 전체 score vector에서 candidate score만 추출.
    해당 scorer universe에 없는 item은 0.
    """

    result = np.zeros(
        len(
            candidate_ids
        ),
        dtype=np.float32,
    )


    for position, app_id in enumerate(
        candidate_ids
    ):

        idx = item_to_idx.get(
            app_id
        )

        if idx is None:

            continue


        value = all_scores[
            idx
        ]


        if np.isfinite(
            value
        ):

            result[
                position
            ] = value


    return result


# ============================================================
# Feature Generation
# ============================================================

def build_user_feature_rows(
    *,
    engine,
    user_id,
    played_app_ids,
    relevant_items,
    precomputed_user_all_scores=None,
):

    """
    한 사용자에 대해 새 4-Retriever 후보를 만들고
    Full14 XGB feature를 생성한다.

    Runtime과 최대한 동일하게 계산하되,
    expensive scorer는 사용자당 한 번씩만 계산한다.
    """

    played_app_ids = (
        runtime_main
        .unique_preserve_order(
            played_app_ids
        )
    )


    if not played_app_ids:

        return pd.DataFrame()


    # --------------------------------------------------------
    # 1. New-user BPR representation
    # --------------------------------------------------------

    (
        new_user_vector,
        bpr_played_idx,
    ) = (
        engine
        .bpr
        .build_user_vector(
            played_app_ids
        )
    )


    # --------------------------------------------------------
    # 2. 각 Retriever 전체 score 계산
    # --------------------------------------------------------

    item_all_scores = (
        engine
        .item_cf
        .calculate_scores(
            played_app_ids
        )
    )


    bpr_all_scores = (
        engine
        .bpr
        .score_all(
            new_user_vector,
            bpr_played_idx,
        )
    )


    content_all_scores = (
        engine
        .content
        .score_all(
            played_app_ids
        )
    )


    if precomputed_user_all_scores is None:

        user_all_scores = (
            engine
            .user_cf
            .calculate_scores(
                played_app_ids
            )
        )

    else:

        user_all_scores = np.asarray(
            precomputed_user_all_scores,
            dtype=np.float32,
        )


    # --------------------------------------------------------
    # 3. Candidate Retrieval
    #    previous Fusion weight -> quota
    # --------------------------------------------------------

    item_candidates = top_positive_ids(
        scores=
            item_all_scores,

        item_ids=
            engine.item_cf.item_ids,

        n=
            ITEM_N,
    )


    bpr_candidates = top_bpr_ids(
        scores=
            bpr_all_scores,

        item_ids=
            engine.bpr.item_ids,

        n=
            BPR_N,
    )


    content_candidates = top_positive_ids(
        scores=
            content_all_scores,

        item_ids=
            engine.content.app_ids,

        n=
            CONTENT_N,
    )


    user_candidates = top_positive_ids(
        scores=
            user_all_scores,

        item_ids=
            engine.user_cf.item_ids,

        n=
            USER_N,
    )


    source_info = {
        "item":
            set(
                item_candidates
            ),

        "bpr":
            set(
                bpr_candidates
            ),

        "content":
            set(
                content_candidates
            ),

        "user":
            set(
                user_candidates
            ),
    }


    candidates = (
        runtime_main
        .unique_preserve_order(
            item_candidates
            +
            bpr_candidates
            +
            content_candidates
            +
            user_candidates
        )
    )


    played_set = set(
        played_app_ids
    )


    candidates = [
        app_id

        for app_id
        in candidates

        if app_id
        not in played_set
    ]


    candidates = candidates[
        :CANDIDATE_SIZE
    ]


    if not candidates:

        return pd.DataFrame()


    # --------------------------------------------------------
    # 4. Candidate score extraction
    # --------------------------------------------------------

    item_score = gather_scores(
        candidate_ids=
            candidates,

        all_scores=
            item_all_scores,

        item_to_idx=
            engine.item_cf.item_to_idx,
    )


    bpr_item_to_idx = {
        app_id: idx

        for idx, app_id
        in enumerate(
            engine.bpr.item_ids
        )
    }


    bpr_score = gather_scores(
        candidate_ids=
            candidates,

        all_scores=
            bpr_all_scores,

        item_to_idx=
            bpr_item_to_idx,
    )


    content_score = gather_scores(
        candidate_ids=
            candidates,

        all_scores=
            content_all_scores,

        item_to_idx=
            engine.content.app_to_idx,
    )


    user_score = gather_scores(
        candidate_ids=
            candidates,

        all_scores=
            user_all_scores,

        item_to_idx=
            engine.user_cf.item_to_idx,
    )


    # --------------------------------------------------------
    # 5. Base data
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        {
            "user_id":
                user_id,

            "app_id":
                candidates,

            "item_score":
                item_score,

            "bpr_score":
                bpr_score,

            "content_score":
                content_score,

            "user_score":
                user_score,
        }
    )


    # --------------------------------------------------------
    # 5.5 Retrieval rank
    #
    # Ratio별 candidate pool을 superset에서 정확히 재구성하기 위해
    # 각 Retriever가 해당 item을 몇 번째로 뽑았는지 저장한다.
    # 0 = 해당 Retriever의 superset 후보가 아님.
    # --------------------------------------------------------

    item_retrieval_rank_map = {
        app_id: rank

        for rank, app_id
        in enumerate(
            item_candidates,
            start=1,
        )
    }


    bpr_retrieval_rank_map = {
        app_id: rank

        for rank, app_id
        in enumerate(
            bpr_candidates,
            start=1,
        )
    }


    content_retrieval_rank_map = {
        app_id: rank

        for rank, app_id
        in enumerate(
            content_candidates,
            start=1,
        )
    }


    user_retrieval_rank_map = {
        app_id: rank

        for rank, app_id
        in enumerate(
            user_candidates,
            start=1,
        )
    }


    feature_df[
        "item_retrieval_rank"
    ] = np.asarray(
        [
            item_retrieval_rank_map.get(
                app_id,
                0,
            )

            for app_id
            in candidates
        ],
        dtype=np.int16,
    )


    feature_df[
        "bpr_retrieval_rank"
    ] = np.asarray(
        [
            bpr_retrieval_rank_map.get(
                app_id,
                0,
            )

            for app_id
            in candidates
        ],
        dtype=np.int16,
    )


    feature_df[
        "content_retrieval_rank"
    ] = np.asarray(
        [
            content_retrieval_rank_map.get(
                app_id,
                0,
            )

            for app_id
            in candidates
        ],
        dtype=np.int16,
    )


    feature_df[
        "user_retrieval_rank"
    ] = np.asarray(
        [
            user_retrieval_rank_map.get(
                app_id,
                0,
            )

            for app_id
            in candidates
        ],
        dtype=np.int16,
    )


    # --------------------------------------------------------
    # 6. Base 8 Features
    # --------------------------------------------------------

    feature_df[
        "item_score_norm"
    ] = (
        runtime_main
        .minmax_normalize(
            item_score
        )
    )


    feature_df[
        "bpr_score_norm"
    ] = (
        runtime_main
        .minmax_normalize(
            bpr_score
        )
    )


    feature_df[
        "content_score_norm"
    ] = (
        runtime_main
        .minmax_normalize(
            content_score
        )
    )


    feature_df[
        "user_score_norm"
    ] = (
        runtime_main
        .minmax_normalize(
            user_score
        )
    )


    feature_df[
        "item_rank"
    ] = (
        runtime_main
        .assign_rank(
            item_score
        )
    )


    feature_df[
        "bpr_rank"
    ] = (
        runtime_main
        .assign_rank(
            bpr_score
        )
    )


    feature_df[
        "content_rank"
    ] = (
        runtime_main
        .assign_rank(
            content_score
        )
    )


    feature_df[
        "user_rank"
    ] = (
        runtime_main
        .assign_rank(
            user_score
        )
    )


    # --------------------------------------------------------
    # 7. Retriever source signals
    # --------------------------------------------------------

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
            int(
                app_id
                in
                item_set
            )

            for app_id
            in candidates
        ],
        dtype=np.int8,
    )


    is_bpr_candidate = np.asarray(
        [
            int(
                app_id
                in
                bpr_set
            )

            for app_id
            in candidates
        ],
        dtype=np.int8,
    )


    is_content_candidate = np.asarray(
        [
            int(
                app_id
                in
                content_set
            )

            for app_id
            in candidates
        ],
        dtype=np.int8,
    )


    is_user_candidate = np.asarray(
        [
            int(
                app_id
                in
                user_set
            )

            for app_id
            in candidates
        ],
        dtype=np.int8,
    )


    # diagnostic:
    # 최종 Full14 XGB에는 직접 넣지 않음
    feature_df[
        "is_item_candidate"
    ] = (
        is_item_candidate
    )


    feature_df[
        "is_bpr_candidate"
    ] = (
        is_bpr_candidate
    )


    feature_df[
        "is_content_candidate"
    ] = (
        is_content_candidate
    )


    feature_df[
        "is_user_candidate"
    ] = (
        is_user_candidate
    )


    # NEW SEMANTICS:
    # 4개 Retriever 모두 포함
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


    # --------------------------------------------------------
    # 8. Context
    # --------------------------------------------------------

    feature_df[
        "item_popularity"
    ] = np.asarray(
        [
            engine
            .item_popularity
            .get(
                app_id,
                0,
            )

            for app_id
            in candidates
        ],
        dtype=np.float32,
    )


    feature_df[
        "user_interaction_count"
    ] = np.full(
        len(
            candidates
        ),
        len(
            played_app_ids
        ),
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # 9. Label
    # --------------------------------------------------------

    relevant_set = set(
        relevant_items
    )


    feature_df[
        "label"
    ] = np.asarray(
        [
            int(
                app_id
                in
                relevant_set
            )

            for app_id
            in candidates
        ],
        dtype=np.int8,
    )


    return feature_df



# ============================================================
# Batch Optimization Helpers
# ============================================================

class BatchUserCFScorer:
    """
    User-CF를 사용자 1명씩 Python/NumPy로 처리하지 않고
    여러 query user를 sparse matrix multiplication으로 묶어서 계산한다.

    기존 main.UserCFRuntimeScorer의 의미는 유지한다:

        cosine(new_user, existing_user)
        -> similarity Top-K users
        -> similarity-weighted item score
        -> abs(similarity) 합으로 normalize

    차이는 계산 방식뿐이다.

    기존:
        user 1명
        -> query @ 13,781,059 users.T
        -> Python 처리
        -> 반복

    변경:
        query users B명
        -> Query[B, items] @ UserMatrix.T
        -> SciPy sparse batch multiplication
        -> Top-K neighbor sparse weight matrix
        -> WeightMatrix @ UserMatrix
    """

    def __init__(
        self,
        base_user_cf,
        batch_size=USER_CF_BATCH_SIZE,
    ):

        self.matrix = (
            base_user_cf
            .matrix
            .tocsr()
            .astype(
                np.float32,
                copy=False,
            )
        )

        # CSR transpose는 CSC 형태.
        # sparse batch matmul에 재사용한다.
        self.matrix_t = (
            self.matrix
            .T
            .tocsc(
                copy=False
            )
        )

        self.item_ids = np.asarray(
            base_user_cf.item_ids
        )

        self.item_to_idx = dict(
            base_user_cf.item_to_idx
        )

        self.user_norms = np.asarray(
            base_user_cf.user_norms,
            dtype=np.float32,
        )

        self.k = int(
            base_user_cf.k
        )

        self.batch_size = int(
            batch_size
        )

        print(
            "\n===== Batch User-CF Ready ====="
        )

        print(
            "Matrix:",
            self.matrix.shape,
        )

        print(
            "NNZ   :",
            f"{self.matrix.nnz:,}",
        )

        print(
            "Batch :",
            self.batch_size,
        )


    def _build_query_matrix(
        self,
        played_lists,
    ):

        row_indices = []
        col_indices = []

        source_indices_by_row = []


        for row_idx, played_app_ids in enumerate(
            played_lists
        ):

            played_app_ids = (
                runtime_main
                .unique_preserve_order(
                    played_app_ids
                )
            )


            source_idx = [
                self.item_to_idx[
                    app_id
                ]

                for app_id
                in played_app_ids

                if app_id
                in self.item_to_idx
            ]


            # 혹시 mapping 차이로 duplicate index가 생기더라도 제거.
            source_idx = list(
                dict.fromkeys(
                    source_idx
                )
            )


            source_indices_by_row.append(
                np.asarray(
                    source_idx,
                    dtype=np.int64,
                )
            )


            row_indices.extend(
                [
                    row_idx
                ]
                *
                len(
                    source_idx
                )
            )

            col_indices.extend(
                source_idx
            )


        if row_indices:

            data = np.ones(
                len(
                    row_indices
                ),
                dtype=np.float32,
            )

            query_matrix = sp.csr_matrix(
                (
                    data,
                    (
                        np.asarray(
                            row_indices,
                            dtype=np.int32,
                        ),
                        np.asarray(
                            col_indices,
                            dtype=np.int32,
                        ),
                    ),
                ),
                shape=(
                    len(
                        played_lists
                    ),
                    self.matrix.shape[1],
                ),
                dtype=np.float32,
            )

        else:

            query_matrix = sp.csr_matrix(
                (
                    len(
                        played_lists
                    ),
                    self.matrix.shape[1],
                ),
                dtype=np.float32,
            )


        return (
            query_matrix,
            source_indices_by_row,
        )


    def calculate_scores_batch(
        self,
        played_lists,
    ):

        """
        Returns:
            list[np.ndarray]
            각 원소 shape=(n_items,), float32
        """

        n_batch = len(
            played_lists
        )


        if n_batch == 0:

            return []


        (
            query_matrix,
            source_indices_by_row,
        ) = self._build_query_matrix(
            played_lists
        )


        # ----------------------------------------------------
        # 1. Batch cosine dot products
        # ----------------------------------------------------

        sims = (
            query_matrix
            @
            self.matrix_t
        ).tocsr()


        # cancellation으로 explicit zero가 남을 수 있으므로 제거.
        sims.eliminate_zeros()


        # ----------------------------------------------------
        # 2. 각 query에서 Top-K neighbor 추출
        #    -> sparse Weight matrix 구성
        # ----------------------------------------------------

        weight_rows = []
        weight_cols = []
        weight_data = []

        sim_sums = np.zeros(
            n_batch,
            dtype=np.float32,
        )


        for row_idx in range(
            n_batch
        ):

            source_idx = (
                source_indices_by_row[
                    row_idx
                ]
            )


            if len(
                source_idx
            ) == 0:

                continue


            start = int(
                sims.indptr[
                    row_idx
                ]
            )

            end = int(
                sims.indptr[
                    row_idx
                    +
                    1
                ]
            )


            if end <= start:

                continue


            neighbor_users = (
                sims.indices[
                    start:end
                ]
                .astype(
                    np.int64,
                    copy=False,
                )
            )


            dot_values = (
                sims.data[
                    start:end
                ]
                .astype(
                    np.float32,
                    copy=False,
                )
            )


            query_norm = np.sqrt(
                float(
                    len(
                        source_idx
                    )
                )
            )


            denominator = (
                query_norm
                *
                self.user_norms[
                    neighbor_users
                ]
            )


            valid = (
                denominator
                >
                0
            )


            if not valid.any():

                continue


            neighbor_users = (
                neighbor_users[
                    valid
                ]
            )

            similarities = (
                dot_values[
                    valid
                ]
                /
                denominator[
                    valid
                ]
            ).astype(
                np.float32,
                copy=False,
            )


            nonzero = (
                similarities
                !=
                0
            )


            if not nonzero.any():

                continue


            neighbor_users = (
                neighbor_users[
                    nonzero
                ]
            )

            similarities = (
                similarities[
                    nonzero
                ]
            )


            k = min(
                self.k,
                len(
                    similarities
                ),
            )


            if k <= 0:

                continue


            if k < len(
                similarities
            ):

                top_pos = np.argpartition(
                    similarities,
                    -k,
                )[
                    -k:
                ]

            else:

                top_pos = np.arange(
                    len(
                        similarities
                    )
                )


            top_users = (
                neighbor_users[
                    top_pos
                ]
            )

            top_sims = (
                similarities[
                    top_pos
                ]
                .astype(
                    np.float32,
                    copy=False,
                )
            )


            sim_sum = float(
                np.abs(
                    top_sims
                )
                .sum()
            )


            if sim_sum <= 0:

                continue


            sim_sums[
                row_idx
            ] = sim_sum


            weight_rows.extend(
                [
                    row_idx
                ]
                *
                len(
                    top_users
                )
            )

            weight_cols.extend(
                top_users.tolist()
            )

            weight_data.extend(
                top_sims.tolist()
            )


        # ----------------------------------------------------
        # 3. Top-K users -> item weighted sum을 한 번에 계산
        # ----------------------------------------------------

        if weight_data:

            weight_matrix = sp.csr_matrix(
                (
                    np.asarray(
                        weight_data,
                        dtype=np.float32,
                    ),
                    (
                        np.asarray(
                            weight_rows,
                            dtype=np.int32,
                        ),
                        np.asarray(
                            weight_cols,
                            dtype=np.int64,
                        ),
                    ),
                ),
                shape=(
                    n_batch,
                    self.matrix.shape[0],
                ),
                dtype=np.float32,
            )


            weighted_items = (
                weight_matrix
                @
                self.matrix
            ).tocsr()


            score_matrix = (
                weighted_items
                .toarray()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

        else:

            score_matrix = np.zeros(
                (
                    n_batch,
                    self.matrix.shape[1],
                ),
                dtype=np.float32,
            )


        # ----------------------------------------------------
        # 4. normalize + seen items mask
        # ----------------------------------------------------

        for row_idx in range(
            n_batch
        ):

            if sim_sums[
                row_idx
            ] > 0:

                score_matrix[
                    row_idx
                ] /= (
                    sim_sums[
                        row_idx
                    ]
                )


            source_idx = (
                source_indices_by_row[
                    row_idx
                ]
            )


            if len(
                source_idx
            ):

                score_matrix[
                    row_idx,
                    source_idx,
                ] = -np.inf


        return [
            score_matrix[
                row_idx
            ]

            for row_idx
            in range(
                n_batch
            )
        ]


def precompute_item_cf_neighbors(
    *,
    item_cf,
    pending_user_ids,
    train_history,
    batch_size=ITEM_CF_PRECOMPUTE_BATCH_SIZE,
):

    """
    남은 사용자들이 실제로 플레이한 source item만 모아
    Item-CF Top-K neighbor를 batch로 미리 계산한다.

    이후 item_cf.calculate_scores()는 neighbor_cache만 읽으므로
    사용자별 item-to-all similarity 계산을 반복하지 않는다.
    """

    print(
        "\n===== Item-CF Neighbor Batch Precompute ====="
    )


    required_indices = set()


    for user_id in pending_user_ids:

        played_app_ids = (
            runtime_main
            .unique_preserve_order(
                train_history.get(
                    user_id,
                    [],
                )
            )
        )


        for app_id in played_app_ids:

            item_idx = (
                item_cf
                .item_to_idx
                .get(
                    app_id
                )
            )


            if item_idx is not None:

                required_indices.add(
                    int(
                        item_idx
                    )
                )


    # 이미 cache에 있는 것은 건너뜀.
    missing_indices = [
        item_idx

        for item_idx
        in sorted(
            required_indices
        )

        if item_idx
        not in item_cf.neighbor_cache
    ]


    print(
        "Required unique source items:",
        len(
            required_indices
        ),
    )

    print(
        "Need to compute:",
        len(
            missing_indices
        ),
    )


    if not missing_indices:

        print(
            "Item-CF neighbor cache already complete."
        )

        return


    matrix = (
        item_cf
        .item_matrix
        .tocsr()
    )

    matrix_t = (
        matrix
        .T
        .tocsc(
            copy=False
        )
    )


    total = len(
        missing_indices
    )

    start_time = time.perf_counter()


    for start_pos in range(
        0,
        total,
        batch_size,
    ):

        chunk = np.asarray(
            missing_indices[
                start_pos:
                start_pos
                +
                batch_size
            ],
            dtype=np.int64,
        )


        similarities = (
            matrix[
                chunk
            ]
            @
            matrix_t
        ).tocsr()


        for local_row, source_item_idx in enumerate(
            chunk
        ):

            row_start = int(
                similarities.indptr[
                    local_row
                ]
            )

            row_end = int(
                similarities.indptr[
                    local_row
                    +
                    1
                ]
            )


            if row_end <= row_start:

                item_cf.neighbor_cache[
                    int(
                        source_item_idx
                    )
                ] = (
                    np.array(
                        [],
                        dtype=np.int64,
                    ),
                    np.array(
                        [],
                        dtype=np.float32,
                    ),
                )

                continue


            indices = (
                similarities.indices[
                    row_start:row_end
                ]
                .astype(
                    np.int64,
                    copy=False,
                )
            )

            values = (
                similarities.data[
                    row_start:row_end
                ]
                .astype(
                    np.float32,
                    copy=False,
                )
            )


            # 기존 get_neighbors()와 동일:
            # self similarity 제거 + positive similarity만 사용.
            valid = (
                (
                    indices
                    !=
                    int(
                        source_item_idx
                    )
                )
                &
                (
                    values
                    >
                    0
                )
            )


            indices = indices[
                valid
            ]

            values = values[
                valid
            ]


            if len(
                indices
            ) == 0:

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

            else:

                k = min(
                    item_cf.k,
                    len(
                        values
                    ),
                )


                if k < len(
                    values
                ):

                    top_pos = np.argpartition(
                        values,
                        -k,
                    )[
                        -k:
                    ]

                else:

                    top_pos = np.arange(
                        len(
                            values
                        )
                    )


                result = (
                    indices[
                        top_pos
                    ]
                    .astype(
                        np.int64,
                        copy=False,
                    ),
                    values[
                        top_pos
                    ]
                    .astype(
                        np.float32,
                        copy=False,
                    ),
                )


            item_cf.neighbor_cache[
                int(
                    source_item_idx
                )
            ] = result


        done = min(
            start_pos
            +
            len(
                chunk
            ),
            total,
        )


        elapsed = (
            time.perf_counter()
            -
            start_time
        )


        rate = (
            elapsed
            /
            done
        )


        eta = (
            total
            -
            done
        ) * rate


        print(
            f"Item neighbor cache: "
            f"{done}/{total} "
            f"| {rate:.3f} sec/item "
            f"| ETA={eta / 60:.1f} min"
        )


        del similarities

        gc.collect()


    print(
        "Item-CF batch precompute complete."
    )


def save_feature_checkpoint(
    *,
    accumulated_parts,
    pending_rows,
    processed,
):

    parts = list(
        accumulated_parts
    )


    if pending_rows:

        parts.append(
            pd.concat(
                pending_rows,
                ignore_index=True,
            )
        )


    if not parts:

        return (
            accumulated_parts,
            [],
            0,
        )


    checkpoint_df = pd.concat(
        parts,
        ignore_index=True,
    )


    checkpoint_df.to_parquet(
        CHECKPOINT_PATH,
        index=False,
    )


    np.save(
        PROCESSED_USERS_PATH,
        np.asarray(
            list(
                processed
            ),
            dtype=object,
        ),
    )


    # 메모리에는 합쳐진 하나만 유지.
    return (
        [
            checkpoint_df
        ],
        [],
        len(
            checkpoint_df
        ),
    )



# ============================================================
# Feature Cache Builder
# ============================================================

def build_or_load_feature_cache(
    *,
    engine,
    train_df,
    test_df,
    target_user_ids,
):

    """
    4-Retriever 전용 feature cache.

    기존 checkpoint를 그대로 resume한다.

    최적화:
    - 이미 처리된 850명 등은 그대로 유지.
    - 남은 user의 source item만 Item-CF neighbor batch precompute.
    - User-CF는 USER_CF_BATCH_SIZE명씩 sparse batch 계산.
    - BPR / Content는 가벼우므로 사용자별 계산 유지.
    """

    if FEATURE_CACHE_PATH.exists():

        print(
            "\n===== Ratio x Size Superset Cache Load ====="
        )

        print(
            FEATURE_CACHE_PATH
        )


        feature_df = pd.read_parquet(
            FEATURE_CACHE_PATH
        )


        print(
            "Feature DF:",
            feature_df.shape,
        )


        return feature_df


    print(
        "\n===== Ratio x Size Superset Cache Build (BATCH) ====="
    )


    target_set = set(
        target_user_ids
    )


    # --------------------------------------------------------
    # 필요한 사용자 행만 축소
    # --------------------------------------------------------

    train_subset = (
        train_df[
            train_df[
                "user_id"
            ]
            .isin(
                target_set
            )
        ][
            [
                "user_id",
                "app_id",
            ]
        ]
        .copy()
    )


    positive_test_subset = (
        test_df[
            (
                test_df[
                    "user_id"
                ]
                .isin(
                    target_set
                )
            )
            &
            (
                test_df[
                    "is_recommended"
                ]
                ==
                True
            )
        ][
            [
                "user_id",
                "app_id",
            ]
        ]
        .copy()
    )


    train_history = (
        train_subset
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


    positive_test = (
        positive_test_subset
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


    del train_subset
    del positive_test_subset

    gc.collect()


    # --------------------------------------------------------
    # Resume checkpoint
    # --------------------------------------------------------

    accumulated_parts = []
    processed = set()


    if (
        CHECKPOINT_PATH.exists()
        and
        PROCESSED_USERS_PATH.exists()
    ):

        checkpoint_df = pd.read_parquet(
            CHECKPOINT_PATH
        )


        accumulated_parts.append(
            checkpoint_df
        )


        processed = set(
            np.load(
                PROCESSED_USERS_PATH,
                allow_pickle=True,
            )
            .tolist()
        )


        print(
            "\n===== CHECKPOINT RESUME ====="
        )

        print(
            "Already processed:",
            len(
                processed
            ),
            "/",
            len(
                target_user_ids
            ),
            "users",
        )

        print(
            "Existing rows:",
            f"{len(checkpoint_df):,}",
        )


    total_users = len(
        target_user_ids
    )


    # --------------------------------------------------------
    # Pending users
    # --------------------------------------------------------

    pending_user_ids = []


    for user_id in target_user_ids:

        if user_id in processed:

            continue


        played = train_history.get(
            user_id,
            [],
        )

        relevant = positive_test.get(
            user_id,
            [],
        )


        # 데이터가 없는 user도 다시 검사할 필요 없도록 processed 처리.
        if (
            not played
            or
            not relevant
        ):

            processed.add(
                user_id
            )

            continue


        pending_user_ids.append(
            user_id
        )


    print(
        "\nPending valid users:",
        len(
            pending_user_ids
        ),
    )


    if not pending_user_ids:

        if not accumulated_parts:

            raise ValueError(
                "생성된 feature가 없습니다."
            )


        feature_df = pd.concat(
            accumulated_parts,
            ignore_index=True,
        )


        feature_df.to_parquet(
            FEATURE_CACHE_PATH,
            index=False,
        )


        if CHECKPOINT_PATH.exists():

            CHECKPOINT_PATH.unlink()


        if PROCESSED_USERS_PATH.exists():

            PROCESSED_USERS_PATH.unlink()


        return feature_df


    # --------------------------------------------------------
    # 1. Item-CF batch precompute
    # --------------------------------------------------------

    precompute_item_cf_neighbors(
        item_cf=
            engine.item_cf,

        pending_user_ids=
            pending_user_ids,

        train_history=
            train_history,

        batch_size=
            ITEM_CF_PRECOMPUTE_BATCH_SIZE,
    )


    # --------------------------------------------------------
    # 2. Batch User-CF
    # --------------------------------------------------------

    batch_user_cf = BatchUserCFScorer(
        engine.user_cf,
        batch_size=
            USER_CF_BATCH_SIZE,
    )


    # --------------------------------------------------------
    # 3. Feature generation
    # --------------------------------------------------------

    pending_rows = []

    session_start = time.perf_counter()

    session_start_processed = len(
        processed
    )

    last_checkpoint_processed = len(
        processed
    )


    n_pending = len(
        pending_user_ids
    )


    for batch_start in range(
        0,
        n_pending,
        USER_CF_BATCH_SIZE,
    ):

        batch_user_ids = (
            pending_user_ids[
                batch_start:
                batch_start
                +
                USER_CF_BATCH_SIZE
            ]
        )


        batch_played = [
            train_history.get(
                user_id,
                [],
            )

            for user_id
            in batch_user_ids
        ]


        batch_start_time = time.perf_counter()


        # expensive User-CF를 batch로 한 번에 계산.
        batch_user_scores = (
            batch_user_cf
            .calculate_scores_batch(
                batch_played
            )
        )


        for (
            user_id,
            played,
            user_all_scores,
        ) in zip(
            batch_user_ids,
            batch_played,
            batch_user_scores,
        ):

            relevant = positive_test.get(
                user_id,
                [],
            )


            user_df = build_user_feature_rows(
                engine=
                    engine,

                user_id=
                    user_id,

                played_app_ids=
                    played,

                relevant_items=
                    relevant,

                precomputed_user_all_scores=
                    user_all_scores,
            )


            if len(
                user_df
            ):

                pending_rows.append(
                    user_df
                )


            processed.add(
                user_id
            )


        batch_seconds = (
            time.perf_counter()
            -
            batch_start_time
        )


        session_done = (
            len(
                processed
            )
            -
            session_start_processed
        )


        session_elapsed = (
            time.perf_counter()
            -
            session_start
        )


        sec_per_user = (
            session_elapsed
            /
            session_done

            if session_done > 0

            else np.nan
        )


        remaining_users = (
            total_users
            -
            len(
                processed
            )
        )


        eta_seconds = (
            remaining_users
            *
            sec_per_user

            if np.isfinite(
                sec_per_user
            )

            else np.nan
        )


        if PROGRESS_EVERY_BATCH:

            print(
                f"Feature batch: "
                f"{len(processed)}/{total_users} users "
                f"| batch={len(batch_user_ids)} "
                f"| batch_time={batch_seconds:.1f}s "
                f"| avg={sec_per_user:.2f} sec/user "
                f"| ETA={eta_seconds / 60:.1f} min"
            )


        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            len(
                processed
            )
            -
            last_checkpoint_processed
            >=
            CHECKPOINT_EVERY
        ):

            (
                accumulated_parts,
                pending_rows,
                checkpoint_rows,
            ) = save_feature_checkpoint(

                accumulated_parts=
                    accumulated_parts,

                pending_rows=
                    pending_rows,

                processed=
                    processed,
            )


            last_checkpoint_processed = len(
                processed
            )


            print(
                f"Checkpoint saved: "
                f"{len(processed)}/{total_users} users "
                f"| rows={checkpoint_rows:,}"
            )


            gc.collect()


    # --------------------------------------------------------
    # Final merge / save
    # --------------------------------------------------------

    if pending_rows:

        accumulated_parts.append(
            pd.concat(
                pending_rows,
                ignore_index=True,
            )
        )


    if not accumulated_parts:

        raise ValueError(
            "생성된 feature가 없습니다."
        )


    feature_df = pd.concat(
        accumulated_parts,
        ignore_index=True,
    )


    feature_df.to_parquet(
        FEATURE_CACHE_PATH,
        index=False,
    )


    # 완료 후 checkpoint 제거.
    if CHECKPOINT_PATH.exists():

        CHECKPOINT_PATH.unlink()


    if PROCESSED_USERS_PATH.exists():

        PROCESSED_USERS_PATH.unlink()


    print(
        "\nFeature Cache 저장:"
    )

    print(
        FEATURE_CACHE_PATH
    )

    print(
        "Feature DF:",
        feature_df.shape,
    )


    print(
        "Batch feature generation time:",
        f"{time.perf_counter() - session_start:.1f} sec",
    )


    return feature_df


# ============================================================
# XGB Dataset
# ============================================================

def prepare_ranker_data(
    df,
    features,
):

    sorted_df = (
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


    X = (
        sorted_df[
            features
        ]
        .astype(
            np.float32
        )
    )


    y = (
        sorted_df[
            "label"
        ]
        .astype(
            np.float32
        )
    )


    group = (
        sorted_df
        .groupby(
            "user_id",
            sort=False,
        )
        .size()
        .to_numpy(
            dtype=np.int32
        )
    )


    return (
        sorted_df,
        X,
        y,
        group,
    )


# ============================================================
# Validation Training
# ============================================================

def train_validation_model(
    ltr_feature_df,
    ltr_user_ids,
):

    (
        train_users,
        valid_users,
    ) = train_test_split(

        ltr_user_ids,

        test_size=
            VALID_RATIO,

        random_state=
            RANDOM_STATE,
    )


    train_set = set(
        train_users.tolist()
    )


    valid_set = set(
        valid_users.tolist()
    )


    train_rank_df = (
        ltr_feature_df[
            ltr_feature_df[
                "user_id"
            ]
            .isin(
                train_set
            )
        ]
        .copy()
    )


    valid_rank_df = (
        ltr_feature_df[
            ltr_feature_df[
                "user_id"
            ]
            .isin(
                valid_set
            )
        ]
        .copy()
    )


    (
        _,
        X_train,
        y_train,
        train_group,
    ) = prepare_ranker_data(
        train_rank_df,
        FULL14_FEATURES,
    )


    (
        _,
        X_valid,
        y_valid,
        valid_group,
    ) = prepare_ranker_data(
        valid_rank_df,
        FULL14_FEATURES,
    )


    print(
        "\n===== Validation Dataset ====="
    )

    print(
        "Train users:",
        len(
            train_group
        ),
    )

    print(
        "Train rows :",
        len(
            X_train
        ),
    )

    print(
        "Valid users:",
        len(
            valid_group
        ),
    )

    print(
        "Valid rows :",
        len(
            X_valid
        ),
    )

    print(
        "Train positives:",
        int(
            y_train.sum()
        ),
    )

    print(
        "Valid positives:",
        int(
            y_valid.sum()
        ),
    )


    model = XGBRanker(
        **XGB_BASE_PARAMS,

        n_estimators=500,

        early_stopping_rounds=
            EARLY_STOPPING_ROUNDS,
    )


    print(
        "\n===== Validation XGB Training ====="
    )


    start = time.perf_counter()


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

        verbose=10,
    )


    elapsed = (
        time.perf_counter()
        -
        start
    )


    best_iteration = int(
        model.best_iteration
    )


    best_score = float(
        model.best_score
    )


    model.save_model(
        VALID_MODEL_PATH
    )


    print(
        "\nBest iteration:",
        best_iteration,
    )

    print(
        "Best Validation NDCG@10:",
        f"{best_score:.9f}",
    )

    print(
        "Training seconds:",
        f"{elapsed:.2f}",
    )


    summary_df = pd.DataFrame(
        [
            {
                "candidate_structure":
                    "Item50+BPR20+Content15+User15",

                "n_features":
                    len(
                        FULL14_FEATURES
                    ),

                "train_users":
                    len(
                        train_group
                    ),

                "valid_users":
                    len(
                        valid_group
                    ),

                "best_iteration":
                    best_iteration,

                "best_validation_ndcg_at_10":
                    best_score,

                "train_seconds":
                    elapsed,
            }
        ]
    )


    summary_df.to_csv(
        VALIDATION_SUMMARY_PATH,
        index=False,
    )


    return (
        model,
        best_iteration,
        best_score,
    )


# ============================================================
# Production Training on all LTR 1000
# ============================================================

def train_final_model(
    ltr_feature_df,
    best_iteration,
):

    (
        _,
        X_all,
        y_all,
        all_group,
    ) = prepare_ranker_data(
        ltr_feature_df,
        FULL14_FEATURES,
    )


    # XGBoost best_iteration is zero-based.
    final_n_estimators = (
        best_iteration
        +
        1
    )


    print(
        "\n===== Final Production XGB Training ====="
    )

    print(
        "LTR users:",
        len(
            all_group
        ),
    )

    print(
        "Rows:",
        len(
            X_all
        ),
    )

    print(
        "n_estimators:",
        final_n_estimators,
    )


    final_model = XGBRanker(
        **XGB_BASE_PARAMS,

        n_estimators=
            final_n_estimators,
    )


    start = time.perf_counter()


    final_model.fit(

        X_all,
        y_all,

        group=
            all_group,

        verbose=False,
    )


    elapsed = (
        time.perf_counter()
        -
        start
    )


    final_model.save_model(
        FINAL_MODEL_PATH
    )


    print(
        "Final model saved:"
    )

    print(
        FINAL_MODEL_PATH
    )

    print(
        "Training seconds:",
        f"{elapsed:.2f}",
    )


    importance_df = (
        pd.DataFrame(
            {
                "feature":
                    FULL14_FEATURES,

                "importance":
                    final_model
                    .feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    importance_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(
                importance_df
            )
            +
            1
        ),
    )


    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )


    print(
        "\n===== Final Feature Importance ====="
    )

    print(
        importance_df
        .to_string(
            index=False
        )
    )


    return (
        final_model,
        final_n_estimators,
        importance_df,
    )


# ============================================================
# Metrics
# ============================================================

def evaluate_user(
    recommended,
    relevant,
    *,
    item_popularity,
    total_interactions,
):

    recommended = list(
        recommended
    )[
        :TOP_N
    ]


    relevant = set(
        relevant
    )


    hits = sum(
        app_id
        in
        relevant

        for app_id
        in recommended
    )


    precision = (
        hits
        /
        TOP_N
    )


    recall = (
        hits
        /
        len(
            relevant
        )

        if relevant

        else 0.0
    )


    hit_rate = float(
        hits > 0
    )


    # NDCG
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
            ideal_hits
            +
            1
        )
    )


    ndcg = (
        dcg
        /
        idcg

        if idcg > 0

        else 0.0
    )


    # AP@10 / MAP component
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


    ap_denominator = min(
        len(
            relevant
        ),
        TOP_N,
    )


    ap = (
        precision_sum
        /
        ap_denominator

        if ap_denominator > 0

        else 0.0
    )


    popularity = (
        runtime_main
        .recommendation_popularity_metrics(
            recommended=
                recommended,

            item_popularity=
                item_popularity,

            total_interactions=
                total_interactions,

            k=
                TOP_N,
        )
    )


    return {
        "hits":
            hits,

        "precision_at_10":
            precision,

        "recall_at_10":
            recall,

        "hit_rate_at_10":
            hit_rate,

        "ndcg_at_10":
            ndcg,

        "ap_at_10":
            ap,

        "mean_log_popularity_at_10":
            popularity[
                "mean_log_popularity_at_k"
            ],

        "novelty_at_10":
            popularity[
                "novelty_at_k"
            ],
    }


# ============================================================
# Final 400 Evaluation
# ============================================================

def evaluate_final400(
    *,
    model,
    final_feature_df,
    test_df,
    final_user_ids,
    engine,
):

    final_set = set(
        final_user_ids.tolist()
    )


    positive_test = (
        test_df[
            (
                test_df[
                    "user_id"
                ]
                .isin(
                    final_set
                )
            )
            &
            (
                test_df[
                    "is_recommended"
                ]
                ==
                True
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


    for index, user_id in enumerate(
        final_user_ids,
        start=1,
    ):

        user_df = (
            final_feature_df[
                final_feature_df[
                    "user_id"
                ]
                ==
                user_id
            ]
            .copy()
        )


        if len(
            user_df
        ) == 0:

            continue


        relevant = positive_test.get(
            user_id,
            [],
        )


        if not relevant:

            continue


        X_user = (
            user_df[
                FULL14_FEATURES
            ]
            .astype(
                np.float32
            )
        )


        predictions = (
            model
            .predict(
                X_user
            )
        )


        user_df[
            "xgb_score"
        ] = predictions


        ranked = (
            user_df
            .sort_values(
                "xgb_score",
                ascending=False,
                kind="stable",
            )
            .head(
                TOP_N
            )
    )


        recommended = (
            ranked[
                "app_id"
            ]
            .tolist()
        )


        metrics = evaluate_user(
            recommended,
            relevant,

            item_popularity=
                engine.item_popularity,

            total_interactions=
                engine.total_interactions,
        )


        rows.append(
            {
                "user_id":
                    user_id,

                "n_candidates":
                    len(
                        user_df
                    ),

                "n_relevant":
                    len(
                        relevant
                    ),

                **metrics,
            }
        )


        if (
            index
            %
            50
            ==
            0
            or
            index
            ==
            len(
                final_user_ids
            )
        ):

            print(
                f"Final evaluation: "
                f"{index}/{len(final_user_ids)}"
            )


    result_df = pd.DataFrame(
        rows
    )


    if len(
        result_df
    ) == 0:

        raise ValueError(
            "Final 400 평가 결과가 비었습니다."
        )


    summary = {
        "n_users":
            len(
                result_df
            ),

        "precision_at_10":
            float(
                result_df[
                    "precision_at_10"
                ]
                .mean()
            ),

        "recall_at_10":
            float(
                result_df[
                    "recall_at_10"
                ]
                .mean()
            ),

        "hit_rate_at_10":
            float(
                result_df[
                    "hit_rate_at_10"
                ]
                .mean()
            ),

        "ndcg_at_10":
            float(
                result_df[
                    "ndcg_at_10"
                ]
                .mean()
            ),

        "map_at_10":
            float(
                result_df[
                    "ap_at_10"
                ]
                .mean()
            ),

        "mean_log_popularity_at_10":
            float(
                result_df[
                    "mean_log_popularity_at_10"
                ]
                .mean()
            ),

        "novelty_at_10":
            float(
                result_df[
                    "novelty_at_10"
                ]
                .mean()
            ),

        "hits":
            int(
                result_df[
                    "hits"
                ]
                .sum()
            ),
    }


    result_df.to_csv(
        FINAL400_USER_RESULT_PATH,
        index=False,
    )


    pd.DataFrame(
        [
            summary
        ]
    ).to_csv(
        FINAL400_SUMMARY_PATH,
        index=False,
    )


    print(
        "\n"
        +
        "=" * 78
    )

    print(
        " FINAL 400 EVALUATION"
    )

    print(
        "=" * 78
    )


    print(
        f"Users                   : "
        f"{summary['n_users']}"
    )

    print(
        f"Precision@10            : "
        f"{summary['precision_at_10']:.4f}"
    )

    print(
        f"Recall@10               : "
        f"{summary['recall_at_10']:.4f}"
    )

    print(
        f"Hit Rate@10             : "
        f"{summary['hit_rate_at_10']:.4f}"
    )

    print(
        f"NDCG@10                 : "
        f"{summary['ndcg_at_10']:.4f}"
    )

    print(
        f"MAP@10                  : "
        f"{summary['map_at_10']:.4f}"
    )

    print(
        f"Mean Log Popularity@10  : "
        f"{summary['mean_log_popularity_at_10']:.4f}"
    )

    print(
        f"Novelty@10              : "
        f"{summary['novelty_at_10']:.4f}"
    )

    print(
        f"Hits                     : "
        f"{summary['hits']}"
    )


    return (
        result_df,
        summary,
    )


# ============================================================
# Diagnostics
# ============================================================

def print_feature_diagnostics(
    feature_df,
):

    print(
        "\n===== Feature Diagnostics ====="
    )


    print(
        "Rows:",
        len(
            feature_df
        ),
    )


    print(
        "Users:",
        feature_df[
            "user_id"
        ]
        .nunique(),
    )


    print(
        "Positive labels:",
        int(
            feature_df[
                "label"
            ]
            .sum()
        ),
    )


    print(
        "\nCandidate rows by source:"
    )


    for column in [
        "is_item_candidate",
        "is_bpr_candidate",
        "is_content_candidate",
        "is_user_candidate",
    ]:

        print(
            f"{column:<24}: "
            f"{int(feature_df[column].sum()):,} "
            f"({feature_df[column].mean():.4f})"
        )


    print(
        "\nRetriever count distribution:"
    )


    print(
        feature_df[
            "retriever_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    candidate_counts = (
        feature_df
        .groupby(
            "user_id"
        )
        .size()
    )


    print(
        "\nCandidates/User"
    )

    print(
        "Mean:",
        f"{candidate_counts.mean():.2f}",
    )

    print(
        "Min :",
        int(
            candidate_counts.min()
        ),
    )

    print(
        "Max :",
        int(
            candidate_counts.max()
        ),
    )



# ============================================================
# Ratio Sweep Helpers
# ============================================================

RETRIEVAL_RANK_COLUMNS = {
    "item":
        "item_retrieval_rank",

    "bpr":
        "bpr_retrieval_rank",

    "content":
        "content_retrieval_rank",

    "user":
        "user_retrieval_rank",
}


def ratio_name(
    config,
):

    return (
        f"I{config['item']}_"
        f"B{config['bpr']}_"
        f"C{config['content']}_"
        f"U{config['user']}"
    )


def materialize_ratio_feature_df(
    superset_df,
    config,
):

    """
    Superset candidate cache에서 특정 quota의 UNION을 재구성한 뒤
    후보 집합이 바뀐 것에 맞춰 Full14를 다시 계산한다.

    중요:
    - score_norm은 candidate set별로 다시 normalize
    - rank도 candidate set별로 다시 계산
    - source flags와 retriever_count도 quota별로 다시 계산
    """

    item_rank = (
        superset_df[
            "item_retrieval_rank"
        ]
    )

    bpr_rank = (
        superset_df[
            "bpr_retrieval_rank"
        ]
    )

    content_rank = (
        superset_df[
            "content_retrieval_rank"
        ]
    )

    user_rank = (
        superset_df[
            "user_retrieval_rank"
        ]
    )


    item_mask = (
        (item_rank > 0)
        &
        (
            item_rank
            <=
            config[
                "item"
            ]
        )
    )

    bpr_mask = (
        (bpr_rank > 0)
        &
        (
            bpr_rank
            <=
            config[
                "bpr"
            ]
        )
    )

    content_mask = (
        (content_rank > 0)
        &
        (
            content_rank
            <=
            config[
                "content"
            ]
        )
    )

    user_mask = (
        (user_rank > 0)
        &
        (
            user_rank
            <=
            config[
                "user"
            ]
        )
    )


    keep_mask = (
        item_mask
        |
        bpr_mask
        |
        content_mask
        |
        user_mask
    )


    df = (
        superset_df[
            keep_mask
        ]
        .copy()
    )


    # quota가 적용된 source flags
    df[
        "is_item_candidate"
    ] = (
        (
            df[
                "item_retrieval_rank"
            ]
            >
            0
        )
        &
        (
            df[
                "item_retrieval_rank"
            ]
            <=
            config[
                "item"
            ]
        )
    ).astype(
        np.int8
    )


    df[
        "is_bpr_candidate"
    ] = (
        (
            df[
                "bpr_retrieval_rank"
            ]
            >
            0
        )
        &
        (
            df[
                "bpr_retrieval_rank"
            ]
            <=
            config[
                "bpr"
            ]
        )
    ).astype(
        np.int8
    )


    df[
        "is_content_candidate"
    ] = (
        (
            df[
                "content_retrieval_rank"
            ]
            >
            0
        )
        &
        (
            df[
                "content_retrieval_rank"
            ]
            <=
            config[
                "content"
            ]
        )
    ).astype(
        np.int8
    )


    df[
        "is_user_candidate"
    ] = (
        (
            df[
                "user_retrieval_rank"
            ]
            >
            0
        )
        &
        (
            df[
                "user_retrieval_rank"
            ]
            <=
            config[
                "user"
            ]
        )
    ).astype(
        np.int8
    )


    df[
        "retriever_count"
    ] = (
        df[
            "is_item_candidate"
        ]
        +
        df[
            "is_bpr_candidate"
        ]
        +
        df[
            "is_content_candidate"
        ]
        +
        df[
            "is_user_candidate"
        ]
    ).astype(
        np.int8
    )


    # --------------------------------------------------------
    # candidate set별 score norm / rank 재계산
    # --------------------------------------------------------

    rebuilt_groups = []


    for _, group_df in df.groupby(
        "user_id",
        sort=False,
    ):

        group_df = group_df.copy()


        for prefix in [
            "item",
            "bpr",
            "content",
            "user",
        ]:

            score_col = (
                prefix
                +
                "_score"
            )

            score_values = (
                group_df[
                    score_col
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )


            group_df[
                prefix
                +
                "_score_norm"
            ] = (
                runtime_main
                .minmax_normalize(
                    score_values
                )
            )


            group_df[
                prefix
                +
                "_rank"
            ] = (
                runtime_main
                .assign_rank(
                    score_values
                )
            )


        rebuilt_groups.append(
            group_df
        )


    if not rebuilt_groups:

        return pd.DataFrame()


    return pd.concat(
        rebuilt_groups,
        ignore_index=True,
    )


def build_cv_user_folds(
    ltr_user_ids,
):

    """
    동일한 user-level 5-fold를 모든 ratio에 공통 사용.
    """

    rng = np.random.RandomState(
        RANDOM_STATE
    )


    shuffled = np.asarray(
        ltr_user_ids
    ).copy()


    rng.shuffle(
        shuffled
    )


    folds = np.array_split(
        shuffled,
        CV_FOLDS,
    )


    return folds


def train_one_ratio_fold(
    *,
    ratio_df,
    train_users,
    valid_users,
    fold_index,
    config,
):

    train_set = set(
        train_users.tolist()
    )

    valid_set = set(
        valid_users.tolist()
    )


    train_df = (
        ratio_df[
            ratio_df[
                "user_id"
            ]
            .isin(
                train_set
            )
        ]
        .copy()
    )


    valid_df = (
        ratio_df[
            ratio_df[
                "user_id"
            ]
            .isin(
                valid_set
            )
        ]
        .copy()
    )


    (
        _,
        X_train,
        y_train,
        train_group,
    ) = prepare_ranker_data(
        train_df,
        FULL14_FEATURES,
    )


    (
        _,
        X_valid,
        y_valid,
        valid_group,
    ) = prepare_ranker_data(
        valid_df,
        FULL14_FEATURES,
    )


    model = XGBRanker(
        **XGB_BASE_PARAMS,

        n_estimators=500,

        early_stopping_rounds=
            EARLY_STOPPING_ROUNDS,
    )


    start = time.perf_counter()


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

        verbose=False,
    )


    elapsed = (
        time.perf_counter()
        -
        start
    )


    return {
        "config":
            config[
                "name"
            ],

        "ratio":
            ratio_name(
                config
            ),

        "base_ratio_name":
            config[
                "base_ratio_name"
            ],

        "candidate_size":
            config[
                "candidate_size"
            ],

        "fold":
            fold_index,

        "item_n":
            config[
                "item"
            ],

        "bpr_n":
            config[
                "bpr"
            ],

        "content_n":
            config[
                "content"
            ],

        "user_n":
            config[
                "user"
            ],

        "train_users":
            len(
                train_group
            ),

        "valid_users":
            len(
                valid_group
            ),

        "train_rows":
            len(
                X_train
            ),

        "valid_rows":
            len(
                X_valid
            ),

        "train_positives":
            int(
                y_train.sum()
            ),

        "valid_positives":
            int(
                y_valid.sum()
            ),

        "best_iteration":
            int(
                model.best_iteration
            ),

        "validation_ndcg_at_10":
            float(
                model.best_score
            ),

        "train_seconds":
            elapsed,
    }


def run_ratio_size_sweep(
    *,
    superset_df,
    ltr_user_ids,
    test_df,
):

    folds = build_cv_user_folds(
        ltr_user_ids
    )


    ltr_user_set = set(
        ltr_user_ids.tolist()
    )


    total_positive_test = int(
        (
            test_df[
                (
                    test_df[
                        "user_id"
                    ]
                    .isin(
                        ltr_user_set
                    )
                )
                &
                (
                    test_df[
                        "is_recommended"
                    ]
                    ==
                    True
                )
            ]
            .shape[0]
        )
    )


    fold_rows = []
    config_rows = []


    print(
        "\n"
        +
        "=" * 78
    )

    print(
        " 4-Retriever Ratio x Candidate Size Joint Sweep"
    )

    print(
        f" {len(EXPERIMENT_CONFIGS)} configs | "
        f"quota=100 fixed | "
        f"{CV_FOLDS}-fold user CV"
    )

    print(
        " Final 400 is NOT used."
    )

    print(
        "=" * 78
    )


    for config_index, config in enumerate(
        EXPERIMENT_CONFIGS,
        start=1,
    ):

        print(
            "\n"
            +
            "-" * 78
        )

        print(
            f"[{config_index}/{len(EXPERIMENT_CONFIGS)}] "
            f"Size{config['candidate_size']} | "
            f"{config['base_ratio_name']} "
            f"=> Item{config['item']} "
            f"+ BPR{config['bpr']} "
            f"+ Content{config['content']} "
            f"+ User{config['user']}"
        )

        print(
            "-" * 78
        )


        ratio_start = time.perf_counter()


        ratio_df = materialize_ratio_feature_df(
            superset_df=
                superset_df,

            config=
                config,
        )


        if len(
            ratio_df
        ) == 0:

            raise ValueError(
                f"Candidate가 없습니다: {config['name']}"
            )


        users_count = int(
            ratio_df[
                "user_id"
            ]
            .nunique()
        )


        avg_candidates = float(
            ratio_df
            .groupby(
                "user_id"
            )
            .size()
            .mean()
        )


        min_candidates = int(
            ratio_df
            .groupby(
                "user_id"
            )
            .size()
            .min()
        )


        max_candidates = int(
            ratio_df
            .groupby(
                "user_id"
            )
            .size()
            .max()
        )


        positive_candidates = int(
            ratio_df[
                "label"
            ]
            .sum()
        )


        candidate_recall = (
            positive_candidates
            /
            total_positive_test

            if total_positive_test > 0

            else np.nan
        )


        config_fold_results = []


        for fold_idx in range(
            CV_FOLDS
        ):

            valid_users = folds[
                fold_idx
            ]


            train_users = np.concatenate(
                [
                    folds[
                        idx
                    ]

                    for idx
                    in range(
                        CV_FOLDS
                    )

                    if idx
                    !=
                    fold_idx
                ]
            )


            fold_result = train_one_ratio_fold(
                ratio_df=
                    ratio_df,

                train_users=
                    train_users,

                valid_users=
                    valid_users,

                fold_index=
                    fold_idx
                    +
                    1,

                config=
                    config,
            )


            fold_rows.append(
                fold_result
            )

            config_fold_results.append(
                fold_result
            )


            print(
                f" Fold {fold_idx + 1}: "
                f"NDCG@10="
                f"{fold_result['validation_ndcg_at_10']:.6f} "
                f"| best_iter="
                f"{fold_result['best_iteration']} "
                f"| {fold_result['train_seconds']:.2f}s"
            )


        ndcg_values = np.asarray(
            [
                row[
                    "validation_ndcg_at_10"
                ]

                for row
                in config_fold_results
            ],
            dtype=np.float64,
        )


        best_iterations = np.asarray(
            [
                row[
                    "best_iteration"
                ]

                for row
                in config_fold_results
            ],
            dtype=np.float64,
        )


        ratio_seconds = (
            time.perf_counter()
            -
            ratio_start
        )


        config_rows.append(
            {
                "config":
                    config[
                        "name"
                    ],

                "ratio":
                    ratio_name(
                        config
                    ),

                "base_ratio_name":
                    config[
                        "base_ratio_name"
                    ],

                "candidate_size":
                    config[
                        "candidate_size"
                    ],

                "item_n":
                    config[
                        "item"
                    ],

                "bpr_n":
                    config[
                        "bpr"
                    ],

                "content_n":
                    config[
                        "content"
                    ],

                "user_n":
                    config[
                        "user"
                    ],

                "users":
                    users_count,

                "rows":
                    len(
                        ratio_df
                    ),

                "avg_candidates_per_user":
                    avg_candidates,

                "min_candidates_per_user":
                    min_candidates,

                "max_candidates_per_user":
                    max_candidates,

                "positive_candidates":
                    positive_candidates,

                "candidate_pool_recall":
                    candidate_recall,

                "cv_mean_ndcg_at_10":
                    float(
                        ndcg_values.mean()
                    ),

                "cv_std_ndcg_at_10":
                    float(
                        ndcg_values.std(
                            ddof=0
                        )
                    ),

                "mean_best_iteration":
                    float(
                        best_iterations.mean()
                    ),

                "ratio_total_seconds":
                    ratio_seconds,
            }
        )


        print(
            " => Mean NDCG@10:",
            f"{ndcg_values.mean():.6f}",
            "| Std:",
            f"{ndcg_values.std(ddof=0):.6f}",
            "| Avg candidates:",
            f"{avg_candidates:.2f}",
            "| Candidate recall:",
            f"{candidate_recall:.4f}",
        )


        del ratio_df

        gc.collect()


    fold_df = pd.DataFrame(
        fold_rows
    )


    summary_df = (
        pd.DataFrame(
            config_rows
        )
        .sort_values(
            [
                "cv_mean_ndcg_at_10",
                "candidate_pool_recall",
            ],
            ascending=[
                False,
                False,
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


    summary_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(
                summary_df
            )
            +
            1,
        ),
    )


    fold_path = (
        RESULT_DIR
        /
        "ratio_size_sweep_fold_results.csv"
    )

    summary_path = (
        RESULT_DIR
        /
        "ratio_size_sweep_summary.csv"
    )

    best_path = (
        RESULT_DIR
        /
        "best_ratio_size.json"
    )


    fold_df.to_csv(
        fold_path,
        index=False,
    )


    summary_df.to_csv(
        summary_path,
        index=False,
    )


    best_row = (
        summary_df
        .iloc[0]
        .to_dict()
    )


    import json

    with open(
        best_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            best_row,
            f,
            ensure_ascii=False,
            indent=2,
        )


    print(
        "\n"
        +
        "=" * 78
    )

    print(
        " RATIO x SIZE SWEEP SUMMARY"
    )

    print(
        "=" * 78
    )


    display_columns = [
        "rank",
        "candidate_size",
        "item_n",
        "bpr_n",
        "content_n",
        "user_n",
        "avg_candidates_per_user",
        "candidate_pool_recall",
        "cv_mean_ndcg_at_10",
        "cv_std_ndcg_at_10",
        "mean_best_iteration",
    ]


    print(
        summary_df[
            display_columns
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\nBEST RATIO x SIZE:"
    )

    print(
        f" Candidate Size {int(best_row['candidate_size'])}"
        f" | Item{int(best_row['item_n'])}"
        f" + BPR{int(best_row['bpr_n'])}"
        f" + Content{int(best_row['content_n'])}"
        f" + User{int(best_row['user_n'])}"
    )

    print(
        "Mean CV NDCG@10:",
        f"{best_row['cv_mean_ndcg_at_10']:.6f}",
    )


    # --------------------------------------------------------
    # Secondary summaries
    # --------------------------------------------------------

    size_summary = (
        summary_df
        .groupby(
            "candidate_size",
            as_index=False,
        )
        .agg(
            mean_cv_ndcg_at_10=(
                "cv_mean_ndcg_at_10",
                "mean",
            ),
            best_cv_ndcg_at_10=(
                "cv_mean_ndcg_at_10",
                "max",
            ),
            mean_candidate_pool_recall=(
                "candidate_pool_recall",
                "mean",
            ),
            mean_actual_candidates=(
                "avg_candidates_per_user",
                "mean",
            ),
        )
        .sort_values(
            "candidate_size"
        )
    )


    ratio_summary = (
        summary_df
        .groupby(
            "base_ratio_name",
            as_index=False,
        )
        .agg(
            mean_cv_ndcg_at_10=(
                "cv_mean_ndcg_at_10",
                "mean",
            ),
            best_cv_ndcg_at_10=(
                "cv_mean_ndcg_at_10",
                "max",
            ),
            mean_candidate_pool_recall=(
                "candidate_pool_recall",
                "mean",
            ),
        )
        .sort_values(
            "mean_cv_ndcg_at_10",
            ascending=False,
        )
    )


    size_path = (
        RESULT_DIR
        /
        "candidate_size_summary.csv"
    )


    ratio_path = (
        RESULT_DIR
        /
        "base_ratio_summary.csv"
    )


    size_summary.to_csv(
        size_path,
        index=False,
    )


    ratio_summary.to_csv(
        ratio_path,
        index=False,
    )


    print(
        "\\n===== Candidate Size Aggregate ====="
    )

    print(
        size_summary
        .to_string(
            index=False
        )
    )


    print(
        "\\n===== Base Ratio Aggregate ====="
    )

    print(
        ratio_summary
        .to_string(
            index=False
        )
    )


    print(
        "\nSaved:"
    )

    print(
        fold_path
    )

    print(
        summary_path
    )

    print(
        best_path
    )

    print(
        size_path
    )

    print(
        ratio_path
    )


    return (
        fold_df,
        summary_df,
    )



# ============================================================
# Main
# ============================================================

def main():

    total_start = time.perf_counter()


    validate_candidate_quota()


    print(
        "=" * 78
    )

    print(
        " Experiment K - 4-Retriever Ratio x Candidate Size Joint Sweep"
    )

    print(
        " Candidate sizes = 90 / 100 / 110 / 125 / 150"
    )

    print(
        " Ranker = XGBoost Full14"
    )

    print(
        f" Validation = {CV_FOLDS}-Fold User CV"
    )

    print(
        " Final 400 = NOT USED"
    )

    print(
        "=" * 78
    )


    # ========================================================
    # 1. LTR 1000 users only
    # ========================================================

    ltr_user_ids = load_user_ids(
        LTR_USERS_PATH
    )


    print(
        "\nLTR users:",
        len(
            ltr_user_ids
        ),
    )


    # ========================================================
    # 2. Runtime Engine
    #
    # 기존 XGB는 필요 없으므로 item mode로 scorer들만 로드.
    # ========================================================

    runtime_main.RANKER_MODE = "item"


    (
        engine,
        _,
        train_df,
        test_df,
    ) = runtime_main.build_engine()


    print(
        "\n===== Batch Optimization ====="
    )

    print(
        f"User-CF batch size        : {USER_CF_BATCH_SIZE}"
    )

    print(
        f"Item-CF precompute batch : {ITEM_CF_PRECOMPUTE_BATCH_SIZE}"
    )

    print(
        "Superset quota:"
    )

    print(
        f"Item{ITEM_N} + "
        f"BPR{BPR_N} + "
        f"Content{CONTENT_N} + "
        f"User{USER_N} "
        f"= max {CANDIDATE_SIZE}"
    )


    # ========================================================
    # 3. Superset feature cache
    #
    # 모든 ratio가 필요로 하는 최대 후보를 한 번만 생성한다.
    # ========================================================

    superset_df = build_or_load_feature_cache(
        engine=
            engine,

        train_df=
            train_df,

        test_df=
            test_df,

        target_user_ids=
            ltr_user_ids,
    )


    print(
        "\n===== Superset Cache ====="
    )

    print(
        "Rows :",
        len(
            superset_df
        ),
    )

    print(
        "Users:",
        superset_df[
            "user_id"
        ]
        .nunique(),
    )

    print(
        "Avg candidates/user:",
        f"{superset_df.groupby('user_id').size().mean():.2f}",
    )


    required_rank_columns = [
        "item_retrieval_rank",
        "bpr_retrieval_rank",
        "content_retrieval_rank",
        "user_retrieval_rank",
    ]


    missing = [
        column

        for column
        in required_rank_columns

        if column
        not in superset_df.columns
    ]


    if missing:

        raise ValueError(
            "Ratio sweep superset cache에 retrieval rank가 없습니다. "
            "기존 다른 실험 cache를 잘못 사용했을 수 있습니다: "
            f"{missing}"
        )


    # ========================================================
    # 4. Ratio Sweep
    # ========================================================

    (
        fold_df,
        summary_df,
    ) = run_ratio_size_sweep(
        superset_df=
            superset_df,

        ltr_user_ids=
            ltr_user_ids,

        test_df=
            test_df,
    )


    # ========================================================
    # 5. Done
    # ========================================================

    total_seconds = (
        time.perf_counter()
        -
        total_start
    )


    print(
        "\n"
        +
        "=" * 78
    )

    print(
        " EXPERIMENT COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        "Total Runtime:",
        f"{total_seconds:.1f} sec",
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "1) 이 실험은 Final 400을 전혀 사용하지 않았습니다."
    )

    print(
        "2) Candidate quota는 90/100/110/125/150만 비교했습니다."
    )

    print(
        "3) Ratio와 Candidate Size를 여기서 동시에 선택합니다."
    )

    print(
        "4) 최종 모델 교체는 #1~#6 실험이 끝난 뒤에 하세요."
    )




if __name__ == "__main__":
    main()
