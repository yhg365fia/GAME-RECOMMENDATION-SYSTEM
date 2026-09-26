# quantitative_evaluation.py
# ============================================================
# Steam Game Recommendation System - Final Quantitative Evaluation (Batch Optimized)
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
# Output
#   Top-10 Recommendation
#
# Offline evaluation
#   Fixed original 400 users
#   Precision / Recall / HR / NDCG / MAP
#   Mean Log Popularity / Novelty / Hits
#
# NOTE
# - Final model choice: E_HR, based on 3-seed CV average NDCG.
# - Final XGB ranker is trained on LTR1000 only.
# - Selection200 artifacts are not referenced.
# - Offline evaluation uses the original fixed Final400 once.
# ============================================================

from __future__ import annotations

import gc
import json
import time
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
RESULT_DIR = SAVED_MODEL_DIR / "results"

# Runtime에서 임의의 신규 사용자를 처리하기 위해 필요한 전역 cache
RUNTIME_CACHE_DIR = SAVED_MODEL_DIR / "runtime_cache"

# 기존 Case 3 Content cache를 가능하면 재사용
CASE3_CACHE_DIR = SAVED_MODEL_DIR / "case3_cache"

RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

# ------------------------------------------------------------
# Run Mode
#
# "recommend":
#   새 사용자가 플레이한 게임을 직접 입력해서 Top-10 추천
#   + 추천 결과의 Popularity / Novelty 출력
#
# "evaluate":
#   기존 Train/Test를 사용한 offline evaluation
#   + Precision / Recall / HR / NDCG / MAP
#   + Mean Log Popularity / Novelty 출력
# ------------------------------------------------------------

RUN_MODE = "evaluate"

TOP_N = 10
CANDIDATE_SIZE = 101

# Offline evaluation: 기존 프로젝트와 동일한 4개 sparsity group
EVAL_LOWER_BOUND = 10
EVAL_UPPER_BOUND = 78
EVAL_SAMPLE_PER_GROUP = 100

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
#
# "item": 현재 안정적인 baseline
# "xgb" : XGBRanker가 최종 확정되면 사용
# ------------------------------------------------------------

RANKER_MODE = "xgb"

FINAL_MODEL_NAME = "E_HR"

XGB_MODEL_PATH = (
    SAVED_MODEL_DIR
    / "xgb_ranker_final_e_hr.json"
)

FINAL_XGB_METADATA_PATH = (
    SAVED_MODEL_DIR
    / "xgb_ranker_final_e_hr_metadata.json"
)

# ------------------------------------------------------------
# Final E_HR Ranker Training
#
# Selection200 산출물은 전혀 사용하지 않는다.
#
# LTR1000 Superset
#   -> E_HR candidate I46/B20/C15/U20 재구성
#   -> Full15
#   -> LTR1000 전체로 XGBRanker 최종 학습
#   -> xgb_ranker_final_e_hr.json 저장
# ------------------------------------------------------------

LTR_SUPERSET_PATH = (
    SAVED_MODEL_DIR
    / "ltr_cache"
    / "xgb_4retriever_ratio_size_sweep"
    / "ratio_size_sweep_superset_features.parquet"
)

LTR_USERS_PATH = (
    SAVED_MODEL_DIR
    / "ltr_cache"
    / "xgb_exp1"
    / "ltr_train_users_1000.csv"
)

TRAIN_FINAL_XGB_IF_MISSING = True
FORCE_RETRAIN_FINAL_XGB = False

# 3-Seed Stability에서 E_HR best_iteration 중앙값 = 53
# 최종 전체 LTR1000 학습은 +1 하여 54 trees 사용.
FINAL_XGB_N_ESTIMATORS = 54

FINAL_XGB_PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",

    "max_depth": 5,
    "min_child_weight": 2,
    "learning_rate": 0.042841786327162734,

    "subsample": 0.8621568879089554,
    "colsample_bytree": 0.7010833568732288,

    "reg_lambda": 0.8133078460477237,
    "reg_alpha": 0.49527026495550774,

    "random_state": RANDOM_STATE,
    "tree_method": "hist",

    "n_estimators": FINAL_XGB_N_ESTIMATORS,
}

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

FINAL400_USERS_PATH = (
    RESULT_DIR
    / "hybrid_sampled_users.csv"
)

INTEGRATED_EVAL_PATH = (
    RESULT_DIR
    / "final_e_hr_full400_eval.csv"
)

INTEGRATED_EVAL_SUMMARY_PATH = (
    RESULT_DIR
    / "final_e_hr_full400_eval_summary.csv"
)

# ------------------------------------------------------------
# Final400 Batch Evaluation Cache
#
# 기존 느린 방식:
#   400명 각각 engine.recommend()
#   -> User-CF의 query @ 전체 user matrix를 400번 개별 수행
#
# 변경:
#   - Item-CF source item neighbor를 batch precompute
#   - User-CF를 8명씩 sparse batch 계산
#   - BPR / Content는 사용자별 계산
#   - 모든 Final400 candidate feature를 만든 뒤
#     XGBoost predict는 전체를 한 번에 수행
# ------------------------------------------------------------

FINAL400_BATCH_CACHE_DIR = (
    RESULT_DIR
    / "final400_batch_cache_e_hr"
)

FINAL400_BATCH_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL400_FEATURE_CACHE_PATH = (
    FINAL400_BATCH_CACHE_DIR
    / "final400_e_hr_i46_b20_c15_u20_ft3_features.parquet"
)

FINAL400_FEATURE_CHECKPOINT_PATH = (
    FINAL400_BATCH_CACHE_DIR
    / "final400_feature_checkpoint.parquet"
)

FINAL400_PROCESSED_USERS_PATH = (
    FINAL400_BATCH_CACHE_DIR
    / "final400_processed_users.npy"
)

USER_CF_BATCH_SIZE = 8
ITEM_CF_PRECOMPUTE_BATCH_SIZE = 64
FEATURE_CHECKPOINT_EVERY_BATCHES = 7

# True로 바꾸면 Final400 feature cache를 무시하고 처음부터 재생성.
FORCE_REBUILD_FINAL400_FEATURE_CACHE = False


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
# Final E_HR XGBoost Training
# ============================================================

def _validate_final_e_hr_superset_schema(
    superset_df,
):

    required_columns = {
        "user_id",
        "app_id",

        "item_score",
        "bpr_score",
        "content_score",
        "user_score",

        "item_retrieval_rank",
        "bpr_retrieval_rank",
        "content_retrieval_rank",
        "user_retrieval_rank",

        "item_popularity",
        "user_interaction_count",

        "label",
    }

    missing = (
        required_columns
        -
        set(
            superset_df.columns
        )
    )

    if missing:

        raise ValueError(
            (
                "LTR Superset에 필요한 column이 없습니다: "
                f"{sorted(missing)}"
            )
        )


def _materialize_final_e_hr_ltr_df(
    superset_df,
):

    """
    기존 Ratio/Size 실험과 동일한 방식으로
    Superset에서 E_HR candidate pool을 재구성한다.

    Final E_HR
    - Item    46
    - BPR     20
    - Content 15
    - User    20
    - max candidate size 101

    후보 집합이 바뀌므로:
    - source flag
    - retriever_count
    - score normalization
    - rank

    을 candidate set 기준으로 다시 계산한다.
    """

    _validate_final_e_hr_superset_schema(
        superset_df
    )

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
        (item_rank <= ITEM_N)
    )

    bpr_mask = (
        (bpr_rank > 0)
        &
        (bpr_rank <= BPR_N)
    )

    content_mask = (
        (content_rank > 0)
        &
        (content_rank <= CONTENT_N)
    )

    user_mask = (
        (user_rank > 0)
        &
        (user_rank <= USER_N)
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

    if len(
        df
    ) == 0:

        raise ValueError(
            "E_HR candidate materialization 결과가 비어 있습니다."
        )

    # --------------------------------------------------------
    # Quota 적용 후 source flag
    # --------------------------------------------------------

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
            ITEM_N
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
            BPR_N
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
            CONTENT_N
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
            USER_N
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
    # E_HR candidate set별 score norm / rank 재계산
    # --------------------------------------------------------

    rebuilt_groups = []

    for (
        _,
        group_df,
    ) in df.groupby(
        "user_id",
        sort=False,
    ):

        group_df = (
            group_df
            .copy()
        )

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

            values = (
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
            ] = minmax_normalize(
                values
            )

            group_df[
                prefix
                +
                "_rank"
            ] = assign_rank(
                values
            )

        rebuilt_groups.append(
            group_df
        )

    if not rebuilt_groups:

        raise ValueError(
            "E_HR training group이 없습니다."
        )

    df = pd.concat(
        rebuilt_groups,
        ignore_index=True,
    )

    return df


def _prepare_final_ranker_data(
    df,
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
            XGB_FEATURES
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


def _validate_ltr_final400_separation(
    ltr_user_ids,
):

    """
    최종 Ranker 학습 사용자와 Final400의 겹침을 확인한다.
    겹치면 Final400이 학습에 들어간 것이므로 즉시 중단한다.
    """

    if not FINAL400_USERS_PATH.exists():

        print(
            "[주의] Final400 user 파일이 없어 "
            "LTR1000/Final400 overlap 검사는 건너뜁니다."
        )

        return

    final400_df = pd.read_csv(
        FINAL400_USERS_PATH
    )

    if (
        "user_id"
        not in
        final400_df.columns
    ):

        raise ValueError(
            "Final400 파일에 user_id column이 없습니다."
        )

    final400_ids = set(
        final400_df[
            "user_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    overlap = (
        set(
            ltr_user_ids
        )
        &
        final400_ids
    )

    print(
        "\n===== LTR1000 / Final400 Leakage Guard ====="
    )

    print(
        "LTR users     :",
        len(
            set(
                ltr_user_ids
            )
        ),
    )

    print(
        "Final400 users:",
        len(
            final400_ids
        ),
    )

    print(
        "Overlap       :",
        len(
            overlap
        ),
    )

    if overlap:

        raise RuntimeError(
            (
                "LTR1000과 Final400 사용자 overlap이 발견됐습니다. "
                "최종 평가를 중단합니다."
            )
        )

    print(
        "OK: Final400은 최종 XGB 학습에 사용되지 않습니다."
    )


def train_final_e_hr_ranker_if_needed():

    """
    Selection200 관련 파일/모델을 전혀 사용하지 않는다.

    Final training source:
        LTR1000 ratio-size superset cache only.

    Final selection:
        3-Seed Avg NDCG 1위 E_HR.

    Final train:
        E_HR fixed config + LTR1000 전체.

    Output:
        models/saved_model/xgb_ranker_final_e_hr.json
    """

    if XGBRanker is None:

        raise ImportError(
            "xgboost가 설치되어 있지 않습니다. "
            "requirements.txt에 xgboost를 확인해주세요."
        )

    should_train = (
        FORCE_RETRAIN_FINAL_XGB
        or
        (
            TRAIN_FINAL_XGB_IF_MISSING
            and
            not XGB_MODEL_PATH.exists()
        )
    )

    if not should_train:

        if XGB_MODEL_PATH.exists():

            print(
                "\n===== Final E_HR XGBRanker ====="
            )

            print(
                "기존 최종 모델 사용:"
            )

            print(
                XGB_MODEL_PATH
            )

            return

        raise FileNotFoundError(
            (
                "최종 E_HR XGB 모델이 없습니다.\n"
                f"{XGB_MODEL_PATH}\n"
                "TRAIN_FINAL_XGB_IF_MISSING=True로 설정하세요."
            )
        )

    if not LTR_SUPERSET_PATH.exists():

        raise FileNotFoundError(
            (
                "LTR1000 Superset cache가 없습니다:\n"
                f"{LTR_SUPERSET_PATH}\n\n"
                "Selection200 파일이 아니라 "
                "기존 LTR1000 ratio-size superset cache가 필요합니다."
            )
        )

    print(
        "\n============================================================"
    )

    print(
        " Final E_HR Ranker Training"
    )

    print(
        " LTR1000 ONLY | Final400 NOT USED"
    )

    print(
        "============================================================"
    )

    print(
        "\n===== LTR1000 Superset Load ====="
    )

    print(
        LTR_SUPERSET_PATH
    )

    superset_df = pd.read_parquet(
        LTR_SUPERSET_PATH
    )

    print(
        "Superset:",
        superset_df.shape,
    )

    superset_users = (
        superset_df[
            "user_id"
        ]
        .drop_duplicates()
        .to_numpy()
    )

    print(
        "Superset users:",
        len(
            superset_users
        ),
    )

    if len(
        superset_users
    ) != 1000:

        raise ValueError(
            (
                "LTR Superset user 수가 1000명이 아닙니다. "
                f"현재={len(superset_users)}"
            )
        )

    # 가능하면 기존 LTR1000 user file과도 일치 확인.
    if LTR_USERS_PATH.exists():

        ltr_users_df = pd.read_csv(
            LTR_USERS_PATH
        )

        if (
            "user_id"
            in
            ltr_users_df.columns
        ):

            expected_ltr_ids = set(
                ltr_users_df[
                    "user_id"
                ]
                .drop_duplicates()
                .tolist()
            )

            actual_ltr_ids = set(
                superset_users.tolist()
            )

            if (
                expected_ltr_ids
                !=
                actual_ltr_ids
            ):

                raise RuntimeError(
                    (
                        "LTR Superset 사용자와 "
                        "ltr_train_users_1000.csv가 일치하지 않습니다."
                    )
                )

            print(
                "LTR1000 user identity check: OK"
            )

    _validate_ltr_final400_separation(
        superset_users
    )

    print(
        "\n===== E_HR Candidate Materialization ====="
    )

    print(
        (
            f"Size={CANDIDATE_SIZE} | "
            f"I{ITEM_N} B{BPR_N} "
            f"C{CONTENT_N} U{USER_N}"
        )
    )

    e_hr_df = (
        _materialize_final_e_hr_ltr_df(
            superset_df
        )
    )

    print(
        "E_HR rows:",
        f"{len(e_hr_df):,}",
    )

    print(
        "E_HR users:",
        e_hr_df[
            "user_id"
        ].nunique(),
    )

    print(
        "Positive labels:",
        int(
            e_hr_df[
                "label"
            ].sum()
        ),
    )

    missing_features = [
        feature
        for feature
        in XGB_FEATURES
        if feature
        not in e_hr_df.columns
    ]

    if missing_features:

        raise ValueError(
            (
                "E_HR final feature가 없습니다: "
                f"{missing_features}"
            )
        )

    (
        _,
        X_all,
        y_all,
        group_all,
    ) = (
        _prepare_final_ranker_data(
            e_hr_df
        )
    )

    print(
        "\n===== Final E_HR Full15 ====="
    )

    for feature in XGB_FEATURES:

        print(
            " -",
            feature,
        )

    print(
        "\nFeature count:",
        len(
            XGB_FEATURES
        ),
    )

    print(
        "n_estimators:",
        FINAL_XGB_N_ESTIMATORS,
    )

    print(
        "Train rows:",
        len(
            X_all
        ),
    )

    print(
        "Train groups:",
        len(
            group_all
        ),
    )

    print(
        "\n===== XGBRanker Fit on ALL LTR1000 ====="
    )

    final_model = XGBRanker(
        **FINAL_XGB_PARAMS
    )

    final_model.fit(
        X_all,
        y_all,
        group=group_all,
        verbose=False,
    )

    XGB_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_model.save_model(
        str(
            XGB_MODEL_PATH
        )
    )

    metadata = {
        "model": FINAL_MODEL_NAME,
        "selection_basis": (
            "3-seed CV average Macro NDCG@10"
        ),
        "three_seed_avg_ndcg_at_10": 0.145155,
        "candidate_size": CANDIDATE_SIZE,
        "quota": {
            "item": ITEM_N,
            "bpr": BPR_N,
            "content": CONTENT_N,
            "user": USER_N,
        },
        "features": XGB_FEATURES,
        "feature_count": len(
            XGB_FEATURES
        ),
        "n_estimators": FINAL_XGB_N_ESTIMATORS,
        "xgb_params": FINAL_XGB_PARAMS,
        "training_source": str(
            LTR_SUPERSET_PATH
        ),
        "training_users": int(
            len(
                superset_users
            )
        ),
        "training_rows": int(
            len(
                X_all
            )
        ),
        "final400_used_for_training": False,
        "selection200_used": False,
    }

    with open(
        FINAL_XGB_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n===== Final E_HR Model Saved ====="
    )

    print(
        XGB_MODEL_PATH
    )

    print(
        FINAL_XGB_METADATA_PATH
    )

    # 메모리 정리
    del superset_df
    del e_hr_df
    del X_all
    del y_all
    del group_all
    del final_model

    gc.collect()



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

    # --------------------------------------------------------
    # 1.5 Final E_HR XGB Ranker
    #
    # 모델이 없으면 LTR1000 전체로 최종 학습 후 저장.
    # Selection200 / Final400은 학습에 사용하지 않음.
    # --------------------------------------------------------

    train_final_e_hr_ranker_if_needed()

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
        train_df,
        test_df,
    )



# ============================================================
# Final400 Batch Evaluation Helpers
# ============================================================

def top_positive_ids(
    scores,
    item_ids,
    n,
):

    """
    Item / Content / User-CF runtime retrieve와 동일:
    finite & score > 0인 item 중 Top-N.
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

    if len(
        valid
    ) == 0:

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
    NewUserBPRScorer.retrieve()와 동일:
    finite score 중 Top-N.
    """

    idx = top_indices(
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
    전체 score vector에서 candidate score만 추출.
    scorer universe에 없는 item은 0.
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


class BatchUserCFScorer:
    """
    기존 UserCFRuntimeScorer의 의미는 유지하면서
    여러 평가 사용자를 한 번의 sparse matrix multiplication으로 묶는다.

    기존:
        query 1명 @ 전체 user matrix.T
        -> 400번 반복

    변경:
        query B명 @ 전체 user matrix.T
        -> batch 반복
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
                unique_preserve_order(
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

        n_batch = len(
            played_lists
        )

        if n_batch == 0:

            return []

        (
            query_matrix,
            source_indices_by_row,
        ) = (
            self._build_query_matrix(
                played_lists
            )
        )

        # ----------------------------------------------------
        # 1. Batch cosine dot products
        # ----------------------------------------------------

        sims = (
            query_matrix
            @
            self.matrix_t
        ).tocsr()

        sims.eliminate_zeros()

        # ----------------------------------------------------
        # 2. 각 query에서 Top-K neighbor 추출
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
        # 3. Top-K users -> item weighted sum
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
        # 4. normalize + seen mask
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
    Final400 사용자가 실제로 플레이한 unique source item만 모아서
    Item-CF Top-K neighbor를 batch로 미리 계산한다.

    이후 item_cf.calculate_scores()는 neighbor_cache를 사용하므로
    item-to-all sparse dot을 사용자별로 반복하지 않는다.
    """

    print(
        "\n===== Item-CF Neighbor Batch Precompute ====="
    )

    required_indices = set()

    for user_id in pending_user_ids:

        played_app_ids = (
            unique_preserve_order(
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


def build_user_feature_rows_fast(
    *,
    engine,
    user_id,
    played_app_ids,
    relevant_items,
    precomputed_user_all_scores,
    bpr_item_to_idx,
):

    """
    Final runtime과 같은 E_HR 후보/feature 정의를 사용하되,
    expensive User-CF score는 batch에서 미리 전달받는다.
    """

    played_app_ids = (
        unique_preserve_order(
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
    # 2. Retriever 전체 score
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

    user_all_scores = np.asarray(
        precomputed_user_all_scores,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # 3. E_HR Candidate Retrieval
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
        unique_preserve_order(
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
    # 5. Base 8
    # Candidate set 기준 normalize / rank
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 6. Retriever source signals
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

    # --------------------------------------------------------
    # 7. Context
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
    # 8. Label - diagnostic / cache validation
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


def _save_final400_checkpoint(
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
        )

    checkpoint_df = pd.concat(
        parts,
        ignore_index=True,
    )

    checkpoint_df.to_parquet(
        FINAL400_FEATURE_CHECKPOINT_PATH,
        index=False,
    )

    np.save(
        FINAL400_PROCESSED_USERS_PATH,
        np.asarray(
            list(
                processed
            ),
            dtype=object,
        ),
    )

    print(
        "Checkpoint saved:"
        f" users={len(processed)}/400"
        f" | rows={len(checkpoint_df):,}"
    )

    return (
        [
            checkpoint_df
        ],
        [],
    )


def _validate_final400_feature_cache(
    feature_df,
    target_user_ids,
):

    required_columns = {
        "user_id",
        "app_id",
        "label",
        *XGB_FEATURES,
    }

    missing = (
        required_columns
        -
        set(
            feature_df.columns
        )
    )

    if missing:

        raise ValueError(
            (
                "Final400 feature cache에 필요한 column이 없습니다: "
                f"{sorted(missing)}"
            )
        )

    expected_users = set(
        target_user_ids
    )

    actual_users = set(
        feature_df[
            "user_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    if (
        expected_users
        !=
        actual_users
    ):

        raise ValueError(
            (
                "Final400 feature cache user가 고정 400명과 다릅니다. "
                f"expected={len(expected_users)}, "
                f"actual={len(actual_users)}"
            )
        )


def build_or_load_final400_feature_cache(
    *,
    engine,
    train_df,
    test_df,
    target_user_ids,
):

    """
    고정 Final400에 대해 E_HR candidate feature를 batch 방식으로 생성.

    Optimizations
    -------------
    1. Item-CF:
       Final400의 unique source item neighbor를 batch precompute.

    2. User-CF:
       USER_CF_BATCH_SIZE명씩 sparse batch score.

    3. BPR / Content:
       사용자별 계산 유지.

    4. XGB:
       이 함수에서는 feature만 만들고,
       evaluator에서 전체 row를 한 번에 predict.
    """

    if FORCE_REBUILD_FINAL400_FEATURE_CACHE:

        for path in [
            FINAL400_FEATURE_CACHE_PATH,
            FINAL400_FEATURE_CHECKPOINT_PATH,
            FINAL400_PROCESSED_USERS_PATH,
        ]:

            if path.exists():

                path.unlink()

    if FINAL400_FEATURE_CACHE_PATH.exists():

        print(
            "\n===== Final400 Feature Cache Load ====="
        )

        print(
            FINAL400_FEATURE_CACHE_PATH
        )

        feature_df = pd.read_parquet(
            FINAL400_FEATURE_CACHE_PATH
        )

        _validate_final400_feature_cache(
            feature_df,
            target_user_ids,
        )

        print(
            "Feature DF:",
            feature_df.shape,
        )

        return feature_df

    print(
        "\n============================================================"
    )

    print(
        " Final400 Feature Build - BATCH OPTIMIZED"
    )

    print(
        "============================================================"
    )

    target_set = set(
        target_user_ids
    )

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

    accumulated_parts = []
    processed = set()

    if (
        FINAL400_FEATURE_CHECKPOINT_PATH.exists()
        and
        FINAL400_PROCESSED_USERS_PATH.exists()
    ):

        checkpoint_df = pd.read_parquet(
            FINAL400_FEATURE_CHECKPOINT_PATH
        )

        accumulated_parts.append(
            checkpoint_df
        )

        processed = set(
            np.load(
                FINAL400_PROCESSED_USERS_PATH,
                allow_pickle=True,
            )
            .tolist()
        )

        print(
            "\n===== Final400 CHECKPOINT RESUME ====="
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
        )

        print(
            "Existing rows:",
            f"{len(checkpoint_df):,}",
        )

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

        if (
            not played
            or
            not relevant
        ):

            raise ValueError(
                (
                    "Final400 사용자에 train history 또는 "
                    f"positive test가 없습니다: {user_id}"
                )
            )

        pending_user_ids.append(
            user_id
        )

    print(
        "\nPending users:",
        len(
            pending_user_ids
        ),
    )

    if pending_user_ids:

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

        batch_user_cf = BatchUserCFScorer(
            engine.user_cf,
            batch_size=
                USER_CF_BATCH_SIZE,
        )

        bpr_item_to_idx = {
            app_id: idx

            for idx, app_id
            in enumerate(
                engine.bpr.item_ids
            )
        }

        pending_rows = []

        session_start = time.perf_counter()
        session_done = 0
        batch_counter = 0

        for batch_start in range(
            0,
            len(
                pending_user_ids
            ),
            USER_CF_BATCH_SIZE,
        ):

            batch_counter += 1

            batch_user_ids = (
                pending_user_ids[
                    batch_start:
                    batch_start
                    +
                    USER_CF_BATCH_SIZE
                ]
            )

            batch_played = [
                train_history[
                    user_id
                ]

                for user_id
                in batch_user_ids
            ]

            batch_start_time = time.perf_counter()

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

                relevant = positive_test[
                    user_id
                ]

                user_df = (
                    build_user_feature_rows_fast(
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

                        bpr_item_to_idx=
                            bpr_item_to_idx,
                    )
                )

                if len(
                    user_df
                ) == 0:

                    raise ValueError(
                        (
                            "Final400 candidate feature가 비었습니다: "
                            f"{user_id}"
                        )
                    )

                pending_rows.append(
                    user_df
                )

                processed.add(
                    user_id
                )

                session_done += 1

            batch_seconds = (
                time.perf_counter()
                -
                batch_start_time
            )

            elapsed = (
                time.perf_counter()
                -
                session_start
            )

            avg_sec_per_user = (
                elapsed
                /
                max(
                    session_done,
                    1,
                )
            )

            remaining = (
                len(
                    pending_user_ids
                )
                -
                session_done
            )

            eta = (
                remaining
                *
                avg_sec_per_user
            )

            print(
                f"Feature batch: "
                f"{len(processed)}/{len(target_user_ids)} users "
                f"| batch={len(batch_user_ids)} "
                f"| batch_time={batch_seconds:.1f}s "
                f"| avg={avg_sec_per_user:.2f}s/user "
                f"| ETA={eta / 60:.1f} min"
            )

            if (
                batch_counter
                %
                FEATURE_CHECKPOINT_EVERY_BATCHES
                ==
                0
            ):

                (
                    accumulated_parts,
                    pending_rows,
                ) = (
                    _save_final400_checkpoint(
                        accumulated_parts=
                            accumulated_parts,

                        pending_rows=
                            pending_rows,

                        processed=
                            processed,
                    )
                )

        if pending_rows:

            accumulated_parts.append(
                pd.concat(
                    pending_rows,
                    ignore_index=True,
                )
            )

    if not accumulated_parts:

        raise ValueError(
            "Final400 feature가 생성되지 않았습니다."
        )

    feature_df = pd.concat(
        accumulated_parts,
        ignore_index=True,
    )

    _validate_final400_feature_cache(
        feature_df,
        target_user_ids,
    )

    feature_df.to_parquet(
        FINAL400_FEATURE_CACHE_PATH,
        index=False,
    )

    if FINAL400_FEATURE_CHECKPOINT_PATH.exists():

        FINAL400_FEATURE_CHECKPOINT_PATH.unlink()

    if FINAL400_PROCESSED_USERS_PATH.exists():

        FINAL400_PROCESSED_USERS_PATH.unlink()

    print(
        "\n===== Final400 Feature Cache Saved ====="
    )

    print(
        FINAL400_FEATURE_CACHE_PATH
    )

    print(
        "Feature DF:",
        feature_df.shape,
    )

    return feature_df



# ============================================================
# Offline Evaluation
# ============================================================

def build_eval_users(
    train_df,
    test_df,
):

    """
    기존 Hybrid 실험과 동일한 고정 400명
    (hybrid_sampled_users.csv)을 그대로 사용한다.

    4개 review_group × 100명 = 400명.
    새로 랜덤 샘플링하지 않는다.
    """

    print(
        "\n===== Fixed Original 400 Evaluation Users ====="
    )

    if not FINAL400_USERS_PATH.exists():

        raise FileNotFoundError(
            (
                "기존 고정 400명 파일이 없습니다. "
                "다른 400명을 새로 샘플링하지 않습니다.\n"
                f"Expected: {FINAL400_USERS_PATH}"
            )
        )

    sampled = pd.read_csv(
        FINAL400_USERS_PATH
    )

    required = {
        "user_id",
        "review_group",
    }

    missing = (
        required
        -
        set(
            sampled.columns
        )
    )

    if missing:

        raise ValueError(
            (
                "Final400 파일에 필요한 column이 없습니다: "
                f"{sorted(missing)}"
            )
        )

    sampled = (
        sampled[
            [
                "user_id",
                "review_group",
            ]
        ]
        .drop_duplicates(
            subset=[
                "user_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        sampled
    ) != 400:

        raise ValueError(
            (
                "고정 Final400 user 수가 400명이 아닙니다. "
                f"현재={len(sampled)}"
            )
        )

    expected_groups = {
        "10-15개": 100,
        "16-25개": 100,
        "26-45개": 100,
        "46-78개": 100,
    }

    actual_groups = (
        sampled[
            "review_group"
        ]
        .value_counts()
        .to_dict()
    )

    for group_name, expected_count in expected_groups.items():

        actual_count = int(
            actual_groups.get(
                group_name,
                0,
            )
        )

        if actual_count != expected_count:

            raise ValueError(
                (
                    f"{group_name}: expected={expected_count}, "
                    f"actual={actual_count}"
                )
            )

    positive_test_users = set(
        test_df.loc[
            test_df[
                "is_recommended"
            ]
            ==
            True,
            "user_id",
        ]
        .drop_duplicates()
        .tolist()
    )

    no_positive = [
        user_id
        for user_id
        in sampled[
            "user_id"
        ].tolist()
        if user_id not in positive_test_users
    ]

    if no_positive:

        raise ValueError(
            (
                "Final400 중 positive test가 없는 사용자가 있습니다: "
                f"{len(no_positive)}명"
            )
        )

    print(
        sampled[
            "review_group"
        ]
        .value_counts()
        .reindex(
            [
                "10-15개",
                "16-25개",
                "26-45개",
                "46-78개",
            ]
        )
    )

    print(
        "Total:",
        len(
            sampled
        ),
    )

    print(
        "Evaluation: fixed original 400-user benchmark"
    )

    return sampled


def evaluate_engine(
    engine,
    train_df,
    test_df,
):

    sampled_users = (
        build_eval_users(
            train_df,
            test_df,
        )
    )

    target_user_ids = (
        sampled_users[
            "user_id"
        ]
        .tolist()
    )

    eval_ids = set(
        target_user_ids
    )

    # --------------------------------------------------------
    # Ground truth / metadata lookup
    # --------------------------------------------------------

    train_eval = (
        train_df[
            train_df[
                "user_id"
            ]
            .isin(
                eval_ids
            )
        ][
            [
                "user_id",
                "app_id",
            ]
        ]
        .copy()
    )

    test_eval = (
        test_df[
            (
                test_df[
                    "user_id"
                ]
                .isin(
                    eval_ids
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
        train_eval
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
        test_eval
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

    group_lookup = (
        sampled_users
        .set_index(
            "user_id"
        )[
            "review_group"
        ]
        .to_dict()
    )

    del train_eval
    del test_eval

    gc.collect()

    print(
        "\n============================================================"
    )

    print(
        " Final E_HR Full400 Quantitative Evaluation"
    )

    print(
        " BATCH / CACHE OPTIMIZED"
    )

    print(
        "============================================================"
    )

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # 1. Final400 candidate/feature build
    # --------------------------------------------------------

    feature_df = (
        build_or_load_final400_feature_cache(
            engine=
                engine,

            train_df=
                train_df,

            test_df=
                test_df,

            target_user_ids=
                target_user_ids,
        )
    )

    feature_seconds = (
        time.perf_counter()
        -
        total_start
    )

    print(
        "\nFeature stage time:",
        f"{feature_seconds / 60:.2f} min",
    )

    # --------------------------------------------------------
    # 2. XGBoost prediction - 전체 row 한 번에
    # --------------------------------------------------------

    print(
        "\n===== XGBoost Batch Predict ====="
    )

    predict_start = time.perf_counter()

    X = (
        feature_df[
            XGB_FEATURES
        ]
        .astype(
            np.float32
        )
    )

    predictions = (
        engine
        .xgb_model
        .predict(
            X
        )
    )

    ranked_df = (
        feature_df[
            [
                "user_id",
                "app_id",
            ]
        ]
        .copy()
    )

    ranked_df[
        "final_score"
    ] = predictions

    ranked_df = (
        ranked_df
        .sort_values(
            [
                "user_id",
                "final_score",
            ],
            ascending=[
                True,
                False,
            ],
            kind="stable",
        )
        .groupby(
            "user_id",
            sort=False,
            group_keys=False,
        )
        .head(
            TOP_N
        )
        .copy()
    )

    predict_seconds = (
        time.perf_counter()
        -
        predict_start
    )

    print(
        "Rows:",
        f"{len(feature_df):,}",
    )

    print(
        "Predict time:",
        f"{predict_seconds:.2f} sec",
    )

    # --------------------------------------------------------
    # 3. User별 Top-10 dictionary
    # --------------------------------------------------------

    recommendation_dict = (
        ranked_df
        .groupby(
            "user_id",
            sort=False,
        )[
            "app_id"
        ]
        .apply(
            list
        )
        .to_dict()
    )

    candidate_dict = (
        feature_df
        .groupby(
            "user_id",
            sort=False,
        )[
            "app_id"
        ]
        .apply(
            list
        )
        .to_dict()
    )

    # --------------------------------------------------------
    # 4. Metrics
    # --------------------------------------------------------

    rows = []

    for count, user_id in enumerate(
        target_user_ids,
        start=1,
    ):

        played = train_history.get(
            user_id,
            [],
        )

        relevant = positive_test.get(
            user_id,
            [],
        )

        recommended = recommendation_dict.get(
            user_id,
            [],
        )

        candidates = candidate_dict.get(
            user_id,
            [],
        )

        relevant_set = set(
            relevant
        )

        hits = sum(
            app_id
            in
            relevant_set

            for app_id
            in recommended[
                :TOP_N
            ]
        )

        candidate_hits = sum(
            app_id
            in
            relevant_set

            for app_id
            in candidates
        )

        candidate_pool_recall = (
            candidate_hits
            /
            len(
                relevant_set
            )

            if relevant_set

            else 0.0
        )

        popularity_metrics = (
            recommendation_popularity_metrics(
                recommended=
                    recommended,

                item_popularity=
                    engine.item_popularity,

                total_interactions=
                    engine.total_interactions,

                k=
                    TOP_N,
            )
        )

        rows.append(
            {
                "user_id":
                    user_id,

                "review_group":
                    group_lookup.get(
                        user_id
                    ),

                "n_played":
                    len(
                        played
                    ),

                "n_relevant":
                    len(
                        relevant
                    ),

                "n_candidates":
                    len(
                        candidates
                    ),

                "n_recommended":
                    len(
                        recommended
                    ),

                "candidate_hits":
                    candidate_hits,

                "candidate_pool_recall":
                    candidate_pool_recall,

                "hits":
                    hits,

                "precision_at_10":
                    precision_at_k(
                        recommended,
                        relevant,
                        TOP_N,
                    ),

                "recall_at_10":
                    recall_at_k(
                        recommended,
                        relevant,
                        TOP_N,
                    ),

                "hit_rate_at_10":
                    hit_rate_at_k(
                        recommended,
                        relevant,
                        TOP_N,
                    ),

                "ndcg_at_10":
                    ndcg_at_k(
                        recommended,
                        relevant,
                        TOP_N,
                    ),

                "ap_at_10":
                    average_precision_at_k(
                        recommended,
                        relevant,
                        TOP_N,
                    ),

                "mean_log_popularity_at_10":
                    popularity_metrics[
                        "mean_log_popularity_at_k"
                    ],

                "novelty_at_10":
                    popularity_metrics[
                        "novelty_at_k"
                    ],
            }
        )

        if (
            count
            %
            100
            ==
            0
            or
            count
            ==
            len(
                target_user_ids
            )
        ):

            print(
                f"Metric calculation: "
                f"{count}/{len(target_user_ids)}"
            )

    eval_df = pd.DataFrame(
        rows
    )

    if len(
        eval_df
    ) != 400:

        raise ValueError(
            (
                "Final400 결과 user 수가 400명이 아닙니다. "
                f"actual={len(eval_df)}"
            )
        )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    eval_df.to_csv(
        INTEGRATED_EVAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    metric_columns = [
        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",
        "ap_at_10",
        "candidate_pool_recall",
        "mean_log_popularity_at_10",
        "novelty_at_10",
    ]

    summary = {
        "model":
            FINAL_MODEL_NAME,

        "candidate_size":
            CANDIDATE_SIZE,

        "item_n":
            ITEM_N,

        "bpr_n":
            BPR_N,

        "content_n":
            CONTENT_N,

        "user_n":
            USER_N,

        "feature_count":
            len(
                XGB_FEATURES
            ),

        "n_users":
            len(
                eval_df
            ),

        "precision_at_10":
            eval_df[
                "precision_at_10"
            ].mean(),

        "recall_at_10":
            eval_df[
                "recall_at_10"
            ].mean(),

        "hit_rate_at_10":
            eval_df[
                "hit_rate_at_10"
            ].mean(),

        "ndcg_at_10":
            eval_df[
                "ndcg_at_10"
            ].mean(),

        "map_at_10":
            eval_df[
                "ap_at_10"
            ].mean(),

        "candidate_pool_recall":
            eval_df[
                "candidate_pool_recall"
            ].mean(),

        "mean_log_popularity_at_10":
            eval_df[
                "mean_log_popularity_at_10"
            ].mean(),

        "novelty_at_10":
            eval_df[
                "novelty_at_10"
            ].mean(),

        "hits":
            int(
                eval_df[
                    "hits"
                ].sum()
            ),

        "feature_build_minutes":
            feature_seconds
            /
            60.0,

        "xgb_predict_seconds":
            predict_seconds,
    }

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    summary_df.to_csv(
        INTEGRATED_EVAL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    group_summary = (
        eval_df
        .groupby(
            "review_group",
            observed=True,
        )[
            metric_columns
        ]
        .mean()
    )

    total_seconds = (
        time.perf_counter()
        -
        total_start
    )

    print(
        "\n============================================================"
    )

    print(
        " Final400 Quantitative Evaluation Summary"
    )

    print(
        "============================================================"
    )

    print(
        f"Model                    : "
        f"{FINAL_MODEL_NAME}"
    )

    print(
        f"Candidate                : "
        f"I{ITEM_N} + B{BPR_N} + C{CONTENT_N} + U{USER_N} "
        f"(max {CANDIDATE_SIZE})"
    )

    print(
        f"Features                 : "
        f"{len(XGB_FEATURES)}"
    )

    print(
        f"Users                    : "
        f"{summary['n_users']}"
    )

    print(
        f"Precision@10             : "
        f"{summary['precision_at_10']:.6f}"
    )

    print(
        f"Recall@10                : "
        f"{summary['recall_at_10']:.6f}"
    )

    print(
        f"Hit Rate@10              : "
        f"{summary['hit_rate_at_10']:.6f}"
    )

    print(
        f"NDCG@10                  : "
        f"{summary['ndcg_at_10']:.6f}"
    )

    print(
        f"MAP@10                   : "
        f"{summary['map_at_10']:.6f}"
    )

    print(
        f"Candidate Recall         : "
        f"{summary['candidate_pool_recall']:.6f}"
    )

    print(
        f"Mean Log Popularity@10   : "
        f"{summary['mean_log_popularity_at_10']:.6f}"
    )

    print(
        f"Novelty@10               : "
        f"{summary['novelty_at_10']:.6f}"
    )

    print(
        f"Hits                     : "
        f"{summary['hits']}"
    )

    print(
        f"Total runtime            : "
        f"{total_seconds / 60:.2f} min"
    )

    print(
        "\n===== Group Summary ====="
    )

    print(
        group_summary
        .to_string()
    )

    print(
        "\nSaved:"
    )

    print(
        INTEGRATED_EVAL_PATH
    )

    print(
        INTEGRATED_EVAL_SUMMARY_PATH
    )

    print(
        FINAL400_FEATURE_CACHE_PATH
    )

    return (
        eval_df,
        summary_df,
    )




# ============================================================
# Console Main
# ============================================================

def main():

    (
        engine,
        meta,
        train_df,
        test_df,
    ) = build_engine()

    # ========================================================
    # Offline Evaluation Mode
    # ========================================================

    if (
        RUN_MODE
        ==
        "evaluate"
    ):

        evaluate_engine(
            engine=engine,
            train_df=train_df,
            test_df=test_df,
        )

        return

    # ========================================================
    # Recommendation Mode
    # ========================================================

    if (
        RUN_MODE
        !=
        "recommend"
    ):

        raise ValueError(
            "RUN_MODE은 'recommend' 또는 'evaluate'여야 합니다."
        )

    print(
        "\n플레이한 게임 이름 또는 app_id를 쉼표(,)로 입력하세요."
    )

    print(
        "예: Hades, Portal 2, 730"
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
            recommended=result[
                "app_id"
            ].tolist(),
            item_popularity=engine.item_popularity,
            total_interactions=engine.total_interactions,
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
        "offline evaluation에서 계산"
    )


if __name__ == "__main__":
    main()
