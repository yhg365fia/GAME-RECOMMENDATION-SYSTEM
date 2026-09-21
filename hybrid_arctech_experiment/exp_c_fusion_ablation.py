import sys
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.cpu.bpr import BayesianPersonalizedRanking
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# =========================================================
# Project Root / Imports
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_split import load_mf_split
from evaluation import run_mf_evaluation, print_evaluation_report
from preprocessing import load_games

try:
    from models.itembase import build_interaction_matrix, ItemBasedCFRecommender
except ImportError:
    # 예전 구조 호환: build_interaction_matrix가 userbase.py에 있었던 경우
    from models.itembase import ItemBasedCFRecommender
    from models.userbase import build_interaction_matrix

from models.userbase import UserBasedCFRecommender


# =========================================================
# Config
# =========================================================

MODEL_TOP_N = 100
TOP_N = 10

ITEM_K = 30
USER_K = 30

# ---------------------------------------------------------
# 4-Model Rank Fusion 초기 가중치
# ---------------------------------------------------------
# Item-Based를 중심 모델로 유지한다.
# 나머지 세 모델은 서로 다른 신호를 보완하는 역할.
#
# 이 값은 "최적값"이 아니라 Case 3 architecture 검증용 초기값이다.
# Retrieval cache가 생성된 뒤에는 아래 숫자만 바꿔 매우 빠르게 재평가 가능.
ITEM_WEIGHT = 0.50
USER_WEIGHT = 0.15
BPR_WEIGHT = 0.20
CONTENT_WEIGHT = 0.15

# True로 바꾸면 해당 cache를 무시하고 다시 생성
FORCE_REBUILD_RETRIEVAL_CACHE = False
FORCE_REBUILD_CONTENT_TFIDF = False
FORCE_REBUILD_EVAL_SUBSET = False

# 기존 Item-Based Top-200 cache가 없을 때 새로 만들 후보 수
ITEM_CACHE_TOP_N = 200

# 최종 BPR 모델명
BPR_MODEL_NAME = "bpr_iter15_factors60_reg0.006_explicit_false.npz"


# =========================================================
# Paths
# =========================================================

SAVED_MODEL_DIR = ROOT / "models" / "saved_model"
RESULT_DIR = SAVED_MODEL_DIR / "results"
CACHE_DIR = SAVED_MODEL_DIR / "case3_cache"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLED_USERS_PATH = RESULT_DIR / "hybrid_sampled_users.csv"
BASELINE_EVAL_PATH = RESULT_DIR / "hybrid_baseline_item_eval.csv"

# Day22에서 이미 생성하기로 한 canonical Item-Based Top-200 cache
ITEM_TOP200_PATH = RESULT_DIR / "item_top200_candidates.csv"

# Case 3 retrieval cache
ITEM_CACHE_PATH = CACHE_DIR / f"case3_item_top{MODEL_TOP_N}.parquet"
USER_CACHE_PATH = CACHE_DIR / f"case3_user_top{MODEL_TOP_N}.parquet"
BPR_CACHE_PATH = CACHE_DIR / f"case3_bpr_top{MODEL_TOP_N}.parquet"
CONTENT_CACHE_PATH = CACHE_DIR / f"case3_content_top{MODEL_TOP_N}.parquet"

# 평가용 400명 subset cache
EVAL_TRAIN_PATH = CACHE_DIR / "case3_eval_train_400.parquet"
EVAL_TEST_PATH = CACHE_DIR / "case3_eval_test_400.parquet"

# Content TF-IDF cache
CONTENT_TFIDF_PATH = CACHE_DIR / "case3_content_tfidf.npz"
CONTENT_APP_IDS_PATH = CACHE_DIR / "case3_content_app_ids.npy"

# 최종 결과
EVAL_PATH = RESULT_DIR / "hybrid_exp_c_4model_fusion_eval.csv"
SUMMARY_PATH = RESULT_DIR / "hybrid_exp_c_4model_fusion_summary.csv"
GROUP_PATH = RESULT_DIR / "hybrid_exp_c_4model_fusion_group_summary.csv"
COMPARE_PATH = RESULT_DIR / "hybrid_exp_c_4model_fusion_vs_baseline.csv"
DIAGNOSTIC_PATH = RESULT_DIR / "hybrid_exp_c_4model_fusion_diagnostics.csv"


# =========================================================
# Generic Utility
# =========================================================

def find_file(base_dir: Path, filename: str):
    matches = list(base_dir.rglob(filename))

    if not matches:
        return None

    if len(matches) > 1:
        print(f"\n주의: {filename} 파일이 여러 개 발견됨")
        for path in matches:
            print(" -", path)
        print("첫 번째 파일을 사용합니다.")

    return matches[0]


def find_sorted_index(sorted_ids, value):
    idx = int(np.searchsorted(sorted_ids, value))

    if idx >= len(sorted_ids):
        return None

    if sorted_ids[idx] != value:
        return None

    return idx


def save_retrieval_cache(rows, path: Path):
    df = pd.DataFrame(rows, columns=["user_id", "rank", "app_id"])
    df.to_parquet(path, index=False)
    return df


def load_retrieval_cache(path: Path, top_n=MODEL_TOP_N):
    df = pd.read_parquet(path)

    required = {"user_id", "rank", "app_id"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{path.name}에 필요한 컬럼이 없습니다: {missing}")

    df = df[df["rank"] <= top_n].copy()
    df = df.sort_values(["user_id", "rank"], kind="stable")
    return df


def cache_to_dict(df):
    """user_id -> rank 순서의 app_id list"""
    return (
        df.sort_values(["user_id", "rank"], kind="stable")
        .groupby("user_id", sort=False)["app_id"]
        .apply(list)
        .to_dict()
    )


def print_cache_coverage(name, df, sampled_users):
    n_users = df["user_id"].nunique()
    total_users = len(sampled_users)
    avg_n = df.groupby("user_id").size().mean() if len(df) else 0.0

    print(
        f"{name}: users={n_users}/{total_users}, "
        f"rows={len(df):,}, avg_items={avg_n:.1f}"
    )


# =========================================================
# Sampled Users / Evaluation Subset
# =========================================================

def load_sampled_users():
    if not SAMPLED_USERS_PATH.exists():
        raise FileNotFoundError(
            f"기존 Hybrid 평가 사용자 파일이 없습니다:\n{SAMPLED_USERS_PATH}"
        )

    sampled_users = pd.read_csv(SAMPLED_USERS_PATH)

    print("\n===== 평가 사용자 =====")
    print("평가 사용자 수:", len(sampled_users))
    print(sampled_users["review_group"].value_counts())

    return sampled_users


def load_or_create_eval_subset(sampled_users, full_split_holder):
    """
    이후 실행에서 37M train 전체를 다시 불러오지 않도록
    평가 대상 400명의 train/test만 별도 저장한다.

    full_split_holder는 dict로 넘겨서 full split을 한 번만 로드한다.
    """

    if (
        EVAL_TRAIN_PATH.exists()
        and EVAL_TEST_PATH.exists()
        and not FORCE_REBUILD_EVAL_SUBSET
    ):
        print("\n===== 평가 subset cache 로드 =====")
        train_eval = pd.read_parquet(EVAL_TRAIN_PATH)
        test_eval = pd.read_parquet(EVAL_TEST_PATH)

        print("Train subset:", train_eval.shape)
        print("Test subset :", test_eval.shape)
        return train_eval, test_eval

    print("\n===== 평가 subset cache 생성 =====")

    if "train" not in full_split_holder:
        train_df, test_df = load_mf_split()
        full_split_holder["train"] = train_df
        full_split_holder["test"] = test_df

    train_df = full_split_holder["train"]
    test_df = full_split_holder["test"]

    sampled_ids = sampled_users["user_id"].to_numpy()

    train_eval = train_df[train_df["user_id"].isin(sampled_ids)].copy()
    test_eval = test_df[test_df["user_id"].isin(sampled_ids)].copy()

    train_eval.to_parquet(EVAL_TRAIN_PATH, index=False)
    test_eval.to_parquet(EVAL_TEST_PATH, index=False)

    print("Train subset:", train_eval.shape)
    print("Test subset :", test_eval.shape)
    print("저장:", EVAL_TRAIN_PATH)
    print("저장:", EVAL_TEST_PATH)

    return train_eval, test_eval


# =========================================================
# Shared Memory-Based CF Objects
# =========================================================

def get_or_build_cf_objects(full_split_holder, cf_holder):
    """
    User-Based와 Item-Based가 동일한 Train interaction matrix를 공유한다.

    - 둘 다 새로 계산해야 하는 경우 matrix를 2번 만들지 않음.
    - 기존 Item Top-200 cache가 있더라도 User-Based cache 최초 생성 시
      global MF Train으로 matrix를 한 번 생성한다.
    """

    if "interaction_matrix" in cf_holder:
        return (
            cf_holder["interaction_matrix"],
            cf_holder["user_to_idx"],
            cf_holder["game_to_idx"],
            cf_holder["idx_to_game"],
        )

    if "train" not in full_split_holder:
        train_df, test_df = load_mf_split()
        full_split_holder["train"] = train_df
        full_split_holder["test"] = test_df

    print("\n===== Shared User×Item Interaction Matrix 생성 =====")
    start = time.time()

    (
        interaction_matrix,
        user_to_idx,
        game_to_idx,
        idx_to_game,
    ) = build_interaction_matrix(full_split_holder["train"])

    cf_holder["interaction_matrix"] = interaction_matrix
    cf_holder["user_to_idx"] = user_to_idx
    cf_holder["game_to_idx"] = game_to_idx
    cf_holder["idx_to_game"] = idx_to_game

    print("Interaction Matrix:", interaction_matrix.shape)
    print(f"생성 시간: {time.time() - start:.1f}초")

    return interaction_matrix, user_to_idx, game_to_idx, idx_to_game


# =========================================================
# Item-Based Cache
# =========================================================

def load_existing_item_top200():
    """
    기존 Day22 cache:
        user_id | rank | app_id

    를 우선 사용한다.
    """

    if not ITEM_TOP200_PATH.exists():
        return None

    print("\n기존 Item-Based Top-200 cache 사용:")
    print(ITEM_TOP200_PATH)

    df = pd.read_csv(ITEM_TOP200_PATH)

    required = {"user_id", "rank", "app_id"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"item_top200_candidates.csv 컬럼 구조가 예상과 다릅니다: {missing}"
        )

    return (
        df[df["rank"] <= MODEL_TOP_N]
        .sort_values(["user_id", "rank"], kind="stable")
        .copy()
    )


def build_item_cache(sampled_users, train_eval, full_split_holder, cf_holder):
    """
    1순위: 기존 item_top200_candidates.csv 재사용
    2순위: Case3 item cache 재사용
    3순위: 정말 없을 때만 기존 ItemBasedCFRecommender를 호출해 생성

    Item-Based 알고리즘 자체는 건드리지 않아 baseline과 동일성을 보존한다.
    """

    if not FORCE_REBUILD_RETRIEVAL_CACHE:
        existing_top200 = load_existing_item_top200()

        if existing_top200 is not None:
            existing_top200.to_parquet(ITEM_CACHE_PATH, index=False)
            return existing_top200

        if ITEM_CACHE_PATH.exists():
            print("\nCase3 Item cache 로드:", ITEM_CACHE_PATH)
            return load_retrieval_cache(ITEM_CACHE_PATH)

    print("\n===== Item-Based cache 새로 생성 =====")
    print("기존 Top-200 cache가 없으므로 Item-Based 추천을 1회 수행합니다.")

    start = time.time()

    interaction_matrix, user_to_idx, game_to_idx, idx_to_game = (
        get_or_build_cf_objects(
            full_split_holder=full_split_holder,
            cf_holder=cf_holder,
        )
    )

    item_model = ItemBasedCFRecommender(
        interaction_matrix=interaction_matrix,
        game_to_idx=game_to_idx,
        idx_to_game=idx_to_game,
        k=ITEM_K,
    )

    history = train_eval.groupby("user_id")["app_id"].apply(list).to_dict()

    rows = []

    for i, user_id in enumerate(sampled_users["user_id"], start=1):
        app_ids = history.get(user_id, [])

        result = item_model.recommend(
            app_id_list=app_ids,
            top_n=ITEM_CACHE_TOP_N,
        )

        if result is None or len(result) == 0:
            continue

        for rank, app_id in enumerate(result["app_id"].tolist(), start=1):
            rows.append((user_id, rank, app_id))

        if i % 25 == 0 or i == len(sampled_users):
            print(f"Item-Based: {i}/{len(sampled_users)} users")

    full_item_df = pd.DataFrame(rows, columns=["user_id", "rank", "app_id"])

    # 앞으로 다른 실험에서도 쓸 canonical Top-200 cache
    full_item_df.to_csv(ITEM_TOP200_PATH, index=False)

    item_df = full_item_df[full_item_df["rank"] <= MODEL_TOP_N].copy()
    item_df.to_parquet(ITEM_CACHE_PATH, index=False)

    print(f"Item cache 생성 시간: {time.time() - start:.1f}초")
    print("저장:", ITEM_TOP200_PATH)

    return item_df


# =========================================================
# User-Based Cache
# =========================================================

def build_user_cache(
    sampled_users,
    train_eval,
    full_split_holder,
    cf_holder,
):
    """
    User-Based CF Top-100을 최초 1회만 계산해서 저장한다.

    중요:
    - interaction matrix는 Item-Based와 공유
    - query는 평가 사용자의 Train app_id만 사용
    - exclude_user_idx를 반드시 전달해 자기 자신을 neighbor에서 제외
    - 일부 sparse user는 후보가 부족해 Top-100보다 적게 반환될 수 있음
    """

    if USER_CACHE_PATH.exists() and not FORCE_REBUILD_RETRIEVAL_CACHE:
        print("\nCase3 User-Based cache 로드:", USER_CACHE_PATH)
        return load_retrieval_cache(USER_CACHE_PATH)

    print("\n===== User-Based cache 생성 =====")
    start = time.time()

    (
        interaction_matrix,
        user_to_idx,
        game_to_idx,
        idx_to_game,
    ) = get_or_build_cf_objects(
        full_split_holder=full_split_holder,
        cf_holder=cf_holder,
    )

    # 기존 models/userbase.py API와 맞춤
    user_model = UserBasedCFRecommender(
        interaction_matrix,
        game_to_idx,
        idx_to_game,
        k=USER_K,
    )

    history = (
        train_eval
        .groupby("user_id")["app_id"]
        .apply(list)
        .to_dict()
    )

    rows = []
    missing_matrix_users = 0
    empty_recommendations = 0

    for i, user_id in enumerate(sampled_users["user_id"], start=1):
        app_ids = history.get(user_id, [])

        if not app_ids:
            empty_recommendations += 1
            continue

        user_idx = user_to_idx.get(user_id)

        if user_idx is None:
            missing_matrix_users += 1
            continue

        result = user_model.recommend(
            app_id_list=app_ids,
            top_n=MODEL_TOP_N,
            exclude_user_idx=user_idx,
        )

        if result is None or len(result) == 0:
            empty_recommendations += 1
            continue

        for rank, app_id in enumerate(
            result["app_id"].tolist()[:MODEL_TOP_N],
            start=1,
        ):
            rows.append((user_id, rank, app_id))

        if i % 25 == 0 or i == len(sampled_users):
            print(f"User-Based: {i}/{len(sampled_users)} users")

    user_df = save_retrieval_cache(rows, USER_CACHE_PATH)

    print("Matrix mapping 없는 사용자:", missing_matrix_users)
    print("추천 결과 없는 사용자:", empty_recommendations)
    print(f"User-Based cache 생성 시간: {time.time() - start:.1f}초")
    print("저장:", USER_CACHE_PATH)

    return user_df


# =========================================================
# BPR Cache
# =========================================================

def find_bpr_files():
    model_path = find_file(SAVED_MODEL_DIR, BPR_MODEL_NAME)

    if model_path is None:
        raise FileNotFoundError(
            f"BPR 모델을 찾을 수 없습니다.\n"
            f"검색 위치: {SAVED_MODEL_DIR}\n"
            f"파일명: {BPR_MODEL_NAME}"
        )

    mapping_path = find_file(SAVED_MODEL_DIR, "implicit_bpr_mapping.npz")

    if mapping_path is None:
        mapping_path = find_file(ROOT / "models", "implicit_bpr_mapping.npz")

    if mapping_path is None:
        raise FileNotFoundError("implicit_bpr_mapping.npz를 찾을 수 없습니다.")

    user_items_path = find_file(SAVED_MODEL_DIR, "implicit_bpr_user_items.npz")

    if user_items_path is None:
        user_items_path = find_file(ROOT / "models", "implicit_bpr_user_items.npz")

    return model_path, mapping_path, user_items_path


def load_bpr_mapping_only():
    _, mapping_path, _ = find_bpr_files()
    mapping = np.load(mapping_path)
    return np.asarray(mapping["user_ids"]), np.asarray(mapping["item_ids"])


def build_bpr_cache(sampled_users, train_eval):
    if BPR_CACHE_PATH.exists() and not FORCE_REBUILD_RETRIEVAL_CACHE:
        print("\nCase3 BPR cache 로드:", BPR_CACHE_PATH)
        return load_retrieval_cache(BPR_CACHE_PATH)

    print("\n===== BPR cache 생성 =====")
    start = time.time()

    model_path, mapping_path, user_items_path = find_bpr_files()

    print("BPR Model  :", model_path)
    print("BPR Mapping:", mapping_path)
    print("User Matrix:", user_items_path if user_items_path else "없음 -> fallback")

    model = BayesianPersonalizedRanking.load(str(model_path))

    mapping = np.load(mapping_path)
    user_ids = np.asarray(mapping["user_ids"])
    item_ids = np.asarray(mapping["item_ids"])

    if model.user_factors.shape[0] != len(user_ids):
        raise ValueError("BPR user_factors와 mapping user 수가 다릅니다.")

    if model.item_factors.shape[0] != len(item_ids):
        raise ValueError("BPR item_factors와 mapping item 수가 다릅니다.")

    if not np.all(user_ids[:-1] <= user_ids[1:]):
        raise ValueError("BPR user_ids가 정렬되어 있지 않습니다.")

    if not np.all(item_ids[:-1] <= item_ids[1:]):
        raise ValueError("BPR item_ids가 정렬되어 있지 않습니다.")

    user_items = load_npz(user_items_path).tocsr() if user_items_path else None

    # fallback에서 이미 본 item 제거용
    seen_dict = train_eval.groupby("user_id")["app_id"].apply(list).to_dict()

    rows = []
    missing_users = 0

    for i, user_id in enumerate(sampled_users["user_id"], start=1):
        user_idx = find_sorted_index(user_ids, user_id)

        if user_idx is None:
            missing_users += 1
            continue

        if user_items is not None:
            item_indices, scores = model.recommend(
                userid=int(user_idx),
                user_items=user_items[user_idx],
                N=MODEL_TOP_N,
                filter_already_liked_items=True,
            )

            app_ids = item_ids[item_indices]

        else:
            # user_items가 없을 때만 numpy dot-product fallback
            user_factor = model.user_factors[user_idx]
            scores = (model.item_factors @ user_factor).astype(np.float32, copy=False)

            seen_ids = np.asarray(seen_dict.get(user_id, []))

            if len(seen_ids):
                seen_idx = np.searchsorted(item_ids, seen_ids)
                valid = seen_idx < len(item_ids)

                positions = np.where(valid)[0]

                if len(positions):
                    matched = item_ids[seen_idx[positions]] == seen_ids[positions]
                    real_seen = seen_idx[positions[matched]]
                    scores[real_seen] = -np.inf

            finite_idx = np.where(np.isfinite(scores))[0]
            n_actual = min(MODEL_TOP_N, len(finite_idx))

            if n_actual == 0:
                continue

            if n_actual < len(finite_idx):
                local = np.argpartition(-scores[finite_idx], n_actual - 1)[:n_actual]
                item_indices = finite_idx[local]
            else:
                item_indices = finite_idx

            order = np.argsort(-scores[item_indices], kind="stable")
            item_indices = item_indices[order]
            app_ids = item_ids[item_indices]

        for rank, app_id in enumerate(app_ids[:MODEL_TOP_N], start=1):
            rows.append((user_id, rank, app_id))

        if i % 100 == 0 or i == len(sampled_users):
            print(f"BPR: {i}/{len(sampled_users)} users")

    bpr_df = save_retrieval_cache(rows, BPR_CACHE_PATH)

    print("BPR mapping 없는 사용자:", missing_users)
    print(f"BPR cache 생성 시간: {time.time() - start:.1f}초")
    print("저장:", BPR_CACHE_PATH)

    return bpr_df


# =========================================================
# Content TF-IDF / Cache
# =========================================================

def load_games_meta():
    meta = load_games()

    if "app_id" not in meta.columns and "AppID" in meta.columns:
        meta = meta.rename(columns={"AppID": "app_id"})

    if "app_id" not in meta.columns:
        raise ValueError("games metadata에 app_id/AppID 컬럼이 없습니다.")

    return meta.drop_duplicates("app_id").reset_index(drop=True)


def build_or_load_content_matrix(valid_app_ids):
    valid_app_ids = np.asarray(valid_app_ids)

    if (
        CONTENT_TFIDF_PATH.exists()
        and CONTENT_APP_IDS_PATH.exists()
        and not FORCE_REBUILD_CONTENT_TFIDF
    ):
        cached_app_ids = np.load(CONTENT_APP_IDS_PATH)

        # mapping이 바뀌지 않았을 때만 cache 사용
        if np.array_equal(cached_app_ids, valid_app_ids):
            print("\nContent TF-IDF cache 로드")
            print(CONTENT_TFIDF_PATH)
            matrix = load_npz(CONTENT_TFIDF_PATH).tocsr()
            return matrix, cached_app_ids

        print("\nContent item mapping이 바뀌어 TF-IDF를 재생성합니다.")

    print("\n===== Content TF-IDF 생성 =====")
    start = time.time()

    meta = load_games_meta()
    valid_set = set(valid_app_ids.tolist())

    meta = (
        meta[meta["app_id"].isin(valid_set)]
        .drop_duplicates("app_id")
        .copy()
    )

    # BPR item mapping 순서와 동일하게 맞춤
    order_df = pd.DataFrame({"app_id": valid_app_ids, "_order": np.arange(len(valid_app_ids))})
    meta = order_df.merge(meta, on="app_id", how="left").sort_values("_order")

    candidate_features = [
        "Genres",
        "Tags",
        "Categories",
        "Developers",
        "Publishers",
    ]

    feature_columns = [c for c in candidate_features if c in meta.columns]

    if not feature_columns:
        raise ValueError(
            "Content feature가 없습니다. "
            "Genres / Tags / Categories / Developers / Publishers 확인 필요"
        )

    print("Content Features:", feature_columns)

    combined = pd.Series("", index=meta.index, dtype="object")

    for col in feature_columns:
        combined = combined + " " + meta[col].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=2,
        dtype=np.float32,
        norm="l2",
    )

    matrix = vectorizer.fit_transform(combined.str.lower()).tocsr()

    save_npz(CONTENT_TFIDF_PATH, matrix)
    np.save(CONTENT_APP_IDS_PATH, valid_app_ids)

    print("TF-IDF Matrix:", matrix.shape)
    print(f"생성 시간: {time.time() - start:.1f}초")
    print("저장:", CONTENT_TFIDF_PATH)

    return matrix, valid_app_ids


def build_content_cache(sampled_users, train_eval, item_ids):
    if CONTENT_CACHE_PATH.exists() and not FORCE_REBUILD_RETRIEVAL_CACHE:
        print("\nCase3 Content cache 로드:", CONTENT_CACHE_PATH)
        return load_retrieval_cache(CONTENT_CACHE_PATH)

    print("\n===== Content cache 생성 =====")
    start = time.time()

    tfidf_matrix, content_app_ids = build_or_load_content_matrix(item_ids)
    app_to_idx = {app_id: idx for idx, app_id in enumerate(content_app_ids)}

    # 핵심 최적화: 전체 1,300만 명이 아니라 평가 대상 400명만 history 생성
    positive_train = train_eval[train_eval["is_recommended"] == True]
    positive_history = positive_train.groupby("user_id")["app_id"].apply(list).to_dict()
    seen_history = train_eval.groupby("user_id")["app_id"].apply(list).to_dict()

    user_order = sampled_users["user_id"].tolist()

    # -----------------------------------------------------
    # 400명의 content profile을 한 번에 sparse matrix로 생성
    # -----------------------------------------------------
    profile_rows = []

    for user_id in user_order:
        source_indices = [
            app_to_idx[app_id]
            for app_id in positive_history.get(user_id, [])
            if app_id in app_to_idx
        ]

        if not source_indices:
            profile_rows.append(sp.csr_matrix((1, tfidf_matrix.shape[1]), dtype=np.float32))
            continue

        # sparse sum -> 평균 profile
        profile = sp.csr_matrix(tfidf_matrix[source_indices].sum(axis=0), dtype=np.float32)
        profile *= (1.0 / len(source_indices))
        profile_rows.append(profile)

    profile_matrix = sp.vstack(profile_rows, format="csr")
    profile_matrix = normalize(profile_matrix, norm="l2", axis=1, copy=False)

    # 400 x Items cosine similarity. sparse @ sparse로 계산
    score_matrix = (profile_matrix @ tfidf_matrix.T).tocsr()

    rows = []

    for row_idx, user_id in enumerate(user_order):
        score_row = score_matrix.getrow(row_idx)

        if score_row.nnz == 0:
            continue

        candidate_indices = score_row.indices
        candidate_scores = score_row.data

        # 이미 본 게임 제외
        seen_ids = set(seen_history.get(user_id, []))

        keep_mask = np.fromiter(
            (content_app_ids[idx] not in seen_ids for idx in candidate_indices),
            dtype=bool,
            count=len(candidate_indices),
        )

        candidate_indices = candidate_indices[keep_mask]
        candidate_scores = candidate_scores[keep_mask]

        positive_mask = candidate_scores > 0
        candidate_indices = candidate_indices[positive_mask]
        candidate_scores = candidate_scores[positive_mask]

        if len(candidate_indices) == 0:
            continue

        n_actual = min(MODEL_TOP_N, len(candidate_indices))

        if n_actual < len(candidate_indices):
            local = np.argpartition(-candidate_scores, n_actual - 1)[:n_actual]
            top_indices = candidate_indices[local]
            top_scores = candidate_scores[local]
        else:
            top_indices = candidate_indices
            top_scores = candidate_scores

        order = np.argsort(-top_scores, kind="stable")
        top_indices = top_indices[order]

        app_ids = content_app_ids[top_indices]

        for rank, app_id in enumerate(app_ids, start=1):
            rows.append((user_id, rank, app_id))

    content_df = save_retrieval_cache(rows, CONTENT_CACHE_PATH)

    print(f"Content cache 생성 시간: {time.time() - start:.1f}초")
    print("저장:", CONTENT_CACHE_PATH)

    return content_df


# =========================================================
# Cached Rank Fusion Recommender
# =========================================================

class CachedRankFusionRecommender:
    """
    4개의 cached Top-N rank만 결합한다.

    Item-Based + User-Based + BPR + Content-Based
        ↓
    Weighted Rank Fusion
        ↓
    Final Top-10

    rank_score = (MODEL_TOP_N - rank + 1) / MODEL_TOP_N

    final_score =
        ITEM_WEIGHT    * item_rank_score
        + USER_WEIGHT  * user_rank_score
        + BPR_WEIGHT   * bpr_rank_score
        + CONTENT_WEIGHT * content_rank_score
    """

    def __init__(
        self,
        item_cache,
        user_cache,
        bpr_cache,
        content_cache,
    ):
        self.item = cache_to_dict(item_cache)
        self.user = cache_to_dict(user_cache)
        self.bpr = cache_to_dict(bpr_cache)
        self.content = cache_to_dict(content_cache)
        self.logs = {}

    @staticmethod
    def rank_score(rank):
        return (MODEL_TOP_N - rank + 1) / MODEL_TOP_N

    def recommend(self, user_id, app_id_list=None, top_n=10):
        item_list = self.item.get(user_id, [])[:MODEL_TOP_N]
        user_list = self.user.get(user_id, [])[:MODEL_TOP_N]
        bpr_list = self.bpr.get(user_id, [])[:MODEL_TOP_N]
        content_list = self.content.get(user_id, [])[:MODEL_TOP_N]

        candidates = {}

        def ensure(app_id):
            if app_id not in candidates:
                candidates[app_id] = {
                    "app_id": app_id,
                    "item_rank": np.nan,
                    "user_rank": np.nan,
                    "bpr_rank": np.nan,
                    "content_rank": np.nan,
                    "item_rank_score": 0.0,
                    "user_rank_score": 0.0,
                    "bpr_rank_score": 0.0,
                    "content_rank_score": 0.0,
                }
            return candidates[app_id]

        for rank, app_id in enumerate(item_list, start=1):
            row = ensure(app_id)
            row["item_rank"] = rank
            row["item_rank_score"] = self.rank_score(rank)

        for rank, app_id in enumerate(user_list, start=1):
            row = ensure(app_id)
            row["user_rank"] = rank
            row["user_rank_score"] = self.rank_score(rank)

        for rank, app_id in enumerate(bpr_list, start=1):
            row = ensure(app_id)
            row["bpr_rank"] = rank
            row["bpr_rank_score"] = self.rank_score(rank)

        for rank, app_id in enumerate(content_list, start=1):
            row = ensure(app_id)
            row["content_rank"] = rank
            row["content_rank_score"] = self.rank_score(rank)

        if not candidates:
            self.logs[user_id] = {
                "item": [],
                "user": [],
                "bpr": [],
                "content": [],
                "fusion": [],
            }
            return pd.DataFrame(columns=["app_id", "fusion_score"])

        fusion_df = pd.DataFrame(candidates.values())

        fusion_df["fusion_score"] = (
            ITEM_WEIGHT * fusion_df["item_rank_score"]
            + USER_WEIGHT * fusion_df["user_rank_score"]
            + BPR_WEIGHT * fusion_df["bpr_rank_score"]
            + CONTENT_WEIGHT * fusion_df["content_rank_score"]
        )

        # 동점 시 중심 모델 Item-Based를 가장 먼저 우선.
        fusion_df["_item_tie"] = fusion_df["item_rank"].fillna(999999)
        fusion_df["_user_tie"] = fusion_df["user_rank"].fillna(999999)
        fusion_df["_bpr_tie"] = fusion_df["bpr_rank"].fillna(999999)
        fusion_df["_content_tie"] = fusion_df["content_rank"].fillna(999999)

        fusion_df = fusion_df.sort_values(
            [
                "fusion_score",
                "_item_tie",
                "_user_tie",
                "_bpr_tie",
                "_content_tie",
            ],
            ascending=[False, True, True, True, True],
            kind="stable",
        ).reset_index(drop=True)

        result = fusion_df.head(top_n).drop(
            columns=[
                "_item_tie",
                "_user_tie",
                "_bpr_tie",
                "_content_tie",
            ]
        )

        self.logs[user_id] = {
            "item": item_list[:TOP_N],
            "user": user_list[:TOP_N],
            "bpr": bpr_list[:TOP_N],
            "content": content_list[:TOP_N],
            "fusion": result["app_id"].tolist(),
        }

        return result


# =========================================================
# Evaluation Summary / Diagnostics
# =========================================================

def make_summary(eval_df):
    total_hits = int(eval_df["hits"].sum())
    total_recommended = int(eval_df["n_recommended"].sum())
    total_test = int(eval_df["n_test"].sum())

    micro_precision = total_hits / total_recommended if total_recommended else 0.0
    micro_recall = total_hits / total_test if total_test else 0.0

    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        "experiment": "case3_4model_cached_rank_fusion",
        "model_top_n": MODEL_TOP_N,
        "top_n": TOP_N,
        "item_weight": ITEM_WEIGHT,
        "user_weight": USER_WEIGHT,
        "bpr_weight": BPR_WEIGHT,
        "content_weight": CONTENT_WEIGHT,
        "n_users": len(eval_df),
        "precision_at_10": eval_df["precision"].mean(),
        "recall_at_10": eval_df["recall"].mean(),
        "hit_rate_at_10": eval_df["hit"].mean(),
        "ndcg_at_10": eval_df["ndcg"].mean(),
        "micro_precision_at_10": micro_precision,
        "micro_recall_at_10": micro_recall,
        "micro_f1_at_10": micro_f1,
        "hits": total_hits,
        "n_recommended": total_recommended,
        "n_test": total_test,
    }


def build_fusion_diagnostics(fusion_model, test_df, sampled_users):
    sampled_ids = set(sampled_users["user_id"])

    positive_test = (
        test_df[
            test_df["user_id"].isin(sampled_ids)
            & (test_df["is_recommended"] == True)
        ]
        .groupby("user_id")["app_id"]
        .apply(list)
        .to_dict()
    )

    rows = []

    for row in sampled_users.itertuples(index=False):
        user_id = row.user_id
        truth = set(positive_test.get(user_id, []))
        logs = fusion_model.logs.get(user_id)

        if logs is None:
            continue

        item_rec = set(logs["item"])
        user_rec = set(logs["user"])
        bpr_rec = set(logs["bpr"])
        content_rec = set(logs["content"])
        fusion_rec = set(logs["fusion"])

        item_hit = item_rec & truth
        user_hit = user_rec & truth
        bpr_hit = bpr_rec & truth
        content_hit = content_rec & truth
        fusion_hit = fusion_rec & truth

        item_unique = item_hit - user_hit - bpr_hit - content_hit
        user_unique = user_hit - item_hit - bpr_hit - content_hit
        bpr_unique = bpr_hit - item_hit - user_hit - content_hit
        content_unique = content_hit - item_hit - user_hit - bpr_hit

        all_four = item_hit & user_hit & bpr_hit & content_hit

        # 기존 중심 baseline인 Item-Based와 비교
        recovered = fusion_hit - item_hit
        lost = item_hit - fusion_hit

        rows.append(
            {
                "user_id": user_id,
                "review_group": row.review_group,
                "n_games": row.n_games,
                "n_test_positive": len(truth),
                "item_hits": len(item_hit),
                "user_hits": len(user_hit),
                "bpr_hits": len(bpr_hit),
                "content_hits": len(content_hit),
                "fusion_hits": len(fusion_hit),
                "item_unique_hits": len(item_unique),
                "user_unique_hits": len(user_unique),
                "bpr_unique_hits": len(bpr_unique),
                "content_unique_hits": len(content_unique),
                "all_four_overlap_hits": len(all_four),
                "fusion_recovered_hits": len(recovered),
                "fusion_lost_hits": len(lost),
                "fusion_delta_hits": len(fusion_hit) - len(item_hit),
                "recovered_app_ids": ",".join(map(str, sorted(recovered))),
                "lost_app_ids": ",".join(map(str, sorted(lost))),
            }
        )

    return pd.DataFrame(rows)


def compare_with_baseline(fusion_eval_df):
    if not BASELINE_EVAL_PATH.exists():
        print("\n기존 Item-Based baseline CSV가 없어 사용자별 비교를 생략합니다.")
        return None

    baseline_df = pd.read_csv(BASELINE_EVAL_PATH)

    columns = ["user_id", "precision", "recall", "hits", "hit", "ndcg"]

    baseline = baseline_df[columns].rename(
        columns={
            "precision": "baseline_precision",
            "recall": "baseline_recall",
            "hits": "baseline_hits",
            "hit": "baseline_hit",
            "ndcg": "baseline_ndcg",
        }
    )

    fusion = fusion_eval_df[columns].rename(
        columns={
            "precision": "fusion_precision",
            "recall": "fusion_recall",
            "hits": "fusion_hits",
            "hit": "fusion_hit",
            "ndcg": "fusion_ndcg",
        }
    )

    comparison = baseline.merge(fusion, on="user_id", how="inner")

    comparison["delta_precision"] = (
        comparison["fusion_precision"] - comparison["baseline_precision"]
    )
    comparison["delta_recall"] = (
        comparison["fusion_recall"] - comparison["baseline_recall"]
    )
    comparison["delta_hits"] = (
        comparison["fusion_hits"] - comparison["baseline_hits"]
    )
    comparison["delta_hit"] = (
        comparison["fusion_hit"] - comparison["baseline_hit"]
    )
    comparison["delta_ndcg"] = (
        comparison["fusion_ndcg"] - comparison["baseline_ndcg"]
    )

    return comparison


# =========================================================
# Main
# =========================================================

def main():
    total_start = time.time()

    print("\n==================================================")
    print(" Hybrid Experiment C - Optimized 4-Model Cached Rank Fusion")
    print(" Item-Based + User-Based + BPR + Content-Based")
    print(f" Each Top-{MODEL_TOP_N} -> Fusion -> Top-{TOP_N}")
    print("==================================================")

    print("\nWeights")
    print(f"Item    : {ITEM_WEIGHT:.2f}")
    print(f"User    : {USER_WEIGHT:.2f}")
    print(f"BPR     : {BPR_WEIGHT:.2f}")
    print(f"Content : {CONTENT_WEIGHT:.2f}")

    total_weight = ITEM_WEIGHT + USER_WEIGHT + BPR_WEIGHT + CONTENT_WEIGHT
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"Fusion weight 합이 1.0이 아닙니다: {total_weight}")

    # full split / CF matrix는 정말 필요한 경우에만 lazy-load
    full_split_holder = {}
    cf_holder = {}

    # -----------------------------------------------------
    # 1. 평가 사용자
    # -----------------------------------------------------
    sampled_users = load_sampled_users()

    # -----------------------------------------------------
    # 2. 400명 평가 subset
    # -----------------------------------------------------
    train_eval, test_eval = load_or_create_eval_subset(
        sampled_users,
        full_split_holder,
    )

    # -----------------------------------------------------
    # 3. Item cache
    # -----------------------------------------------------
    print("\n===== 3. Item-Based Retrieval Cache =====")
    item_cache = build_item_cache(
        sampled_users,
        train_eval,
        full_split_holder,
        cf_holder,
    )
    print_cache_coverage("Item", item_cache, sampled_users)

    # -----------------------------------------------------
    # 4. User-Based cache
    # -----------------------------------------------------
    print("\n===== 4. User-Based Retrieval Cache =====")
    user_cache = build_user_cache(
        sampled_users,
        train_eval,
        full_split_holder,
        cf_holder,
    )
    print_cache_coverage("User", user_cache, sampled_users)

    # -----------------------------------------------------
    # 5. BPR cache
    # -----------------------------------------------------
    print("\n===== 5. BPR Retrieval Cache =====")
    bpr_cache = build_bpr_cache(sampled_users, train_eval)
    print_cache_coverage("BPR", bpr_cache, sampled_users)

    # Content item space를 BPR mapping과 동일하게 맞춤
    _, item_ids = load_bpr_mapping_only()

    # -----------------------------------------------------
    # 6. Content cache
    # -----------------------------------------------------
    print("\n===== 6. Content Retrieval Cache =====")
    content_cache = build_content_cache(
        sampled_users,
        train_eval,
        item_ids,
    )
    print_cache_coverage("Content", content_cache, sampled_users)

    # -----------------------------------------------------
    # 7. Cached Fusion
    # -----------------------------------------------------
    print("\n===== 7. Cached Rank Fusion =====")

    fusion_model = CachedRankFusionRecommender(
        item_cache=item_cache,
        user_cache=user_cache,
        bpr_cache=bpr_cache,
        content_cache=content_cache,
    )

    # -----------------------------------------------------
    # 8. Evaluation
    # -----------------------------------------------------
    print("\n===== 8. Case 3 평가 시작 =====")
    eval_start = time.time()

    eval_df = run_mf_evaluation(
        recommender=fusion_model,
        train_df=train_eval,
        test_df=test_eval,
        sampled_users=sampled_users,
        top_n=TOP_N,
        positive_only=True,
    )

    print(f"Fusion 평가 시간: {time.time() - eval_start:.2f}초")

    # -----------------------------------------------------
    # 9. Report
    # -----------------------------------------------------
    print("\n===== 9. Case 3 평가 결과 =====")

    group_summary = print_evaluation_report(eval_df, top_n=TOP_N)
    summary = make_summary(eval_df)

    # -----------------------------------------------------
    # 10. Diagnostics
    # -----------------------------------------------------
    print("\n===== 10. Hit Overlap / Unique Hit =====")

    diagnostic_df = build_fusion_diagnostics(
        fusion_model=fusion_model,
        test_df=test_eval,
        sampled_users=sampled_users,
    )

    if len(diagnostic_df):
        item_hits = int(diagnostic_df["item_hits"].sum())
        user_hits = int(diagnostic_df["user_hits"].sum())
        bpr_hits = int(diagnostic_df["bpr_hits"].sum())
        content_hits = int(diagnostic_df["content_hits"].sum())
        fusion_hits = int(diagnostic_df["fusion_hits"].sum())

        item_unique = int(diagnostic_df["item_unique_hits"].sum())
        user_unique = int(diagnostic_df["user_unique_hits"].sum())
        bpr_unique = int(diagnostic_df["bpr_unique_hits"].sum())
        content_unique = int(diagnostic_df["content_unique_hits"].sum())
        all_four = int(diagnostic_df["all_four_overlap_hits"].sum())
        recovered = int(diagnostic_df["fusion_recovered_hits"].sum())
        lost = int(diagnostic_df["fusion_lost_hits"].sum())

        print("\n각 모델 Top-10 Hits")
        print("Item-Based :", item_hits)
        print("User-Based :", user_hits)
        print("BPR        :", bpr_hits)
        print("Content    :", content_hits)
        print("Fusion     :", fusion_hits)

        print("\nUnique Hits")
        print("Item only    :", item_unique)
        print("User only    :", user_unique)
        print("BPR only     :", bpr_unique)
        print("Content only :", content_unique)
        print("4개 모델 공통:", all_four)

        print("\nItem-Based 대비 Fusion")
        print("복구된 정답:", recovered)
        print("손실된 정답:", lost)
        print("순증가:", recovered - lost)

        summary.update(
            {
                "item_top10_hits": item_hits,
                "user_top10_hits": user_hits,
                "bpr_top10_hits": bpr_hits,
                "content_top10_hits": content_hits,
                "fusion_top10_hits_diagnostic": fusion_hits,
                "item_unique_hits": item_unique,
                "user_unique_hits": user_unique,
                "bpr_unique_hits": bpr_unique,
                "content_unique_hits": content_unique,
                "all_four_overlap_hits": all_four,
                "fusion_recovered_hits": recovered,
                "fusion_lost_hits": lost,
            }
        )

    # -----------------------------------------------------
    # 10. Baseline comparison
    # -----------------------------------------------------
    print("\n===== 11. Item-Based Baseline 비교 =====")

    comparison = compare_with_baseline(eval_df)

    if comparison is not None:
        baseline_hits = int(comparison["baseline_hits"].sum())
        fusion_hits = int(comparison["fusion_hits"].sum())

        print("공통 평가 사용자:", len(comparison))
        print("Baseline Hits:", baseline_hits)
        print("Fusion Hits  :", fusion_hits)
        print("Delta Hits   :", fusion_hits - baseline_hits)

        print("평균 Delta Precision:", f"{comparison['delta_precision'].mean():.6f}")
        print("평균 Delta Recall   :", f"{comparison['delta_recall'].mean():.6f}")
        print("평균 Delta NDCG     :", f"{comparison['delta_ndcg'].mean():.6f}")

        improved = int((comparison["delta_hits"] > 0).sum())
        worsened = int((comparison["delta_hits"] < 0).sum())
        same = int((comparison["delta_hits"] == 0).sum())

        print("사용자별 Hit 개선/악화/동일:", improved, worsened, same)

    # -----------------------------------------------------
    # 11. Save
    # -----------------------------------------------------
    print("\n===== 12. 결과 저장 =====")

    eval_df.to_csv(EVAL_PATH, index=False)
    pd.DataFrame([summary]).to_csv(SUMMARY_PATH, index=False)
    group_summary.to_csv(GROUP_PATH, index=False)
    diagnostic_df.to_csv(DIAGNOSTIC_PATH, index=False)

    if comparison is not None:
        comparison.to_csv(COMPARE_PATH, index=False)

    print(EVAL_PATH)
    print(SUMMARY_PATH)
    print(GROUP_PATH)
    print(DIAGNOSTIC_PATH)

    if comparison is not None:
        print(COMPARE_PATH)

    # -----------------------------------------------------
    # 12. Final
    # -----------------------------------------------------
    print("\n==================================================")
    print(" Experiment C Summary")
    print("==================================================")

    print(f"Precision@10: {summary['precision_at_10']:.6f}")
    print(f"Recall@10:    {summary['recall_at_10']:.6f}")
    print(f"Hit Rate@10:  {summary['hit_rate_at_10']:.6f}")
    print(f"NDCG@10:      {summary['ndcg_at_10']:.6f}")
    print(f"Hits:         {summary['hits']}")
    print(f"전체 실행 시간: {time.time() - total_start:.1f}초")

    print("\n다음 weight 실험부터는 retrieval cache를 그대로 재사용합니다.")
    print("ITEM_WEIGHT / USER_WEIGHT / BPR_WEIGHT / CONTENT_WEIGHT만 바꿔 다시 실행하면 됩니다.")


if __name__ == "__main__":
    main()
