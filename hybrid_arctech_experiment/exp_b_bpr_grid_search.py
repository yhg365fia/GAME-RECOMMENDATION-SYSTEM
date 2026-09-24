import gc
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

import exp_c_fusion_ablation as base
import exp_b_item_ranker_ratio_sweep as ratio_base
import exp_b_retriever_ratio_finetune as ratio_tune
from models.bpr import create_bpr_model, fine_tune_with_explicit_false

# =========================================================
# BPR Grid Search
# - Retriever ratio fixed: BPR 56 / Content 39 / User 5
# - Final ranker fixed: Item-Based
# - Base BPR tuning only: factors x regularization x iterations
# - learning_rate / Explicit-False FT settings fixed
# =========================================================

TOP_N = 10
BPR_N = 56
CONTENT_N = 39
USER_N = 5

FACTORS_GRID = [40, 60, 80]
REG_GRID = [0.002, 0.006, 0.010]
ITER_GRID = [10, 15, 20]

BPR_LEARNING_RATE = 0.05
BPR_VERIFY_NEGATIVE_SAMPLES = True
BPR_NUM_THREADS = 0
RANDOM_STATE = 42

# Explicit False fine-tuning: 이번 grid에서는 통제변수
USE_EXPLICIT_FALSE = True
FT_EPOCHS = 1
FT_LEARNING_RATE = 0.01
FT_REGULARIZATION = 0.001
FT_BATCH_SIZE = 65536

# 27개 모델 파일을 모두 남기면 용량이 매우 커질 수 있으므로 기본 False.
# True로 바꾸면 각 case 모델을 저장한다.
SAVE_EACH_MODEL = False

RESULT_DIR = base.RESULT_DIR / "bpr_grid_search"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = RESULT_DIR / "bpr_grid_search_summary.csv"
MODEL_DIR = base.SAVED_MODEL_DIR / "bpr_grid_search"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def first_existing(paths, label):
    for path in paths:
        path = Path(path)
        if path.exists():
            print(f"{label}: {path}")
            return path
    raise FileNotFoundError(
        f"{label} 파일을 찾을 수 없습니다. 확인한 경로:\n" +
        "\n".join(str(Path(p)) for p in paths)
    )


def find_bpr_data_files():
    """기존 프로젝트의 BPR matrix/mapping을 재사용한다."""
    root = base.ROOT
    saved = base.SAVED_MODEL_DIR

    user_items_path = first_existing([
        saved / "implicit_bpr_user_items.npz",
        root / "models" / "implicit_bpr_user_items.npz",
    ], "Seen matrix")

    positive_path = first_existing([
        saved / "implicit_bpr_positive_user_items.npz",
        root / "models" / "implicit_bpr_positive_user_items.npz",
    ], "Positive matrix")

    negative_path = first_existing([
        saved / "implicit_bpr_negative_user_items.npz",
        root / "models" / "implicit_bpr_negative_user_items.npz",
    ], "Negative matrix")

    mapping_path = first_existing([
        saved / "implicit_bpr_mapping.npz",
        root / "models" / "implicit_bpr_mapping.npz",
    ], "Mapping")

    return user_items_path, positive_path, negative_path, mapping_path


def load_bpr_data():
    user_items_path, positive_path, negative_path, mapping_path = find_bpr_data_files()

    print("\n===== BPR 학습 데이터 로드 =====")
    user_items = load_npz(user_items_path).tocsr()
    positive_items = load_npz(positive_path).tocsr()
    negative_items = load_npz(negative_path).tocsr()
    mapping = np.load(mapping_path)
    user_ids = np.asarray(mapping["user_ids"])
    item_ids = np.asarray(mapping["item_ids"])

    expected_shape = (len(user_ids), len(item_ids))
    for name, matrix in [
        ("user_items", user_items),
        ("positive_items", positive_items),
        ("negative_items", negative_items),
    ]:
        if matrix.shape != expected_shape:
            raise ValueError(f"{name} shape 불일치: {matrix.shape} != {expected_shape}")

    print("Users:", len(user_ids))
    print("Items:", len(item_ids))
    print("Seen interactions:", f"{user_items.nnz:,}")
    print("Positive interactions:", f"{positive_items.nnz:,}")
    print("Negative interactions:", f"{negative_items.nnz:,}")

    return user_items, positive_items, negative_items, user_ids, item_ids


def generate_bpr_topn(model, sampled_users, user_items, user_ids, item_ids, n=BPR_N):
    """학습된 한 BPR 모델에서 동일 400명의 Top-N 후보를 생성."""
    rows = []
    missing_users = 0
    start = time.perf_counter()

    for i, user_id in enumerate(sampled_users["user_id"], start=1):
        user_idx = np.searchsorted(user_ids, user_id)
        if user_idx >= len(user_ids) or user_ids[user_idx] != user_id:
            missing_users += 1
            continue

        item_indices, scores = model.recommend(
            userid=int(user_idx),
            user_items=user_items[user_idx],
            N=n,
            filter_already_liked_items=True,
        )
        app_ids = item_ids[item_indices]

        for rank, (app_id, score) in enumerate(zip(app_ids, scores), start=1):
            rows.append((user_id, rank, app_id, float(score)))

        if i % 100 == 0 or i == len(sampled_users):
            print(f"BPR Top{n}: {i}/{len(sampled_users)}")

    df = pd.DataFrame(rows, columns=["user_id", "rank", "app_id", "bpr_score"])
    print("BPR mapping 없는 사용자:", missing_users)
    print(f"BPR Top{n} 생성 시간: {time.perf_counter() - start:.2f}초")
    return df


def train_one_bpr(factors, regularization, iterations, positive_items, negative_items):
    print("\n===== BPR 학습 =====")
    print("factors       :", factors)
    print("regularization:", regularization)
    print("iterations    :", iterations)
    print("learning_rate :", BPR_LEARNING_RATE)
    print("explicit false:", USE_EXPLICIT_FALSE)

    model = create_bpr_model(
        factors=factors,
        learning_rate=BPR_LEARNING_RATE,
        regularization=regularization,
        iterations=iterations,
        verify_negative_samples=BPR_VERIFY_NEGATIVE_SAMPLES,
        num_threads=BPR_NUM_THREADS,
        random_state=RANDOM_STATE,
    )

    start = time.perf_counter()
    model.fit(positive_items, show_progress=True)
    base_train_seconds = time.perf_counter() - start

    ft_seconds = 0.0
    if USE_EXPLICIT_FALSE:
        ft_start = time.perf_counter()
        model = fine_tune_with_explicit_false(
            model=model,
            positive_items=positive_items,
            negative_items=negative_items,
            epochs=FT_EPOCHS,
            learning_rate=FT_LEARNING_RATE,
            regularization=FT_REGULARIZATION,
            batch_size=FT_BATCH_SIZE,
            random_state=RANDOM_STATE,
        )
        ft_seconds = time.perf_counter() - ft_start

    return model, base_train_seconds, ft_seconds


def main():
    total_start = time.perf_counter()

    print("\n==================================================")
    print(" BPR Grid Search - Hybrid Pipeline Evaluation")
    print(" BPR56 + Content39 + User5 -> Item-Based -> Top10")
    print("==================================================")
    print("Factors:", FACTORS_GRID)
    print("Regularization:", REG_GRID)
    print("Iterations:", ITER_GRID)
    print("총 Cases:", len(FACTORS_GRID) * len(REG_GRID) * len(ITER_GRID))

    # 1) 동일 평가 사용자 / subset
    sampled_users = base.load_sampled_users()
    holder = {}
    train_eval, test_eval = base.load_or_create_eval_subset(sampled_users, holder)

    # 2) Content / User cache는 모든 case에서 고정
    print("\n===== 고정 Retriever Cache =====")
    _, bpr_item_ids = base.load_bpr_mapping_only()
    content_cache = base.build_content_cache(sampled_users, train_eval, bpr_item_ids)
    cf_holder = {}
    user_cache = base.build_user_cache(sampled_users, train_eval, holder, cf_holder)
    content_dict = ratio_tune.cache_to_dict(content_cache)
    user_dict = ratio_tune.cache_to_dict(user_cache)

    # 3) Item-Based final ranker 고정
    item_ranker, fast_path = ratio_base.prepare_item_ranker(
        sampled_users=sampled_users,
        train_eval=train_eval,
    )
    print("Item Ranker Fast Path:", fast_path)

    # 4) BPR 전체 학습 matrix 로드
    user_items, positive_items, negative_items, user_ids, item_ids = load_bpr_data()

    configs = list(itertools.product(FACTORS_GRID, REG_GRID, ITER_GRID))
    summaries = []

    fixed_hybrid_config = {
        "candidate_size": 100,
        "bpr_n": BPR_N,
        "content_n": CONTENT_N,
        "user_n": USER_N,
    }

    for case_idx, (factors, reg, iterations) in enumerate(configs, start=1):
        case_name = f"bpr_f{factors}_reg{reg:g}_iter{iterations}"
        print("\n" + "#" * 70)
        print(f"CASE {case_idx}/{len(configs)}: {case_name}")
        print("#" * 70)

        model, train_seconds, ft_seconds = train_one_bpr(
            factors=factors,
            regularization=reg,
            iterations=iterations,
            positive_items=positive_items,
            negative_items=negative_items,
        )

        if SAVE_EACH_MODEL:
            model_path = MODEL_DIR / f"{case_name}_explicit_false.npz"
            model.save(str(model_path))
            print("모델 저장:", model_path)

        bpr_cache = generate_bpr_topn(
            model=model,
            sampled_users=sampled_users,
            user_items=user_items,
            user_ids=user_ids,
            item_ids=item_ids,
            n=BPR_N,
        )
        bpr_dict = ratio_tune.cache_to_dict(bpr_cache)

        summary = ratio_tune.run_case(
            case_name=case_name,
            config=fixed_hybrid_config,
            bpr_dict=bpr_dict,
            content_dict=content_dict,
            user_dict=user_dict,
            item_ranker=item_ranker,
            train_eval=train_eval,
            test_eval=test_eval,
            sampled_users=sampled_users,
        )

        summary.update({
            "factors": factors,
            "regularization": reg,
            "iterations": iterations,
            "learning_rate": BPR_LEARNING_RATE,
            "explicit_false": USE_EXPLICIT_FALSE,
            "ft_epochs": FT_EPOCHS if USE_EXPLICIT_FALSE else 0,
            "ft_learning_rate": FT_LEARNING_RATE if USE_EXPLICIT_FALSE else np.nan,
            "ft_regularization": FT_REGULARIZATION if USE_EXPLICIT_FALSE else np.nan,
            "bpr_train_seconds": train_seconds,
            "ft_seconds": ft_seconds,
            "total_model_train_seconds": train_seconds + ft_seconds,
        })
        summaries.append(summary)

        # 중간 저장: 중간에 중단되어도 완료 case 결과 보존
        pd.DataFrame(summaries).to_csv(SUMMARY_PATH, index=False)
        print("중간 Summary 저장:", SUMMARY_PATH)

        del model, bpr_cache, bpr_dict
        gc.collect()

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values(
        ["ndcg_at_10", "precision_at_10", "hit_rate_at_10"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary_df.insert(0, "rank_by_ndcg", np.arange(1, len(summary_df) + 1))
    summary_df.to_csv(SUMMARY_PATH, index=False)

    display_cols = [
        "rank_by_ndcg", "factors", "regularization", "iterations",
        "bpr_candidate_recall", "union_candidate_recall",
        "scoreable_candidate_recall", "precision_at_10", "recall_at_10",
        "hit_rate_at_10", "ndcg_at_10", "hits",
        "bpr_train_seconds", "ft_seconds",
    ]

    print("\n==================================================")
    print(" BPR Grid Search Final Comparison")
    print("==================================================")
    print(summary_df[display_cols].to_string(index=False))

    best = summary_df.iloc[0]
    print("\n==================================================")
    print(" Best BPR Parameters (sorted by NDCG@10)")
    print("==================================================")
    print("Factors       :", int(best["factors"]))
    print("Regularization:", best["regularization"])
    print("Iterations    :", int(best["iterations"]))
    print("P@10          :", f"{best['precision_at_10']:.6f}")
    print("R@10          :", f"{best['recall_at_10']:.6f}")
    print("HR@10         :", f"{best['hit_rate_at_10']:.6f}")
    print("NDCG@10       :", f"{best['ndcg_at_10']:.6f}")
    print("Hits          :", int(best["hits"]))
    print("BPR Recall    :", f"{best['bpr_candidate_recall']:.6f}")
    print("Union Recall  :", f"{best['union_candidate_recall']:.6f}")

    print("\nSummary 저장:", SUMMARY_PATH)
    print("전체 실행 시간:", f"{(time.perf_counter() - total_start) / 60:.2f}분")


if __name__ == "__main__":
    main()
