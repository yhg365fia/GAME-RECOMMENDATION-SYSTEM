import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def build_user_review_groups(user_history, lower_bound=10, upper_bound=78,
                              bins=None, labels=None):
    user_counts = user_history.groupby("user_id").size().reset_index(name="n_games")

    eligible_users = user_counts[
        (user_counts["n_games"] >= lower_bound) &
        (user_counts["n_games"] <= upper_bound)
    ].copy()

    print(f"평가 대상 유저 수: {len(eligible_users):,}")
    print(eligible_users["n_games"].describe())

    if bins is None:
        bins = [9, 15, 25, 45, 78]
    if labels is None:
        labels = ["10-15개", "16-25개", "26-45개", "46-78개"]

    eligible_users["review_group"] = pd.cut(
        eligible_users["n_games"], bins=bins, labels=labels
    )

    print(eligible_users["review_group"].value_counts().sort_index())
    return eligible_users


def stratified_sample_users(user_counts, sample_per_group=100, random_state=42):
    """
    구간(review_group)마다 동일한 인원 수(sample_per_group)를 무작위 추출.
    해당 구간 인원이 sample_per_group보다 적으면 있는 만큼만 뽑음.
    """
    sampled_groups = []

    for group_name, group_df in user_counts.groupby("review_group", observed=True):
        n = min(len(group_df), sample_per_group)
        sampled_groups.append(group_df.sample(n=n, random_state=random_state))

    sampled = pd.concat(sampled_groups, ignore_index=True)

    print(sampled["review_group"].value_counts().sort_index())
    return sampled


# ===== 변경 1: user_idx=None 파라미터 추가 =====
def evaluate_user(recommender, train_app_ids, test_app_ids, top_n=10, user_idx=None):
    if len(train_app_ids) == 0 or len(test_app_ids) == 0:
        return None

    result = recommender.recommend(
        app_id_list=train_app_ids,
        top_n=top_n,
        exclude_user_idx=user_idx,
    )

    if result is None or len(result) == 0:
        return None

    # 추천 순서 보존 (NDCG 계산용)
    recommended_list = result["app_id"].tolist()

    # 집합 연산용 (hits 계산)
    recommended_ids = set(recommended_list)
    test_ids = set(test_app_ids)

    hits = len(recommended_ids & test_ids)

    # NDCG@K 계산
    dcg = 0.0

    # recommended_ids(set)가 아니라
    # recommended_list를 사용해서 실제 추천 순서를 유지
    for rank, app_id in enumerate(recommended_list, start=1):
        relevance = 1 if app_id in test_ids else 0
        dcg += relevance / np.log2(rank + 1)

    ideal_hits = min(len(test_ids), top_n)

    idcg = sum(
        1 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    # 추천한 게임 중 실제 사용자가 좋아한 게임의 개수
    precision = hits / len(recommended_ids) if len(recommended_ids) > 0 else 0.0
    recall = hits / len(test_ids) if len(test_ids) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "hits": hits,
        "hit": 1 if hits > 0 else 0,
        "ndcg": ndcg,
        "n_recommended": len(recommended_ids),
        "n_test": len(test_ids),
    }


# ===== 변경 3: user_to_idx=None 파라미터 추가 =====
def run_evaluation(recommender, user_history, sampled_users, user_to_idx=None,
                    top_n=10, test_size=0.3, random_state=42):
    sampled_ids = set(sampled_users["user_id"])
    target_history = user_history[user_history["user_id"].isin(sampled_ids)]

    results, skipped = [], 0
    start = time.time()

    for user_id, group in target_history.groupby("user_id"):
        app_ids = group["app_id"].tolist()
        if len(app_ids) < 2:
            skipped += 1
            continue

        train_ids, test_ids = train_test_split(app_ids, test_size=test_size, random_state=random_state)

        # ===== 변경 4: user_id를 interaction_matrix 상의 인덱스로 변환 =====
        user_idx = user_to_idx.get(user_id) if user_to_idx is not None else None

        # ===== 변경 5: user_idx를 evaluate_user에 전달 =====
        metrics = evaluate_user(recommender, train_ids, test_ids, top_n=top_n, user_idx=user_idx)

        if metrics is None:
            skipped += 1
            continue

        metrics["user_id"] = user_id
        metrics["n_games"] = len(app_ids)
        results.append(metrics)

    print(f"평가 완료: 유저 {len(results)}명 (스킵 {skipped}명), 소요 시간 {time.time()-start:.1f}초")

    eval_df = pd.DataFrame(results)
    eval_df = eval_df.merge(sampled_users[["user_id", "review_group"]], on="user_id", how="left")
    return eval_df


def print_evaluation_report(eval_df, top_n=10):

    # =========================
    # 1. Macro Average
    # =========================
    macro_precision = eval_df["precision"].mean()
    macro_recall = eval_df["recall"].mean()
    hit_rate = eval_df["hit"].mean()
    ndcg = eval_df["ndcg"].mean()

    print("===== Macro Average =====")
    print(f"Precision@{top_n}: {macro_precision:.4f}")
    print(f"Recall@{top_n}:    {macro_recall:.4f}")
    print(f"Hit Rate@{top_n}:  {hit_rate:.4f}")
    print(f"NDCG@{top_n}:      {ndcg:.4f}")


    # =========================
    # 2. Micro Average
    # =========================
    total_hits = eval_df["hits"].sum()
    total_recommended = eval_df["n_recommended"].sum()
    total_test = eval_df["n_test"].sum()

    micro_precision = (
        total_hits / total_recommended
        if total_recommended > 0
        else 0.0
    )

    micro_recall = (
        total_hits / total_test
        if total_test > 0
        else 0.0
    )

    micro_f1 = (
        2 * micro_precision * micro_recall
        / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    print()
    print("===== Micro Average =====")
    print(f"Precision@{top_n}: {micro_precision:.4f}")
    print(f"Recall@{top_n}:    {micro_recall:.4f}")
    print(f"F1@{top_n}:        {micro_f1:.4f}")


    # =========================
    # 3. 평가 정보
    # =========================
    print()
    print(f"평가 대상 유저 수: {len(eval_df)}명")
    print(f"전체 Hits: {total_hits}")
    print(f"전체 추천 개수: {total_recommended}")
    print(f"전체 Test 개수: {total_test}")


    # =========================
    # 4. review_group별 Macro 결과
    # =========================
    summary = (
        eval_df.groupby("review_group", observed=True)
        .agg(
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            hit_rate=("hit", "mean"),
            ndcg_mean=("ndcg", "mean"),
            n_users=("user_id", "count"),
        )
        .reset_index()
    )

    print()
    print("===== Review Group별 결과 =====")
    print(summary.to_string(index=False))

     # =========================
    # 5. Pearson Correlation
    # =========================
    from scipy.stats import pearsonr

    # 유저의 게임 수 ↔ 실제 추천 개수
    r1, p1 = pearsonr(
        eval_df["n_games"],
        eval_df["n_recommended"]
    )

    # 유저의 게임 수 ↔ Precision
    r2, p2 = pearsonr(
        eval_df["n_games"],
        eval_df["precision"]
    )

    print()
    print("===== Sparsity Correlation =====")

    print("n_games ↔ n_recommended")
    print(f"Pearson r: {r1:.4f}")
    print(f"p-value:   {p1:.4f}")

    print()

    print("n_games ↔ precision")
    print(f"Pearson r: {r2:.4f}")
    print(f"p-value:   {p2:.4f}")

    return summary


def evaluate_pipeline(recommender, user_history, lower_bound=10, upper_bound=78,
                       n_groups=3, sample_per_group=1000, top_n=10, random_state=42):
    user_counts = build_user_review_groups(user_history, lower_bound, upper_bound, n_groups)
    sampled_users = stratified_sample_users(user_counts, sample_per_group, random_state)
    eval_df = run_evaluation(recommender, user_history, sampled_users, top_n, random_state=random_state)
    summary = print_evaluation_report(eval_df, top_n)
    return eval_df, summary