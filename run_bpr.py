import os
import gc

import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from implicit.cpu.bpr import BayesianPersonalizedRanking

from data_split import load_mf_split

from evaluation import (
    build_mf_user_review_groups,
    stratified_sample_users,
    run_mf_evaluation,
)

from models.bpr import (
    create_bpr_model,
    ImplicitBPRAdapter,
    fine_tune_with_explicit_false,
)


# ============================================================
# FINAL BPR EXPERIMENT
#
# 기준 최고:
# factors = 60
# reg = 0.006
#
# 마지막 4개:
# 60 / 0.007
# 60 / 0.008
# 63 / 0.007
# 63 / 0.008
# ============================================================

EXPERIMENTS = [
    {
        "name": "factor60_reg007",
        "factors": 60,
        "regularization": 0.007,
    },
    {
        "name": "factor60_reg008",
        "factors": 60,
        "regularization": 0.008,
    },
    {
        "name": "factor63_reg007",
        "factors": 63,
        "regularization": 0.007,
    },
    {
        "name": "factor63_reg008",
        "factors": 63,
        "regularization": 0.008,
    },
]


# ============================================================
# Base BPR 고정값
# ============================================================

BPR_ITERATIONS = 15

BPR_LEARNING_RATE = 0.05

BPR_VERIFY_NEGATIVE_SAMPLES = True

BPR_NUM_THREADS = 0

RANDOM_STATE = 42


# ============================================================
# Explicit False Fine-Tuning
# ============================================================

BPR_EXPLICIT_FALSE_EPOCHS = 1

BPR_EXPLICIT_FALSE_LEARNING_RATE = 0.01

BPR_EXPLICIT_FALSE_REGULARIZATION = 0.001

BPR_EXPLICIT_FALSE_BATCH_SIZE = 65536


# ============================================================
# Evaluation
# ============================================================

TOP_N = 10

LOWER_BOUND = 10
UPPER_BOUND = 78

SAMPLE_PER_GROUP = 100

POSITIVE_ONLY = True


# ============================================================
# Paths
# ============================================================

MATRIX_PATH = (
    "models/implicit_bpr_user_items.npz"
)

POSITIVE_MATRIX_PATH = (
    "models/implicit_bpr_positive_user_items.npz"
)

NEGATIVE_MATRIX_PATH = (
    "models/implicit_bpr_negative_user_items.npz"
)

MAPPING_PATH = (
    "models/implicit_bpr_mapping.npz"
)

RESULT_PATH = (
    "results/bpr_final_last_search_results.csv"
)


# ============================================================
# 현재 최고 기준 모델
# ============================================================

REFERENCE_RESULT = {
    "name": "REFERENCE_factor60_reg006",

    "factors": 60,

    "regularization": 0.006,

    "precision@10": 0.05225,

    "recall@10": 0.068695,

    "hit_rate@10": 0.3850,

    "ndcg@10": 0.069681,

    "micro_f1@10": 0.057473,

    "hits": 209,

    "type": "reference",
}


# ============================================================
# 모델 저장 경로
# ============================================================

def get_model_path(
    factors,
    regularization,
):

    os.makedirs(
        "models/bpr_grid",
        exist_ok=True,
    )

    reg_string = (
        f"{regularization:g}"
    )

    return (
        "models/bpr_grid/"
        f"bpr_iter{BPR_ITERATIONS}_"
        f"factors{factors}_"
        f"reg{reg_string}_"
        f"explicit_false.npz"
    )


# ============================================================
# Train / Load
# ============================================================

def train_or_load_model(
    factors,
    regularization,
    positive_items,
    negative_items,
):

    model_path = (
        get_model_path(
            factors,
            regularization,
        )
    )


    # ========================================================
    # 동일 모델이 이미 있으면 로드
    # ========================================================

    if os.path.exists(
        model_path
    ):

        print(
            "\n===== 저장 모델 로드 ====="
        )

        print(
            model_path
        )

        model = (
            BayesianPersonalizedRanking.load(
                model_path
            )
        )

        return model


    # ========================================================
    # Base BPR Training
    # ========================================================

    print(
        "\n===== Base BPR Training ====="
    )

    print(
        "Factors:",
        factors
    )

    print(
        "Regularization:",
        regularization
    )


    model = (
        create_bpr_model(

            factors=(
                factors
            ),

            learning_rate=(
                BPR_LEARNING_RATE
            ),

            regularization=(
                regularization
            ),

            iterations=(
                BPR_ITERATIONS
            ),

            verify_negative_samples=(
                BPR_VERIFY_NEGATIVE_SAMPLES
            ),

            num_threads=(
                BPR_NUM_THREADS
            ),

            random_state=(
                RANDOM_STATE
            ),
        )
    )


    # ========================================================
    # Stage 1
    #
    # True > Unseen
    # ========================================================

    model.fit(

        positive_items,

        show_progress=True,
    )


    # ========================================================
    # Stage 2
    #
    # True > False
    # ========================================================

    print(
        "\n===== Explicit False Fine-Tuning ====="
    )


    model = (
        fine_tune_with_explicit_false(

            model=model,

            positive_items=(
                positive_items
            ),

            negative_items=(
                negative_items
            ),

            epochs=(
                BPR_EXPLICIT_FALSE_EPOCHS
            ),

            learning_rate=(
                BPR_EXPLICIT_FALSE_LEARNING_RATE
            ),

            regularization=(
                BPR_EXPLICIT_FALSE_REGULARIZATION
            ),

            batch_size=(
                BPR_EXPLICIT_FALSE_BATCH_SIZE
            ),

            random_state=(
                RANDOM_STATE
            ),
        )
    )


    # ========================================================
    # Save
    # ========================================================

    model.save(
        model_path
    )


    print(
        "\n모델 저장:"
    )

    print(
        model_path
    )


    return model


# ============================================================
# Result Summary
# ============================================================

def summarize_result(
    eval_df,
    experiment_name,
    factors,
    regularization,
):

    # ========================================================
    # Macro
    # ========================================================

    precision = (
        eval_df[
            "precision"
        ].mean()
    )


    recall = (
        eval_df[
            "recall"
        ].mean()
    )


    hit_rate = (
        eval_df[
            "hit"
        ].mean()
    )


    ndcg = (
        eval_df[
            "ndcg"
        ].mean()
    )


    # ========================================================
    # Micro
    # ========================================================

    total_hits = (
        eval_df[
            "hits"
        ].sum()
    )


    total_recommended = (
        eval_df[
            "n_recommended"
        ].sum()
    )


    total_test = (
        eval_df[
            "n_test"
        ].sum()
    )


    micro_precision = (
        total_hits
        / total_recommended

        if total_recommended > 0

        else 0.0
    )


    micro_recall = (
        total_hits
        / total_test

        if total_test > 0

        else 0.0
    )


    micro_f1 = (
        2
        * micro_precision
        * micro_recall
        /
        (
            micro_precision
            + micro_recall
        )

        if (
            micro_precision
            + micro_recall
        ) > 0

        else 0.0
    )


    return {

        "name":
            experiment_name,

        "iterations":
            BPR_ITERATIONS,

        "factors":
            factors,

        "regularization":
            regularization,

        "explicit_false":
            True,

        "positive_only":
            POSITIVE_ONLY,

        "precision@10":
            precision,

        "recall@10":
            recall,

        "hit_rate@10":
            hit_rate,

        "ndcg@10":
            ndcg,

        "micro_precision@10":
            micro_precision,

        "micro_recall@10":
            micro_recall,

        "micro_f1@10":
            micro_f1,

        "hits":
            int(
                total_hits
            ),

        "n_users":
            len(
                eval_df
            ),

        "n_test":
            int(
                total_test
            ),

        "type":
            "new_experiment",
    }


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        "results",
        exist_ok=True,
    )


    print(
        "\n============================================"
    )

    print(
        "FINAL BPR LAST SEARCH"
    )

    print(
        "============================================"
    )


    # ========================================================
    # 1. Train / Test
    # ========================================================

    train_df, test_df = (
        load_mf_split()
    )


    print(
        "Train:",
        train_df.shape
    )

    print(
        "Test :",
        test_df.shape
    )


    # ========================================================
    # 2. Required files
    # ========================================================

    required_files = [

        MATRIX_PATH,

        POSITIVE_MATRIX_PATH,

        NEGATIVE_MATRIX_PATH,

        MAPPING_PATH,
    ]


    for path in required_files:

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                f"필요한 파일 없음: {path}"
            )


    # ========================================================
    # 3. Matrix Load
    # ========================================================

    print(
        "\n===== BPR Matrix 로드 ====="
    )


    user_items = (
        load_npz(
            MATRIX_PATH
        )
    )


    positive_items = (
        load_npz(
            POSITIVE_MATRIX_PATH
        )
    )


    negative_items = (
        load_npz(
            NEGATIVE_MATRIX_PATH
        )
    )


    mapping = (
        np.load(
            MAPPING_PATH
        )
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


    print(
        "Users:",
        len(
            user_ids
        )
    )

    print(
        "Items:",
        len(
            item_ids
        )
    )

    print(
        "All interactions:",
        user_items.nnz
    )

    print(
        "Positive interactions:",
        positive_items.nnz
    )

    print(
        "Negative interactions:",
        negative_items.nnz
    )


    # ========================================================
    # 4. 평가 사용자 고정
    # ========================================================

    print(
        "\n===== 평가 사용자 고정 ====="
    )


    eligible_users = (
        build_mf_user_review_groups(

            train_df,

            test_df,

            lower_bound=(
                LOWER_BOUND
            ),

            upper_bound=(
                UPPER_BOUND
            ),
        )
    )


    sampled_users = (
        stratified_sample_users(

            eligible_users,

            sample_per_group=(
                SAMPLE_PER_GROUP
            ),

            random_state=(
                RANDOM_STATE
            ),
        )
    )


    print(
        "\n평가 사용자:",
        len(
            sampled_users
        )
    )


    # ========================================================
    # 5. Experiments
    # ========================================================

    experiment_results = []


    for experiment in (
        EXPERIMENTS
    ):

        experiment_name = (
            experiment[
                "name"
            ]
        )


        factors = (
            experiment[
                "factors"
            ]
        )


        regularization = (
            experiment[
                "regularization"
            ]
        )


        print(
            "\n\n"
            "##################################################"
        )

        print(
            "FINAL BPR EXPERIMENT"
        )

        print(
            "name           =",
            experiment_name
        )

        print(
            "iterations     =",
            BPR_ITERATIONS
        )

        print(
            "factors        =",
            factors
        )

        print(
            "regularization =",
            regularization
        )

        print(
            "explicit_false = True"
        )

        print(
            "positive_only  = True"
        )

        print(
            "##################################################"
        )


        # ====================================================
        # Train / Load
        # ====================================================

        model = (
            train_or_load_model(

                factors=(
                    factors
                ),

                regularization=(
                    regularization
                ),

                positive_items=(
                    positive_items
                ),

                negative_items=(
                    negative_items
                ),
            )
        )


        # ====================================================
        # Adapter
        # ====================================================

        recommender = (
            ImplicitBPRAdapter(

                model=model,

                user_items=(
                    user_items
                ),

                user_ids=(
                    user_ids
                ),

                item_ids=(
                    item_ids
                ),
            )
        )


        # ====================================================
        # Evaluation
        # ====================================================

        eval_df = (
            run_mf_evaluation(

                recommender=(
                    recommender
                ),

                train_df=(
                    train_df
                ),

                test_df=(
                    test_df
                ),

                sampled_users=(
                    sampled_users
                ),

                top_n=(
                    TOP_N
                ),

                positive_only=(
                    POSITIVE_ONLY
                ),
            )
        )


        row = (
            summarize_result(

                eval_df=(
                    eval_df
                ),

                experiment_name=(
                    experiment_name
                ),

                factors=(
                    factors
                ),

                regularization=(
                    regularization
                ),
            )
        )


        experiment_results.append(
            row
        )


        print(
            "\n===== 이번 실험 결과 ====="
        )


        print(
            pd.DataFrame(
                [row]
            )[
                [
                    "factors",
                    "regularization",
                    "precision@10",
                    "recall@10",
                    "hit_rate@10",
                    "ndcg@10",
                    "micro_f1@10",
                    "hits",
                ]
            ]
            .to_string(
                index=False
            )
        )


        # ====================================================
        # 중간 저장
        # ====================================================

        pd.DataFrame(
            experiment_results
        ).to_csv(

            RESULT_PATH,

            index=False,
        )


        # ====================================================
        # Memory
        # ====================================================

        del model
        del recommender
        del eval_df

        gc.collect()


    # ========================================================
    # 6. New Results
    # ========================================================

    new_result_df = (
        pd.DataFrame(
            experiment_results
        )
    )


    print(
        "\n\n============================================"
    )

    print(
        "NEW FINAL SEARCH RESULTS"
    )

    print(
        "============================================"
    )


    print(
        new_result_df[
            [
                "factors",
                "regularization",
                "precision@10",
                "recall@10",
                "hit_rate@10",
                "ndcg@10",
                "micro_f1@10",
                "hits",
            ]
        ]
        .sort_values(
            [
                "factors",
                "regularization",
            ]
        )
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 7. 기존 최고 + 새 실험 비교
    # ========================================================

    reference_df = (
        pd.DataFrame(
            [
                REFERENCE_RESULT
            ]
        )
    )


    all_results = (
        pd.concat(
            [
                reference_df,
                new_result_df,
            ],

            ignore_index=True,

            sort=False,
        )
    )


    print(
        "\n\n============================================"
    )

    print(
        "REFERENCE + FINAL SEARCH"
    )

    print(
        "============================================"
    )


    print(
        all_results[
            [
                "factors",
                "regularization",
                "precision@10",
                "recall@10",
                "hit_rate@10",
                "ndcg@10",
                "micro_f1@10",
                "hits",
                "type",
            ]
        ]
        .sort_values(
            [
                "factors",
                "regularization",
            ]
        )
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 8. Metric별 Best
    # ========================================================

    metrics = [
        "precision@10",
        "recall@10",
        "hit_rate@10",
        "ndcg@10",
        "micro_f1@10",
    ]


    print(
        "\n============================================"
    )

    print(
        "BEST CONFIG BY METRIC"
    )

    print(
        "============================================"
    )


    for metric in metrics:

        best_row = (
            all_results
            .sort_values(
                metric,
                ascending=False,
            )
            .iloc[0]
        )


        print(
            f"\n[{metric}]"
        )

        print(
            "Factors:",
            int(
                best_row[
                    "factors"
                ]
            )
        )

        print(
            "Regularization:",
            best_row[
                "regularization"
            ]
        )

        print(
            "Score:",
            f"{best_row[metric]:.6f}"
        )


    # ========================================================
    # 9. NDCG 기준 최고
    # ========================================================

    best_ndcg = (
        all_results
        .sort_values(
            "ndcg@10",
            ascending=False,
        )
        .iloc[0]
    )


    print(
        "\n============================================"
    )

    print(
        "BEST NDCG CONFIG"
    )

    print(
        "============================================"
    )


    print(
        "Factors:",
        int(
            best_ndcg[
                "factors"
            ]
        )
    )

    print(
        "Regularization:",
        best_ndcg[
            "regularization"
        ]
    )

    print(
        "Precision@10:",
        f"{best_ndcg['precision@10']:.6f}"
    )

    print(
        "Recall@10:",
        f"{best_ndcg['recall@10']:.6f}"
    )

    print(
        "HR@10:",
        f"{best_ndcg['hit_rate@10']:.6f}"
    )

    print(
        "NDCG@10:",
        f"{best_ndcg['ndcg@10']:.6f}"
    )

    print(
        "Micro F1:",
        f"{best_ndcg['micro_f1@10']:.6f}"
    )

    print(
        "Hits:",
        int(
            best_ndcg[
                "hits"
            ]
        )
    )


    # ========================================================
    # 10. Save
    # ========================================================

    all_result_path = (
        "results/"
        "bpr_final_last_search_all_results.csv"
    )


    all_results.to_csv(

        all_result_path,

        index=False,
    )


    print(
        "\n============================================"
    )

    print(
        "저장 완료"
    )

    print(
        RESULT_PATH
    )

    print(
        all_result_path
    )

    print(
        "============================================"
    )


if __name__ == "__main__":

    main()