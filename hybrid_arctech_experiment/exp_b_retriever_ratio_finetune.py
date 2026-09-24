# hybrid_arctech_experiment/exp_b_retriever_ratio_finetune.py



import sys

import time

from pathlib import Path



import numpy as np

import pandas as pd



from scipy.sparse import load_npz

from implicit.cpu.bpr import BayesianPersonalizedRanking





# =========================================================

# Project Root

# =========================================================



ROOT = Path(__file__).resolve().parents[1]

CURRENT_DIR = Path(__file__).resolve().parent



if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))



if str(CURRENT_DIR) not in sys.path:

    sys.path.insert(0, str(CURRENT_DIR))





import exp_c_fusion_ablation as base



# 직전 Ratio Sweep에서 만든 최적화 Item Ranker 재사용

import exp_b_item_ranker_ratio_sweep as ratio_base





# =========================================================

# Experiment Config

# =========================================================



TOP_N = 10



# ---------------------------------------------------------

# 현재 최종 Retriever 비율

#

# BPR : Content : User

# 56  : 39      : 5

#

# 변경하는 것은 Candidate Size뿐.

# ---------------------------------------------------------



EXPERIMENTS = {
    "ratio_50_45_5": {"candidate_size": 100, "bpr_n": 50, "content_n": 45, "user_n": 5},
    "ratio_53_42_5": {"candidate_size": 100, "bpr_n": 53, "content_n": 42, "user_n": 5},
    "ratio_56_39_5_baseline": {"candidate_size": 100, "bpr_n": 56, "content_n": 39, "user_n": 5},
    "ratio_59_36_5": {"candidate_size": 100, "bpr_n": 59, "content_n": 36, "user_n": 5},
    "ratio_62_33_5": {"candidate_size": 100, "bpr_n": 62, "content_n": 33, "user_n": 5},
}


# =========================================================

# Paths

# =========================================================



RESULT_DIR = (

    base.RESULT_DIR

    / "case2_retriever_ratio_finetune"

)



RESULT_DIR.mkdir(

    parents=True,

    exist_ok=True,

)





CACHE_DIR = (

    base.SAVED_MODEL_DIR

    / "case2_cache"

    / "retriever_ratio_finetune"

)



CACHE_DIR.mkdir(

    parents=True,

    exist_ok=True,

)





SUMMARY_PATH = (

    RESULT_DIR

    / "retriever_ratio_finetune_summary.csv"

)





# ---------------------------------------------------------

# 기존 Case3 BPR는 Top100이므로

# Candidate Size 200에서 필요한 BPR112를 못 줌.

#

# 한 번 Top200 cache 생성.

# ---------------------------------------------------------



BPR_TOP200_PATH = (

    CACHE_DIR

    / "bpr_top200.parquet"

)





BPR_CACHE_N = 200





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

# BPR Top-200 Cache

#

# Case3 BPR cache와 알고리즘 완전히 동일.

#

# 차이:

# MODEL_TOP_N = 100 -> 여기서는 200

#

# 모델 재학습 X

# 파라미터 변경 X

# scoring 변경 X

# seen filtering 변경 X

# =========================================================



def build_or_load_bpr_top200(

    sampled_users,

    train_eval,

):



    # -----------------------------------------------------

    # 기존 cache 확인

    # -----------------------------------------------------



    if BPR_TOP200_PATH.exists():



        print(

            "\nBPR Top-200 cache 로드:"

        )



        print(

            BPR_TOP200_PATH

        )





        df = pd.read_parquet(

            BPR_TOP200_PATH

        )





        required = {

            "user_id",

            "rank",

            "app_id",

        }





        missing = (

            required

            -

            set(

                df.columns

            )

        )





        if missing:



            raise ValueError(

                f"BPR Top200 cache 컬럼 오류: "

                f"{missing}"

            )





        return (

            df

            .sort_values(

                ["user_id", "rank"],

                kind="stable",

            )

            .copy()

        )





    # =====================================================

    # 새로 생성

    # =====================================================



    print(

        "\n===== BPR Top-200 Cache 생성 ====="

    )





    start = (

        time.perf_counter()

    )





    (

        model_path,

        mapping_path,

        user_items_path,

    ) = (

        base.find_bpr_files()

    )





    print(

        "BPR Model:",

        model_path

    )



    print(

        "Mapping:",

        mapping_path

    )



    print(

        "User Items:",

        user_items_path

    )





    # -----------------------------------------------------

    # 동일한 최종 BPR 모델 로드

    # -----------------------------------------------------



    model = (

        BayesianPersonalizedRanking

        .load(

            str(

                model_path

            )

        )

    )





    # -----------------------------------------------------

    # Mapping

    # -----------------------------------------------------



    mapping = np.load(

        mapping_path

    )





    user_ids = np.asarray(

        mapping[

            "user_ids"

        ]

    )





    item_ids = np.asarray(

        mapping[

            "item_ids"

        ]

    )





    if (

        model.user_factors.shape[0]

        !=

        len(

            user_ids

        )

    ):



        raise ValueError(

            "BPR user mapping 불일치"

        )





    if (

        model.item_factors.shape[0]

        !=

        len(

            item_ids

        )

    ):



        raise ValueError(

            "BPR item mapping 불일치"

        )





    if not np.all(

        user_ids[:-1]

        <=

        user_ids[1:]

    ):



        raise ValueError(

            "BPR user_ids가 정렬되어 있지 않습니다."

        )





    if not np.all(

        item_ids[:-1]

        <=

        item_ids[1:]

    ):



        raise ValueError(

            "BPR item_ids가 정렬되어 있지 않습니다."

        )





    # -----------------------------------------------------

    # 이미 본 interaction matrix

    # -----------------------------------------------------



    user_items = (



        load_npz(

            user_items_path

        ).tocsr()



        if user_items_path

        is not None



        else None

    )





    # fallback용

    seen_dict = (



        train_eval

        .groupby(

            "user_id"

        )[

            "app_id"

        ]

        .apply(list)

        .to_dict()

    )





    rows = []



    missing_users = 0





    # =====================================================

    # 400명 추천

    # =====================================================



    for i, user_id in enumerate(

        sampled_users[

            "user_id"

        ],

        start=1,

    ):



        user_idx = (

            base.find_sorted_index(

                user_ids,

                user_id,

            )

        )





        if user_idx is None:



            missing_users += 1

            continue





        # =================================================

        # 정상 implicit recommend

        # =================================================



        if user_items is not None:



            (

                item_indices,

                scores,

            ) = (

                model.recommend(



                    userid=

                        int(

                            user_idx

                        ),



                    user_items=

                        user_items[

                            user_idx

                        ],



                    N=

                        BPR_CACHE_N,



                    filter_already_liked_items=

                        True,

                )

            )





            app_ids = (

                item_ids[

                    item_indices

                ]

            )





        # =================================================

        # fallback

        # =================================================



        else:



            user_factor = (

                model

                .user_factors[

                    user_idx

                ]

            )





            scores = (



                model

                .item_factors

                @

                user_factor



            ).astype(

                np.float32,

                copy=False,

            )





            seen_ids = np.asarray(

                seen_dict.get(

                    user_id,

                    [],

                )

            )





            if len(

                seen_ids

            ):



                seen_idx = (

                    np.searchsorted(

                        item_ids,

                        seen_ids,

                    )

                )





                valid = (

                    seen_idx

                    <

                    len(

                        item_ids

                    )

                )





                positions = (

                    np.where(

                        valid

                    )[0]

                )





                if len(

                    positions

                ):



                    matched = (



                        item_ids[

                            seen_idx[

                                positions

                            ]

                        ]



                        ==



                        seen_ids[

                            positions

                        ]

                    )





                    real_seen = (

                        seen_idx[

                            positions[

                                matched

                            ]

                        ]

                    )





                    scores[

                        real_seen

                    ] = -np.inf





            finite_idx = (

                np.where(

                    np.isfinite(

                        scores

                    )

                )[0]

            )





            n_actual = min(

                BPR_CACHE_N,

                len(

                    finite_idx

                ),

            )





            if n_actual == 0:

                continue





            if (

                n_actual

                <

                len(

                    finite_idx

                )

            ):



                local = np.argpartition(



                    -scores[

                        finite_idx

                    ],



                    n_actual - 1,



                )[

                    :n_actual

                ]





                item_indices = (

                    finite_idx[

                        local

                    ]

                )





            else:



                item_indices = (

                    finite_idx

                )





            order = np.argsort(



                -scores[

                    item_indices

                ],



                kind="stable",

            )





            item_indices = (

                item_indices[

                    order

                ]

            )





            app_ids = (

                item_ids[

                    item_indices

                ]

            )





        # =================================================

        # 저장

        # =================================================



        for rank, app_id in enumerate(

            app_ids[

                :BPR_CACHE_N

            ],

            start=1,

        ):



            rows.append(

                (

                    user_id,

                    rank,

                    app_id,

                )

            )





        if (

            i % 100 == 0



            or



            i

            ==

            len(

                sampled_users

            )

        ):



            print(

                f"BPR Top200: "

                f"{i}/"

                f"{len(sampled_users)}"

            )





    bpr_df = pd.DataFrame(



        rows,



        columns=[

            "user_id",

            "rank",

            "app_id",

        ],

    )





    bpr_df.to_parquet(

        BPR_TOP200_PATH,

        index=False,

    )





    elapsed = (

        time.perf_counter()

        -

        start

    )





    print(

        "BPR mapping 없는 사용자:",

        missing_users

    )





    print(

        "BPR Top200 rows:",

        f"{len(bpr_df):,}"

    )





    print(

        "BPR Top200 생성 시간:",

        f"{elapsed:.2f}초"

    )





    print(

        "저장:",

        BPR_TOP200_PATH

    )





    return bpr_df





# =========================================================

# Candidate Size Sweep Recommender

# =========================================================



class CandidateSizeSweepRecommender:



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





        self.bpr = (

            bpr_dict

        )



        self.content = (

            content_dict

        )



        self.user = (

            user_dict

        )





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

            %

            100

            ==

            0

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

        # 1. 각 Retriever에서 정해진 수만큼

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

        # 2. Seen 제거

        #

        # 부족하더라도 다른 Retriever로 보충하지 않음.

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

        # 3. UNION

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

        # 4. Empty

        # =================================================



        if len(

            union_ids

        ) == 0:



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

        # 5. 기존 Item-Based Ranker

        #

        # Item Ranker 조건 변경 없음.

        # ITEM_K=30 그대로.

        #

        # 이미 저장된 400명 Item score cache 사용.

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

        # 6. Log

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

# Diagnostics

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





                # -----------------------------------------

                # Actual Candidate Count

                # -----------------------------------------



                "bpr_count":

                    len(

                        bpr_set

                    ),



                "content_count":

                    len(

                        content_set

                    ),



                "user_count":

                    len(

                        user_set

                    ),



                "union_count":

                    len(

                        union_set

                    ),



                "scoreable_count":

                    len(

                        scoreable_set

                    ),





                # -----------------------------------------

                # Candidate Recall

                # -----------------------------------------



                "bpr_recall":

                    (

                        len(

                            bpr_set

                            &

                            relevant

                        )

                        /

                        n_test

                    ),



                "content_recall":

                    (

                        len(

                            content_set

                            &

                            relevant

                        )

                        /

                        n_test

                    ),



                "user_recall":

                    (

                        len(

                            user_set

                            &

                            relevant

                        )

                        /

                        n_test

                    ),



                "union_recall":

                    (

                        len(

                            union_set

                            &

                            relevant

                        )

                        /

                        n_test

                    ),



                "scoreable_recall":

                    (

                        len(

                            scoreable_set

                            &

                            relevant

                        )

                        /

                        n_test

                    ),





                # -----------------------------------------

                # Final

                # -----------------------------------------



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

    elapsed,

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



        if total_recommended



        else 0.0

    )





    micro_recall = (



        total_hits

        /

        total_test



        if total_test



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

        )



        else 0.0

    )





    return {



        "case_name":

            case_name,



        "requested_candidate_size":

            config[

                "candidate_size"

            ],



        "bpr_n":

            config[

                "bpr_n"

            ],



        "content_n":

            config[

                "content_n"

            ],



        "user_n":

            config[

                "user_n"

            ],





        # ---------------------------------------------

        # 실제 Candidate 수

        # ---------------------------------------------



        "avg_bpr_candidates":

            diagnostics[

                "bpr_count"

            ].mean(),



        "avg_content_candidates":

            diagnostics[

                "content_count"

            ].mean(),



        "avg_user_candidates":

            diagnostics[

                "user_count"

            ].mean(),



        "avg_union_candidates":

            diagnostics[

                "union_count"

            ].mean(),



        "avg_scoreable_candidates":

            diagnostics[

                "scoreable_count"

            ].mean(),





        # ---------------------------------------------

        # Candidate Recall

        # ---------------------------------------------



        "bpr_candidate_recall":

            diagnostics[

                "bpr_recall"

            ].mean(),



        "content_candidate_recall":

            diagnostics[

                "content_recall"

            ].mean(),



        "user_candidate_recall":

            diagnostics[

                "user_recall"

            ].mean(),



        "union_candidate_recall":

            diagnostics[

                "union_recall"

            ].mean(),



        "scoreable_candidate_recall":

            diagnostics[

                "scoreable_recall"

            ].mean(),





        # ---------------------------------------------

        # Final

        # ---------------------------------------------



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





        # ---------------------------------------------

        # Micro

        # ---------------------------------------------



        "micro_precision":

            micro_precision,



        "micro_recall":

            micro_recall,



        "micro_f1":

            micro_f1,





        "evaluation_seconds":

            elapsed,

    }





# =========================================================

# Run One Experiment

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

        "Requested Candidate Size:",

        config[

            "candidate_size"

        ]

    )





    print(

        "BPR     :",

        config[

            "bpr_n"

        ]

    )



    print(

        "Content :",

        config[

            "content_n"

        ]

    )



    print(

        "User    :",

        config[

            "user_n"

        ]

    )





    recommender = (

        CandidateSizeSweepRecommender(



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

        f"{elapsed:.3f}초"

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



            elapsed=

                elapsed,

        )

    )





    # =====================================================

    # Save

    # =====================================================



    eval_df.to_csv(



        RESULT_DIR

        /

        f"{case_name}_eval.csv",



        index=False,

    )





    group_summary.to_csv(



        RESULT_DIR

        /

        f"{case_name}_group.csv",



        index=False,

    )





    diagnostics.to_csv(



        RESULT_DIR

        /

        f"{case_name}_diagnostics.csv",



        index=False,

    )





    # =====================================================

    # Console Summary

    # =====================================================



    print(

        "\n===== Candidate Stage ====="

    )





    print(

        "요청 Candidate:",

        summary[

            "requested_candidate_size"

        ]

    )





    print(

        "실제 평균 UNION:",

        f"{summary['avg_union_candidates']:.2f}"

    )





    print(

        "실제 평균 Scoreable:",

        f"{summary['avg_scoreable_candidates']:.2f}"

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

        "User Recall:",

        f"{summary['user_candidate_recall']:.6f}"

    )





    print(

        "UNION Candidate Recall:",

        f"{summary['union_candidate_recall']:.6f}"

    )





    print(

        "Item Scoreable Recall:",

        f"{summary['scoreable_candidate_recall']:.6f}"

    )





    print(

        "\n===== Final Top10 ====="

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





    return summary





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

        " Case 2 - Retriever Ratio Fine-Tuning"

    )



    print(

        " Candidate Size = 100 / User = 5 Fixed"

    )



    print(

        " Final Ranker = Item-Based"

    )



    print(

        "=================================================="

    )





    print(

        "\nRetriever Ratios:"

    )



    for (

        case_name,

        config,

    ) in EXPERIMENTS.items():



        print(

            f"{case_name}: "

            f"BPR={config['bpr_n']} / "

            f"Content={config['content_n']} / "

            f"User={config['user_n']}"

        )





    # =====================================================

    # 1. 동일 평가 사용자 400명

    # =====================================================



    sampled_users = (

        base.load_sampled_users()

    )





    # =====================================================

    # 2. 동일 평가 subset

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





    # -----------------------------------------------------

    # BPR는 Top200으로 확장

    # -----------------------------------------------------



    bpr_cache = (

        build_or_load_bpr_top200(



            sampled_users=

                sampled_users,



            train_eval=

                train_eval,

        )

    )





    # -----------------------------------------------------

    # Content는 최대 78개 필요

    # 기존 Top100이면 충분

    # -----------------------------------------------------



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





    # -----------------------------------------------------

    # User는 최대 10개 필요

    # 기존 cache 충분

    # -----------------------------------------------------



    cf_holder = {}





    user_cache = (

        base.build_user_cache(



            sampled_users,

            train_eval,

            holder,

            cf_holder,

        )

    )





    base.print_cache_coverage(

        "BPR Top200",

        bpr_cache,

        sampled_users,

    )





    base.print_cache_coverage(

        "Content Top100",

        content_cache,

        sampled_users,

    )





    base.print_cache_coverage(

        "User",

        user_cache,

        sampled_users,

    )





    # =====================================================

    # 4. Dictionary 변환

    # =====================================================



    print(

        "\nRetrieval cache -> dict"

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





    # =====================================================

    # 5. Item Ranker

    #

    # 직전 Ratio Sweep에서 만든

    # item_user_score_cache.pkl 사용.

    #

    # 정상이라면:

    #

    # FAST PATH 사용

    # 37M Train load 생략

    # Interaction Matrix 생성 생략

    # Item cosine 계산 생략

    # =====================================================



    item_ranker, fast_path = (

        ratio_base.prepare_item_ranker(



            sampled_users=

                sampled_users,



            train_eval=

                train_eval,

        )

    )





    print(

        "\nItem Ranker Fast Path:",

        fast_path

    )





    # =====================================================

    # 6. Sweep

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

            f"######## "

            f"CASE {case_idx}/{len(EXPERIMENTS)} "

            f"########"

        )





        summary = (

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





    # =====================================================

    # 7. Final Summary

    # =====================================================



    summary_df = (

        pd.DataFrame(

            summaries

        )

    )





    summary_df = (

        summary_df

        .sort_values(

            "requested_candidate_size"

        )

        .reset_index(

            drop=True

        )

    )





    summary_df.to_csv(

        SUMMARY_PATH,

        index=False,

    )





    display_columns = [



        "requested_candidate_size",



        "bpr_n",

        "content_n",

        "user_n",



        "avg_union_candidates",

        "avg_scoreable_candidates",



        "union_candidate_recall",

        "scoreable_candidate_recall",



        "precision_at_10",

        "recall_at_10",

        "hit_rate_at_10",

        "ndcg_at_10",



        "hits",



        "evaluation_seconds",

    ]





    print(

        "\n"

        "=================================================="

    )



    print(

        " Retriever Ratio Fine-Tuning Final Comparison"

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

    # 8. 변화량

    #

    # 이전 Size 대비 Candidate Recall / Final Recall 변화

    # =====================================================



    summary_df[

        "delta_union_candidate_recall"

    ] = (



        summary_df[

            "union_candidate_recall"

        ]

        .diff()

    )





    summary_df[

        "delta_final_recall"

    ] = (



        summary_df[

            "recall_at_10"

        ]

        .diff()

    )





    summary_df[

        "delta_ndcg"

    ] = (



        summary_df[

            "ndcg_at_10"

        ]

        .diff()

    )





    print(

        "\n"

        "=================================================="

    )



    print(

        " Marginal Gain"

    )



    print(

        "=================================================="

    )





    print(

        summary_df[

            [

                "requested_candidate_size",

                "delta_union_candidate_recall",

                "delta_final_recall",

                "delta_ndcg",

            ]

        ]

        .to_string(

            index=False

        )

    )





    # 변경량까지 포함해 다시 저장



    summary_df.to_csv(

        SUMMARY_PATH,

        index=False,

    )





    # =====================================================

    # 9. Best Final Performance

    # =====================================================



    best_ndcg_idx = (

        summary_df[

            "ndcg_at_10"

        ]

        .idxmax()

    )





    best_row = (

        summary_df

        .loc[

            best_ndcg_idx

        ]

    )





    print(

        "\n"

        "=================================================="

    )



    print(

        " Best NDCG Ratio"

    )



    print(

        "=================================================="

    )





    print(

        "Candidate Size:",

        int(

            best_row[

                "requested_candidate_size"

            ]

        )

    )





    print(

        "P@10:",

        f"{best_row['precision_at_10']:.6f}"

    )





    print(

        "R@10:",

        f"{best_row['recall_at_10']:.6f}"

    )





    print(

        "HR@10:",

        f"{best_row['hit_rate_at_10']:.6f}"

    )





    print(

        "NDCG@10:",

        f"{best_row['ndcg_at_10']:.6f}"

    )





    print(

        "Candidate Recall:",

        f"{best_row['union_candidate_recall']:.6f}"

    )





    # =====================================================

    # 10. Total Runtime

    # =====================================================



    total_elapsed = (

        time.perf_counter()

        -

        total_start

    )





    print(

        "\nSummary 저장:"

    )



    print(

        SUMMARY_PATH

    )





    print(

        "\n전체 실행 시간:",

        f"{total_elapsed:.2f}초"

    )





    print(

        "전체 실행 시간:",

        f"{total_elapsed / 60:.2f}분"

    )





if __name__ == "__main__":

    main()