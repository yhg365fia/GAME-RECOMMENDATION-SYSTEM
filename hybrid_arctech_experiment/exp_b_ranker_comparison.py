# hybrid_arctech_experiment/exp_b_ranker_comparison.py

import sys
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
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


from data_split import load_mf_split
from models.itembase import build_interaction_matrix

import exp_c_fusion_ablation as base


# =========================================================
# Config
# =========================================================

TOP_N = 10
ITEM_K = 30


# =========================================================
# 3 Ranker Experiments
# =========================================================
#
# 기본 3-model 중요도
#
# Item    = 0.5882
# BPR     = 0.2353
# Content = 0.1765
#
# Candidate Slot 총합 = 100
#
# 핵심 실험 원칙:
#
# "최종 Ranker로 사용하는 모델은
#  Candidate Retriever에서는 제외한다."
#
# 따라서 자기 자신의 Top-N을
# 다시 자기 점수로 정렬하는 중복 구조를 제거한다.
# =========================================================

EXPERIMENTS = {

    # =====================================================
    # Case 1
    #
    # BPR + Content
    #       ↓
    #     UNION
    #       ↓
    # Item-Based Ranker
    #
    # Item Retriever는 제거
    #
    # BPR : Content
    # 0.2353 : 0.1765
    # ≈ 57.1 : 42.9
    #
    # 슬롯 반올림:
    # 59 + 41 = 100
    # =====================================================

    "item_rank_without_item": {

        "ranker": "item",

        "item_n": 0,
        "bpr_n": 59,
        "content_n": 41,
    },


    # =====================================================
    # Case 2
    #
    # Item + BPR
    #      ↓
    #    UNION
    #      ↓
    # Content-Based Ranker
    #
    # Content Retriever는 제거
    #
    # Item : BPR
    # 0.5882 : 0.2353
    # ≈ 71.4 : 28.6
    # =====================================================

    "content_rank_without_content": {

        "ranker": "content",

        "item_n": 71,
        "bpr_n": 29,
        "content_n": 0,
    },


    # =====================================================
    # Case 3
    #
    # Item + Content
    #       ↓
    #     UNION
    #       ↓
    # BPR Ranker
    #
    # BPR Retriever는 제거
    #
    # Item : Content
    # 0.5882 : 0.1765
    # ≈ 76.9 : 23.1
    #
    # 기존 실험 배분 유지:
    # 78 + 22 = 100
    # =====================================================

    "bpr_rank_without_bpr": {

        "ranker": "bpr",

        "item_n": 78,
        "bpr_n": 0,
        "content_n": 22,
    },
}


# =========================================================
# Paths
# =========================================================

RESULT_DIR = (
    base.RESULT_DIR
    / "case2_ranker_comparison"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH = (
    RESULT_DIR
    / "case2_ranker_comparison_summary.csv"
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
# Item-Based Candidate Ranker
# =========================================================

class ItemBasedCandidateRanker:

    """
    기존 Item-Based 방식으로
    이미 만들어진 Candidate Pool만 재정렬한다.

    사용자 history
        ↓
    각 source item의 Top-K neighbor
        ↓
    similarity score 합산
        ↓
    Candidate Pool 안에서만 ranking
    """

    def __init__(
        self,
        interaction_matrix,
        game_to_idx,
        idx_to_game,
        k=30,
    ):

        self.game_to_idx = game_to_idx
        self.idx_to_game = idx_to_game

        self.k = k


        # User × Item
        # ->
        # Item × User

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
            "\nItem Ranker matrix normalization..."
        )


        self.item_matrix = (
            normalize(
                item_matrix,
                norm="l2",
                axis=1,
                copy=False,
            )
            .tocsr()
        )


        self.n_games = (
            self.item_matrix.shape[0]
        )


        self.neighbor_cache = {}


        print(
            "Item Ranker 준비 완료"
        )


    # =====================================================
    # Neighbor Cache
    # =====================================================

    def _cache_missing_neighbors(
        self,
        source_indices,
    ):

        source_indices = list(
            dict.fromkeys(
                source_indices
            )
        )


        missing = [

            idx

            for idx in source_indices

            if idx
            not in
            self.neighbor_cache
        ]


        if len(missing) == 0:
            return


        source_matrix = (
            self.item_matrix[
                missing
            ]
        )


        # normalized dot-product
        # = cosine similarity

        sims = (
            source_matrix
            @
            self.item_matrix.T
        ).tocsr()


        for (
            local_row,
            item_idx,
        ) in enumerate(
            missing
        ):

            row = sims.getrow(
                local_row
            )


            indices = (
                row.indices
            )

            data = (
                row.data
            )


            # 자기 자신 제외
            # positive similarity만 사용

            valid = (

                (indices != item_idx)

                &

                (data > 0)
            )


            indices = (
                indices[
                    valid
                ]
            )

            data = (
                data[
                    valid
                ]
            )


            if len(data) == 0:

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
                len(data),
            )


            pos = np.argpartition(
                data,
                -k,
            )[-k:]


            self.neighbor_cache[
                item_idx
            ] = (

                indices[
                    pos
                ].copy(),

                data[
                    pos
                ].copy(),
            )


    # =====================================================
    # 전체 Item-Based score
    # =====================================================

    def predict_scores(
        self,
        app_id_list,
    ):

        source_indices = [

            self.game_to_idx[
                app_id
            ]

            for app_id in app_id_list

            if app_id
            in
            self.game_to_idx
        ]


        if len(source_indices) == 0:
            return None


        self._cache_missing_neighbors(
            source_indices
        )


        scores = np.zeros(
            self.n_games,
            dtype=np.float64,
        )


        for item_idx in source_indices:

            (
                neighbor_idx,
                neighbor_score,
            ) = self.neighbor_cache[
                item_idx
            ]


            scores[
                neighbor_idx
            ] += neighbor_score


        # 이미 interaction한 게임 제외

        scores[
            source_indices
        ] = -np.inf


        return scores


    # =====================================================
    # Candidate Ranking
    # =====================================================

    def rank_candidates(
        self,
        app_id_list,
        candidate_ids,
        top_n=10,
    ):

        scores = (
            self.predict_scores(
                app_id_list
            )
        )


        if scores is None:

            return (
                [],
                [],
                [],
            )


        scoreable_ids = []
        scoreable_scores = []


        for app_id in candidate_ids:

            idx = (
                self.game_to_idx
                .get(
                    app_id
                )
            )


            if idx is None:
                continue


            score = (
                scores[
                    idx
                ]
            )


            # 기존 Item-Based 조건 유지
            # positive score만 사용

            if (
                not np.isfinite(
                    score
                )
                or
                score <= 0
            ):
                continue


            scoreable_ids.append(
                app_id
            )

            scoreable_scores.append(
                float(
                    score
                )
            )


        if len(scoreable_ids) == 0:

            return (
                [],
                [],
                [],
            )


        scoreable_ids = np.asarray(
            scoreable_ids
        )

        scoreable_scores = np.asarray(
            scoreable_scores
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


# =========================================================
# Content-Based Candidate Ranker
# =========================================================

class ContentCandidateRanker:

    """
    사용자의 positive Train item으로
    Content Profile을 생성한 뒤

    Candidate Pool 안의 게임만
    TF-IDF cosine similarity로 재정렬한다.
    """

    def __init__(
        self,
        tfidf_matrix,
        app_ids,
        positive_history,
    ):

        self.tfidf_matrix = (
            tfidf_matrix
            .tocsr()
        )

        self.app_ids = np.asarray(
            app_ids
        )


        self.app_to_idx = {

            app_id: idx

            for idx, app_id
            in enumerate(
                self.app_ids
            )
        }


        self.positive_history = (
            positive_history
        )


    # =====================================================
    # Candidate Ranking
    # =====================================================

    def rank_candidates(
        self,
        user_id,
        candidate_ids,
        top_n=10,
    ):

        source_app_ids = (
            self.positive_history
            .get(
                user_id,
                [],
            )
        )


        source_indices = [

            self.app_to_idx[
                app_id
            ]

            for app_id in source_app_ids

            if app_id
            in
            self.app_to_idx
        ]


        if len(source_indices) == 0:

            return (
                [],
                [],
                [],
            )


        # =================================================
        # User Content Profile
        # =================================================

        profile = (
            self.tfidf_matrix[
                source_indices
            ]
            .sum(
                axis=0
            )
        )


        profile = csr_matrix(
            profile,
            dtype=np.float32,
        )


        profile = normalize(
            profile,
            norm="l2",
            axis=1,
            copy=False,
        )


        # =================================================
        # Candidate Mapping
        # =================================================

        scoreable_ids = []
        scoreable_indices = []


        for app_id in candidate_ids:

            idx = (
                self.app_to_idx
                .get(
                    app_id
                )
            )


            if idx is None:
                continue


            scoreable_ids.append(
                app_id
            )

            scoreable_indices.append(
                idx
            )


        if len(scoreable_ids) == 0:

            return (
                [],
                [],
                [],
            )


        candidate_matrix = (
            self.tfidf_matrix[
                scoreable_indices
            ]
        )


        # cosine similarity

        scores = np.asarray(

            (
                profile
                @
                candidate_matrix.T
            )

            .toarray()

            .ravel()
        )


        order = np.argsort(
            -scores,
            kind="stable",
        )


        scoreable_ids = np.asarray(
            scoreable_ids
        )


        ranked_ids = (
            scoreable_ids[
                order
            ]
        )

        ranked_scores = (
            scores[
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


# =========================================================
# BPR Candidate Ranker
# =========================================================

class BPRCandidateRanker:

    """
    이미 생성된 Candidate Pool에 대해서만

        user_factor · item_factor

    BPR score를 계산해서 정렬한다.
    """

    def __init__(
        self,
        model,
        user_ids,
        item_ids,
    ):

        self.model = model

        self.user_ids = np.asarray(
            user_ids
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


    # =====================================================
    # User Index
    # =====================================================

    def find_user_idx(
        self,
        user_id,
    ):

        idx = int(
            np.searchsorted(
                self.user_ids,
                user_id,
            )
        )


        if idx >= len(
            self.user_ids
        ):
            return None


        if (
            self.user_ids[
                idx
            ]
            != user_id
        ):
            return None


        return idx


    # =====================================================
    # Candidate Ranking
    # =====================================================

    def rank_candidates(
        self,
        user_id,
        candidate_ids,
        top_n=10,
    ):

        user_idx = (
            self.find_user_idx(
                user_id
            )
        )


        if user_idx is None:

            return (
                [],
                [],
                [],
            )


        scoreable_ids = []
        item_indices = []


        for app_id in candidate_ids:

            idx = (
                self.item_to_idx
                .get(
                    app_id
                )
            )


            if idx is None:
                continue


            scoreable_ids.append(
                app_id
            )

            item_indices.append(
                idx
            )


        if len(scoreable_ids) == 0:

            return (
                [],
                [],
                [],
            )


        item_indices = np.asarray(
            item_indices,
            dtype=np.int64,
        )


        user_factor = (
            self.model
            .user_factors[
                user_idx
            ]
        )


        item_factors = (
            self.model
            .item_factors[
                item_indices
            ]
        )


        scores = (
            item_factors
            @
            user_factor
        )


        scoreable_ids = np.asarray(
            scoreable_ids
        )


        order = np.argsort(
            -scores,
            kind="stable",
        )


        ranked_ids = (
            scoreable_ids[
                order
            ]
        )

        ranked_scores = (
            scores[
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


# =========================================================
# Experimental Recommender
# =========================================================

class CandidateRankerExperiment:

    def __init__(
        self,
        case_name,
        config,
        item_cache,
        bpr_cache,
        content_cache,
        item_ranker,
        content_ranker,
        bpr_ranker,
    ):

        self.case_name = (
            case_name
        )

        self.config = (
            config
        )


        self.item = (
            cache_to_dict(
                item_cache
            )
        )

        self.bpr = (
            cache_to_dict(
                bpr_cache
            )
        )

        self.content = (
            cache_to_dict(
                content_cache
            )
        )


        self.item_ranker = (
            item_ranker
        )

        self.content_ranker = (
            content_ranker
        )

        self.bpr_ranker = (
            bpr_ranker
        )


        self.logs = {}

        self.pool_logs = []

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
            % 50
            == 0
        ):

            print(
                f"[{self.case_name}] "
                f"{self.call_count} users 완료"
            )


        if app_id_list is None:
            app_id_list = []


        played_set = set(
            app_id_list
        )


        # =================================================
        # 1. Retrieval
        # =================================================

        item_ids = (

            self.item
            .get(
                user_id,
                [],
            )
            [
                :self.config[
                    "item_n"
                ]
            ]
        )


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


        # =================================================
        # 2. Seen Item 제거
        # =================================================

        item_ids = [

            app_id

            for app_id in item_ids

            if app_id
            not in
            played_set
        ]


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


        # =================================================
        # 3. UNION
        # =================================================

        union_ids = (
            unique_preserve_order(

                item_ids

                +

                bpr_ids

                +

                content_ids
            )
        )


        if len(union_ids) == 0:

            self.logs[
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

                "scoreable":
                    [],

                "final":
                    [],

                "pure_ranker_top10":
                    [],
            }


            return pd.DataFrame(
                columns=[
                    "app_id",
                    "ranker_score",
                ]
            )


        # =================================================
        # 4. Final Ranker
        # =================================================

        ranker_name = (
            self.config[
                "ranker"
            ]
        )


        # -------------------------------------------------
        # Item-Based Ranker
        # -------------------------------------------------

        if (
            ranker_name
            ==
            "item"
        ):

            (
                final_ids,
                final_scores,
                scoreable_ids,
            ) = (
                self.item_ranker
                .rank_candidates(

                    app_id_list=
                        app_id_list,

                    candidate_ids=
                        union_ids,

                    top_n=
                        top_n,
                )
            )


            # 비교용 Pure Item Top10
            # 후보군에는 Item을 넣지 않았지만
            # 결과가 원래 Item 결과와 얼마나 비슷한지 확인

            pure_ranker_top10 = (

                self.item
                .get(
                    user_id,
                    [],
                )
                [:TOP_N]
            )


        # -------------------------------------------------
        # Content-Based Ranker
        # -------------------------------------------------

        elif (
            ranker_name
            ==
            "content"
        ):

            (
                final_ids,
                final_scores,
                scoreable_ids,
            ) = (
                self.content_ranker
                .rank_candidates(

                    user_id=
                        user_id,

                    candidate_ids=
                        union_ids,

                    top_n=
                        top_n,
                )
            )


            pure_ranker_top10 = (

                self.content
                .get(
                    user_id,
                    [],
                )
                [:TOP_N]
            )


        # -------------------------------------------------
        # BPR Ranker
        # -------------------------------------------------

        elif (
            ranker_name
            ==
            "bpr"
        ):

            (
                final_ids,
                final_scores,
                scoreable_ids,
            ) = (
                self.bpr_ranker
                .rank_candidates(

                    user_id=
                        user_id,

                    candidate_ids=
                        union_ids,

                    top_n=
                        top_n,
                )
            )


            pure_ranker_top10 = (

                self.bpr
                .get(
                    user_id,
                    [],
                )
                [:TOP_N]
            )


        else:

            raise ValueError(
                f"Unknown ranker: "
                f"{ranker_name}"
            )


        # =================================================
        # 5. Logs
        # =================================================

        self.logs[
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

            "scoreable":
                scoreable_ids,

            "final":
                final_ids,

            "pure_ranker_top10":
                pure_ranker_top10,
        }


        # =================================================
        # 6. Candidate Pool Log
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


        final_rank = {

            app_id:
                rank + 1

            for rank, app_id
            in enumerate(
                final_ids
            )
        }


        final_score = dict(
            zip(
                final_ids,
                final_scores,
            )
        )


        rows = []


        for app_id in union_ids:

            rows.append(
                {

                    "case_name":
                        self.case_name,

                    "ranker":
                        ranker_name,

                    "user_id":
                        user_id,

                    "app_id":
                        app_id,

                    "from_item":
                        int(
                            app_id
                            in
                            item_rank
                        ),

                    "from_bpr":
                        int(
                            app_id
                            in
                            bpr_rank
                        ),

                    "from_content":
                        int(
                            app_id
                            in
                            content_rank
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

                    "ranker_score":
                        final_score.get(
                            app_id,
                            np.nan,
                        ),

                    "final_rank":
                        final_rank.get(
                            app_id,
                            np.nan,
                        ),

                    "selected_top10":
                        int(
                            app_id
                            in
                            final_rank
                        ),

                    "pure_ranker_top10":
                        int(
                            app_id
                            in
                            pure_ranker_top10
                        ),
                }
            )


        self.pool_logs.append(
            pd.DataFrame(
                rows
            )
        )


        return pd.DataFrame(
            {

                "app_id":
                    final_ids,

                "ranker_score":
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
            test_df[
                "user_id"
            ].isin(
                sampled_ids
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


        if len(relevant) == 0:
            continue


        log = (
            recommender
            .logs
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


        pure_set = set(
            log.get(
                "pure_ranker_top10",
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

                "scoreable_candidate_count":
                    len(
                        scoreable_set
                    ),


                # =========================================
                # Candidate Recall
                # =========================================

                "item_candidate_recall":
                    (
                        len(
                            item_set
                            &
                            relevant
                        )
                        /
                        n_test
                    ),

                "bpr_candidate_recall":
                    (
                        len(
                            bpr_set
                            &
                            relevant
                        )
                        /
                        n_test
                    ),

                "content_candidate_recall":
                    (
                        len(
                            content_set
                            &
                            relevant
                        )
                        /
                        n_test
                    ),

                "union_candidate_recall":
                    (
                        len(
                            union_set
                            &
                            relevant
                        )
                        /
                        n_test
                    ),

                "scoreable_candidate_recall":
                    (
                        len(
                            scoreable_set
                            &
                            relevant
                        )
                        /
                        n_test
                    ),


                # =========================================
                # Final vs Pure Ranker
                # =========================================

                "final_pure_ranker_overlap":
                    len(
                        final_set
                        &
                        pure_set
                    ),

                "final_exactly_pure_ranker":
                    int(

                        final_set
                        ==
                        pure_set

                        and

                        len(
                            final_set
                        )
                        ==
                        TOP_N
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
    diagnostic_df,
):

    hits = int(
        eval_df[
            "hits"
        ].sum()
    )


    n_recommended = int(
        eval_df[
            "n_recommended"
        ].sum()
    )


    n_test = int(
        eval_df[
            "n_test"
        ].sum()
    )


    micro_p = (

        hits
        /
        n_recommended

        if n_recommended > 0

        else 0.0
    )


    micro_r = (

        hits
        /
        n_test

        if n_test > 0

        else 0.0
    )


    micro_f1 = (

        2
        *
        micro_p
        *
        micro_r
        /
        (
            micro_p
            +
            micro_r
        )

        if (
            micro_p
            +
            micro_r
        ) > 0

        else 0.0
    )


    return {

        "case_name":
            case_name,

        "ranker":
            config[
                "ranker"
            ],


        # Candidate Allocation

        "item_candidate_n":
            config[
                "item_n"
            ],

        "bpr_candidate_n":
            config[
                "bpr_n"
            ],

        "content_candidate_n":
            config[
                "content_n"
            ],


        # Final Metrics

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
            hits,

        "micro_precision":
            micro_p,

        "micro_recall":
            micro_r,

        "micro_f1":
            micro_f1,


        # Candidate Metrics

        "avg_union_candidates":
            diagnostic_df[
                "union_candidate_count"
            ].mean(),

        "union_candidate_recall":
            diagnostic_df[
                "union_candidate_recall"
            ].mean(),

        "scoreable_candidate_recall":
            diagnostic_df[
                "scoreable_candidate_recall"
            ].mean(),


        # Pure model comparison

        "avg_pure_ranker_overlap":
            diagnostic_df[
                "final_pure_ranker_overlap"
            ].mean(),

        "pure_ranker_exact_ratio":
            diagnostic_df[
                "final_exactly_pure_ranker"
            ].mean(),
    }


# =========================================================
# Run One Experiment
# =========================================================

def run_case(
    case_name,
    config,
    item_cache,
    bpr_cache,
    content_cache,
    item_ranker,
    content_ranker,
    bpr_ranker,
    train_eval,
    test_eval,
    sampled_users,
):

    print(
        "\n"
        "=================================================="
    )

    print(
        f" {case_name}"
    )

    print(
        "=================================================="
    )


    print(
        f"Ranker  : "
        f"{config['ranker']}"
    )

    print(
        f"Item    : "
        f"{config['item_n']}"
    )

    print(
        f"BPR     : "
        f"{config['bpr_n']}"
    )

    print(
        f"Content : "
        f"{config['content_n']}"
    )


    print(
        "\nCandidate UNION"
        " -> "
        f"{config['ranker'].upper()} Ranking"
        " -> Top10"
    )


    recommender = (
        CandidateRankerExperiment(

            case_name=
                case_name,

            config=
                config,

            item_cache=
                item_cache,

            bpr_cache=
                bpr_cache,

            content_cache=
                content_cache,

            item_ranker=
                item_ranker,

            content_ranker=
                content_ranker,

            bpr_ranker=
                bpr_ranker,
        )
    )


    # =====================================================
    # Evaluation
    # =====================================================

    start = time.time()


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
        time.time()
        -
        start
    )


    print(
        f"\n평가 시간: "
        f"{elapsed:.2f}초"
    )


    # =====================================================
    # Evaluation Report
    # =====================================================

    group_summary = (
        base.print_evaluation_report(

            eval_df,

            top_n=
                TOP_N,
        )
    )


    # =====================================================
    # Diagnostics
    # =====================================================

    diagnostic_df = (
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

            diagnostic_df=
                diagnostic_df,
        )
    )


    summary[
        "evaluation_seconds"
    ] = elapsed


    # =====================================================
    # Candidate Pool
    # =====================================================

    if len(
        recommender.pool_logs
    ) > 0:

        pool_df = (
            pd.concat(

                recommender
                .pool_logs,

                ignore_index=True,
            )
        )

    else:

        pool_df = (
            pd.DataFrame()
        )


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

    pool_path = (
        RESULT_DIR
        /
        f"{case_name}_pool.csv"
    )


    eval_df.to_csv(
        eval_path,
        index=False,
    )

    group_summary.to_csv(
        group_path,
        index=False,
    )

    diagnostic_df.to_csv(
        diagnostic_path,
        index=False,
    )

    pool_df.to_csv(
        pool_path,
        index=False,
    )


    # =====================================================
    # Console Summary
    # =====================================================

    print(
        "\n===== Candidate Stage ====="
    )

    print(
        "평균 UNION 후보:",
        f"{summary['avg_union_candidates']:.2f}"
    )

    print(
        "Candidate Recall:",
        f"{summary['union_candidate_recall']:.6f}"
    )

    print(
        "Ranker Scoreable Recall:",
        f"{summary['scoreable_candidate_recall']:.6f}"
    )


    print(
        "\n===== Final Ranking ====="
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


    print(
        "\n===== Pure Ranker 비교 ====="
    )

    print(
        "Final Top10 중 Pure Ranker Top10 평균 겹침:",
        f"{summary['avg_pure_ranker_overlap']:.2f}"
    )

    print(
        "Pure Ranker Top10과 완전 동일 비율:",
        f"{summary['pure_ranker_exact_ratio']:.4f}"
    )


    print(
        "\n저장:"
    )

    print(
        eval_path
    )

    print(
        group_path
    )

    print(
        diagnostic_path
    )

    print(
        pool_path
    )


    return (
        summary,
        eval_df,
    )


# =========================================================
# Main
# =========================================================

def main():

    total_start = (
        time.time()
    )


    print(
        "\n"
        "=================================================="
    )

    print(
        " Case 2 - Cross-Ranker Comparison"
    )

    print(
        " Ranker Model Excluded From Retrieval"
    )

    print(
        "=================================================="
    )


    print(
        "\n실험 구조:"
    )

    print(
        "1. BPR59 + Content41"
        " -> Item Ranker"
    )

    print(
        "2. Item71 + BPR29"
        " -> Content Ranker"
    )

    print(
        "3. Item78 + Content22"
        " -> BPR Ranker"
    )


    # =====================================================
    # 1. 평가 사용자
    # =====================================================

    sampled_users = (
        base.load_sampled_users()
    )


    # =====================================================
    # 2. 평가 subset
    # =====================================================

    holder = {}


    train_eval, test_eval = (
        base.load_or_create_eval_subset(

            sampled_users,

            holder,
        )
    )


    # =====================================================
    # 3. Retrieval Cache
    # =====================================================

    print(
        "\n===== Retrieval Cache ====="
    )


    cf_holder = {}


    item_cache = (
        base.build_item_cache(

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
        bpr_mapping_item_ids,
    ) = (
        base.load_bpr_mapping_only()
    )


    content_cache = (
        base.build_content_cache(

            sampled_users,
            train_eval,
            bpr_mapping_item_ids,
        )
    )


    base.print_cache_coverage(
        "Item",
        item_cache,
        sampled_users,
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


    # =====================================================
    # 4. BPR Ranker 준비
    # =====================================================

    print(
        "\n===== BPR Ranker 준비 ====="
    )


    (
        bpr_model_path,
        bpr_mapping_path,
        _,
    ) = (
        base.find_bpr_files()
    )


    bpr_model = (
        BayesianPersonalizedRanking
        .load(
            str(
                bpr_model_path
            )
        )
    )


    mapping = np.load(
        bpr_mapping_path
    )


    bpr_user_ids = np.asarray(
        mapping[
            "user_ids"
        ]
    )

    bpr_item_ids = np.asarray(
        mapping[
            "item_ids"
        ]
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


    bpr_ranker = (
        BPRCandidateRanker(

            model=
                bpr_model,

            user_ids=
                bpr_user_ids,

            item_ids=
                bpr_item_ids,
        )
    )


    # =====================================================
    # 5. Content Ranker 준비
    # =====================================================

    print(
        "\n===== Content Ranker 준비 ====="
    )


    (
        tfidf_matrix,
        content_app_ids,
    ) = (
        base.build_or_load_content_matrix(
            bpr_item_ids
        )
    )


    positive_history = (

        train_eval[
            train_eval[
                "is_recommended"
            ]
            == True
        ]

        .groupby(
            "user_id"
        )[
            "app_id"
        ]

        .apply(list)

        .to_dict()
    )


    content_ranker = (
        ContentCandidateRanker(

            tfidf_matrix=
                tfidf_matrix,

            app_ids=
                content_app_ids,

            positive_history=
                positive_history,
        )
    )


    print(
        "Content Ranker 준비 완료"
    )


    # =====================================================
    # 6. Item Ranker 준비
    # =====================================================

    print(
        "\n===== Item Ranker 준비 ====="
    )


    # 평가 subset cache를 사용했다면
    # full train이 holder에 없으므로 다시 load.

    if (
        "train"
        in
        holder
    ):

        full_train = (
            holder[
                "train"
            ]
        )

    else:

        full_train, _ = (
            load_mf_split()
        )


    (
        interaction_matrix,
        user_to_idx,
        game_to_idx,
        idx_to_game,
    ) = (
        build_interaction_matrix(
            full_train
        )
    )


    print(
        "Interaction Matrix:",
        interaction_matrix.shape,
    )


    item_ranker = (
        ItemBasedCandidateRanker(

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


    # Full dataframe는 이후 필요 없음

    del full_train

    holder.clear()
    cf_holder.clear()

    gc.collect()


    # =====================================================
    # 7. 3 Experiments
    # =====================================================

    summaries = []


    for (
        case_name,
        config,
    ) in (
        EXPERIMENTS
        .items()
    ):

        summary, _ = (
            run_case(

                case_name=
                    case_name,

                config=
                    config,

                item_cache=
                    item_cache,

                bpr_cache=
                    bpr_cache,

                content_cache=
                    content_cache,

                item_ranker=
                    item_ranker,

                content_ranker=
                    content_ranker,

                bpr_ranker=
                    bpr_ranker,

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
    # 8. Final Comparison
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
        "ranker",

        "item_candidate_n",
        "bpr_candidate_n",
        "content_candidate_n",

        "avg_union_candidates",
        "union_candidate_recall",
        "scoreable_candidate_recall",

        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",

        "hits",

        "avg_pure_ranker_overlap",
        "pure_ranker_exact_ratio",
    ]


    print(
        "\n"
        "=================================================="
    )

    print(
        " 3-Case Final Comparison"
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


    print(
        "\nSummary 저장:"
    )

    print(
        SUMMARY_PATH
    )


    print(
        "\n전체 실행 시간:",
        f"{time.time() - total_start:.1f}초"
    )


if __name__ == "__main__":
    main()