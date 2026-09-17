import os
import gc

from collections import Counter

import numpy as np
import pandas as pd

from scipy.sparse import (
    csr_matrix,
    save_npz,
    load_npz
)

# 저장된 BPR 모델 load 용도로만 사용
from implicit.cpu.bpr import BayesianPersonalizedRanking

from data_split import load_mf_split

from evaluation import (
    build_mf_user_review_groups,
    stratified_sample_users,
    run_mf_evaluation,
    print_evaluation_report
)

# ==========================================
# BPR 관련 구현은 models/bpr.py에서 가져옴
# ==========================================

from models.bpr import (
    create_bpr_model,
    ImplicitBPRAdapter,
    fine_tune_with_explicit_false
)


print(os.getcwd())


# ==========================================
# BPR Experiment Config
# ==========================================

BPR_ITERATIONS = 10
BPR_FACTORS = 40
BPR_LEARNING_RATE = 0.05
BPR_REGULARIZATION = 0.001

BPR_VERIFY_NEGATIVE_SAMPLES = True
BPR_NUM_THREADS = 0


# ==========================================
# Explicit False 실험
# ==========================================
#
# False:
#   True + False 모두 positive interaction으로 취급
#
# True:
#   True만 base positive로 BPR 학습
#   False는 explicit negative로 추가 fine-tuning
# ==========================================

BPR_USE_EXPLICIT_FALSE = True

BPR_EXPLICIT_FALSE_EPOCHS = 1
BPR_EXPLICIT_FALSE_LEARNING_RATE = 0.01
BPR_EXPLICIT_FALSE_REGULARIZATION = 0.001
BPR_EXPLICIT_FALSE_BATCH_SIZE = 65536


RANDOM_STATE = 42


# ==========================================
# 추천 / 평가 설정
# ==========================================

TOP_N = 10

LOWER_BOUND = 10
UPPER_BOUND = 78

SAMPLE_PER_GROUP = 100


# False:
# Test의 True + False interaction 전체 평가
#
# True:
# Test의 True interaction만 정답으로 평가
#
# Explicit False 실험에서는
# True만 정답으로 보는 것이 기본
POSITIVE_ONLY = True


# ==========================================
# 저장 경로
# ==========================================

if BPR_USE_EXPLICIT_FALSE:

    MODEL_PATH = (
        f"models/implicit_bpr_{BPR_ITERATIONS}epoch_"
        f"explicit_false_{BPR_EXPLICIT_FALSE_EPOCHS}ft_model.npz"
    )

else:

    MODEL_PATH = (
        f"models/implicit_bpr_{BPR_ITERATIONS}epoch_model.npz"
    )


# 전체 interaction
# 추천 시 seen item 제거용
MATRIX_PATH = (
    "models/implicit_bpr_user_items.npz"
)

# True interaction만 저장
POSITIVE_MATRIX_PATH = (
    "models/implicit_bpr_positive_user_items.npz"
)

# False interaction만 저장
NEGATIVE_MATRIX_PATH = (
    "models/implicit_bpr_negative_user_items.npz"
)

# 실제 user_id / app_id ↔ 내부 index mapping
MAPPING_PATH = (
    "models/implicit_bpr_mapping.npz"
)


# ==========================================
# Games Metadata
# ==========================================

def load_games_meta():

    candidate_paths = [
        "data/cache/games.parquet",
        "data/cache/games_inc.parquet",
        "data/games.parquet",
        "data/raw/games.csv"
    ]

    for path in candidate_paths:

        if not os.path.exists(path):
            continue

        print(
            "캐시에서 games 데이터 불러오는 중..."
        )

        if path.endswith(".parquet"):

            return pd.read_parquet(
                path
            )

        return pd.read_csv(
            path
        )

    raise FileNotFoundError(
        "games metadata 파일을 찾을 수 없습니다."
    )


# ==========================================
# 게임 이름 → app_id
# ==========================================

def resolve_name_to_appid(
    meta,
    name
):

    candidates = meta[
        meta["Name"] == name
    ]

    if len(candidates) == 0:

        print(
            f"'{name}' 게임을 찾을 수 없습니다."
        )

        return None

    if len(candidates) == 1:

        return candidates[
            "app_id"
        ].iloc[0]

    print(
        f"\n'{name}'과 일치하는 게임이 여러 개 있습니다."
    )

    candidates = (
        candidates
        .reset_index(
            drop=True
        )
    )

    for i, row in candidates.iterrows():

        print(
            f"[{i}] "
            f"app_id: {row['app_id']} "
            f"- {row['Name']}"
        )

    while True:

        choice = input(
            "번호를 입력하세요: "
        ).strip()

        if (
            choice.isdigit()
            and int(choice) < len(candidates)
        ):

            return candidates.iloc[
                int(choice)
            ]["app_id"]

        print(
            "올바른 번호를 입력해주세요."
        )


# ==========================================
# Mask 기반 User × Item Matrix 생성
# ==========================================
#
# True-only / False-only matrix를 만들 때 사용
# ==========================================

def build_filtered_matrix(
    train_df,
    mask,
    user_ids,
    item_ids,
    shape
):

    filtered_df = train_df.loc[
        mask,
        [
            "user_id",
            "app_id"
        ]
    ]

    user_codes = np.searchsorted(
        user_ids,
        filtered_df["user_id"].to_numpy()
    ).astype(
        np.int32,
        copy=False
    )

    item_codes = np.searchsorted(
        item_ids,
        filtered_df["app_id"].to_numpy()
    ).astype(
        np.int32,
        copy=False
    )

    values = np.ones(
        len(filtered_df),
        dtype=np.float32
    )

    matrix = csr_matrix(
        (
            values,
            (
                user_codes,
                item_codes
            )
        ),
        shape=shape,
        dtype=np.float32
    )

    matrix.sum_duplicates()

    # 중복 interaction이 있어도 binary interaction으로 사용
    matrix.data[:] = 1.0

    del user_codes
    del item_codes
    del values
    del filtered_df

    gc.collect()

    return matrix


# ==========================================
# main
# ==========================================

def main():

    # ==========================================
    # 1. Games Metadata
    # ==========================================

    meta = load_games_meta()

    meta = (
        meta[
            [
                "app_id",
                "Name"
            ]
        ]
        .drop_duplicates(
            subset="app_id"
        )
    )


    # ==========================================
    # 2. MF Train / Test 데이터
    # ==========================================

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


    os.makedirs(
        "models",
        exist_ok=True
    )


    # ==========================================
    # 3. 전체 User × Item Sparse Matrix
    # ==========================================
    #
    # user_items:
    #
    # True + False interaction 전체 포함
    #
    # 학습용이라기보다
    # 추천 시 이미 interaction한 게임을
    # 제거하기 위한 Seen Matrix 역할도 함.
    # ==========================================

    if (
        os.path.exists(MATRIX_PATH)
        and
        os.path.exists(MAPPING_PATH)
    ):

        print(
            "\n===== implicit BPR 데이터 로드 ====="
        )

        user_items = load_npz(
            MATRIX_PATH
        )

        mapping = np.load(
            MAPPING_PATH
        )

        user_ids = mapping[
            "user_ids"
        ]

        item_ids = mapping[
            "item_ids"
        ]

        print(
            "Users:",
            len(user_ids)
        )

        print(
            "Items:",
            len(item_ids)
        )

        print(
            "Interactions:",
            user_items.nnz
        )


    # ==========================================
    # 저장된 Matrix가 없는 경우
    # ==========================================

    else:

        print(
            "\n===== implicit BPR Sparse Matrix 생성 ====="
        )

        # --------------------------------------
        # 실제 user_id → 내부 user index
        # --------------------------------------

        user_codes, user_ids = pd.factorize(
            train_df["user_id"],
            sort=True
        )

        # --------------------------------------
        # 실제 app_id → 내부 item index
        # --------------------------------------

        item_codes, item_ids = pd.factorize(
            train_df["app_id"],
            sort=True
        )

        user_codes = user_codes.astype(
            np.int32,
            copy=False
        )

        item_codes = item_codes.astype(
            np.int32,
            copy=False
        )

        print(
            "Users:",
            len(user_ids)
        )

        print(
            "Items:",
            len(item_ids)
        )

        print(
            "Raw Interactions:",
            len(train_df)
        )


        # 모든 Train interaction
        values = np.ones(
            len(train_df),
            dtype=np.float32
        )

        user_items = csr_matrix(
            (
                values,
                (
                    user_codes,
                    item_codes
                )
            ),
            shape=(
                len(user_ids),
                len(item_ids)
            ),
            dtype=np.float32
        )

        user_items.sum_duplicates()

        user_items.data[:] = 1.0


        print(
            "\nSparse Matrix:",
            user_items.shape
        )

        print(
            "Interactions:",
            user_items.nnz
        )


        # ======================================
        # 저장
        # ======================================

        print(
            "\nSparse Matrix 저장 중..."
        )

        save_npz(
            MATRIX_PATH,
            user_items,
            compressed=False
        )

        np.savez(
            MAPPING_PATH,

            user_ids=np.asarray(
                user_ids
            ),

            item_ids=np.asarray(
                item_ids
            )
        )

        print(
            "Sparse Matrix / Mapping 저장 완료"
        )


        del user_codes
        del item_codes
        del values

        gc.collect()


    # ==========================================
    # 4. BPR 모델
    # ==========================================

    if os.path.exists(
        MODEL_PATH
    ):

        print(
            "\n===== 저장된 implicit BPR 모델 로드 ====="
        )

        # 직접 BPR 생성은 하지 않음.
        # BayesianPersonalizedRanking은
        # 저장 모델 load 용도로만 사용.
        bpr_model = (
            BayesianPersonalizedRanking.load(
                MODEL_PATH
            )
        )


    # ==========================================
    # 저장 모델 없음
    # → 새로 학습
    # ==========================================

    else:

        print(
            "\n===== implicit BPR 전체 Train 학습 ====="
        )


        # ======================================
        # 기본 BPR
        # ======================================
        #
        # False일 경우:
        #
        # True + False interaction 전체를
        # positive interaction으로 취급
        # ======================================

        training_user_items = user_items


        # ======================================
        # Explicit False BPR
        # ======================================
        #
        # True:
        #
        # 1. True-only matrix로 기본 BPR
        # 2. False-only matrix를 explicit negative로
        #    fine-tuning
        # ======================================

        if BPR_USE_EXPLICIT_FALSE:

            # ----------------------------------
            # True / False mask
            # ----------------------------------

            positive_mask = (
                train_df[
                    "is_recommended"
                ]
                .to_numpy(
                    dtype=bool,
                    copy=False
                )
            )

            negative_mask = (
                ~positive_mask
            )


            # ==================================
            # True-only Matrix
            # ==================================

            if os.path.exists(
                POSITIVE_MATRIX_PATH
            ):

                print(
                    "\n===== True-only BPR Matrix 로드 ====="
                )

                positive_user_items = load_npz(
                    POSITIVE_MATRIX_PATH
                )

            else:

                print(
                    "\n===== True-only BPR Matrix 생성 ====="
                )

                positive_user_items = (
                    build_filtered_matrix(
                        train_df=train_df,
                        mask=positive_mask,
                        user_ids=user_ids,
                        item_ids=item_ids,
                        shape=user_items.shape
                    )
                )

                save_npz(
                    POSITIVE_MATRIX_PATH,
                    positive_user_items,
                    compressed=False
                )

                print(
                    "True-only interactions:",
                    positive_user_items.nnz
                )


            # ==================================
            # False-only Matrix
            # ==================================

            if os.path.exists(
                NEGATIVE_MATRIX_PATH
            ):

                print(
                    "\n===== False-only BPR Matrix 로드 ====="
                )

                negative_user_items = load_npz(
                    NEGATIVE_MATRIX_PATH
                )

            else:

                print(
                    "\n===== False-only BPR Matrix 생성 ====="
                )

                negative_user_items = (
                    build_filtered_matrix(
                        train_df=train_df,
                        mask=negative_mask,
                        user_ids=user_ids,
                        item_ids=item_ids,
                        shape=user_items.shape
                    )
                )

                save_npz(
                    NEGATIVE_MATRIX_PATH,
                    negative_user_items,
                    compressed=False
                )

                print(
                    "False-only interactions:",
                    negative_user_items.nnz
                )


            # 기본 implicit BPR 학습은
            # True interaction만 사용
            training_user_items = (
                positive_user_items
            )

            del positive_mask
            del negative_mask

            gc.collect()


        # ======================================
        # BPR 모델 생성
        # ======================================
        #
        # 모델 생성 로직은
        # models/bpr.py에서 관리
        # ======================================

        bpr_model = create_bpr_model(

            factors=BPR_FACTORS,

            learning_rate=(
                BPR_LEARNING_RATE
            ),

            regularization=(
                BPR_REGULARIZATION
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
            )
        )


        # ======================================
        # Base BPR 학습
        # ======================================

        print(
            "\n===== Training Start ====="
        )

        bpr_model.fit(
            training_user_items,
            show_progress=True
        )

        print(
            "\n===== Training Complete ====="
        )


        # ======================================
        # Explicit False Fine-Tuning
        # ======================================
        #
        # models/bpr.py의
        # fine_tune_with_explicit_false()
        # 사용
        # ======================================

        if BPR_USE_EXPLICIT_FALSE:

            bpr_model = (
                fine_tune_with_explicit_false(

                    model=bpr_model,

                    positive_items=(
                        positive_user_items
                    ),

                    negative_items=(
                        negative_user_items
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
                    )
                )
            )


        # ======================================
        # 모델 저장
        # ======================================

        bpr_model.save(
            MODEL_PATH
        )

        print(
            "\nimplicit BPR 모델 저장 완료:"
        )

        print(
            MODEL_PATH
        )


    # ==========================================
    # 5. 기존 Evaluation Pipeline 연결
    # ==========================================
    #
    # Adapter 역시 models/bpr.py 사용
    # ==========================================

    recommender = (
        ImplicitBPRAdapter(

            model=bpr_model,

            # True + False interaction 전체
            # → 이미 본 게임 필터링 용도
            user_items=user_items,

            user_ids=user_ids,

            item_ids=item_ids
        )
    )


    # ==========================================
    # 6. 랜덤 사용자 추천 테스트
    # ==========================================

    user_id = (
        test_df[
            "user_id"
        ]
        .drop_duplicates()
        .sample(
            n=1,
            random_state=RANDOM_STATE
        )
        .iloc[0]
    )

    print(
        f"\n랜덤 선택 user_id: {user_id}"
    )


    # ==========================================
    # 사용자의 Train interaction
    # ==========================================

    played_app_ids = (
        train_df.loc[
            train_df[
                "user_id"
            ] == user_id,
            "app_id"
        ]
        .tolist()
    )

    if len(played_app_ids) == 0:

        print(
            f"user_id {user_id}는 "
            "Train 데이터에 존재하지 않습니다."
        )

        return


    print(
        f"Train 상호작용 게임 수: "
        f"{len(played_app_ids)}"
    )


    # ==========================================
    # 7. Top-N 추천
    # ==========================================

    result = (
        recommender.recommend(

            user_id=user_id,

            app_id_list=(
                played_app_ids
            ),

            top_n=TOP_N
        )
    )


    # 게임 이름 추가

    result = result.merge(
        meta[
            [
                "app_id",
                "Name"
            ]
        ],
        on="app_id",
        how="left"
    )


    print(
        "\n===== BPR 추천 결과 ====="
    )

    print(
        result[
            [
                "app_id",
                "Name",
                "score"
            ]
        ].to_string(
            index=False
        )
    )


    # ==========================================
    # 진단 0
    # Seen Item 제외 확인
    # ==========================================

    seen_items = set(
        played_app_ids
    )

    recommended_items = set(
        result[
            "app_id"
        ].tolist()
    )

    overlap = (
        seen_items
        & recommended_items
    )

    print(
        "\nSeen item overlap:",
        overlap
    )

    if len(overlap) == 0:

        print(
            "정상: 이미 본 게임이 추천되지 않음"
        )

    else:

        print(
            "문제: 이미 본 게임이 추천됨"
        )


    # ==========================================
    # 8. 실제 Test 데이터
    # ==========================================

    user_test = test_df[
        test_df[
            "user_id"
        ] == user_id
    ]

    test_result = (
        user_test[
            [
                "app_id",
                "is_recommended"
            ]
        ]
        .merge(
            meta[
                [
                    "app_id",
                    "Name"
                ]
            ],
            on="app_id",
            how="left"
        )
    )

    print(
        "\n===== 실제 Test 데이터 ====="
    )

    print(
        test_result[
            [
                "app_id",
                "Name",
                "is_recommended"
            ]
        ].to_string(
            index=False
        )
    )


    # ==========================================
    # 추천 게임 학습 통계
    # ==========================================

    item_stats = (
        train_df
        .groupby(
            "app_id"
        )
        .agg(

            interaction_count=(
                "user_id",
                "size"
            ),

            true_count=(
                "is_recommended",
                "sum"
            ),

            true_ratio=(
                "is_recommended",
                "mean"
            )
        )
        .reset_index()
    )


    diagnosis_result = (
        result[
            [
                "app_id",
                "Name"
            ]
        ]
        .merge(
            item_stats,
            on="app_id",
            how="left"
        )
    )

    print(
        "\n===== 추천 게임 학습 통계 ====="
    )

    print(
        diagnosis_result.to_string(
            index=False
        )
    )


    # ==========================================
    # 9. BPR 평가
    # ==========================================

    print(
        "\n===== BPR 평가 시작 ====="
    )


    # ==========================================
    # 9-1. 평가 대상 사용자
    # ==========================================

    eligible_users = (
        build_mf_user_review_groups(

            train_df,

            test_df,

            lower_bound=(
                LOWER_BOUND
            ),

            upper_bound=(
                UPPER_BOUND
            )
        )
    )


    # ==========================================
    # 9-2. Stratified Sample
    # ==========================================

    sampled_users = (
        stratified_sample_users(

            eligible_users,

            sample_per_group=(
                SAMPLE_PER_GROUP
            ),

            random_state=(
                RANDOM_STATE
            )
        )
    )


    # ==========================================
    # 9-3. 평가 실행
    # ==========================================

    eval_df = (
        run_mf_evaluation(

            recommender=recommender,

            train_df=train_df,

            test_df=test_df,

            sampled_users=(
                sampled_users
            ),

            top_n=TOP_N,

            positive_only=(
                POSITIVE_ONLY
            )
        )
    )


    # ==========================================
    # 진단 2
    # 반복 추천되는 게임 확인
    # ==========================================

    sampled_ids = set(
        sampled_users[
            "user_id"
        ]
    )


    target_train = train_df[
        train_df[
            "user_id"
        ].isin(
            sampled_ids
        )
    ]


    train_by_user = (
        target_train
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


    recommend_counter = (
        Counter()
    )


    for sampled_user_id in sampled_ids:

        sampled_played_app_ids = (
            train_by_user.get(
                sampled_user_id,
                []
            )
        )

        if len(
            sampled_played_app_ids
        ) == 0:

            continue


        rec_result = (
            recommender.recommend(

                user_id=(
                    sampled_user_id
                ),

                app_id_list=(
                    sampled_played_app_ids
                ),

                top_n=TOP_N
            )
        )


        if (
            rec_result is None
            or len(rec_result) == 0
        ):

            continue


        recommend_counter.update(
            rec_result[
                "app_id"
            ].tolist()
        )


    # ==========================================
    # 반복 추천 TOP 30
    # ==========================================

    top_common = pd.DataFrame(

        recommend_counter.most_common(
            30
        ),

        columns=[
            "app_id",
            "recommend_count"
        ]
    )


    top_common = (
        top_common.merge(

            meta[
                [
                    "app_id",
                    "Name"
                ]
            ],

            on="app_id",

            how="left"
        )
    )


    top_common = (
        top_common.merge(

            item_stats,

            on="app_id",

            how="left"
        )
    )


    print(
        "\n===== 여러 사용자에게 반복 추천되는 게임 TOP 30 ====="
    )

    print(
        top_common.to_string(
            index=False
        )
    )


    # ==========================================
    # 9-4. 평가 결과 출력
    # ==========================================

    summary = (
        print_evaluation_report(

            eval_df,

            top_n=TOP_N
        )
    )


    # ==========================================
    # 9-5. 추천 개수 확인
    # ==========================================

    n_rec_summary = (
        eval_df
        .groupby(
            "review_group",
            observed=True
        )[
            "n_recommended"
        ]
        .agg(
            [
                "mean",
                "min",
                "max",
                "count"
            ]
        )
    )


    print(
        "\n===== 그룹별 추천 개수 ====="
    )

    print(
        n_rec_summary
    )


    print(
        f"\n전체 n_recommended 평균: "
        f"{eval_df['n_recommended'].mean():.2f} / 10"
    )


    full_ratio = (
        eval_df[
            "n_recommended"
        ] == TOP_N
    ).mean()


    print(
        f"{TOP_N}개 꽉 채워 추천된 유저 비율: "
        f"{full_ratio:.1%}"
    )


# ==========================================
# 실행
# ==========================================

if __name__ == "__main__":

    main()