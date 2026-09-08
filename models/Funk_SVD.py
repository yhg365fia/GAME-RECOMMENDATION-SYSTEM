import os
import numpy as np
import pandas as pd
import joblib

from surprise import Dataset, Reader, SVD


class FunkSVDRecommender:

    def __init__(
        self,
        n_factors=40,
        n_epochs=5,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
        model_path="models/funk_svd_model.pkl"
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state

        self.model_path = model_path

        self.model = None
        self.trainset = None
        self.raw_item_ids = None


    # ==========================================
    # 1. 학습 또는 기존 모델 불러오기
    # ==========================================

    def fit(self, train_df):
        """
        저장된 모델이 있으면 불러오고,
        없으면 새로 학습한 뒤 저장한다.
        """

        # --------------------------------------
        # 기존 모델 존재 -> 학습 생략
        # --------------------------------------

        if os.path.exists(self.model_path):

            print("저장된 Funk SVD 모델 발견")
            print("모델 불러오는 중...")

            saved = joblib.load(
                self.model_path
            )

            self.model = saved["model"]
            self.trainset = saved["trainset"]
            self.raw_item_ids = saved["raw_item_ids"]

            print("Funk SVD 모델 로드 완료")

            return self


        # --------------------------------------
        # 기존 모델 없음 -> 새로 학습
        # --------------------------------------

        print("저장된 모델 없음")
        print("새로운 Funk SVD 모델 학습 준비...")

        ratings = train_df[
            ["user_id", "app_id", "is_recommended"]
        ].copy()

        ratings["is_recommended"] = (
            ratings["is_recommended"]
            .astype(int)
        )

        reader = Reader(
            rating_scale=(0, 1)
        )

        data = Dataset.load_from_df(
            ratings[
                ["user_id", "app_id", "is_recommended"]
            ],
            reader
        )

        self.trainset = (
            data.build_full_trainset()
        )

        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state
        )

        print("Funk SVD 학습 시작...")

        self.model.fit(
            self.trainset
        )

        print("Funk SVD 학습 완료")


        # --------------------------------------
        # app_id 저장
        # --------------------------------------

        self.raw_item_ids = np.array([
            self.trainset.to_raw_iid(
                inner_iid
            )
            for inner_iid
            in self.trainset.all_items()
        ])


        # --------------------------------------
        # 저장 폴더 생성
        # --------------------------------------

        model_dir = os.path.dirname(
            self.model_path
        )

        if model_dir:
            os.makedirs(
                model_dir,
                exist_ok=True
            )


        # --------------------------------------
        # 모델 저장
        # --------------------------------------

        print("Funk SVD 모델 저장 중...")

        joblib.dump(
            {
                "model": self.model,
                "trainset": self.trainset,
                "raw_item_ids": self.raw_item_ids
            },
            self.model_path
        )

        print(
            f"모델 저장 완료: {self.model_path}"
        )

        return self


    # ==========================================
    # 2. 특정 user-item 점수 예측
    # ==========================================

    def predict_score(
        self,
        user_id,
        app_id
    ):

        if self.model is None:
            raise ValueError(
                "모델이 준비되지 않았습니다."
            )

        prediction = self.model.predict(
            user_id,
            app_id
        )

        return prediction.est


    # ==========================================
    # 3. Top-N 추천
    # ==========================================

    def recommend(
        self,
        user_id,
        app_id_list,
        top_n=10
    ):

        if self.model is None:
            raise ValueError(
                "모델이 준비되지 않았습니다."
            )


        # --------------------------------------
        # user_id -> 내부 user index
        # --------------------------------------

        try:

            inner_uid = (
                self.trainset.to_inner_uid(
                    user_id
                )
            )

        except ValueError:

            return pd.DataFrame(
                columns=["app_id"]
            )


        # --------------------------------------
        # 전체 게임 점수 계산
        # --------------------------------------

        user_vector = (
            self.model.pu[
                inner_uid
            ]
        )

        scores = (
            self.model.qi
            @ user_vector
        )


        # bias 포함
        if self.model.biased:

            scores = (
                self.trainset.global_mean
                + self.model.bu[inner_uid]
                + self.model.bi
                + scores
            )


        # --------------------------------------
        # 이미 본 게임 제외
        # --------------------------------------

        for app_id in app_id_list:

            try:

                inner_iid = (
                    self.trainset.to_inner_iid(
                        app_id
                    )
                )

                scores[inner_iid] = -np.inf

            except ValueError:
                continue


        # --------------------------------------
        # 추천 가능한 게임
        # --------------------------------------

        valid_idx = np.where(
            np.isfinite(scores)
        )[0]

        if len(valid_idx) == 0:

            return pd.DataFrame(
                columns=["app_id"]
            )


        n_actual = min(
            top_n,
            len(valid_idx)
        )


        # --------------------------------------
        # Top-N
        # --------------------------------------

        top_n_idx = valid_idx[
            np.argpartition(
                scores[valid_idx],
                -n_actual
            )[-n_actual:]
        ]

        top_n_idx = top_n_idx[
            np.argsort(
                -scores[top_n_idx]
            )
        ]


        recommended_app_ids = (
            self.raw_item_ids[
                top_n_idx
            ]
        )


        return pd.DataFrame({
            "app_id":
                recommended_app_ids
        })