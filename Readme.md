# 🎮 Steam Game Recommendation System

Steam 게임 데이터를 활용하여 다양한 추천 시스템 알고리즘을 구현하고 성능을 비교하는 프로젝트입니다.

현재는 **Content-Based Recommendation System**, **User-Based Collaborative Filtering**, **Item-Based Collaborative Filtering**, **Funk SVD**, **BPR**의 Standalone 구현·평가를 완료하였으며,

* Steam 메타데이터 전처리
* User-based / Global Train-Test Split 설계
* TF-IDF Vectorization / Cosine Similarity 기반 추천
* Multi-Game Recommendation
* 리뷰 수 구간별 층화 표집 기반 평가 시스템 구축
* Precision@K, Recall@K, Hit Rate@K, NDCG@K + Macro/Micro 평가 구현
* **Content-Based / User-Based / Item-Based CF 구현 및 정량·정성평가 완료**
* **User-Based CF 데이터 누수 버그 발견 및 수정** + Sparsity 한계 정량 검증
* **Item-Based CF의 강한 co-consumption signal 확인** 및 Standalone 최고 성능 확보
* **Model-Based CF 전환 및 Global MF Train/Test 구조 설계** (총 41,154,794건 interaction)
* **Funk SVD 구현·평가·원인 진단·Ablation 완료 후 단계 종료** — item bias 영향은 확인했지만 근본 원인은 **rating prediction objective와 Top-N ranking objective의 불일치**로 결론
* **BPR 이론 학습 → 소규모 직접 구현 → `implicit` 기반 37M 전체 학습 완료**
* **BPR의 `is_recommended=False` signal 취급 문제 발견** 및 `True > Unseen → True > False` 2단계 학습 구조 구현
* **BPR 공정 재평가 및 최종 튜닝 완료** (iteration / regularization / factors / Factors×Regularization / Local Search)
* **Final BPR 확정** — P@10 0.05225 / R@10 0.06870 / HR@10 0.3850 / NDCG@10 0.06968
* **Item-Based / User-Based / BPR의 Collaborative 관계 재해석** — Item↔Item / User↔User / User↔Item latent preference
* **BPR의 latent factor / preference space와 PCA·Funk-SVD의 차이 재정리**
* **Item-Based를 Hybrid Anchor Model로 설정**하고 BPR 단독 Candidate Filtering의 Recall Ceiling 문제 발견
* **Hybrid Architecture A/B/C 실험안 설계** — Item-Based Candidate + Reranking / Multi-Retriever / Score·Rank Fusion
* **Hybrid 전용 동일 평가환경 구축 및 Item-Based Baseline 평가 완료** (400명, P@10 0.06325 / R@10 0.08144 / HR@10 0.3925 / NDCG@10 0.09129)

를 완료하였으며, 현재는 **Item-Based Top-200 Candidate를 1회 생성·저장한 뒤 Candidate Recall@20/50/100/200의 포화 구간을 확인하는 단계**입니다.

향후에는 Candidate Recall 결과를 바탕으로 **Architecture A → B → C**를 순차 비교하고, **Common Hit / Unique Hit / Recovered Hit / Lost Hit** 분석을 통해 각 모델의 실제 보완성을 검증한 뒤 Hybrid 구조를 최적화하고 웹 서비스 배포까지 확장하는 것을 목표로 합니다.

---

# 📊 전체 모델 성능 비교 (Top-10, 평가 유저 400명)

Standalone 모델의 대표 성능을 기준으로 비교했습니다.

| 모델                     |        P@10 |        R@10 |      HR@10 |     NDCG@10 | 비고                                             |
| ---------------------- | ----------: | ----------: | ---------: | ----------: | ---------------------------------------------- |
| Content-Based (TF-IDF) |      0.0268 |      0.0238 |     0.2250 |      0.0286 | 콘텐츠 semantic 유사성                               |
| User-Based CF          |      0.0545 |      0.0427 |     0.3011 |      0.0655 | User neighborhood, Sparsity에 취약                |
| **Item-Based CF**      |  **0.0783** |  **0.0880** | **0.4800** |  **0.1078** | **Standalone 최고 성능**, 강한 co-consumption signal |
| Funk SVD (biased)      |      0.0003 |      0.0006 |     0.0025 |      0.0004 | item bias가 랭킹 지배                               |
| Funk SVD (unbiased)    |      0.0037 |      0.0030 |     0.0350 |      0.0042 | bias 제거로 개선되나 절대 성능은 낮음                        |
| **Final BPR**          | **0.05225** | **0.06870** | **0.3850** | **0.06968** | Explicit False 활용 + 최종 튜닝                      |

**핵심 관찰**:

* 가장 복잡한 모델이 가장 높은 성능을 보이지 않았습니다. 추천 성능은 **모델 복잡도보다 데이터 구조 + 추천 목적 + 모델이 사용하는 signal**에 크게 좌우되었습니다.
* Item-Based는 실제 공동소비 관계를 직접 사용하는 **local item-item neighborhood signal**이 Steam 데이터에서 강하게 작동했습니다.
* BPR은 Item-Based/User-Based와 같은 Collaborative Interaction을 출발점으로 사용하지만, 직접 similarity를 만드는 대신 **User / Item latent vector를 학습해 개인별 상대적 선호 순서를 표현**합니다.
* 따라서 Item-Based와 BPR의 추천은 일부 겹칠 수 있지만 완전히 같은 정보를 표현하는 것은 아닙니다. Hybrid 단계에서는 **Standalone 성능 순위보다 Unique Hit / Complementarity가 더 중요한가?**를 검증합니다.

### Hybrid 전용 Item-Based Baseline

Architecture 비교를 위해 기존 Global MF Split과 동일한 평가 사용자 400명을 고정하고 `positive_only=True` 조건으로 Item-Based를 다시 평가했습니다.

| Metric       | Hybrid Baseline |
| ------------ | --------------: |
| Precision@10 |     **0.06325** |
| Recall@10    |     **0.08144** |
| Hit Rate@10  |      **0.3925** |
| NDCG@10      |     **0.09129** |
| Hits         |         **253** |

기존 Item-Based 결과와 절대값은 다르지만, 앞으로 Architecture A/B/C는 **같은 Global Split / 같은 400명 / 같은 Positive 정의 / 같은 Top-K / 같은 Evaluator**를 사용하므로 위 결과를 Hybrid의 실제 비교 기준선으로 사용합니다.

---

# 📌 Project Goals

## ✅ Current

* Steam 메타데이터 전처리 / Global Train-Test Split / Parquet 캐싱 / 프로젝트 모듈화
* Content-Based Recommendation (TF-IDF, Cosine Similarity, Multi-Game, AppID 식별 구조)
* 리뷰 수 구간별 층화 표집 평가 시스템 (Precision/Recall/Hit Rate/NDCG@K, Macro/Micro)
* **User-Based CF / Item-Based CF 전체 구현 및 정량·정성평가 완료**
* **Funk SVD 구현·평가·원인 진단·Ablation 완료 및 단계 종료**
* **BPR 전체 구현·False signal 개선·공정 재평가·최종 튜닝 완료**
* **Final BPR 확정** (P@10 0.05225 / R@10 0.06870 / HR@10 0.3850 / NDCG@10 0.06968)
* **Item-Based / User-Based / BPR의 Collaborative 관계 및 latent preference 구조 재정리**
* **Hybrid Architecture A/B/C 실험안 설계 완료**
* **Hybrid 평가 사용자 400명 고정 및 Item-Based Baseline 구축 완료**

---

## 🚧 In Progress

* **Item-Based Top-200 Candidate 생성 및 Cache**
* **Candidate Recall@20/50/100/200 Saturation 분석**
* Candidate Size 결정 후 Architecture A → B → C 순차 실험
* **Common Hit / Unique Hit / Recovered Hit / Lost Hit 분석**

---

## 🚀 Future

* 선택된 Hybrid Architecture의 Weight / Candidate Size / 모델 역할 최적화
* 필요 시 Situation-Aware Hybrid (Sparse/Dense User, Confidence 기반 Routing 등)
* Accuracy 외 Sparsity / Coverage / Personalization / Explainability / 실행시간 비교
* MAP@K
* Popularity Baseline / Popularity Bias / Near-Duplicate / Coverage 분석
* (선택적) ALS / Clustering 비교 실험
* FastAPI & Streamlit Deployment

---

# 🛠 Tech Stack

## Language

* Python

## Data Processing

* Pandas / NumPy / PyArrow (Parquet 캐싱) / SciPy (Sparse Matrix, CSR/LIL)

## Machine Learning

* Scikit-learn
* **Surprise** (Funk SVD 구현)
* **implicit** (대규모 BPR 학습 — Cython/C 최적화, CSR 기반, 멀티코어)

### Algorithms

* TF-IDF Vectorization / Cosine Similarity
* User-Based CF (Neighborhood-based, Top-K)
* Item-Based CF (행별 Top-K, Similarity Sum Aggregation)
* **Matrix Factorization / Funk SVD** (Surprise `SVD`, biased/unbiased 실험 완료 — 단계 종료)
* **BPR (Bayesian Personalized Ranking)** — 직접 구현(원리 이해용) + `implicit`(전체 학습용), **Explicit Negative Fine-Tuning(`True > False`) 자체 구현**

## Evaluation

* Precision@K / Recall@K / Hit Rate@K / NDCG@K (Macro) + Micro Precision/Recall/F1
* Stratified Sampling (리뷰 수 구간 기반), Qualitative Experiment
* Leave-N-Out 유저별 Split (Memory-based CF) / **Global Train/Test Split (Model-Based CF)**
* Pearson Correlation 기반 가설 검증
* 추천 게임 단위 Train 통계 분석(`groupby("app_id")`), `Counter` 기반 반복 추천 빈도 분석
* Hybrid Candidate Recall / Candidate Hit Rate / Hit Overlap 분석

## Visualization / Deployment

* Matplotlib / FastAPI / Streamlit

---

# 📂 Project Structure

```text
Game-Recommendation-System/
│
├── data/                                      # Git 제외
│   ├── raw/
│   ├── cache/                                 # 전처리/로딩 결과 캐싱 (Parquet)
│   └── split/                                 # Global MF Train/Test
│       ├── mf_train.parquet
│       └── mf_test.parquet
│
├── docs/
│   ├── Day01.md
│   ├── Day02.md
│   ├── ...
│   ├── Day20.md
│   ├── Day21.md
│   └── Day22.md
│
├── hybrid_arctech_experiment/
│   ├── baseline_item.py                       # Hybrid Item-Based 기준선
│   ├── exp_a_reranking.py                     # Architecture A
│   ├── exp_b_multi_retriever.py              # Architecture B placeholder
│   └── exp_c_fusion.py                        # Architecture C placeholder
│
├── models/
│   ├── content_base.py
│   ├── userbase.py
│   ├── itembase.py
│   ├── Funk_SVD.py
│   ├── bpr.py
│   ├── hybrid.py                              # 최종 Hybrid 구현용 placeholder
│   │
│   ├── run_model/
│   │   ├── runsvd.py
│   │   └── run_bpr.py
│   │
│   └── saved_model/                           # Git 제외
│       ├── 저장 모델 파일
│       └── results/
│           ├── BPR / Hybrid 실험 결과
│           ├── hybrid_sampled_users.csv
│           └── Candidate Cache / 평가 결과
│
├── notebook/
│   ├── 01_data_exploration.ipynb
│   ├── 02_vectorization.ipynb
│   ├── 03_content_based_recommendation.ipynb
│   ├── 04_construct_evaluation_system.ipynb
│   ├── 05_outpt_experiment.ipynb
│   ├── 06_userbased_vectorization.ipynb
│   └── 07_knn_modelbase.ipynb
│
├── preprocessing.py
├── data_split.py
├── evaluation.py
├── main.py
│
├── Readme.md
├── requirements.txt
└── .gitignore
```

`models/`는 모델 구현, `models/run_model/`은 개별 모델 실행, `models/saved_model/`은 학습된 모델 저장, `models/saved_model/results/`는 실험 결과, `hybrid_arctech_experiment/`는 Hybrid Architecture 실험, `notebook/`은 탐색·초기 실험, `docs/`는 일별 개발 기록으로 역할을 분리했습니다.

`exp_b_multi_retriever.py`, `exp_c_fusion.py`, `models/hybrid.py`는 현재 Architecture 설계에 맞춰 만들어둔 **placeholder 파일**이며, 실험 결과에 따라 실제 구현을 채워갈 예정입니다.

---

# ⚙️ Current Recommendation Pipeline

## Content-Based Pipeline

```text
Raw Steam Dataset → Data Loading (Parquet Cache) → Data Validation
      ↓
User-based Train/Test Split (런타임 유저 단위 적용)
      ↓
Data Preprocessing (Name Dedup, AppID 기준 정리) → Combined Features
      ↓
TF-IDF Vectorization → Name → AppID → Index 변환
      ↓
Cosine Similarity → Top-N Recommendation (AppID 기준 중복 제거)
      ↓
Recommendation Evaluation (정량 지표 + 정성적 실험)
```

## User-Based CF Pipeline

```text
User History (user_id, app_id, is_recommended)
      ↓
build_interaction_matrix() → Sparse User x Item Matrix (+1 / -1 인코딩)
      ↓
유저별 Train/Test Split (Leave-N-Out, 70/30)
      ↓
Query Vector 생성 (Train App ID만 사용, Test 누수 방지)
      ↓
Cosine Similarity → 자기 자신 제외(exclude_user_idx) → Top-K Neighbor (k=30)
      ↓
이웃 가중합 예측 점수 (Σ 유사도×상호작용 / Σ|유사도|)
      ↓
이미 플레이한 게임 제외 + 양수 점수만 채택 → Top-N 추천
      ↓
Recommendation Evaluation (구간별 Breakdown 포함)
```

## Item-Based CF Pipeline

```text
build_interaction_matrix() (User-Based와 동일 함수 재사용)
      ↓
interaction_matrix.T → item_matrix (Item x User)
      ↓
유저의 Train App ID → Source Items 행 조회 (Query Vector 불필요)
      ↓
Cosine Similarity (Source Items vs 전체 item_matrix)
      ↓
Self-Similarity 0 처리 → Source Item별 행 단위 Positive Top-K (k=30)
      ↓
동일 Candidate Similarity Sum Aggregation
      ↓
Train Item 제외 (-np.inf) → Positive Score만 채택 → Top-N 반환
      ↓
Recommendation Evaluation (기존 evaluation.py 재사용)
```

> Item-Based는 similarity 결과가 이미 output 공간(Item)에 존재하므로 User-Based처럼 "유사 이웃 → 이웃의 interaction"으로 한 단계 더 연결할 필요가 없습니다. 대신 여러 Source Item의 similarity를 하나의 candidate score로 합치는 aggregation이 추가 설계 지점이며, 현재는 similarity sum을 baseline으로 채택했습니다. **중요한 점은 이 matrix가 `True→+1, False→-1, 없음→0`으로 구성되어 있어 좋아요/싫어요 패턴이 모두 similarity에 반영된다는 것입니다** — 이 차이가 BPR 대비 높은 성능의 원인 중 하나일 가능성을 확인했습니다.

## Model-Based CF (Funk SVD) Pipeline — 완료·단계 종료

```text
Global Train/Test Split → mf_train.parquet (37,113,471건) / mf_test.parquet (4,041,323건)
      ↓
Surprise Dataset 변환 → SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02).fit()
      ↓
recommend() — bias 반영 score, seen-item 제거, 벡터화 Top-N
      ↓
400명 평가: Precision@10 0.0003, Hits 1/4000
      ↓
[가설1] positive_only=False 재평가 → Hits 그대로 1 → 기각
      ↓
[가설2] 추천 게임 Train 통계 + 반복 추천 분석 → item bias(b_i) 지배 패턴 발견
      ↓
[Ablation] biased=False → Hit Rate 0.0025→0.0350 개선, 그러나 여전히 낮음
      ↓
rating-vs-ranking objective mismatch가 근본 원인 → Funk SVD 단계 종료
```

## Model-Based CF (BPR) Pipeline — 완료·단계 종료

```text
[1단계] True interaction만 사용
      ↓
implicit BPR 학습 (True > Unseen)
iterations=15 / factors=60 / learning_rate=0.05 / regularization=0.006
      ↓
[2단계] Explicit Negative Fine-Tuning
각 User의 True item > False item pair 학습
epochs=1 / lr=0.01 / reg=0.001
      ↓
positive_only=True 조건으로 공정 재평가
      ↓
Regularization / Factors / Factors×Regularization / Local Search
      ↓
Final BPR 확정
      ↓
predict_all(user) → seen item 제외 → Top-K 추출
```

> **핵심 발견 1 — False Signal**: `implicit` BPR은 sparse matrix 값이 `+1`인지 `-1`인지보다 non-zero interaction 여부를 중심으로 해석하기 때문에 Steam의 `is_recommended=False`를 의도한 explicit negative로 직접 사용하지 못했습니다. 이를 해결하기 위해 **True > Unseen → True > False**의 2단계 학습 구조를 적용했습니다.
>
> **핵심 발견 2 — Problem Definition > Hyperparameter**: 가장 큰 개선은 factor나 iteration의 미세 조정이 아니라 **False를 모델이 어떤 signal로 이해해야 하는가**라는 문제 정의를 바로잡은 데서 발생했습니다.
>
> **Final BPR**: P@10 0.05225 / R@10 0.06870 / HR@10 0.3850 / NDCG@10 0.06968 / Micro F1@10 0.05747 / Hits 209. Standalone BPR 단계는 종료하고, Hybrid 실험에서 BPR의 실제 보완성이 확인될 경우에만 추가 튜닝을 다시 검토합니다.

---

# 🧩 Recommendation Identifier Flow

```text
Game Name (사용자 입력) → [동명이인 시 후보 선택] → AppID (고유 식별자)
      ↓ game_to_idx
Index (tfidf_matrix 상의 위치) → TF-IDF Vector → Cosine Similarity
      ↓
추천 Index → AppID → Game Name
```

> Game Name은 중복될 수 있지만 AppID는 고유하므로 내부 로직은 전부 **AppID 기준**으로 동작합니다. User/Item-Based CF, Funk SVD, BPR 모두 동일한 원칙을 유지합니다.

---

# 🔍 Qualitative Experiment Findings (Content-Based)

### 발견 1 — 텍스트에 없는 특성은 포착 불가

`Party Animals` 입력 시 장르/인원수는 유사했지만 "동물 캐릭터" 테마는 전혀 반영되지 않음. TF-IDF는 텍스트 메타데이터에 명시된 정보만 학습하므로 비주얼/테마적 특성은 원천적으로 포착 불가.

### 발견 2 — 장르 혼합 시 쏠림 현상

카드/덱빌딩 + 슈팅을 함께 입력했을 때 추천이 카드/덱빌딩 계열로 완전히 쏠리고 슈팅은 하나도 포함되지 않음. 이 발견은 협업 필터링에서 동일한 쏠림이 재현되는지 비교하는 기준점으로 활용했습니다.

---

# 🐛 User-Based CF 데이터 누수 진단 및 수정

첫 평가에서 Precision@10이 **0.6254**로 비정상적으로 높게 나와 원인을 진단했습니다.

**원인**: 평가용 `interaction_matrix`가 Train/Test 미분리 원본으로 생성되어, 평가 대상 유저 본인의 전체 데이터가 이웃 후보 풀에 남아 Test 정답이 예측 점수에 직접 새어 들어감.

**수정 과정**: `exclude_user_idx` 파라미터 추가 → 1차 수정 후에도 지표 동일 → 호출 체인 추적 → `run_evaluation()` 호출 시 `user_to_idx` 인자 전달 누락으로 제외 로직이 무력화된 것 확인 → 수정 후 Precision@10 0.0545로 정상화.

**교훈**: 여러 파일/함수로 나뉜 파이프라인에서는 내부 로직보다 **호출 체인(인자 전달 경로) 검증을 우선시**하는 디버깅 원칙(Outside-In)을 세우는 계기가 되었습니다.

---

# 🐛 NDCG 계산 구현 오류 수정

추천 결과를 `set(result["app_id"])`로 바로 변환해 **추천 순위 정보가 소실**되고 있는 것을 발견. `recommended_list`(순서 보존, NDCG용)와 `recommended_ids`(집합, Precision/Recall용)로 역할을 분리해 수정했고, 수정 후 NDCG@10은 **0.0655**로 확인했습니다.

---

# 📐 Macro vs Micro 평가 및 Sparsity 정량 검증

`n_recommended`가 유저마다 크게 달라(평균 7.28/10) Macro 평균에 편향이 생길 수 있어 **Micro Precision/Recall/F1**을 추가 도입했습니다.

| 비교 변수                   | Pearson r |  p-value |
| ----------------------- | --------: | -------: |
| n_games ↔ n_recommended |   +0.3096 | < 0.0001 |
| n_games ↔ precision     |   +0.2504 | < 0.0001 |

review_group별 추이와 함께 **"interaction이 적을수록 안정적인 이웃 형성이 어렵다"는 가설과 일관된 방향**으로 해석했습니다.

---

# 🔗 Self-Similarity 처리 원칙 (User-Based / Item-Based 공통)

* **User-Based CF**: 본인의 interaction(Test 포함)이 neighbor로 사용되는 **데이터 누수** 문제
* **Item-Based CF**: `A↔A=1.0`이 항상 최고 유사도로 Top-K를 차지하는 **trivial self-similarity** 문제

```python
similarities = cosine_similarity(...)
similarities[self_idx] = 0
# 이후 Top-K 선정
```

---

# 🧠 Item-Based CF 구조 이해 및 구현

| 구분              | User-Based CF                                      | Item-Based CF                         |
| --------------- | -------------------------------------------------- | ------------------------------------- |
| Matrix 방향       | User × Item                                        | Item × User (`interaction_matrix.T`)  |
| Query 필요 여부     | 필요                                                 | 불필요 (Item이 이미 행으로 존재)                 |
| Similarity 대상   | User ↔ User                                        | Item ↔ Item                           |
| Candidate Score | User similarity × interaction의 weighted prediction | 동일 candidate로 들어오는 item similarity 합산 |

**구현 완료 사항**: Self-similarity 0 처리 → Source Item별 행 단위 Positive Top-K(k=30) → Candidate Aggregation → Train Item Exclusion 및 Positive Score Filtering → Top-N 반환.

---

# 🧮 Model-Based CF: Funk SVD → BPR (완료·종료)

### Memory-based CF와의 핵심 차이

| 구분           | Memory-based CF        | Model-Based CF                         |
| ------------ | ---------------------- | -------------------------------------- |
| User/Item 표현 | 이미 존재하는 interaction 벡터 | 데이터로부터 학습된 latent vector               |
| 학습 단위        | 유저 평가 시점마다 개별 계산       | 전체 interaction을 한 번에 학습하는 Global Model |

### Global Train/Test 설계

* 평가 대상 유저(interaction 10~78개, **666,781명**): 70% Train / 30% Test, 그 외 유저: 100% Train
* 전체 41,154,794건 = Train 37,113,471건 + Test 4,041,323건
* `train_test_split()` 호출을 666,781회 → 최대 69회로 최적화 (interaction 개수별 position 재사용, 약 5초)

### Funk SVD — 원인 진단과 최종 결론

**원인 진단**: 가설 1(`positive_only` 정답 축소)은 재평가로 기각, 가설 2(item bias 지배)는 추천 게임 대부분이 True Ratio 0.94~1.00인 패턴으로 확보.

**Unbiased Ablation 결과** (400명, Top-10):

| Metric       |  Macro |  Micro |
| ------------ | -----: | -----: |
| Precision@10 | 0.0037 | 0.0037 |
| Recall@10    | 0.0030 | 0.0039 |
| Hit Rate@10  | 0.0350 |      - |
| NDCG@10      | 0.0042 |      - |
| F1@10        |      - | 0.0038 |

전체 Hit 15 / 전체 추천 4,000 / Test positive 3,850.

Hit Rate가 약 14배 개선되어 bias의 영향은 확인했지만 절대 수치는 여전히 매우 낮았고, Half-Life 시리즈 등 특정 게임의 반복 추천도 해소되지 않아 **rating-prediction objective와 Top-N ranking objective의 근본적 불일치**를 최종 원인으로 결론짓고 단계를 종료했습니다.

### BPR — 이론, 전체 학습, False signal 개선

**이론**: `(u,i,j)` triplet으로 positive가 negative보다 높은 점수를 갖도록 직접 학습합니다.

$$
x_{uij} = p_u^T(q_i-q_j)
$$

$$
L = -\log\sigma(x_{uij})
+\lambda(\|p_u\|^2+\|q_i\|^2+\|q_j\|^2)
$$

**직접 구현 vs implicit**: BPR 알고리즘 자체가 다른 게 아니라 **구현 최적화 수준이 다르다**는 것을 확인했습니다.

| 구분    | 직접 구현                           | implicit                    |
| ----- | ------------------------------- | --------------------------- |
| 처리 단위 | single interaction, Python loop | CSR Sparse Matrix 기반        |
| 자료구조  | dict / set / 함수 호출              | 연속된 factor array            |
| 실행 환경 | 순수 Python                       | Cython/C 최적화, 멀티코어          |
| 역할    | **BPR 원리를 이해하기 위한 구현**          | **대규모 데이터를 실제 학습시키기 위한 구현** |

### 초기 Iteration 실험

전체 데이터 학습 후 먼저 5 / 10 / 15 / 30 iteration을 비교했습니다.

| Iteration |       P@10 |       R@10 |      HR@10 |    NDCG@10 |   Micro F1 |    Hits |
| --------: | ---------: | ---------: | ---------: | ---------: | ---------: | ------: |
|         5 |     0.0315 |     0.0349 |     0.2475 |     0.0398 |     0.0321 |     126 |
|    **10** | **0.0335** | **0.0380** | **0.2600** | **0.0452** | **0.0341** | **134** |
|        15 |     0.0278 |     0.0319 |     0.2225 |     0.0369 |     0.0283 |     111 |
|        30 |     0.0295 |     0.0368 |     0.2175 |     0.0424 |     0.0301 |     118 |

특히 15 iteration에서는 **Train AUC는 증가했는데 Test 추천 성능은 감소**해, BPR training objective가 더 잘 최적화되는 것과 Top-N 추천 지표가 좋아지는 것은 같은 일이 아님을 확인했습니다.

### False signal 문제와 Explicit Negative Fine-Tuning

Item-Based는 `True→+1, False→-1`로 싫어요 정보까지 similarity에 반영하는 반면, `implicit` BPR은 sparse matrix의 non-zero interaction을 중심으로 학습하기 때문에 False를 의도한 explicit negative로 사용하지 못하고 있었습니다.

이를 해결하기 위해

```text
1단계: True > Unseen
      ↓
implicit BPR

2단계: True > False
      ↓
Explicit Negative Fine-Tuning
```

구조를 구현했습니다.

초기 실험에서는 다음과 같은 큰 개선이 관찰됐습니다.

| Metric       | 15 iter 기존 | 15 iter + False |
| ------------ | ---------: | --------------: |
| Precision@10 |     0.0278 |      **0.0520** |
| Recall@10    |     0.0319 |      **0.0670** |
| Hit Rate@10  |     0.2225 |      **0.3950** |
| NDCG@10      |     0.0369 |      **0.0718** |
| Micro F1     |     0.0283 |      **0.0572** |
| Hits         |        111 |         **208** |

다만 이 초기 비교는 기존 BPR이 `positive_only=False`, Explicit-False BPR은 `positive_only=True`였기 때문에 **평가 정답 정의가 달랐습니다.**

따라서 Day21에서는 이 결과를 그대로 결론으로 사용하지 않고 **두 모델을 `positive_only=True`로 통일하여 공정 재평가**했습니다.

### Explicit False 효과 공정 재검증

|   Iter | Explicit False |        P@10 |        R@10 |      HR@10 |     NDCG@10 |    Hits |
| -----: | :------------: | ----------: | ----------: | ---------: | ----------: | ------: |
|     10 |        X       |     0.02850 |     0.03731 |     0.2125 |     0.04032 |     114 |
|     10 |        O       |     0.02925 |     0.03855 |     0.2350 |     0.04282 |     117 |
|     15 |        X       |     0.02525 |     0.03415 |     0.2150 |     0.03863 |     101 |
| **15** |      **O**     | **0.03650** | **0.04630** | **0.2825** | **0.04875** | **146** |

10 iteration에서는 Explicit False의 개선폭이 작았지만 15 iteration에서는 훨씬 큰 차이가 나타났습니다.

즉 **False fine-tuning의 효과도 fine-tuning이 시작되는 latent representation의 상태에 따라 달라질 수 있다**고 해석했습니다.

학습 구조 자체가 바뀌면 기존의 optimal hyperparameter 역시 다시 달라질 수 있다는 것을 확인했습니다.

### Regularization 실험

Factors=40 / Iterations=15를 고정하고 Regularization을 비교했습니다.

|       Reg |        P@10 |        R@10 |      HR@10 |     NDCG@10 |    Hits |
| --------: | ----------: | ----------: | ---------: | ----------: | ------: |
|    0.0001 |     0.03225 |     0.04107 |     0.2550 |     0.04373 |     129 |
|     0.001 |     0.03650 |     0.04630 |     0.2825 |     0.04875 |     146 |
| **0.005** | **0.04575** | **0.05823** | **0.3625** | **0.06127** | **183** |
|     0.010 |     0.04200 |     0.05252 |     0.3250 |     0.05865 |     168 |
|     0.020 |     0.03150 |     0.04029 |     0.2650 |     0.04180 |     126 |

Train의 pair 관계를 더 정확하게 맞추는 것과 새로운 Test interaction에 일반화하는 것은 다른 문제라는 점을 다시 확인했습니다.

Regularization을 단순히 "학습을 방해하는 penalty"가 아니라 **Train interaction에 과도하게 맞춰지는 것을 제한해 unseen interaction에 더 잘 일반화하도록 만드는 장치**로 이해했습니다.

### Factors 실험

Reg=0.005를 고정했습니다.

| Factors |        P@10 |        R@10 |      HR@10 |     NDCG@10 |    Hits |
| ------: | ----------: | ----------: | ---------: | ----------: | ------: |
|      20 |     0.04300 |     0.05666 |     0.3375 |     0.06197 |     172 |
|  **40** | **0.04575** |     0.05823 | **0.3625** |     0.06127 | **183** |
|      80 |     0.04475 | **0.05924** |     0.3350 | **0.06400** |     179 |

Factors=80에서는 Recall과 NDCG가 높았지만 Precision / HR / Hits는 Factors=40보다 낮았습니다.

이를 단순히 "80이 더 좋다" 또는 "80이 과적합됐다"로 처리하지 않고,

* Precision → 추천 목록의 정확도
* Recall → 실제 정답을 얼마나 찾아냈는가
* HR → 몇 명의 사용자에게 최소 1개 이상 성공했는가
* NDCG → 맞힌 아이템을 얼마나 높은 순위에 배치했는가
* Hits → 전체 정답 개수

라는 **서로 다른 모델 행동을 측정하는 지표**로 해석했습니다.

### Factors × Regularization 상호작용 실험

Factors와 Regularization을 독립적으로 보지 않고 함께 변화시켰습니다.

| Factors |       Reg |        P@10 |        R@10 |      HR@10 |     NDCG@10 |    Hits |
| ------: | --------: | ----------: | ----------: | ---------: | ----------: | ------: |
|      20 |     0.001 |     0.03150 |     0.04089 |     0.2450 |     0.04148 |     126 |
|  **60** | **0.005** | **0.05125** | **0.06567** | **0.3750** | **0.06807** | **205** |
|      80 |    0.0075 |     0.04575 |     0.06024 |     0.3475 |     0.06439 |     183 |
|      80 |     0.010 |     0.04250 |     0.05772 |     0.3375 |     0.05868 |     170 |
|     100 |     0.010 |     0.04175 |     0.05277 |     0.3100 |     0.05927 |     167 |

더 큰 Factor를 강한 Regularization으로 제어하는 것이 항상 좋은 것이 아니라 **현재 데이터에 필요한 representation capacity 자체가 60 근처일 수 있다**는 가설을 얻었습니다.

### Local Search

`20 → 40 → 80 → 100`과 같은 넓은 탐색에서 `60 / 0.005`가 가장 좋게 나오자 탐색 범위를 좁혔습니다.

| Factors |       Reg |        P@10 |        R@10 |      HR@10 |     NDCG@10 |    Hits |
| ------: | --------: | ----------: | ----------: | ---------: | ----------: | ------: |
|      55 |     0.005 |     0.04875 |     0.06092 |     0.3500 |     0.06918 |     195 |
|      60 |     0.005 |     0.05125 |     0.06567 |     0.3750 |     0.06807 |     205 |
|  **60** | **0.006** | **0.05225** | **0.06870** | **0.3850** | **0.06968** | **209** |
|      65 |     0.005 |     0.04575 |     0.06132 |     0.3650 |     0.06256 |     183 |

추가로

```text
60 / 0.007
60 / 0.008
63 / 0.007
63 / 0.008
```

을 확인했지만 `60 / 0.006`을 넘지 못했습니다.

더 세밀한 탐색을 계속하면 미세하게 점수를 높일 가능성은 있지만, 프로젝트의 목적은 BPR benchmark competition이 아니라 **Hybrid Recommendation System 완성**이므로 이 지점에서 Standalone BPR 탐색을 종료했습니다.

### Final BPR

```python
iterations = 15
factors = 60
learning_rate = 0.05
regularization = 0.006

explicit_false = True
explicit_false_epochs = 1
explicit_false_learning_rate = 0.01
explicit_false_regularization = 0.001

positive_only = True
random_state = 42
```

| Metric       | Final Result |
| ------------ | -----------: |
| Precision@10 |  **0.05225** |
| Recall@10    | **0.068695** |
| Hit Rate@10  |   **0.3850** |
| NDCG@10      | **0.069681** |
| Micro F1@10  | **0.057473** |
| Hits         |      **209** |

BPR 단계 전체에서 얻은 가장 중요한 결론은 다음과 같습니다.

```text
Problem Definition
       >
Data Signal
       >
Hyperparameter
```

`is_recommended=False`를 모델이 어떤 의미로 받아들이는지를 바로잡는 것이 factors / regularization / iteration을 미세 조정하는 것보다 먼저였습니다.

또한 Standalone 최적 BPR과 Hybrid 내부에서의 최적 BPR은 반드시 동일하지 않을 수 있기 때문에, Hybrid에서 BPR의 역할이 실제로 중요하다고 확인될 경우에만 추가 튜닝을 다시 진행하기로 했습니다.

---

# 🧩 Hybrid Recommendation 설계 방향

Hybrid 단계에서는 단순히 Standalone 점수가 높은 모델을 섞는 것이 아니라, **같은 interaction을 각 모델이 어떤 관계로 표현하고 다른 모델의 실패를 실제로 얼마나 보완하는가**를 중심으로 설계합니다.

### Collaborative 모델 관계 재해석

Item-Based / User-Based / BPR은 모두 User-Item Interaction을 출발점으로 사용하므로 추천 결과가 일정 부분 겹칠 수 있습니다. 그러나 직접 표현하는 관계가 다릅니다.

| 모델             | 직접적으로 보는 관계                   | 핵심 질문                             |
| -------------- | ----------------------------- | --------------------------------- |
| **Item-Based** | Item ↔ Item                   | 이 게임과 같이 소비되는 게임은?                |
| **User-Based** | User ↔ User                   | 이 사용자와 비슷한 사용자는?                  |
| **BPR**        | User ↔ Item latent preference | 이 사용자에게 어떤 item을 더 높은 순위에 둬야 하는가? |

Item-Based는 사용자가 소비한 게임 주변의 실제 공동소비 관계를 직접 이용하는 **local neighborhood 방식**이고, BPR은 수많은 interaction을 User / Item latent vector로 표현하여 **개인별 상대적 preference ranking**을 학습합니다.

`factors=60`은 사용자의 취향을 RPG/Indie처럼 사람이 직접 지정한 60개 feature로 나눈다는 의미가 아니라, **ranking objective에 유리하도록 자동으로 학습된 60개의 latent preference 좌표**로 표현한다는 의미입니다.

PCA 역시 큰 차원을 작은 차원으로 줄인다는 표면적 공통점이 있지만,

```text
PCA → 원본 데이터의 분산을 잘 보존하는 저차원 공간

BPR → Positive Item > Negative/Unseen Item의 Ranking을 잘 만드는 latent 공간
```

이라는 목적 차이가 있습니다.

Funk-SVD와 BPR은 모두 User / Item latent vector 구조를 사용하지만, Funk-SVD는 prediction error를 줄이는 방향이고 BPR은 pairwise ranking을 직접 최적화합니다. 프로젝트에서 BPR이 Funk-SVD보다 훨씬 높은 Top-N 성능을 보인 것도 **Top-N Recommendation이라는 문제 정의와 objective의 정합성** 관점으로 해석했습니다.

### 첫 Hybrid 가설 — BPR Candidate Filter

처음에는 다음 구조를 생각했습니다.

```text
BPR
 ↓ Latent Preference 기반 Candidate Filtering
Item-Based / Content-Based
 ↓
Final Ranking
```

하지만 BPR이 정답 Item을 Candidate Pool에서 제거하면 뒤 단계의 Item-Based나 Content-Based가 그 Item을 아무리 잘 평가할 수 있어도 복구할 수 없습니다.

즉 **BPR을 유일한 Candidate Generator로 두면 전체 Hybrid Recall의 상한이 BPR Candidate Recall에 묶일 수 있다**는 문제를 발견했습니다.

또 처음에는 "BPR은 filtering에는 어울리지만 최종 ranking에는 덜 어울린다"고 생각했지만, BPR 자체가 pairwise ranking objective를 가진 모델이므로 이 해석은 수정했습니다.

정확한 표현은 **현재 Steam 데이터와 현재 BPR 구현에서 BPR의 Standalone Top-N Ranking 성능이 Item-Based보다 낮다**입니다.

### Multi-Retriever로 확장

```text
Item-Based Candidate ─┐
BPR Candidate ────────┼→ UNION → Candidate Pool → Ranking
Content Candidate ────┘
```

하나의 Retriever에 Candidate Generation을 모두 맡기지 않고, Item-Based가 놓친 후보를 BPR이, Collaborative 모델들이 놓친 후보를 Content-Based가 보완할 수 있는 구조를 비교하기로 했습니다.

Hybrid에서의 질문도 다음처럼 바뀌었습니다.

```text
기존: 어떤 모델이 가장 좋은가?

현재: 이 모델은 다른 모델이 모르는 무엇을 알고 있는가?
```

따라서 Standalone 지표뿐 아니라 **Common Hit / Item-Based Unique Hit / BPR Unique Hit / Content Unique Hit / Recovered Hit / Lost Hit**를 함께 분석합니다.

특히 **Item-Based 실패 & BPR 성공**이 충분히 존재한다면 Item-Based의 직접적인 item-item neighborhood가 포착하지 못한 latent preference를 BPR이 실제로 보완하고 있다는 근거가 됩니다.

### 현재 모델별 Hybrid 역할 가설

| 모델                | 현재 역할 가설                                           |
| ----------------- | -------------------------------------------------- |
| **Item-Based**    | Hybrid Anchor. 가장 강한 Candidate / Ranking 기본 signal |
| **BPR**           | User-Item latent preference를 통한 Item-Based 보완 가능성  |
| **Content-Based** | Collaborative 모델과 다른 semantic information 제공       |
| **User-Based**    | 유사 사용자 행동을 이용하는 추가 collaborative signal            |
| Funk SVD          | 현재 성능상 핵심 Hybrid 후보에서는 우선 제외                       |

위 역할은 아직 **실험 전 가설**이며 Hit Overlap / Unique Hit / 정성평가를 통해 검증합니다.

### Architecture A — Item-Based Candidate + Reranking

```text
Item-Based Candidate
      ↓
Item Score + BPR Score + Content Score
      ↓
Reranking
      ↓
Top-10
```

> **질문**: Item-Based가 Candidate 자체는 충분히 잘 찾고 있고, 다른 모델은 순위만 보완하면 되는가?

### Architecture B — Multi-Retriever

```text
Item-Based Candidate ─┐
BPR Candidate ────────┼→ Candidate UNION → Ranking → Top-10
Content Candidate ────┘
```

> **질문**: Item-Based가 놓친 정답 Candidate를 다른 모델이 실제로 복구해야 하는가?

### Architecture C — Score / Rank Fusion

```text
Item-Based Recommendation ─┐
BPR Recommendation ────────┼→ Score / Rank Fusion → Top-10
Content Recommendation ────┘
```

> **질문**: 복잡한 Retrieval-Reranking Pipeline 없이 단순 결합만으로도 개선되는가?

정밀 Weight Search / Situation-Aware Routing / Rule-Based Reranking / Learning-to-Rank는 Architecture가 선택되기 전에는 진행하지 않습니다.

현재 원칙은

```text
Architecture
    ↓
Model Role
    ↓
Weight Tuning
```

순서입니다.

### Hybrid 전용 Item-Based Baseline

기존 Global MF Split과 고정 평가 사용자 400명을 이용해 Item-Based를 다시 평가했습니다.

| Metric                        |                   Result |
| ----------------------------- | -----------------------: |
| Precision@10                  |              **0.06325** |
| Recall@10                     |              **0.08144** |
| Hit Rate@10                   |               **0.3925** |
| NDCG@10                       |              **0.09129** |
| Micro Precision / Recall / F1 | 0.0633 / 0.0773 / 0.0696 |
| Hits                          |                  **253** |
| 추천 수 / Test Relevant Items    |            4,000 / 3,273 |

400명 모두 정확히 10개 추천을 받았고 평가에는 약 **1811.1초(30분 11초)**가 걸렸습니다.

Review Group별 결과:

| Review Group |      P@10 |       R@10 |    HR@10 |        NDCG |
| ------------ | --------: | ---------: | -------: | ----------: |
| 10–15        |     0.041 | **0.1110** |     0.26 |     0.09657 |
| 16–25        |     0.040 |    0.06986 |     0.28 |     0.06752 |
| 26–45        |     0.064 |    0.07468 |     0.43 |     0.07999 |
| 46–78        | **0.108** |    0.07021 | **0.60** | **0.12109** |

Interaction 수와 Precision 사이에는 `Pearson r = 0.2893, p < 0.0001`의 양의 관계가 관찰됐습니다.

### Item-Based Inference 비용과 Candidate Cache

Item-Based는 모델이 저장되어 있어도 `recommend()` 시점마다 다음 연산을 다시 수행합니다.

```python
cosine_similarity(
    source_items,
    self.item_matrix,
    dense_output=False
)
```

즉 **학습된 모델을 저장했다는 것과 Item-Item similarity 또는 추천 결과를 저장했다는 것은 다릅니다.**

전체 37,567개 Item의 similarity / Top-K neighbor를 미리 계산하는 방법도 있지만, 현재 목적은 Serving 최적화가 아니라 **Hybrid Architecture 검증**이므로 우선 보류했습니다.

대신 고정된 400명의 **Item-Based Top-200 Candidate를 한 번만 계산해 Cache**하고 이후 실험에서 재사용합니다.

```text
저장 예정:

models/saved_model/results/item_top200_candidates.csv

형식:

user_id | rank | app_id

최대 약 400 × 200 = 80,000 rows
```

### 다음 실험 — Candidate Recall Saturation

```text
Item-Based Top-200 생성 + 저장
        ↓
Candidate Recall@20
Candidate Recall@50
Candidate Recall@100
Candidate Recall@200
        ↓
Recall 증가폭이 포화되는 구간 확인
        ↓
Architecture A Candidate Size 결정
```

Top-20 / 50 / 100 / 200을 각각 다시 추천하지 않고 Top-200을 한 번 계산한 뒤 앞에서부터 잘라 사용합니다.

Candidate Recall이 빠르게 높은 수준으로 포화되면 **Architecture A(Item-Based Retriever + Reranking)**에 근거가 생기고, 200까지 늘려도 정답이 많이 누락되면 **Architecture B(Multi-Retriever)**의 필요성이 커집니다.

---

# 🐛 알려진 이슈 (To-Do)

* **Name = NaN metadata mismatch**: Item-Based CF 정성평가 중 일부 게임 이름이 NaN으로 조회. 원인 분석 보류
* **Party Animals 입력 시 빈 DataFrame 반환**: 정성평가 대상에서 제외. 원인 분석 보류
* **`ModuleNotFoundError: No module named 'data_split'`**: 하위 폴더 파일 직접 실행 시 import 경로 문제. Project Root import 구조로 해결 방향 정리
* **Item-Based inference 비용**: 400명 평가에 약 30분 소요. Hybrid 반복 실험에서는 Top-200 Candidate Cache를 우선 사용
* **Hybrid 모델 역할은 아직 가설 단계**: Item-Based Anchor / BPR latent 보완 / Content semantic 보완이 실제로 유효한지는 Hit Overlap / Unique Hit 분석으로 검증 예정
* **실험 저장 경로 통일 필요**: 현재 프로젝트의 기준 저장 구조는 `models/saved_model/` 및 `models/saved_model/results/`이지만 일부 이전 BPR 실험 Runner에는 과거 실험 경로가 남아 있어 최종 리팩토링 시 통일 필요
* **Dependency 동기화 필요**: 실제 프로젝트에서는 `implicit`, Surprise, PyArrow 등을 사용하고 있으므로 최종 정리 단계에서 `requirements.txt`와 실제 실행 환경을 다시 동기화할 필요가 있음
* **BPR Config 단일화 필요**: Final BPR은 `iterations=15 / factors=60 / reg=0.006`으로 확정되었지만 일부 일반 실행 코드에는 이전 실험용 기본값이 남아 있어 최종 리팩토링 시 Config Source를 하나로 통일할 예정

---

# ✅ Implemented Features

## Content-Based

Steam Metadata Loading (Parquet Caching), Data Validation, User-based Train/Test Split, Combined Features, TF-IDF Vectorization, Cosine Similarity Recommendation, Multi-Game Recommendation, AppID 기반 식별 및 동명이인 처리, 정성적 실험

## User-Based CF

Sparse Interaction Matrix 구축, Cosine Similarity 기반 Top-K 이웃 탐색, 자기 자신 제외 로직, Query Vector 생성(Test 누수 방지), 이웃 가중합 예측 점수 계산, Sparsity 정량 검증

## Item-Based CF

User×Item → Item×User 구조 변환, Self-Similarity 0 처리, Source Item별 행 단위 Positive Top-K, Candidate Aggregation, Train Item Exclusion, 정량·정성평가

## Model-Based CF — Funk SVD (완료·종료)

Global Train/Test Split 설계 및 최적화, Surprise 기반 학습, `recommend()` Top-N 구현, MF 전용 Evaluation, 400명 정량평가(Biased/Unbiased), `positive_only` 가설 기각, item bias 가설 확보, `biased=False` Ablation 완료, objective mismatch 분석

## Model-Based CF — BPR (완료·종료)

* BPR 이론 학습 및 소규모 프로토타입 구현·검증
* **`implicit` 기반 전체 데이터(37M) 학습 및 5/10/15/30 iteration 비교 실험**
* **False signal 취급 문제 발견 및 2단계 학습 구조 구현** (True > Unseen → True > False)
* **`positive_only=True` 공정 재평가 완료**
* **Regularization / Factors / Factors×Regularization / Local Search 완료**
* **Final BPR 확정** (P@10 0.05225 / R@10 0.06870 / HR@10 0.3850 / NDCG@10 0.06968)

## Hybrid Recommendation — 진행 중

* **Item-Based / User-Based / BPR의 Collaborative signal 차이 및 latent preference 구조 재정리**
* **Item-Based를 Hybrid Anchor Model로 설정**
* **BPR 단독 Candidate Filtering의 Recall Ceiling 문제 발견**
* **Architecture A / B / C 실험안 설계**
* **동일 Global MF Split + 동일 평가 사용자 400명 Hybrid 평가환경 구축**
* **Item-Based Hybrid Baseline 평가 완료**
* **Top-200 Candidate Cache + Candidate Recall Saturation 실험 설계**

## Evaluation (공통)

리뷰 수 구간 기반 층화 표집, Precision/Recall/Hit Rate/NDCG@K, Macro/Micro Precision·Recall·F1, Pearson Correlation 가설 검증, 구간별 Breakdown, 정성평가, Hybrid Candidate Recall / Hit Overlap 분석 예정

---

# 🚀 Development Roadmap

## ✅ V1. Content-Based Recommendation

전체 완료

## ✅ V2. Collaborative Filtering

### User-Based CF / Item-Based CF

전체 완료

### Model-Based CF

#### 1. Funk SVD — 완료·종료

* [x] Memory-based vs Model-Based(Global Model) 학습 구조 차이 파악
* [x] Global Train/Test Split 설계 및 대규모 split 성능 최적화
* [x] Surprise 기반 구현, Top-N 추천 함수, evaluation 연결, 400명 정량평가
* [x] 성능 저조 원인 진단 (`positive_only` 가설 기각 → item bias 가설 확보)
* [x] `biased=False` Ablation 실행 및 결과 해석
* [x] Funk SVD 단계 최종 종료 (objective mismatch로 결론)

#### 2. BPR — Top-N Ranking 직접 학습 (완료·종료)

* [x] Funk-SVD와 BPR의 차이 이해, `(u,i,j)` triplet·sigmoid·likelihood·loss 이해
* [x] 소규모 프로토타입 구현 및 학습 검증
* [x] `implicit` 기반 전체 데이터(37M) 학습
* [x] iteration 비교 실험 (5/10/15/30)
* [x] `implicit`이 False를 explicit negative로 처리하지 않는 문제 발견
* [x] True-only + `True > False` Explicit Negative Fine-Tuning 구조 구현
* [x] 기존 BPR을 `positive_only=True`로 공정 재평가
* [x] Regularization 비교
* [x] Factors 비교
* [x] Factors × Regularization 상호작용 실험
* [x] Local Search
* [x] **Final BPR 결정 및 단계 종료**

#### 3. ALS / Clustering — 선택적, 보류

* [ ] ALS: Alternating Least Squares 학습 구조 및 성능·실행시간 비교
* [ ] Clustering: MiniBatch K-Means 기반 사용자 segmentation 분석

### Model Comparison & Selection

* [x] Content-Based / User-Based / Item-Based / Funk SVD / BPR 성능 비교
* [x] 각 모델이 사용하는 signal과 관계 구조 재해석
* [ ] Accuracy 외 Sparsity, Coverage, Personalization, Explainability, 실행시간 관점 비교

---

## 🚧 V3. Hybrid Recommendation

* [x] Hands-On Recommendation Hybrid 파트 학습
* [x] Item-Based / User-Based / BPR 관계 재해석
* [x] Item-Based를 Hybrid Anchor Model로 설정
* [x] BPR Candidate Filtering 가설 및 Recall Ceiling 문제 분석
* [x] Architecture A/B/C 실험 구조 설계
* [x] Hybrid 평가 사용자 400명 고정
* [x] Item-Based Hybrid Baseline 평가
* [ ] **Item-Based Top-200 Candidate 생성 및 저장**
* [ ] **Candidate Recall@20/50/100/200 Saturation 분석**
* [ ] Architecture A — Item-Based Candidate + Reranking
* [ ] Architecture B — Multi-Retriever
* [ ] Architecture C — Score / Rank Fusion
* [ ] Common Hit / Unique Hit / Recovered Hit / Lost Hit 분석
* [ ] 정량 + 정성평가 후 Hybrid Baseline Architecture 선택
* [ ] Weight / Candidate Size / 모델 역할 최적화
* [ ] 필요 시 Situation-Aware Hybrid
* [ ] 최종 장점 / 한계 분석

### 추가 평가

* [ ] MAP@K
* [ ] Popularity Baseline / Popularity Bias / Coverage
* [ ] Sparsity / Personalization / Explainability / 실행시간 비교

---

## ⏳ V4. Deployment

* [ ] FastAPI
* [ ] Streamlit

---

# 📊 Evaluation

Memory-based CF는 유저 단위 런타임 Split을, Model-Based CF와 Hybrid는 저장된 **Global MF Train/Test Split**을 재사용합니다.

Hybrid 단계에서는 `hybrid_sampled_users.csv`에 저장한 **동일 평가 사용자 400명**을 모든 Architecture에서 사용해 실험 조건을 통제합니다.

### Recommendation Quality

Precision@K / Recall@K / Hit Rate@K / NDCG@K / Micro Precision·Recall·F1 / Candidate Recall / Candidate Hit Rate

### Sampling Strategy

리뷰 수 구간별 층화 표집:

| Review Group |   Users |
| ------------ | ------: |
| 10–15        |     100 |
| 16–25        |     100 |
| 26–45        |     100 |
| 46–78        |     100 |
| **Total**    | **400** |

### Hybrid Evaluation

* **Item-Based Hybrid Baseline**: P@10 0.06325 / R@10 0.08144 / HR@10 0.3925 / NDCG@10 0.09129 / Hits 253
* Architecture A/B/C는 동일 Global Split + 동일 400명 + 동일 Positive 정의 + 동일 Top-10에서 비교
* Candidate Recall@20/50/100/200으로 Retriever가 정답을 Candidate Pool에 포함시키는 능력과 Saturation 확인
* Common Hit / Unique Hit / Recovered Hit / Lost Hit으로 모델 간 보완성 분석 예정
* 최종적으로 정량 지표와 실제 추천 결과의 정성평가를 함께 사용

### Performance

* 대규모 Split 연산 최적화
* BPR 전체 학습은 `implicit`의 Cython/C 최적화 및 CSR 구조 활용
* Item-Based는 400명 평가에 약 1811.1초 소요되어 Hybrid 반복 실험에서는 **Top-200 Candidate Cache**를 우선 사용

---

# 📈 Current Progress

| Module                                              | Status |
| :-------------------------------------------------- | :----: |
| Data Loading / Caching / Validation / Preprocessing |    ✅   |
| Content-Based Recommendation 전체                     |    ✅   |
| User-Based Collaborative Filtering 전체               |    ✅   |
| Item-Based Collaborative Filtering 전체               |    ✅   |
| Model-Based CF Global Train/Test Split              |    ✅   |
| Funk SVD 학습·추천·평가·Ablation 전체                       |    ✅   |
| BPR 전체 데이터 학습 / False Fine-Tuning / 최종 튜닝           |    ✅   |
| Final BPR 결정                                        |    ✅   |
| 전체 모델 signal 재분석                                    |    ✅   |
| Item/User/BPR 관계 및 latent preference 재정리            |    ✅   |
| Hybrid Architecture A/B/C 설계                        |    ✅   |
| Hybrid 평가 사용자 400명 고정                               |    ✅   |
| Item-Based Hybrid Baseline                          |    ✅   |
| Item-Based Top-200 Candidate Cache                  |   🚧   |
| Candidate Recall Saturation 실험                      |   🚧   |
| Architecture A / B / C 비교                           |    ⏳   |
| Hit Overlap / Unique Hit 분석                         |    ⏳   |
| Hybrid 최적화                                          |    ⏳   |
| MAP / Popularity / Coverage 추가 분석                   |    ⏳   |
| Deployment                                          |    ⏳   |

---

# 📌 Project Status

**Current Version:** `V3.2 - Hybrid Architecture 설계 및 Item-Based Baseline 구축 완료`

### Completed

* Content-Based / User-Based CF / Item-Based CF 전체 파이프라인 구현 및 정량·정성평가
* Model-Based CF Global Train/Test Split 설계 및 저장 (41,154,794건)
* Funk SVD 구현·평가·원인 진단·Ablation 완료 및 단계 종료
* **BPR 전체 구현·False signal 개선·공정 재평가·최종 튜닝 완료**
* **Final BPR 확정** (P@10 0.05225 / R@10 0.06870 / HR@10 0.3850 / NDCG@10 0.06968)
* **Item-Based / User-Based / BPR의 Collaborative 관계 및 latent preference 차이 재정리**
* **Item-Based를 Hybrid Anchor Model로 설정**
* **BPR 단독 Candidate Filter의 Recall Ceiling 문제 발견 및 Multi-Retriever 방향 도출**
* **Architecture A/B/C 실험안 설계**
* 프로젝트 모델 / saved model / Hybrid 실험 폴더 구조 정리
* Hybrid 평가 사용자 400명 고정
* **Item-Based Hybrid Baseline 평가 완료** (P@10 0.06325 / R@10 0.08144 / HR@10 0.3925 / NDCG@10 0.09129 / Hits 253)

### In Progress

* **Item-Based Top-200 Candidate 생성 및 Cache**
* **Candidate Recall@20/50/100/200 Saturation 실험**

### Next Milestone

➡️ 동일 400명의 Item-Based Top-200 Candidate 생성 → `item_top200_candidates.csv` 저장

➡️ Candidate Recall@20/50/100/200 비교 → Recall 포화 구간 확인 → Candidate Size 결정

➡️ Architecture A — Item-Based Candidate + BPR / Content Reranking

➡️ Architecture B — Multi-Retriever + Ranking

➡️ Architecture C — Score / Rank Fusion

➡️ Common Hit / Unique Hit / Recovered Hit / Lost Hit 분석

➡️ 정량 + 정성평가를 통해 Hybrid Baseline Architecture 선택

➡️ 선택 구조의 Weight / Candidate Size / 모델 역할 최적화

➡️ 필요 시 Situation-Aware Hybrid

➡️ 최종 평가 / 장점·한계 분석 → README / GitHub 정리 → Deployment
