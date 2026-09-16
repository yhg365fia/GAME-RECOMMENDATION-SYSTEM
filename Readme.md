# 🎮 Steam Game Recommendation System

Steam 게임 데이터를 활용하여 다양한 추천 시스템 알고리즘을 구현하고 성능을 비교하는 프로젝트입니다.

현재는 **Content-Based Recommendation System**, **User-Based Collaborative Filtering**, **Item-Based Collaborative Filtering**, **Funk SVD**, **BPR**을 구현하였으며,

- Steam 메타데이터 전처리
- User-based Train/Test Split
- TF-IDF Vectorization
- Cosine Similarity 기반 추천
- Multi-Game Recommendation
- 리뷰 수 구간별 층화 표집 기반 평가 시스템 구축
- Precision@K, Recall@K, Hit Rate@K, NDCG@K 지표 구현 및 결과 분석
- **정성적 실험을 통한 콘텐츠 기반 추천의 구조적 한계 발견** (장르 혼합 시 쏠림 현상 등)
- AppID 기반 안정적인 게임 식별 구조 설계
- 데이터 로딩 캐싱(Parquet)을 통한 성능 최적화
- 프로젝트 모듈화 및 객체지향 설계
- **User-Based CF 구현 (Sparse Interaction Matrix, +1/-1 인코딩)**
- **평가 파이프라인 데이터 누수 버그 발견 및 수정** (자기 자신 제외 로직 무력화 → 인자 전달 누락 확인 후 수정)
- **User-Based CF의 데이터 희소성(Sparsity) 한계 정량 검증** → Item-Based CF 전환 근거 확보
- **NDCG 계산 구현 오류 발견 및 수정** (순위 정보가 소실되는 `set` 사용 → 순서 보존하는 `list`로 수정)
- **Macro Average 외 Micro Precision/Recall/F1 도입**으로 유저별 추천 개수 편차가 평가에 미치는 영향 다각도로 확인
- **Pearson Correlation을 통한 Sparsity 가설 정량 검증**
- **User-Based/Item-Based CF의 Self-Similarity 처리 원칙 정립**
- **Item-Based CF 구현 및 정량·정성평가 완료** (400명 규모, User-Based 대비 전 지표 상승, Content-Based와는 정반대 방향의 쏠림 발견)
- **Model-Based CF 전환 및 MF Train/Test 구조 재설계** (Global Model 특성을 반영해 평가 대상 유저 666,781명(70:30) + 그 외 유저(100% Train)로 구조 변경, 총 41,154,794건 interaction을 Parquet으로 저장)
- **Funk SVD 구현·평가·원인 진단·Ablation 완료 후 단계 종료** (`positive_only` 가설 기각 → item bias 가설 확보 → `biased=False` Ablation으로 Hit Rate 0.0025→0.0350 개선 확인, 그러나 근본 원인은 **rating prediction objective와 Top-N ranking objective의 불일치**로 결론)
- **BPR 이론 학습 및 소규모 프로토타입 검증 완료** (유저 1,000명, loss 0.693→0.382 정상 감소 확인)
- **`implicit` 라이브러리 기반 BPR 전체 데이터(37M) 학습 완료 및 iteration 비교 실험** (5/10/15/30 → 기존 BPR은 10 iterations가 최고 성능, Train AUC 상승과 Top-N 성능 상승이 일치하지 않음을 확인)
- **실험 파라미터 Config 구조 개선** (`main.py` 상단 집중 관리 + 모델 파일명 자동 생성으로 반복 실험 환경 개선)
- **BPR이 Steam의 `is_recommended=False`를 positive처럼 취급하던 문제 발견** — `implicit`은 sparse matrix 값의 부호가 아니라 non-zero 여부로 interaction을 해석하기 때문에, `True/False/Unseen` 3분 구조가 `positive/positive/negative candidate`로 잘못 학습되고 있었음
- **True-only BPR + `True > False` Explicit Negative Fine-Tuning 구조 구현 및 성능 대폭 개선** (3,928,771 pairs, pair_accuracy 0.9079 / 15 iter 기준 P@10 0.0278 → 0.0520, HR@10 0.2225 → 0.3950)

를 완료하였으며, 현재는 **BPR 최종 버전을 결정하기 위한 소규모 개선·공정 재평가 단계**입니다 (평가 조건 `positive_only` 차이를 통제한 baseline 재평가 → iteration/regularization 소규모 실험 → BPR 단계 종료 → Hybrid 설계 시작).

향후에는 BPR 최종화 → 전체 모델 비교표 정리 → **Hybrid Recommendation(Candidate Generation → Ranking → Re-ranking 구조)** 설계 → 웹 서비스 배포까지 확장하는 것을 목표로 합니다.

---

# 📊 전체 모델 성능 비교 (Top-10, 평가 유저 400명)

모든 모델은 동일한 층화 표집 400명(리뷰 수 구간별 100명씩) 기준으로 평가했습니다.

| 모델 | P@10 | R@10 | HR@10 | NDCG@10 | 비고 |
|---|---:|---:|---:|---:|---|
| Content-Based (TF-IDF) | 0.0268 | - | - | - | 콘텐츠 semantic 유사성 |
| User-Based CF | 0.0545 | 0.0427 | 0.3011 | 0.0655 | Sparsity에 취약 |
| **Item-Based CF** | **0.0783** | **0.0880** | **0.4800** | **0.1078** | **현재 최고 성능** (강한 co-consumption signal) |
| Funk SVD (biased) | 0.0003 | 0.0006 | 0.0025 | 0.0004 | item bias가 랭킹 지배 |
| Funk SVD (unbiased) | 0.0037 | 0.0030 | 0.0350 | 0.0042 | bias 제거로 개선되나 여전히 낮음 |
| BPR (implicit, 5 iter) | 0.0315 | 0.0349 | 0.2475 | 0.0398 | Hits 126 |
| BPR (implicit, 10 iter) | 0.0335 | 0.0380 | 0.2600 | 0.0452 | 기존 BPR 중 최고, Hits 134 |
| BPR (implicit, 15 iter) | 0.0278 | 0.0319 | 0.2225 | 0.0369 | Hits 111 |
| BPR (implicit, 30 iter) | 0.0295 | 0.0368 | 0.2175 | 0.0424 | Hits 118 |
| **BPR + Explicit False (15 iter)** | **0.0520** | **0.0670** | **0.3950** | **0.0718** | **False를 negative로 학습, Hits 208** |
| BPR + Explicit False (10 iter) | 0.0348 | 0.0462 | 0.2825 | 0.0483 | False 적용 후 10 < 15로 역전, Hits 139 |

> **주의(통제 변수)**: 기존 BPR은 `positive_only=False`(Test relevant 3,850), Explicit-False BPR은 `positive_only=True`(Test relevant 3,273) 조건으로 평가되었습니다. 따라서 성능 상승폭 전체를 False 학습 효과로 단정할 수 없으며, **동일 조건(`positive_only=True`)에서의 재평가가 다음 단계 최우선 작업**입니다.

**핵심 관찰**:

- 가장 복잡한 모델(Funk SVD, BPR)이 가장 단순한 Item-Based CF보다 낮게 나왔습니다. 추천 성능은 모델 복잡도가 아니라 **데이터 구조 + 추천 목적 + 각 모델이 사용하는 signal**에 의해 결정된다는 것을 확인했습니다.
- 현재 Steam 데이터(37M interaction / 37K items)는 매우 강한 co-consumption 정보를 갖고 있어 Item-Based CF가 강력한 baseline이 됩니다. 반면 latent space로 압축하는 MF 계열은 일반화를 얻는 대신 정보 손실이 발생할 수 있습니다.
- **factor 수·learning rate·iteration 튜닝보다 "False를 무엇으로 정의하는가"가 훨씬 큰 성능 변화를 만들었습니다** (P@10 약 1.9배, Hits 111 → 208).

---

# 📌 Project Goals

## ✅ Current

- Steam 메타데이터 전처리 / User-based Train/Test Split / Parquet 캐싱 / 프로젝트 모듈화
- Content-Based Recommendation (TF-IDF, Cosine Similarity, Multi-Game, AppID 식별 구조)
- 리뷰 수 구간별 층화 표집 평가 시스템 (Precision/Recall/Hit Rate/NDCG@K, Macro/Micro)
- **User-Based CF** (Sparse Matrix, Self-Exclusion, 데이터 누수 진단·수정, Sparsity 한계 분석)
- **Item-Based CF** (행별 Top-K, Candidate Aggregation, Train Item Exclusion, 정량·정성평가 완료)
- **NDCG 순위 보존 버그 수정 / Micro 평가 도입 / Pearson Correlation 가설 검증 / Self-Similarity 처리 원칙 정립**
- **Model-Based CF용 Global Train/Test Split 설계 및 저장** (41,154,794건)
- **Funk SVD 구현·평가·원인 진단·Ablation 완료 및 단계 종료**
- **BPR 이론 학습 및 소규모 프로토타입 검증** (loss 0.693→0.382)
- **`implicit` 기반 BPR 전체 데이터 학습 및 iteration 비교 실험 (5/10/15/30)**
- **실험 파라미터 Config 구조 개선** (main.py 상단 집중 + 모델명 자동 생성)
- **BPR의 False signal 취급 문제 발견 및 Explicit Negative Fine-Tuning 구현**

---

## 🚧 In Progress

- **기존 BPR을 `positive_only=True`로 재평가** (False 학습 효과의 공정한 비교)
- **Explicit-False BPR 기준 iteration 재정리 (10 / 15 중심) 및 regularization 소규모 실험**
- **BPR 최종 버전 결정 및 BPR 단계 종료**

---

## 🚀 Future

- **Hybrid Recommendation 설계** (Candidate Generation → Ranking → Re-ranking 구조, 각 모델 score를 feature로 활용)
- 전체 모델 비교표 최종 정리 (Accuracy 외 Sparsity/Coverage/Personalization/Explainability/실행시간 관점)
- MAP@K
- Popularity Baseline 비교 / Popularity Bias / Near-Duplicate / Coverage 분석
- (선택적) ALS 짧은 비교 실험, Clustering 사용자 군집화 분석
- FastAPI & Streamlit Deployment

---

# 🛠 Tech Stack

## Language
- Python

## Data Processing
- Pandas / NumPy / PyArrow (Parquet 캐싱) / SciPy (Sparse Matrix, CSR/LIL)

## Machine Learning
- Scikit-learn
- **Surprise** (Funk SVD 구현)
- **implicit** (대규모 BPR 학습 — Cython/C 최적화, CSR 기반, 멀티코어)

### Algorithms
- TF-IDF Vectorization / Cosine Similarity
- User-Based CF (Neighborhood-based, Top-K)
- Item-Based CF (행별 Top-K, Similarity Sum Aggregation)
- **Matrix Factorization / Funk SVD** (Surprise `SVD`, biased/unbiased 실험 완료 — 단계 종료)
- **BPR (Bayesian Personalized Ranking)** — 직접 구현(원리 이해용) + `implicit`(전체 학습용), **Explicit Negative Fine-Tuning(`True > False`) 자체 구현**

## Evaluation
- Precision@K / Recall@K / Hit Rate@K / NDCG@K (Macro) + Micro Precision/Recall/F1
- Stratified Sampling (리뷰 수 구간 기반), Qualitative Experiment
- Leave-N-Out 유저별 Split (Memory-based CF) / **Global Train/Test Split (Model-Based CF)**
- Pearson Correlation 기반 가설 검증
- 추천 게임 단위 Train 통계 분석(`groupby("app_id")`), `Counter` 기반 반복 추천 빈도 분석

## Visualization / Deployment
- Matplotlib / FastAPI / Streamlit

---

# 📂 Project Structure

```text
Game-Recommendation-System/
│
├── data/
│   ├── raw/
│   ├── cache/                # 전처리/로딩 결과 캐싱 (Parquet)
│   └── split/                 # Model-Based CF용 Global Train/Test (Parquet)
│       ├── mf_train.parquet
│       └── mf_test.parquet
│
├── docs/
│   ├── Day01.md
│   ├── ...
│   ├── Day19.md
│   └── Day20.md
│
├── models/
│   ├── content_base.py
│   ├── userbase.py
│   ├── itembase.py                          # 구현 완료
│   ├── funk_svd.py                           # 완료·단계 종료
│   ├── funk_svd_model.pkl                           # Biased Funk SVD
│   ├── funk_svd_unbiased_model.pkl                  # Unbiased Funk SVD (Ablation)
│   ├── bpr_experiment.py                     # 소규모 프로토타입 (원리 검증용)
│   ├── bpr.py                                 # implicit 기반 정식 구현 + Explicit False Fine-Tuning
│   └── implicit_bpr_{N}epoch_model.npz        # iteration별 자동 생성 모델 파일
│
├── notebooks/
├── preprocessing.py
├── data_split.py               # Global Train/Test 생성·저장·로드
├── evaluation.py                # build_mf_user_review_groups / evaluate_mf_user / run_mf_evaluation
├── main.py                       # 상단 Config 영역에 실험 파라미터 집중 관리
│
├── README.md
├── requirements.txt
└── .gitignore
```

**실험 Config 구조 (main.py 상단)**

```python
BPR_ITERATIONS = 15
BPR_FACTORS = 40
BPR_LEARNING_RATE = 0.05
BPR_REGULARIZATION = 0.001
BPR_EXPLICIT_FALSE_EPOCHS = 1

MODEL_PATH = f"implicit_bpr_{BPR_ITERATIONS}epoch_model.npz"  # 모델명 자동 생성
```

`BPR_ITERATIONS` 한 줄만 수정하면 학습 iteration과 저장 모델 이름이 동시에 바뀌도록 구성해, 반복 실험 비용을 줄였습니다.

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

> Item-Based는 similarity 결과가 이미 output 공간(Item)에 존재하므로 User-Based처럼 "유사 이웃 → 이웃의 interaction"으로 한 단계 더 연결할 필요가 없습니다. 대신 여러 Source Item의 similarity를 하나의 candidate score로 합치는 aggregation이 추가 설계 지점이며, 현재는 similarity sum을 baseline으로 채택했습니다. **중요한 점은 이 matrix가 `True→+1, False→-1, 없음→0`으로 구성되어 있어 좋아요/싫어요 패턴이 모두 similarity에 반영된다는 것입니다** — 이 차이가 BPR 대비 높은 성능의 원인 중 하나로 확인되었습니다.

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

## Model-Based CF (BPR) Pipeline — 전체 학습 완료, 최종 버전 결정 중

```text
[1단계] True interaction만 사용
      ↓
implicit BPR 학습 (True > Unseen)
  BPR_ITERATIONS / BPR_FACTORS / BPR_LEARNING_RATE / BPR_REGULARIZATION
      ↓
[2단계] Explicit Negative Fine-Tuning
  각 User의 True item > False item pair 학습
  (3,928,771 pairs, pair_accuracy ≈ 0.9079)
      ↓
predict_all(user) → seen item 제외 → Top-K 추출
      ↓
기존 evaluation.py 재사용 → 400명 동일 조건 평가
```

> **핵심 발견**: `implicit` BPR은 sparse matrix 값이 `+1`인지 `-1`인지가 아니라 **non-zero 여부**로 interaction을 해석합니다. 따라서 Steam의 `True=좋아함 / False=싫어함 / Unseen=모름` 구조가 실제로는 `True=positive / False=positive / Unseen=negative candidate`로 학습되고 있었습니다. 이는 hyperparameter 문제가 아니라 **추천 문제 정의 자체의 문제**였고, `implicit`에서 False를 직접 explicit negative로 넣을 수 없어 위와 같은 **2단계 학습 구조**를 직접 구현했습니다. 그 결과 15 iter 기준 P@10이 0.0278 → 0.0520, HR@10이 0.2225 → 0.3950으로 크게 상승했습니다.
>
> 또한 False fine-tuning 이후 **optimal iteration이 10 → 15로 역전**되었습니다. False 학습량은 동일하지만 **False 학습이 시작되는 latent representation의 상태가 다르기** 때문이며, 문제 정의가 바뀌면 optimal hyperparameter도 함께 바뀔 수 있다는 가설을 얻었습니다.

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

| 비교 변수 | Pearson r | p-value |
|---|---|---|
| n_games ↔ n_recommended | +0.3096 | < 0.0001 |
| n_games ↔ precision | +0.2504 | < 0.0001 |

review_group별 추이와 함께 **"interaction이 적을수록 안정적인 이웃 형성이 어렵다"는 가설과 일관된 방향**으로 해석했습니다.

---

# 🔗 Self-Similarity 처리 원칙 (User-Based / Item-Based 공통)

- **User-Based CF**: 본인의 interaction(Test 포함)이 neighbor로 사용되는 **데이터 누수** 문제
- **Item-Based CF**: `A↔A=1.0`이 항상 최고 유사도로 Top-K를 차지하는 **trivial self-similarity** 문제

```python
similarities = cosine_similarity(...)
similarities[self_idx] = 0
# 이후 Top-K 선정
```

---

# 🧠 Item-Based CF 구조 이해 및 구현

| 구분 | User-Based CF | Item-Based CF |
|---|---|---|
| Matrix 방향 | User × Item | Item × User (`interaction_matrix.T`) |
| Query 필요 여부 | 필요 | 불필요 (Item이 이미 행으로 존재) |
| Similarity 대상 | User ↔ User | Item ↔ Item |
| Candidate Score | User similarity × interaction의 weighted prediction | 동일 candidate로 들어오는 item similarity 합산 |

**구현 완료 사항**: Self-similarity 0 처리 → Source Item별 행 단위 Positive Top-K(k=30) → Candidate Aggregation → Train Item Exclusion 및 Positive Score Filtering → Top-N 반환.

---

# 🧮 Model-Based CF: Funk SVD (완료·종료) → BPR (진행 중)

### Memory-based CF와의 핵심 차이

| 구분 | Memory-based CF | Model-Based CF |
|---|---|---|
| User/Item 표현 | 이미 존재하는 interaction 벡터 | 데이터로부터 학습된 latent vector |
| 학습 단위 | 유저 평가 시점마다 개별 계산 | 전체 interaction을 한 번에 학습하는 Global Model |

### Global Train/Test 설계

- 평가 대상 유저(interaction 10~78개, **666,781명**): 70% Train / 30% Test, 그 외 유저: 100% Train
- 전체 41,154,794건 = Train 37,113,471건 + Test 4,041,323건
- `train_test_split()` 호출을 666,781회 → 최대 69회로 최적화 (interaction 개수별 position 재사용, 약 5초)

### Funk SVD — 원인 진단과 최종 결론

**원인 진단**: 가설 1(`positive_only` 정답 축소)은 재평가로 기각, 가설 2(item bias 지배)는 추천 게임 대부분이 True Ratio 0.94~1.00인 패턴으로 확보.

**Unbiased Ablation 결과** (400명, Top-10):

| Metric | Macro | Micro |
|---|---:|---:|
| Precision@10 | 0.0037 | 0.0037 |
| Recall@10 | 0.0030 | 0.0039 |
| Hit Rate@10 | 0.0350 | - |
| NDCG@10 | 0.0042 | - |
| F1@10 | - | 0.0038 |

전체 Hit 15 / 전체 추천 4,000 / Test positive 3,850.

Hit Rate가 약 14배 개선되어 bias의 영향은 확인했지만 절대 수치는 여전히 매우 낮았고, Half-Life 시리즈 등 특정 게임의 반복 추천도 해소되지 않아 **rating-prediction objective와 Top-N ranking objective의 근본적 불일치**를 최종 원인으로 결론짓고 단계를 종료했습니다.

### BPR — 이론, 전체 학습, False signal 개선

**이론**: `(u,i,j)` triplet으로 positive가 negative보다 높은 점수를 갖도록 직접 학습합니다.

$$ x_{uij} = p_u^T(q_i-q_j), \quad L = -\log\sigma(x_{uij}) + \lambda(\|p_u\|^2+\|q_i\|^2+\|q_j\|^2) $$

**직접 구현 vs implicit**: BPR 알고리즘 자체가 다른 게 아니라 **구현 최적화 수준이 다르다**는 것을 확인했습니다.

| 구분 | 직접 구현 | implicit |
|---|---|---|
| 처리 단위 | single interaction, Python loop | CSR Sparse Matrix 기반 |
| 자료구조 | dict / set / 함수 호출 | 연속된 factor array |
| 실행 환경 | 순수 Python | Cython/C 최적화, 멀티코어 |
| 역할 | **BPR 원리를 이해하기 위한 구현** | **대규모 데이터를 실제 학습시키기 위한 구현** |

**Iteration 실험 결과** (400명 동일 조건):

| Iteration | P@10 | R@10 | HR@10 | NDCG@10 | Micro F1 | Hits |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.0315 | 0.0349 | 0.2475 | 0.0398 | 0.0321 | 126 |
| **10** | **0.0335** | **0.0380** | **0.2600** | **0.0452** | **0.0341** | **134** |
| 15 | 0.0278 | 0.0319 | 0.2225 | 0.0369 | 0.0283 | 111 |
| 30 | 0.0295 | 0.0368 | 0.2175 | 0.0424 | 0.0301 | 118 |

특히 15 iteration에서는 **Train AUC는 증가했는데 Test 추천 성능은 감소**해, BPR training objective가 더 잘 최적화되는 것과 Top-N 추천 지표가 좋아지는 것은 같은 일이 아님을 확인했습니다.

**False signal 문제와 Explicit Negative Fine-Tuning**: Item-Based는 `True→+1, False→-1`로 싫어요 정보까지 similarity에 반영하는 반면, `implicit` BPR은 non-zero 여부만 보기 때문에 False를 positive처럼 취급하고 있었습니다. 이를 해결하기 위해 `1단계: True > Unseen (implicit BPR)` → `2단계: True > False (Explicit Negative Fine-Tuning)` 구조를 구현했고, 결과는 다음과 같습니다.

| Metric | 15 iter (기존) | 15 iter + False |
|---|---:|---:|
| Precision@10 | 0.0278 | **0.0520** |
| Recall@10 | 0.0319 | **0.0670** |
| Hit Rate@10 | 0.2225 | **0.3950** |
| NDCG@10 | 0.0369 | **0.0718** |
| Micro F1 | 0.0283 | **0.0572** |
| Hits | 111 | **208** |

**단, 두 실험의 `positive_only` 조건이 달라(False 포함 3,850 vs True만 3,273) 상승폭 전체를 False 학습 효과로 단정할 수 없습니다.** 동일 조건 재평가가 다음 단계의 최우선 작업입니다.

---

# 🧩 Hybrid Recommendation 설계 방향

각 모델은 서로 다른 signal을 사용합니다.

```text
Content-Based → 게임 자체가 무엇과 비슷한가
User-Based    → 비슷한 유저가 무엇을 소비했는가
Item-Based    → 이 게임과 함께 소비되는 게임은 무엇인가
BPR           → 유저의 latent preference에서 어떤 item이 다른 item보다 위에 있어야 하는가
```

따라서 Hybrid는 단순히 "모델 A 3개 + 모델 B 3개 + 모델 C 4개"를 합치는 것이 아니라, **multi-stage recommender** 구조로 설계할 수 있습니다.

```text
Content-Based ─┐
Item-Based CF ─┤
BPR ───────────┤
Popularity ────┤
               ▼
         Candidate Pool
               ↓
   ItemCF score / BPR score / Content similarity
   Popularity / Genre affinity / True ratio / interaction count
               ↓
        LightGBM / XGBoost
               ↓
            Final Rank → Top-10
```

머신러닝 관점에서는 stacking/ensemble과 유사하지만, 추천시스템에서는 `Candidate Generation → Ranking → Re-ranking` 구조로 이해할 수 있습니다. 이 방식은 **지금까지 만든 모델을 버리지 않고 각 모델이 잘 잡는 signal을 feature로 활용**할 수 있어 현재 프로젝트와 잘 맞습니다.

---

# 🐛 알려진 이슈 (To-Do)

- **Name = NaN metadata mismatch**: Item-Based CF 정성평가 중 일부 게임 이름이 NaN으로 조회. 원인 분석 보류
- **Party Animals 입력 시 빈 DataFrame 반환**: 정성평가 대상에서 제외. 원인 분석 보류
- **`ModuleNotFoundError: No module named 'data_split'`**: `models/` 내부 파일 직접 실행 시 import 경로 문제. `python -m models.funk_svd` 형태로 해결 방향 확인
- **Funk SVD 단계 종료**: bias 영향은 확인했지만 근본 원인은 objective mismatch로 결론. 추가 튜닝 없이 BPR로 전환
- **BPR 평가 조건 불일치 (`positive_only`)**: 기존 BPR(False 포함)과 Explicit-False BPR(True만)의 Test relevant 수가 달라 직접 비교 불가. **동일 조건 재평가가 최우선 작업**
- **Explicit False 적용 후 optimal iteration 역전 (10 → 15)**: False 학습이 시작되는 latent representation 상태 차이로 추정. 재평가 후 iteration 재정리 필요

---

# ✅ Implemented Features

## Content-Based
Steam Metadata Loading (Parquet Caching), Data Validation, User-based Train/Test Split, Combined Features, TF-IDF Vectorization, Cosine Similarity Recommendation, Multi-Game Recommendation, AppID 기반 식별 및 동명이인 처리, 정성적 실험

## User-Based CF
Sparse Interaction Matrix 구축, Cosine Similarity 기반 Top-K 이웃 탐색, 자기 자신 제외 로직, Query Vector 생성(Test 누수 방지), 이웃 가중합 예측 점수 계산

## Item-Based CF
User×Item → Item×User 구조 변환, Self-Similarity 0 처리, Source Item별 행 단위 Positive Top-K, Candidate Aggregation, Train Item Exclusion, 정성적 실험 6종

## Model-Based CF — Funk SVD (완료·종료)
Global Train/Test Split 설계 및 최적화, Surprise 기반 학습, `recommend()` Top-N 구현(bias 반영·seen-item 제거·벡터화), MF 전용 Evaluation, 400명 정량평가(Biased/Unbiased), `positive_only` 가설 기각, item bias 가설 확보, `biased=False` Ablation 완료

## Model-Based CF — BPR
- BPR 이론 학습 및 소규모 프로토타입(`bpr_experiment.py`) 구현·검증 (loss 0.693→0.382)
- **`implicit` 기반 전체 데이터(37M) 학습 및 5/10/15/30 iteration 비교 실험**
- **실험 파라미터 Config 구조 개선** (main.py 상단 집중, 모델명 자동 생성)
- **False signal 취급 문제 발견 및 2단계 학습 구조 구현** (True > Unseen → True > False Fine-Tuning, 3,928,771 pairs / pair_accuracy 0.9079)
- **Explicit False 적용 후 성능 대폭 개선 확인** (P@10 0.0278 → 0.0520)
- [진행 중] `positive_only=True` 공정 재평가, regularization/factors 소규모 실험, 최종 버전 결정

## Evaluation (공통)
리뷰 수 구간 기반 층화 표집, Precision/Recall/Hit Rate/NDCG@K, Macro/Micro Precision·Recall·F1, Pearson Correlation 가설 검증, 구간별 Breakdown 리포트

---

# 🚀 Development Roadmap

## ✅ V1. Content-Based Recommendation
전체 완료

## ✅ V2. Collaborative Filtering

### User-Based CF / Item-Based CF
전체 완료

### Model-Based CF

#### 1. Funk SVD — 완료·종료
- [x] Memory-based vs Model-Based(Global Model) 학습 구조 차이 파악
- [x] Global Train/Test Split 설계 및 대규모 split 성능 최적화
- [x] Surprise 기반 구현, Top-N 추천 함수, evaluation 연결, 400명 정량평가
- [x] 성능 저조 원인 진단 (`positive_only` 가설 기각 → item bias 가설 확보)
- [x] `biased=False` Ablation 실행 및 결과 해석
- [x] Funk SVD 단계 최종 종료 (objective mismatch로 결론)

#### 2. BPR — Top-N Ranking 직접 학습 (진행 중)
- [x] Funk-SVD와 BPR의 차이 이해, `(u,i,j)` triplet·sigmoid·likelihood·loss 이해
- [x] 소규모 프로토타입 구현 및 학습 검증 (loss 0.693→0.382)
- [x] `implicit` 기반 전체 데이터(37M) 학습
- [x] iteration 비교 실험 (5/10/15/30 → 10 iter 최고)
- [x] 실험 파라미터 Config 구조 개선
- [x] `implicit`이 False를 explicit negative로 처리하지 않는 문제 발견
- [x] True-only + `True > False` Explicit Negative Fine-Tuning 구조 구현 및 성능 개선 확인
- [ ] **기존 BPR을 `positive_only=True`로 재평가 (공정한 baseline 확보)**
- [ ] Explicit-False BPR 기준 iteration 재정리 (10 / 15 중심)
- [ ] regularization 소규모 실험 (0.0001 / 0.001 / 0.01)
- [ ] 필요 시 factors 40 vs 80 비교
- [ ] BPR 최종 버전 결정 및 단계 종료

#### 3. ALS / Clustering — 선택적, 보류
- [ ] ALS: Alternating Least Squares 학습 구조 및 성능·실행시간 비교
- [ ] Clustering: MiniBatch K-Means 기반 사용자 segmentation 분석

### Model Comparison & Selection
- [x] Content-Based / User-Based / Item-Based / Funk SVD / BPR 성능 비교표 작성 (README 상단)
- [ ] Accuracy 외 Sparsity, Coverage, Personalization, Explainability, 실행 시간 관점 비교
- [ ] 성능과 역할이 가장 뚜렷한 1~2개 모델만 소규모 튜닝

### MAP & Popularity Baseline (Hybrid 이전 진행 예정)
- [ ] MAP@K 구현
- [ ] Popularity Baseline 비교
- [ ] Hit item의 Popularity Bias 분석

---

## 🚧 V3. Hybrid Recommendation

- Candidate Generation → Ranking → Re-ranking 구조의 multi-stage recommender 설계
- 각 모델 score(ItemCF, BPR, Content similarity, Popularity, Genre affinity, True ratio 등)를 feature로 활용
- LightGBM/XGBoost 기반 Final Ranking
- Content-Based와 Item-Based CF가 정반대 방향으로 쏠린 결과를 반영한 aggregation 설계

---

## 🚧 V4. Deployment

- FastAPI / Streamlit

---

# 📊 Evaluation

Memory-based CF는 유저 단위 런타임 70:30 split(Leave-N-Out)을, Model-Based CF(Funk SVD/BPR)는 전체 사용자 interaction을 한 번에 학습하는 Global Model이므로 별도의 Global Train/Test를 Parquet으로 저장해 재사용합니다.

### Recommendation Quality
Precision@K / Recall@K / Hit Rate@K / NDCG@K / Micro Precision·Recall·F1 / MAP(예정)

### Sampling Strategy
리뷰 수 구간별 층화 표집 — User-Based 100명/그룹, Item-Based·Funk SVD·BPR 모두 400명 규모 본 평가 완료

### Qualitative Evaluation
- Content-Based ↔ Item-Based 간 동일 입력 비교를 통한 쏠림 방향 차이 분석
- Funk SVD는 랜덤 사용자 정성 확인 + 추천 게임 Train 통계 분석으로 item bias 패턴 확인

### Model Comparison
전체 비교표는 README 상단 「전체 모델 성능 비교」 참고. 평가 조건(`positive_only`) 통제 후 최종 비교표를 갱신할 예정입니다.

### Performance
- 대규모 Split 연산 최적화 (Python loop 66만 회 → interaction count별 최대 69회, 약 5초)
- 추천 게임 통계·반복 빈도 분석에 `groupby`/`Counter` 기반 벡터화 집계 활용
- BPR 전체 학습은 Python loop 직접 구현 대신 `implicit`(Cython/C, 멀티코어)으로 37M 규모 학습 실현

---

# 📈 Current Progress

| Module | Status |
| :--- | :---: |
| Data Loading / Caching / Validation / Preprocessing | ✅ |
| Content-Based Recommendation 전체 | ✅ |
| User-Based Collaborative Filtering 전체 | ✅ |
| Item-Based Collaborative Filtering 전체 | ✅ |
| Model-Based CF Global Train/Test Split | ✅ |
| Funk SVD 학습·추천·평가·Ablation 전체 | ✅ |
| BPR 이론 학습 및 소규모 프로토타입 검증 | ✅ |
| BPR 전체 데이터(implicit) 학습 및 iteration 실험 | ✅ |
| 실험 Config 구조 개선 | ✅ |
| BPR Explicit False Fine-Tuning 구현 및 성능 확인 | ✅ |
| 전체 모델 성능 비교표 작성 | ✅ |
| BPR 공정 재평가 및 최종 버전 결정 | 🚧 |
| Model Comparison & Selection (다차원 관점) | ⏳ |
| MAP | ⏳ |
| Popularity Baseline | ⏳ |
| ALS / Clustering (선택적) | ⏳ |
| Hybrid Recommendation | 🚧 |
| Deployment | 🚧 |

---

# 📌 Project Status

**Current Version:** `V3.1 - BPR 전체 학습 및 False signal 개선 완료, 공정 재평가 후 BPR 단계 종료 예정`

### Completed

- Content-Based / User-Based CF / Item-Based CF 전체 파이프라인 구현 및 정량·정성 평가
- Model-Based CF Global Train/Test Split 설계 및 저장 (41,154,794건)
- Funk SVD 구현·평가·원인 진단·Ablation 완료 및 단계 종료
- BPR 이론 학습 및 소규모 프로토타입 검증
- **`implicit` 기반 BPR 전체 데이터(37M) 학습 및 5/10/15/30 iteration 비교 실험**
- **실험 파라미터 Config 구조 개선 (main.py 상단 집중 + 모델명 자동 생성)**
- **BPR의 False signal 취급 문제 발견** (`implicit`은 non-zero 여부로만 interaction 해석)
- **True-only + Explicit Negative Fine-Tuning 2단계 학습 구조 구현 및 성능 대폭 개선** (P@10 0.0278 → 0.0520)
- **전체 모델 성능 비교표 작성 및 signal 관점 해석** (모델 복잡도보다 문제 정의가 더 중요함을 실험으로 확인)

### In Progress

- **기존 BPR `positive_only=True` 공정 재평가**
- **Explicit-False BPR 기준 iteration/regularization 소규모 실험 및 최종 버전 결정**

### Next Milestone

➡️ 기존 BPR을 `positive_only=True`로 재평가 → False 학습 효과 공정 비교
➡️ Explicit-False BPR 기준 iteration 결과 재정리 (10 / 15 중심)
➡️ regularization 소규모 실험 (0.0001 / 0.001 / 0.01), 필요 시 factors 40 vs 80 비교
➡️ BPR 최종 버전 결정 → BPR 단계 종료
➡️ Content / User / Item / Funk SVD / BPR 전체 모델 비교표 최종 정리
➡️ **Hybrid Recommendation 설계 시작** (Candidate Generation → Ranking → Re-ranking)
➡️ MAP@K, Popularity Baseline 및 Popularity Bias / Coverage 분석
➡️ Web Service Deployment