# 🎮 Steam Game Recommendation System

Steam 게임 데이터를 활용하여 다양한 추천 시스템 알고리즘을 구현하고 성능을 비교하는 프로젝트입니다.

현재는 **Content-Based Recommendation System**, **User-Based Collaborative Filtering**, **Item-Based Collaborative Filtering**을 구현하였으며,

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
- **Pearson Correlation을 통한 Sparsity 가설 정량 검증** (`n_games ↔ n_recommended`, `n_games ↔ precision`)
- **User-Based/Item-Based CF의 Self-Similarity 처리 원칙 정립** (유사도 계산 자체는 문제없음 → Top-K 선정 전 자기 자신 제거로 통일)
- **Item-Based CF 구현 완료** (Source Item별 행 단위 Top-K 추출, Candidate Aggregation, Train Item Exclusion, Top-N 반환까지 전체 파이프라인 완성)
- **Item-Based CF 정량평가 완료** (400명 규모, User-Based 대비 Precision/Recall/HitRate/NDCG 전 지표 상승, `n_recommended` 100% 충족으로 Sparsity 한계 완화 확인)
- **Item-Based CF 정성평가 완료** (6개 실험을 통해 콘텐츠 기반과는 다른 방향으로 쏠리는 현상 발견, Content-Based vs Item-Based가 서로 다른 signal을 포착한다는 결론 도출)
- **Model-Based CF 전환 및 MF Train/Test 구조 재설계** (Memory-based CF의 사용자별 70:30 split 구조가 Global Model인 MF에는 적합하지 않음을 확인하고, 평가 대상 유저 666,781명(70:30) + 그 외 유저(100% Train)로 구조 변경, 총 41,154,794건 interaction을 Parquet으로 저장)
- **Surprise 기반 Funk SVD 모델 구현 및 학습 성공, Top-N 추천 함수 구현, 400명 정량평가 완료** (Precision@10 0.0003으로 매우 낮은 baseline 확인)
- **Funk SVD 성능 저조 원인 1차 진단 완료** — `positive_only=False`로 재평가해 "정답 축소가 원인"이라는 가설을 실험으로 기각하고, 추천 게임의 Train 통계·400명 반복 추천 분석을 통해 **높은 True 비율(긍정률)을 가진 일부 게임이 item bias($b_i$)로 인해 랭킹 상단에 반복 등장**하는 패턴을 발견. Rating Prediction objective와 Top-N Ranking objective의 근본적인 불일치를 원인 가설로 정리
- **Bias 제거(`biased=False`) Ablation 실험 및 BPR 전환을 다음 단계로 확정**

를 완료하였으며, 현재는 **Funk SVD의 bias 영향을 검증하는 Ablation 실험을 준비하고, 이후 rating prediction이 아닌 Top-N ranking을 직접 학습하는 BPR로 전환할 계획을 확정한 단계**입니다.

향후에는 Funk SVD bias ablation → **BPR(Bayesian Personalized Ranking) 구현 및 Funk SVD와 성능 비교** → (필요 시) ALS 짧은 비교 실험 → Clustering(사용자 군집화 분석) 순서로 서로 다른 모델의 구조와 성능을 경험한 뒤, 전체 모델 비교·소규모 튜닝·Hybrid Recommendation·웹 서비스 배포까지 확장하는 것을 목표로 합니다.

---

# 📌 Project Goals

## ✅ Current

- Steam 메타데이터 전처리
- User-based Train/Test Split
- Content-Based Recommendation
- TF-IDF Vectorization
- Cosine Similarity
- Multi-Game Recommendation
- AppID 기반 게임 식별 및 동명이인 게임 선택 기능
- 리뷰 수 구간별 층화 표집 평가 시스템
- Precision@K / Recall@K / Hit Rate@K / NDCG@K
- 정성적 실험 (단일 입력 / 장르 혼합 입력 분석)
- 데이터 로딩 캐싱 (Parquet)
- 프로젝트 구조 모듈화
- **User-Based Collaborative Filtering (Sparse Matrix 기반)**
- **평가 파이프라인 데이터 누수(Self-Leakage) 진단 및 수정**
- **User-Based CF Sparsity 한계 분석**
- **NDCG 순위 보존 버그 수정**
- **Macro/Micro Precision, Recall, F1 비교 평가**
- **Pearson Correlation 기반 Sparsity 가설 정량 검증**
- **Self-Similarity 처리 원칙 정립 (User-Based/Item-Based 공통)**
- **Item-Based Collaborative Filtering 구현 완료** (행별 Top-K, Candidate Aggregation, Train Item Exclusion, Top-N 반환)
- **Item-Based CF 정량평가 완료** (User-Based 대비 전 지표 상승, Sparsity 강건성 확인)
- **Item-Based CF 정성평가 완료** (Content-Based와의 쏠림 방향 비교, Hybrid 설계 필요성 확인)
- **Model-Based CF용 대규모 Train/Test Split 설계 및 Parquet 저장** (41,154,794건, Global Model 특성을 반영한 구조로 재설계)
- **Surprise 기반 Funk SVD 학습 파이프라인 및 Top-N 추천 함수 구현, 400명 정량평가 완료**
- **`positive_only=False` 재평가를 통해 "정답 축소가 원인" 가설 기각**
- **추천 게임 Train 통계 및 400명 반복 추천 분석으로 item bias 원인 가설 확보**

---

## 🚧 In Progress

- **Funk SVD `biased=False` Ablation 실험 구현 및 실행** (신규 모델 `models/funk_svd_unbiased_model.pkl`로 저장, 동일 조건 재평가)
- **Ablation 결과 해석 및 Funk SVD 단계 최종 종료 판단**

---

## 🚀 Future

- **BPR(Bayesian Personalized Ranking) 구현** (Positive/Negative item 정의, 기존 MF Train/Test split 재사용, Top-10 추천 연결, 기존 평가 지표로 Funk SVD와 성능 비교)
- ALS (Matrix Factorization 최적화 방식 비교, 짧은 실험)
- Clustering (사용자 군집화/세그먼트 분석 중심의 짧은 실험)
- 전체 모델 동일 evaluation pipeline 비교 및 최상위 1~2개 모델 소규모 튜닝
- MAP@K
- Popularity Baseline 비교
- Popularity Bias / Near-Duplicate / Coverage 분석
- Hybrid Recommendation (실패 유형 기반 설계)
- FastAPI & Streamlit Deployment

---

# 🛠 Tech Stack

## Language

- Python

## Data Processing

- Pandas
- NumPy
- PyArrow (Parquet 캐싱)
- SciPy (Sparse Matrix, CSR/LIL)

## Machine Learning

- Scikit-learn
- **Surprise** (Funk SVD 구현)

### Algorithms

- TF-IDF Vectorization
- Cosine Similarity
- User-Based Collaborative Filtering (Neighborhood-based, Top-K)
- Item-Based Collaborative Filtering (Neighborhood-based, 행별 Top-K, Similarity Sum Aggregation)
- **Matrix Factorization / Funk SVD** (Surprise `SVD`, SGD 기반 Latent Factor 학습, bias(μ, bu, bi) 포함 — 현재 bias 영향 검증 중)
- **BPR (예정)** — pairwise ranking 직접 학습 방식

## Evaluation

- Precision@K / Recall@K / Hit Rate@K / NDCG@K (Macro)
- Micro Precision / Micro Recall / Micro F1
- Stratified Sampling (리뷰 수 구간 기반)
- Qualitative Experiment (단일/혼합 입력 결과 분석)
- Leave-N-Out 유저별 Train/Test Split (Memory-based CF)
- **Global Train/Test Split (Model-Based CF, 평가 대상 유저 70:30 + 그 외 유저 100% Train)**
- Pearson Correlation (SciPy) 기반 가설 검증
- **추천 게임 단위 Train 통계 분석 (`groupby("app_id")` 기반 interaction/True ratio 집계), `Counter` 기반 반복 추천 빈도 분석**

## Future Libraries

- Implicit

## Visualization

- Matplotlib

## Deployment

- FastAPI
- Streamlit

---

# 📂 Project Structure

```text
Game-Recommendation-System/
│
├── data/
│   ├── raw/
│   ├── cache/               # 전처리/로딩 결과 캐싱 (Parquet)
│   │   ├── games.parquet
│   │   ├── train.parquet
│   │   ├── test.parquet
│   │   └── recommendations.parquet
│   └── split/                # Model-Based CF용 Global Train/Test (Parquet)
│       ├── mf_train.parquet
│       └── mf_test.parquet
│
├── docs/
│   ├── Day01.md
│   ├── ...
│   ├── Day17.md
│   └── Day18.md
│
├── models/
│   ├── content_base.py
│   ├── userbase.py
│   ├── itembase.py           # 구현 완료
│   ├── funk_svd.py            # 학습·추천·평가 구현 완료, bias 영향 진단 중
│   ├── funk_svd_model.pkl              # Biased Funk SVD 저장 모델
│   └── funk_svd_unbiased_model.pkl     # [예정] biased=False Ablation 모델
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_content_based_recommendation.ipynb
│   └── 04_evaluation_dataset.ipynb
│
├── preprocessing.py
├── data_split.py              # Model-Based CF용 Global Train/Test 생성·저장·로드
├── evaluation.py               # build_mf_user_review_groups / evaluate_mf_user / run_mf_evaluation 추가
├── main.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# ⚙️ Current Recommendation Pipeline

## Content-Based Pipeline

```text
Raw Steam Dataset
      │
      ▼
Data Loading (with Parquet Cache)
      │
      ▼
Data Validation
      │
      ▼
User-based Train/Test Split (유저별 게임 수가 제각각이라
sklearn train_test_split을 런타임에 유저 단위로 적용,
별도 데이터셋 파일로 저장하지 않음)
      │
      ├──────────────┐
      ▼              ▼
 Train           Test
      │
      ▼
Data Preprocessing (Name Dedup, AppID 기준 정리)
      │
      ▼
Combined Features
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Name → AppID → Index 변환
      │
      ▼
Content-Based Recommendation (Cosine Similarity)
      │
      ▼
Top-N Recommendation (AppID 기준 중복 제거)
      │
      ▼
Recommendation Evaluation (정량 지표 + 정성적 실험)
```

## User-Based CF Pipeline

```text
User History (user_id, app_id, is_recommended)
      │
      ▼
build_interaction_matrix()
  → Sparse User x Item Matrix (+1 / -1 인코딩)
  → user_to_idx / game_to_idx / idx_to_game 생성
      │
      ▼
유저별 Train/Test Split (Leave-N-Out, 70/30)
      │
      ▼
Query Vector 생성 (Train App ID만 사용, Test 누수 방지)
      │
      ▼
Cosine Similarity (Query vs 전체 Interaction Matrix)
      │
      ▼
자기 자신 제외 (exclude_user_idx → sims 0 처리)
      │
      ▼
Top-K Neighbor 추출 (기본 k=30)
      │
      ▼
이웃 가중합 예측 점수 계산 (Σ 유사도×상호작용 / Σ|유사도|)
      │
      ▼
이미 플레이한 게임 제외 + 양수 점수만 후보로 채택
      │
      ▼
Top-N 추천 (n_recommended ≤ N, 후보 부족 시 N보다 적을 수 있음)
      │
      ▼
Recommendation Evaluation (Precision/Recall/Hit Rate/NDCG, 구간별 Breakdown)
```

## Item-Based CF Pipeline

```text
User History (user_id, app_id, is_recommended)
      │
      ▼
build_interaction_matrix() (User-Based와 동일 함수 재사용)
      │
      ▼
interaction_matrix.T → item_matrix (Item x User 구조로 전환)
      │
      ▼
유저의 Train App ID → item_matrix에서 해당 게임 행(Source Items) 조회
  (별도 Query Vector 생성 불필요 — Item이 이미 행으로 존재)
      │
      ▼
Cosine Similarity (Source Items vs 전체 item_matrix)
  → Train item 개수 × 전체 Item 개수 shape의 similarity 행렬
      │
      ▼
각 Source Item의 Self-Similarity 0 처리 (A↔A 제거)
      │
      ▼
Source Item별 행(row) 단위 Positive Top-K 추출 (기본 k=30)
      │
      ▼
동일 Candidate로 모이는 Similarity 합산 (Similarity Sum Aggregation)
      │
      ▼
이미 플레이한 게임(Train) 제외 (predicted_scores[col_idx] = -np.inf)
      │
      ▼
양수 점수(Positive Score)만 후보로 채택 후 Top-N 반환
      │
      ▼
Recommendation Evaluation (기존 evaluation.py 재사용, User-Based와 동일 조건 비교)
```

> Item-Based는 similarity 계산 결과가 이미 output 공간(Item)에 존재하기 때문에, User-Based처럼 "유사 이웃 → 이웃의 interaction"으로 한 단계 더 연결할 필요가 없습니다. 대신 여러 Source Item의 similarity를 하나의 candidate score로 합치는 aggregation 단계가 User-Based에는 없던 추가 설계 지점이며, 현재는 가장 기본적인 similarity sum 방식을 baseline으로 채택했습니다.

## Model-Based CF (Funk SVD) Pipeline — 학습·추천·평가 완료, 성능 저조 원인 진단 및 Ablation 준비 중

Memory-based CF(User/Item-Based)는 평가 시점에 사용자별로 70:30 split을 적용해도 문제가 없었지만, Funk SVD는 **모든 사용자 interaction으로 User/Item latent vector를 동시에 학습하는 Global Model**이라 평가 대상 유저마다 모델을 다시 학습하는 기존 구조를 쓸 수 없습니다. Train/Test 구조를 재설계하고, 학습된 모델로부터 실제 추천을 만들고 평가하는 부분까지 완성했으며, 낮은 baseline 성능의 원인을 진단하는 단계까지 진행했습니다.

```text
전체 Interaction (recommendations)
      │
      ▼
평가 대상 유저 판별 (interaction 10~78개, 666,781명)
      │
      ├─ 평가 대상 유저 ── 70% Train / 30% Test
      └─ 그 외 유저     ── 100% Train
      │
      ▼
data_split.py → mf_train.parquet (37,113,471건) / mf_test.parquet (4,041,323건) 저장
      │
      ▼
Surprise Dataset 변환 → SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02).fit(trainset)
      │
      ▼
recommend(user_id, app_id_list, top_n)
  ├─ scores = model.qi @ model.pu[inner_uid]
  ├─ scores += global_mean + bu[inner_uid] + bi   # bias 반영 (μ + b_u + b_i + qᵀp)
  ├─ scores[seen_items] = -inf
  └─ argpartition → argsort → raw_item_ids 복원
      │
      ▼
MF 전용 Evaluation (build_mf_user_review_groups / evaluate_mf_user / run_mf_evaluation)
      │
      ▼
[결과] 400명 평가: Precision@10 0.0003, Hits 1/4000 — 매우 낮은 baseline 확인
      │
      ▼
[가설1 검증] positive_only=False 재평가 → Test 정답 3,273→3,850 증가에도 Hits는 그대로 1
  → "정답 축소가 원인"이라는 가설 기각
      │
      ▼
[가설2 탐색] 추천 게임 Train 통계(interaction 수 · True ratio) + 400명 반복 추천 빈도 분석
  → 대부분 True Ratio 0.94~1.00인 게임이 랭킹 상단에 반복 등장하는 패턴 발견
  → item bias(b_i)가 사용자 개인화(p_u^Tq_i)를 압도할 가능성 + rating-vs-ranking objective mismatch로 해석
      │
      ▼
[다음] biased=False Ablation 실험 → 성능 변화로 bias 영향 정도 확인
      │
      ▼
[다음] Funk SVD 단계 종료 → BPR(Positive/Negative pairwise ranking) 구현 및 성능 비교
```

> Funk SVD의 예측값 $\hat r_{ui}=\mu+b_u+b_i+q_i^Tp_u$은 정확한 평점이 아니라 **ranking score**로 사용합니다. 400명 평가 결과 Precision@10 **0.0003**으로 매우 낮게 나왔고, 1차로 `positive_only=True`(정답 축소) 가설을 세워 `positive_only=False`로 재평가했지만 Hits는 여전히 1개로 변화가 없어 이 가설은 기각했습니다. 대신 추천 게임의 Train 통계와 400명 전체의 반복 추천 패턴을 분석한 결과, **True 비율(긍정률)이 매우 높은 일부 게임이 item bias($b_i$)로 인해 여러 사용자의 Top-10에 체계적으로 반복 등장**하는 현상을 발견했습니다. 이는 Funk SVD가 학습하는 "rating을 얼마나 잘 예측하는가"라는 objective와, 실제로 필요한 "미래에 interaction할 게임을 Top-N 안에 넣을 수 있는가"라는 ranking objective가 근본적으로 다르다는 것과 연결됩니다. 다음 단계로 `biased=False` Ablation 실험을 거쳐 원인을 한 번 더 좁힌 뒤, Positive item의 점수가 Negative item보다 높도록 직접 학습하는 **BPR**로 전환할 계획입니다.

---

# 🧩 Recommendation Identifier Flow

콘텐츠 기반 추천에서 게임을 식별하고 유사도를 계산하는 내부 흐름은 다음과 같습니다.

```text
Game Name (사용자 입력)
      │  동명이인 게임 존재 시 후보 목록 출력 후 선택
      ▼
AppID (고유 식별자)
      │  game_to_idx 딕셔너리로 조회
      ▼
Index (tfidf_matrix 상의 위치)
      │
      ▼
TF-IDF Vector
      │
      ▼
Cosine Similarity
      │
      ▼
추천 Index → AppID → Game Name
```

> Game Name은 중복될 수 있지만 AppID는 고유하므로, 내부 로직은 전부 **AppID 기준**으로 동작하도록 설계하였습니다. User-Based/Item-Based CF에서도 동일한 원칙을 적용하여, `user_to_idx` / `game_to_idx` / `idx_to_game`을 통해 ID ↔ 행렬 인덱스 변환을 일관되게 관리합니다. Funk SVD 역시 Surprise 내부의 raw id ↔ inner id 매핑을 통해 동일한 AppID 기준 식별 원칙을 유지합니다.

---

# 🔍 Qualitative Experiment Findings (Content-Based)

정량 지표만으로는 "왜 이런 결과가 나오는가"를 확인하기 어려워, 실제 게임을 직접 입력하고 추천 결과를 눈으로 확인하는 정성적 실험을 병행하였습니다.

### 발견 1 — 텍스트에 없는 특성은 포착 불가

`Party Animals` 입력 시 장르/인원수는 유사하게 나왔지만, 기대했던 "동물 캐릭터" 테마는 전혀 반영되지 않음을 확인. TF-IDF는 Genres/Tags 등 텍스트 메타데이터에 명시된 정보만 학습하므로, 비주얼/테마적 특성은 원천적으로 포착할 수 없다는 한계를 확인하였습니다.

### 발견 2 — 장르 혼합 시 쏠림 현상

카드/덱빌딩(`Slay the Spire`, `Balatro`) + 슈팅(`Counter-Strike 2`, `Left 4 Dead 2`)을 함께 입력했을 때, 추천 결과가 카드/덱빌딩 계열로 완전히 쏠리고 슈팅 계열은 단 하나도 포함되지 않는 현상을 확인하였습니다. 두 가지 가설을 세웠습니다:

- **가설 1 (IDF 희귀도)**: 카드/덱빌딩류 태그(`deck-building`, `roguelike` 등)는 카탈로그 내 등장 빈도가 낮아 TF-IDF 가중치가 크게 작동하는 반면, FPS류 태그(`action`, `shooter`, `multiplayer`)는 흔해서 변별력이 낮았을 가능성
- **가설 2 (벡터 응집력)**: 여러 게임 벡터를 평균 낼 때, 방향이 비슷하고 크기가 큰 쪽(카드게임군)이 평균을 지배했을 가능성

> 이 발견을 우연이 아닌 "콘텐츠 기반 + 단순 평균 방식"의 구조적 특성일 가능성으로 보고, 최소 검증(단독 입력 시 같은 장르끼리 서로 상위에 오르는지 확인 및 추가 장르 조합 실험)을 다음 단계로 남겨두었습니다. 이 발견은 향후 하이브리드 모델의 장르별 가중치 설계, 그리고 협업 필터링에서 동일한 쏠림이 재현되는지 비교하는 기준점으로 활용했으며, 실제로 Item-Based CF 정성평가에서 비교 기준으로 사용했습니다 (아래 「Item-Based CF Qualitative Experiment Findings」 참고).

---

# 🐛 User-Based CF 데이터 누수 진단 및 수정

User-Based CF 첫 평가에서 Precision@10이 **0.6254**로, Content-Based(0.0268) 대비 비정상적으로 높게 나오는 것을 발견하고 원인을 진단하였습니다.

### 원인

평가용 `interaction_matrix`가 Train/Test 미분리 원본 데이터로 생성되어, 평가 대상 유저 본인의 전체 데이터가 이웃 후보 풀에 그대로 남아있었습니다. 쿼리 벡터(Train 기준)가 자기 자신의 원본 벡터와 매우 높은 유사도(실측 0.7746, 전체 유저 중 1위)를 가져, 사실상 자기 자신이 최상위 이웃으로 선택되어 Test 정답이 예측 점수에 직접 새어 들어가는 구조였습니다.

### 수정 과정

1. `recommend()`에 `exclude_user_idx` 파라미터를 추가하고, 유사도 계산 직후 자기 자신 유사도를 0으로 처리
2. 1차 수정 후에도 지표가 완전히 동일하게 나오는 현상 발견 → 함수 내부 로직이 아니라 **호출 체인**을 추적
3. 최종적으로 `run_evaluation()` 호출 시 `user_to_idx` 인자 자체를 넘기지 않아, `exclude_user_idx`가 항상 `None`으로 고정되어 제외 로직이 한 번도 실행되지 않았던 것을 확인
4. `user_to_idx`를 명시적으로 전달하도록 수정 후 재평가 → Precision@10이 0.0545 수준(Content-Based와 유사한 자릿수)으로 정상화

### 교훈

개별 함수(클래스 내부 로직, 평가 함수)는 모두 정확했음에도, 최상위 호출부에서 인자 하나가 누락되며 전체 로직이 무력화되었습니다. 여러 파일/함수로 나뉜 파이프라인에서는 내부 로직보다 **호출 체인(인자 전달 경로) 검증을 우선시**하는 디버깅 원칙(Outside-In)을 세우는 계기가 되었습니다.

---

# 🐛 NDCG 계산 구현 오류 수정

평가 지표를 Macro/Micro로 확장하는 과정에서 기존 `evaluate_user()`의 NDCG 계산 코드를 재점검하다, 추천 결과를 `set(result["app_id"])`로 바로 변환해 **추천 순위 정보가 소실**되고 있는 것을 발견했습니다.

- Precision/Recall은 포함 여부와 개수만 필요해 순서 무관이지만, NDCG는 "정답이 몇 번째 순위에 있는가"를 평가하는 지표라 순서 보존이 필수입니다.
- `recommended_list`(순서 보존, NDCG용)와 `recommended_ids`(집합, Precision/Recall 교집합 연산용)로 역할을 분리해 수정했습니다.
- 수정 후 재평가한 NDCG@10은 **0.0655**로 확인했습니다.

---

# 📐 Macro vs Micro 평가 및 Sparsity 정량 검증

### Macro → Micro 확장 배경

기존 평가는 유저별 Precision/Recall을 계산 후 평균 내는 Macro 방식이었는데, `n_recommended`가 유저마다 1~10개로 크게 달라(평균 7.28/10, Top-10 완전 채움 54.4%), 추천을 적게 받은 유저와 많이 받은 유저가 Macro 평균에서 동일한 가중치를 갖는 문제를 확인했습니다. 이를 보완하기 위해 **Micro Precision/Recall**과 **Micro F1**을 추가로 도입했습니다.

> Micro 수치가 Macro보다 높게 나온 것은 **모델 성능이 개선된 것이 아니라, 동일한 결과를 다른 가중치 기준으로 재집계**한 것입니다.

### Sparsity 가설 정량 검증 (Pearson Correlation)

| 비교 변수 | Pearson r | p-value |
|---|---|---|
| n_games ↔ n_recommended | +0.3096 | < 0.0001 |
| n_games ↔ precision | +0.2504 | < 0.0001 |

r 값은 약~중간 수준의 양의 상관이며, 상관관계가 인과관계를 증명하지는 않습니다. 다만 review_group별 추이와 함께 **"interaction이 적을수록 안정적인 이웃 형성이 어렵다"는 기존 가설과 일관된 방향의 정황**으로 해석했습니다.

---

# 🔗 Self-Similarity 처리 원칙 (User-Based / Item-Based 공통)

- **자기 자신과의 유사도를 계산하는 것 자체는 문제가 아닙니다.** 문제는 자기 자신이 **Top-K neighbor로 선정되어 실제 추천 계산에 쓰이는 순간**부터 발생합니다.
- **User-Based CF**: 본인의 interaction(Test 포함)이 neighbor로 사용되며 정답 정보가 추천 점수로 유출되는 **데이터 누수** 문제
- **Item-Based CF**: `A↔A=1.0`이라는 자명한 값이 항상 최고 유사도로 Top-K를 차지해, 자기 자신에게 다시 가중치를 부여하는 **trivial self-similarity** 문제

두 모델 모두 다음 구현 원칙으로 통일했습니다:

```python
similarities = cosine_similarity(...)
similarities[self_idx] = 0
# 이후 Top-K 선정
```

---

# 🧠 Item-Based CF 구조 이해 및 구현

### User-Based vs Item-Based 구조 비교

| 구분 | User-Based CF | Item-Based CF |
|---|---|---|
| Matrix 방향 | User × Item | Item × User (`interaction_matrix.T`) |
| Query 필요 여부 | 필요 (User를 벡터로 재구성) | 불필요 (Item이 이미 행으로 존재) |
| Similarity 대상 | User ↔ User | Item ↔ Item |
| Similarity 결과 shape | 1 × n_users (한 행) | Train item 개수 × n_items (여러 행) |
| Top-K 방식 | 전체에서 한 번 | Source item마다 행별로 |
| Similarity → Output 거리 | User 유사도 → 이웃의 interaction까지 연결해야 Item(output) 도출 | Similarity 자체가 이미 Item(output) 공간에 존재 |
| Candidate 생성 | Neighbor User의 interaction에서 생성 | 각 Source Item의 similar item에서 직접 생성 |
| Candidate Score | User similarity × interaction의 weighted prediction | 동일 candidate로 들어오는 item similarity 합산(Sum Aggregation) |
| 주요 구조적 문제 | User overlap 부족 시 neighbor/candidate 부족 | 여러 source에서 candidate가 생성되어 coverage가 넓어짐 |

### 구현 완료 사항

- `evaluation.py`와 main pipeline 그대로 유지 — User-Based/Item-Based 비교 조건 통일
- 기존 `build_interaction_matrix()` 재사용, `interaction_matrix.T`로 Item 관점 추가
- Self-similarity 0 처리 후 Source Item별 행 단위 Positive Top-K 추출 (k=30)
- 동일 candidate로 모이는 similarity를 합산하는 Candidate Aggregation 구현
- Train Item Exclusion 및 Positive Score Filtering 후 Top-N 반환

---

# 🧮 Model-Based CF 전환 및 Funk SVD 구현

Item-Based CF까지의 두 모델은 모두 이미 존재하는 interaction vector 간 유사도를 계산하는 Memory-based 방식이었습니다. Funk SVD부터는 User/Item의 특징 벡터 자체를 데이터로부터 학습하는 **Model-Based(Matrix Factorization)** 방식으로 전환했습니다.

### Memory-based CF와의 핵심 차이

| 구분 | Memory-based CF (User/Item-Based) | Model-Based CF (Funk SVD) |
|---|---|---|
| User/Item 표현 | 이미 존재하는 interaction 벡터 | 데이터로부터 학습된 latent vector |
| 적합도 계산 | 존재하는 벡터 간 cosine similarity | 학습된 latent vector의 dot product |
| 학습 단위 | 유저 평가 시점마다 개별 계산 | 전체 interaction을 한 번에 학습하는 Global Model |
| Train/Test 구조 | 유저별 런타임 70:30 split으로 충분 | 평가 전 전체 데이터로 1회 학습 필요 → 별도 Global Split 설계 필요 |

### Global Train/Test 설계

- 평가 대상 유저(interaction 10~78개, **666,781명**, interaction 12,564,053건): 70% Train / 30% Test
- 그 외 유저: 100% Train
- 결과: 전체 41,154,794건 = Train 37,113,471건 + Test 4,041,323건 (원본과 정확히 일치)
- `random_state` 고정 시 동일 interaction 개수 유저는 동일 split position pattern을 가진다는 점을 이용해 `train_test_split()` 호출을 666,781회 → 최대 69회로 최적화

### Funk SVD 모델 학습 및 추천 (Surprise)

```python
model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
model.fit(trainset)
```

```python
scores = model.qi @ model.pu[inner_uid]                                 # latent 내적
scores = trainset.global_mean + model.bu[inner_uid] + model.bi + scores  # μ + b_u + b_i + q_i^T p_u
scores[seen_item_idx] = -np.inf                                          # 이미 본 게임 제외
top_idx = np.argsort(-scores[np.argpartition(scores, -top_n)[-top_n:]]) # 벡터화 Top-N
```

Funk SVD의 예측값은 정확한 평점이 아니라 **ranking score**로 사용하기로 했으며, rating 기반 loss가 pairwise ranking을 직접 최적화하지는 않는다는 한계를 확인했습니다.

### 성능 저조 원인 분석 — 가설 검증과 기각

**가설 1 (기각)**: 평가 시 `positive_only=True`로 True인 Test만 정답 처리해 정답 수가 줄어든 것이 원인이라 추정 → `positive_only=False`로 재평가

| Metric | 기존(True만) | False 포함 |
|---|---:|---:|
| Precision@10 | 0.0003 | 0.0003 |
| Hit Rate@10 | 0.0025 | 0.0025 |
| 전체 Hits | 1 | 1 |
| Test 정답 | 3,273 | 3,850 |

정답 수가 577개 늘었음에도 Hits는 그대로 1이라 **이 가설은 기각**했습니다.

**가설 2 (유력)**: 추천 게임의 Train 통계와 400명 전체의 반복 추천 게임을 분석한 결과, 아래처럼 **True 비율(긍정률)이 매우 높은 게임이 여러 사용자의 Top-10에 반복적으로 등장**하는 패턴을 발견했습니다.

| 게임(400명 반복 추천 상위) | 추천 횟수 | interaction | True Ratio |
|---|---:|---:|---:|
| DUSK '82: ULTIMATE EDITION | 26 | 108 | 0.9815 |
| OXXO | 22 | 95 | 0.9684 |
| Paperball | 22 | 92 | 0.9783 |
| 100 hidden turtles | 17 | 73 | **1.0000** |
| Hook 2 | 15 | 198 | **1.0000** |

$\hat r_{ui}=\mu+b_u+b_i+p_u^Tq_i$에서 **item bias $b_i$**가 개인화 항($p_u^Tq_i$)보다 지배적으로 작용해, "누가 평가하든 거의 항상 긍정적으로 평가되는 게임"이 특정 사용자의 취향과 무관하게 Top-N에 자주 오르는 것으로 해석했습니다. 이는 곧 **rating prediction objective**(사용자가 이 게임에 줄 rating 예측)와 **ranking objective**(미래에 실제로 interaction할 게임을 Top-N에 배치)가 서로 다른 문제라는 것과 연결됩니다.

### 다음 실험 — Bias 제거 Ablation

```python
SVD(..., biased=False)   # μ + b_u + b_i 제거, p_u^Tq_i만으로 예측
```

기존 모델과 구분하기 위해 `models/funk_svd_unbiased_model.pkl`로 별도 저장해 재평가할 예정이며, 결과에 따라 다음 세 가지로 해석합니다.

- 성능이 크게 상승 → item bias가 Top-N 랭킹을 방해한 것이 주요 원인
- 여전히 매우 낮음 → rating prediction objective와 ranking 문제의 mismatch가 더 근본적인 원인
- 성능이 더 감소 → bias도 어느 정도 유용했지만 Funk-SVD 구조 자체가 현재 문제와 맞지 않음

Ablation 결과 해석 후 Funk SVD 단계를 종료하고, Top-N ranking을 직접 학습하는 **BPR(Bayesian Personalized Ranking)**로 전환합니다. Funk SVD는 $r_{ui}\approx\hat r_{ui}$를 학습하지만 BPR은 $score(u,i^+) > score(u,j^-)$를 직접 학습해 Precision@10/Recall@10/HR@10/NDCG@10 평가와 더 직접적으로 연결됩니다.

---

# 📊 User-Based CF Evaluation Results

그룹당 100명 샘플링 기준 (유효 평가 유저 362명, 스킵 38명).

| 구분 | Precision@10 | Recall@10 | F1@10 | Hit Rate@10 | NDCG@10 |
|---|---|---|---|---|---|
| Macro | 0.0545 | 0.0427 | - | 0.3011 | 0.0655 |
| Micro | 0.0626 | 0.0454 | 0.0526 | - | - |

| review_group | precision_mean | recall_mean | hit_rate | ndcg_mean | n_users |
|---|---|---|---|---|---|
| 10-15개 | 0.0242 | 0.0303 | 0.1039 | 0.0346 | 77 |
| 16-25개 | 0.0388 | 0.0451 | 0.2151 | 0.0530 | 93 |
| 26-45개 | 0.0482 | 0.0445 | 0.3478 | 0.0599 | 92 |
| 46-78개 | 0.0982 | 0.0483 | 0.4900 | 0.1061 | 100 |

**핵심 발견**: 리뷰 수가 많은 유저일수록 Precision/Hit Rate/NDCG가 함께 상승하는 패턴을 확인. **User-Based CF는 데이터가 희소(sparse)한 유저에게는 이웃 매칭 자체가 어려워 성능이 급격히 저하되는 구조적 한계**를 가짐을 확인하고 Item-Based CF로 전환.

---

# 📊 Item-Based CF Evaluation Results

그룹당 100명씩 총 400명 평가 (스킵 0명).

| 구분 | Precision@10 | Recall@10 | F1@10 | Hit Rate@10 | NDCG@10 |
|---|---|---|---|---|---|
| Macro | 0.0783 | 0.0880 | - | 0.4800 | 0.1078 |
| Micro | 0.0783 | 0.0813 | 0.0797 | - | - |

### User-Based vs Item-Based 비교

| Model | Aggregate | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| User-Based | Macro | 0.0545 | 0.0427 | 0.3011 | 0.0655 |
| **Item-Based** | **Macro** | **0.0783** | **0.0880** | **0.4800** | **0.1078** |

모든 주요 지표가 User-Based 대비 상승했고, Item-Based는 400명 전원이 Top-10을 전부 채워(`n_recommended` 100%) Sparsity 문제가 해소됨을 확인했습니다.

---

# 📊 Funk SVD (Model-Based CF) Evaluation Results

그룹당 100명씩 총 400명 평가.

| Metric | positive_only=True | positive_only=False |
|---|---:|---:|
| Precision@10 | 0.0003 | 0.0003 |
| Recall@10 | 0.0006 | 0.0006 수준 |
| Hit Rate@10 | 0.0025 | 0.0025 |
| 전체 Hits | 1 | 1 |
| 전체 추천 | 4,000 | 4,000 |
| Test 정답 | 3,273 | 3,850 |

**주의**: User/Item-Based 평가는 True/False를 모두 정답으로 인정하는 조건이라, `positive_only=True` 결과를 위 두 모델과 직접 비교할 수 없습니다. 다만 `positive_only=False`(더 관대한 조건)로 재평가해도 Hits가 1개로 동일해, **평가 조건 차이가 낮은 성능의 원인이 아님을 확인**했습니다.

**핵심 발견**: 400명에게 총 4,000개를 추천했으나 실제 Test 정답과 일치한 것은 단 1개. 추천 게임 대부분이 **True Ratio 0.94~1.00**에 몰려 있어 item bias($b_i$)가 랭킹을 지배하고 있을 가능성이 유력한 원인으로 확인됨 (자세한 통계는 위 「Model-Based CF 전환 및 Funk SVD 구현」 섹션 참고). 모든 사용자에게 정확히 10개씩 추천되어 `n_recommended`가 상수가 되면서 Pearson 상관 계산 시 `ConstantInputWarning`(r=nan)이 발생했는데, 이는 모델 에러가 아니라 현재 MF 조건에서는 이 상관 분석 자체가 의미가 없기 때문입니다.

---

# 🔍 Item-Based CF Qualitative Experiment Findings

Item-Based CF의 정량적 우위가 실제 추천 결과에서도 납득 가능한지, 그리고 Content-Based에서 발견한 장르 혼합 쏠림 현상이 재현되는지 확인하기 위해 6개 실험을 진행했습니다.

### 발견 1 — Content-Based와 정반대 방향의 쏠림

| 모델 | 동일한 4개 입력(카드 2 + 슈팅 2) 결과 |
|---|---|
| Content-Based | 카드/덱빌딩 쪽으로 강하게 쏠림 |
| Item-Based CF | 슈팅(CS2·L4D2) 쪽으로 강하게 쏠림 |

### 발견 2 — Content-Based와 Item-Based는 서로 다른 signal을 포착

- **Content-Based**: "이 게임은 무엇인가?" — 장르·태그·테마 등 콘텐츠 자체의 semantic 유사성
- **Item-Based**: "이 게임을 플레이한 사람은 다른 무엇을 함께 소비하는가?" — 공통 interaction 기반의 인접 취향

### 발견 3 — 세부 의미 특징 보존의 한계

Left 4 Dead 2 입력 시 좀비/공포/협동이라는 세부 특징은 강하게 유지되지 않고 전반적인 슈팅 취향만 두드러졌습니다.

> 이 발견들을 종합해 Content-Based + Item-Based를 결합하는 **Hybrid 설계의 필요성**을 확인했습니다. 구체적 설계는 Model-Based baseline 완료 이후로 보류했습니다.

---

# 🐛 알려진 이슈 (To-Do)

- **Name = NaN metadata mismatch**: Item-Based CF 정성평가 과정에서 일부 추천 결과의 게임 이름이 NaN으로 조회되는 현상. 원인 분석 보류
- **Party Animals 입력 시 빈 DataFrame 반환**: 정성평가 대상에서 제외. 원인 분석 보류
- **`ModuleNotFoundError: No module named 'data_split'`**: `models/` 내부 파일 직접 실행 시 import 경로 문제. `python -m models.funk_svd` 형태로 해결 방향 확인
- **Funk SVD baseline 성능이 극단적으로 낮음 (P@10=0.0003, Hits 1/4000)**: `positive_only` 조건 차이는 원인이 아님을 확인(기각). **item bias($b_i$)가 랭킹을 지배하는 것으로 추정**되며, `biased=False` Ablation으로 검증 예정
- **Rating Prediction objective와 Top-N Ranking objective의 근본적 불일치**: Funk SVD 구조 자체의 한계일 가능성이 있어, Ablation 결과와 무관하게 BPR 전환을 계획

---

# ✅ Implemented Features

## Content-Based
- Steam Metadata Loading (with Parquet Caching), Data Validation, User-based Train/Test Split, Combined Features Generation, TF-IDF Vectorization, Cosine Similarity Recommendation, Multi-Game Recommendation, AppID 기반 게임 식별 및 동명이인 처리, 정성적 실험

## User-Based Collaborative Filtering
- Sparse Interaction Matrix 구축, Cosine Similarity 기반 Top-K 이웃 탐색, 자기 자신 제외 로직, Train 게임 목록 기반 Query Vector 생성, 이웃 가중합 기반 예측 점수 계산

## Item-Based Collaborative Filtering
- User×Item → Item×User 구조 변환, Self-Similarity 0 처리, Source Item별 행 단위 Positive Top-K 추출, Candidate Aggregation, Train Item Exclusion, 정성적 실험 6종

## Model-Based Collaborative Filtering (Funk SVD)

- Global Train/Test Split 설계 및 최적화 (interaction count별 position 재사용)
- 41,154,794건 interaction Parquet 저장/로드
- Surprise `Dataset`/`Reader` 기반 데이터 변환
- Funk SVD(`SVD`) 학습 파이프라인 및 **`recommend()` Top-N 추천 함수 구현** (bias 반영, seen-item 제거, 벡터화 Top-N)
- MF 전용 Evaluation 구현 (`build_mf_user_review_groups`, `evaluate_mf_user`, `run_mf_evaluation`)
- **400명 정량평가 실행** (Precision@10 0.0003, Hits 1/4,000)
- **`positive_only=False` 재평가로 "정답 축소" 가설 기각**
- **추천 게임 Train 통계 분석(`groupby("app_id")`) 및 400명 반복 추천 빈도 분석(`Counter`)으로 item bias 원인 가설 확보**
- [진행 중] `biased=False` Ablation 실험 구현
- [예정] BPR 구현 및 Funk SVD와 성능 비교

## Evaluation (공통)
- 리뷰 수 구간 기반 층화 표집 평가 시스템, Precision@K/Recall@K/Hit Rate@K/NDCG@K, Macro/Micro Precision·Recall·F1, Pearson Correlation 기반 가설 검증, 구간별 평가 결과 Breakdown 리포트

---

# 🚀 Development Roadmap

## ✅ V1. Content-Based Recommendation
전체 완료 (세부 항목은 이전 버전 참고)

## ✅ V2. Collaborative Filtering

### User-Based CF / Item-Based CF
전체 완료

### Model-Based CF 학습 및 적용 순서

#### 1. Funk SVD — 핵심 Model-Based baseline

- [x] Memory-based CF와 Model-Based CF(Global Model)의 학습 구조 차이 파악
- [x] Global Train/Test Split 설계 및 대규모 split 성능 최적화
- [x] Surprise 기반 Funk SVD 기본형 구현 및 학습 파이프라인 정상 작동 확인
- [x] Top-N 추천 함수 구현 및 기존 evaluation pipeline 연결, 400명 정량평가 완료
- [x] 성능 저조 원인 진단 1단계: `positive_only=False` 재평가 → 가설 기각
- [x] 성능 저조 원인 진단 2단계: 추천 게임 Train 통계 + 400명 반복 추천 분석 → item bias 가설 확보
- [ ] `biased=False` Ablation 실험 실행 및 결과 해석
- [ ] Funk SVD 단계 최종 종료 판단

#### 2. BPR — Top-N Ranking 직접 학습 (다음 단계로 확정)

- [ ] Funk-SVD와 BPR의 차이 이해
- [ ] Positive / Negative item 정의
- [ ] 기존 MF Train/Test split 그대로 사용
- [ ] 기본 BPR 구현
- [ ] Top-10 추천 연결
- [ ] 기존 평가 지표(P@10, R@10, HR@10, NDCG@10)로 평가
- [ ] Funk-SVD와 성능 비교

#### 3. ALS — 짧은 비교 실험 (선택적)

- [ ] Funk SVD와 동일한 latent factor 계열이지만 Alternating Least Squares로 학습되는 구조 확인
- [ ] Sparse interaction에서의 학습 방식, 실행 시간, 추천 성능을 SVD와 비교
- [ ] 학습 중복도가 높거나 프로젝트 기간이 길어질 경우 baseline 비교까지만 진행 후 종료

#### 4. Clustering — 사용자 집단 구조 분석 (선택적)

- [ ] Sparse User-Item representation을 이용한 MiniBatch K-Means baseline 실험
- [ ] Cluster별 대표 게임/선호 패턴을 확인하여 사용자 segmentation 가능성 분석

### Model Comparison & Selection (Model-Based 이후)

- [ ] Content-Based / User-Based / Item-Based / Funk SVD / BPR을 동일 조건으로 비교
- [ ] Accuracy 지표뿐 아니라 Sparsity, Coverage, Personalization, Explainability, 실행 시간 관점 비교
- [ ] 성능과 역할이 가장 뚜렷한 1~2개 모델만 소규모 튜닝

### MAP & Popularity Baseline (Model-Based 이후, Hybrid 이전 진행 예정)

- [ ] MAP@K 구현
- [ ] Popularity Baseline 비교
- [ ] Hit item의 Popularity Bias 분석

---

## 🚧 V3. Hybrid Recommendation

- Hybrid Recommendation
- Content-Based와 Item-Based CF가 정반대 방향으로 쏠린 결과를 반영한 Hybrid aggregation 설계

---

## 🚧 V4. Deployment

- FastAPI
- Streamlit

---

# 📊 Evaluation

추천 시스템은 **Train Dataset**으로 학습하고 **Test Dataset**으로 평가합니다. Memory-based CF는 유저 단위 런타임 70:30 split(Leave-N-Out)을 사용하고, Model-Based CF(Funk SVD)는 전체 사용자 interaction을 한 번에 학습하는 Global Model이므로 별도의 Global Train/Test를 Parquet으로 저장해 재사용합니다.

### Recommendation Quality
- Precision@K / Recall@K / Hit Rate@K / NDCG@K / Micro Precision·Recall·F1 / MAP(예정)

### Sampling Strategy
- 리뷰 수 구간별 층화 표집 (User-Based 100명/그룹, Item-Based·Funk SVD 400명 규모 본 평가 완료)

### Qualitative Evaluation
- Content-Based ↔ Item-Based 간 동일 입력 비교를 통한 쏠림 방향 차이 분석
- Funk SVD는 랜덤 사용자 정성 확인 + 추천 게임 Train 통계 분석으로 item bias 패턴 확인

### Model Comparison
- Content-Based / User-Based / Item-Based CF 완료
- Model-Based CF(Funk SVD): 학습·추천·평가 완료, 원인 진단 및 Ablation 진행 중
- BPR: 다음 단계로 구현 예정
- Hybrid Recommendation / Popularity Baseline: 예정

### Performance
- 대규모 Split 연산 최적화 (Python loop 66만 회 → interaction count별 최대 69회, 약 5초 소요)
- 추천 게임 통계·반복 빈도 분석에 `groupby`/`Counter` 기반 벡터화 집계 활용

---

# 📈 Current Progress

| Module | Status |
| :--- | :---: |
| Data Loading / Caching / Validation / Preprocessing | ✅ |
| Content-Based Recommendation 전체 | ✅ |
| User-Based Collaborative Filtering 전체 | ✅ |
| Item-Based Collaborative Filtering 전체 | ✅ |
| Model-Based CF Global Train/Test Split | ✅ |
| Funk SVD 학습 파이프라인 및 Top-N 추천 함수 | ✅ |
| Funk SVD Evaluation 연결 및 400명 정량평가 | ✅ |
| Funk SVD 원인 진단 (positive_only 재평가, 통계 분석) | ✅ |
| Funk SVD `biased=False` Ablation | 🚧 |
| BPR 구현 및 비교 | ⏳ |
| ALS (짧은 비교 실험) | ⏳ |
| Clustering (사용자 군집화 분석) | ⏳ |
| Model Comparison & Selection | ⏳ |
| MAP | 🚧 |
| Popularity Baseline | 🚧 |
| Hybrid Recommendation | 🚧 |
| Deployment | 🚧 |

---

# 📌 Project Status

**Current Version:** `V2.9 - Funk SVD 원인 진단 완료(item bias 가설 확보), Ablation 준비 및 BPR 전환 계획 확정`

### Completed

- Content-Based / User-Based CF / Item-Based CF 전체 파이프라인 구현 및 정량·정성 평가
- Model-Based CF Global Train/Test Split 설계 및 저장 완료 (41,154,794건)
- Surprise 기반 Funk SVD 학습 파이프라인 및 Top-N 추천 함수(`recommend()`) 구현
- Funk SVD Evaluation 연결 및 400명 정량평가 완료 (Precision@10 0.0003)
- **`positive_only=False` 재평가로 "정답 축소가 원인" 가설을 실험으로 기각**
- **추천 게임 Train 통계 및 400명 반복 추천 빈도 분석으로 item bias 원인 가설 확보**
- **Rating Prediction objective와 Top-N Ranking objective의 불일치를 개념적으로 정리**

### In Progress

- **Funk SVD `biased=False` Ablation 실험 구현 및 실행**
- **Ablation 결과 해석**

### Next Milestone

➡️ `biased=False` Ablation 실행 및 기존 Biased Funk SVD와 비교
➡️ Ablation 결과 해석 후 Funk SVD 단계 종료
➡️ **BPR(Bayesian Personalized Ranking) 구현**: Funk-SVD와의 차이 이해 → Positive/Negative item 정의 → 기존 MF Train/Test split 재사용 → 기본 BPR 구현 → Top-10 추천 연결 → 기존 평가 지표(P@10/R@10/HR@10/NDCG@10)로 Funk SVD와 성능 비교
➡️ (필요 시) ALS 짧은 비교 실험, Clustering 사용자 군집화/세그먼트 분석
➡️ Model-Based 단계 종료 후 전체 모델 비교 및 최상위 1~2개 모델만 소규모 튜닝
➡️ MAP@K, Popularity Baseline 및 Popularity Bias / Coverage 분석
➡️ Hybrid Recommendation (모델별 failure mode 및 쏠림 방향 차이 기반 설계)
➡️ Web Service Deployment