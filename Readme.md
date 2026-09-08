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
- **Surprise 기반 Funk SVD 모델 구현 및 학습 성공** (`SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)`, 10만 건 샘플 및 전체 Train 3,700만 건 규모에서 학습 파이프라인 정상 작동 확인)
- **Funk SVD Top-N 추천 함수 구현 완료** (latent vector 내적 + bias 반영, seen-item 제거, `argpartition`/`argsort` 기반 벡터화 Top-N)
- **Funk SVD 전용 Evaluation 파이프라인 구축 및 400명 정량평가 완료** (기존 Precision/Recall/HitRate/NDCG 지표 재사용, Global Train/Test 구조에 맞는 데이터 공급 로직만 신규 작성)
- **Funk SVD baseline 성능이 매우 낮음을 확인하고 원인 후보 정리** (Precision@10 0.0003, `positive_only=True`로 인한 기존 CF 평가와의 조건 불일치 발견 → 원인 진단은 다음 단계로 명시적으로 분리)

를 완료하였으며, 현재는 **Funk SVD baseline의 낮은 성능 원인을 진단하는 단계**입니다 (`positive_only` 조건 재검토, Top-10 overlap, item bias 영향, sparsity 영향 등을 확인한 뒤 Funk SVD 유지/개선, BPR·implicit MF 검토, 기존 CF 유지 중 방향을 결정할 예정).

향후에는 원인 진단 및 모델 방향 결정 → (필요 시) ALS 짧은 비교 실험 → Clustering(사용자 군집화 분석) → BPR(진행 상황에 따라 선택적 확장) 순서로 서로 다른 모델의 구조와 성능을 경험한 뒤, 전체 모델 비교·소규모 튜닝·Hybrid Recommendation·웹 서비스 배포까지 확장하는 것을 목표로 합니다.

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
- **Surprise 기반 Funk SVD 학습 파이프라인 구현 및 정상 작동 확인**
- **Funk SVD Top-N 추천 함수 구현 완료** (bias 포함 score 계산, seen-item 제거, 벡터화 Top-N 추출)
- **Funk SVD 정량평가 완료** (400명, 기존 evaluation 지표 재사용 + MF 전용 데이터 공급 로직 신규 구현)

---

## 🚧 In Progress

- **Funk SVD 낮은 성능(P@10=0.0003)의 원인 진단** (`positive_only=False` 재평가, Top-10 overlap, item bias 영향, Train vocabulary 커버리지, sparsity 영향 확인)
- **모델 방향 의사결정** (Funk SVD 유지·개선 / BPR·implicit MF 검토 / 기존 CF 계열 유지)

---

## 🚀 Future

- 진단 결과에 따른 Funk SVD 개선 실험 (n_factors, epoch, learning rate, regularization 등) 또는 BPR/implicit MF 검토
- ALS (Matrix Factorization 최적화 방식 비교, 짧은 실험)
- Clustering (사용자 군집화/세그먼트 분석 중심의 짧은 실험)
- BPR (Top-N Ranking 학습, 프로젝트 진행 상황에 따라 선택적 진행)
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
- **Matrix Factorization / Funk SVD** (Surprise `SVD`, SGD 기반 Latent Factor 학습, bias(μ, bu, bi) 포함)

## Evaluation

- Precision@K / Recall@K / Hit Rate@K / NDCG@K (Macro)
- Micro Precision / Micro Recall / Micro F1
- Stratified Sampling (리뷰 수 구간 기반)
- Qualitative Experiment (단일/혼합 입력 결과 분석)
- Leave-N-Out 유저별 Train/Test Split (Memory-based CF)
- **Global Train/Test Split (Model-Based CF, 평가 대상 유저 70:30 + 그 외 유저 100% Train)**
- Pearson Correlation (SciPy) 기반 가설 검증

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
│   ├── Day16.md
│   └── Day17.md
│
├── models/
│   ├── content_base.py
│   ├── userbase.py
│   ├── itembase.py          # 구현 완료
│   └── funk_svd.py           # 학습·추천(Top-N) 구현 완료, 성능 원인 진단 진행 중
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

## Model-Based CF (Funk SVD) Pipeline — 학습·추천·평가 파이프라인 구현 완료, 성능 원인 진단 중

Memory-based CF(User/Item-Based)는 평가 시점에 사용자별로 70:30 split을 적용해도 문제가 없었지만, Funk SVD는 **모든 사용자 interaction으로 User/Item latent vector를 동시에 학습하는 Global Model**이라 평가 대상 유저마다 모델을 다시 학습하는 기존 구조를 쓸 수 없습니다. 이에 따라 Train/Test 구조 자체를 재설계했고, 학습된 모델로부터 실제 추천을 만들고 평가하는 부분까지 완성했습니다.

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
data_split.py
  → interaction 개수별 동일 position pattern 재사용 (66만 회 → 최대 69회 split 연산으로 최적화)
  → mf_train.parquet (37,113,471건) / mf_test.parquet (4,041,323건) 저장
      │
      ▼
Surprise Dataset 변환 → SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02).fit(trainset)
      │
      ▼
recommend(user_id, app_id_list, top_n)
  ├─ inner_uid = trainset.to_inner_uid(user_id)
  ├─ user_vector = model.pu[inner_uid]           # 40차원 latent vector
  ├─ scores = model.qi @ user_vector              # 전체 게임과 내적
  ├─ scores += global_mean + bu[inner_uid] + bi   # bias 반영 (μ + b_u + b_i + qᵀp)
  ├─ scores[seen_items] = -inf                     # Train에서 이미 본 게임 제외
  └─ argpartition → argsort → raw_item_ids 복원   # 벡터화 Top-N 추출
      │
      ▼
MF 전용 Evaluation (build_mf_user_review_groups / evaluate_mf_user / run_mf_evaluation)
  → 기존 Precision/Recall/Hit Rate/NDCG 지표 재사용, MF Train/Test 구조에 맞는 데이터 공급만 신규 구현
      │
      ▼
[결과] 400명 평가: Precision@10 0.0003, Hits 1/4000 — 매우 낮은 baseline 확인
      │
      ▼
[진행 중] 원인 진단: positive_only 조건 재검토, Top-10 overlap, item bias 영향, sparsity 영향 등
```

> Funk SVD의 예측값 $\hat r_{ui}=\mu+b_u+b_i+q_i^Tp_u$은 정확한 평점이 아니라 **ranking score**로 사용합니다. 다만 400명 평가 결과 Precision@10 **0.0003**으로 매우 낮게 나왔고, 원인 후보 중 하나로 현재 MF 평가가 `positive_only=True`(is_recommended=True만 정답)를 쓰는 반면 기존 User/Item-Based 평가는 True/False를 모두 정답으로 인정한다는 **평가 조건 불일치**를 발견했습니다. 따라서 현재 수치를 User-based(P@10≈0.0545)·Item-based(P@10≈0.078)와 그대로 비교할 수 없으며, `positive_only=False` 재평가를 포함한 원인 진단을 다음 단계로 진행합니다.

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

> Game Name은 중복될 수 있지만 AppID는 고유하므로, 내부 로직은 전부 **AppID 기준**으로 동작하도록 설계하였습니다. User-Based/Item-Based CF에서도 동일한 원칙을 적용하여, `user_to_idx` / `game_to_idx` / `idx_to_game`을 통해 ID ↔ 행렬 인덱스 변환을 일관되게 관리합니다. Funk SVD 역시 Surprise 내부의 raw id ↔ inner id 매핑(`to_inner_uid`, `raw_item_ids`)을 통해 동일한 AppID 기준 식별 원칙을 유지합니다.

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
- 수정 후 재평가한 NDCG@10은 **0.0655**로 확인했습니다. 다만 이 값을 "`set → list` 수정 때문에 NDCG가 올랐다"고 단정하지는 않았습니다 — 순서를 반영하지 않던 이전 구현과 이번 값을 직접 비교할 근거가 없으므로, 순서를 올바르게 보존하는 구현으로 수정한 뒤 최종적으로 확인된 값이 0.0655라는 사실만 기록하기로 했습니다.

---

# 📐 Macro vs Micro 평가 및 Sparsity 정량 검증

### Macro → Micro 확장 배경

기존 평가는 유저별 Precision/Recall을 계산 후 평균 내는 Macro 방식이었는데, `n_recommended`가 유저마다 1~10개로 크게 달라(평균 7.28/10, Top-10 완전 채움 54.4%), 추천을 적게 받은 유저와 많이 받은 유저가 Macro 평균에서 동일한 가중치를 갖는 문제를 확인했습니다. 이를 보완하기 위해 **Micro Precision/Recall**(전체 Hits를 전체 추천 수·Test 수로 나눈 값)과 **Micro F1**을 추가로 도입했습니다. Hit Rate와 NDCG는 각각 이미 유저 단위/정규화된 지표라 별도 Micro 확장이 필요하지 않다고 판단해 제외했습니다.

> Micro 수치가 Macro보다 높게 나온 것은 **모델 성능이 개선된 것이 아니라, 동일한 결과를 다른 가중치 기준으로 재집계**한 것입니다.

### Sparsity 가설 정량 검증 (Pearson Correlation)

| 비교 변수 | Pearson r | p-value |
|---|---|---|
| n_games ↔ n_recommended | +0.3096 | < 0.0001 |
| n_games ↔ precision | +0.2504 | < 0.0001 |

r 값은 약~중간 수준의 양의 상관이며, 상관관계가 인과관계를 증명하지는 않습니다. 다만 review_group별 추이(추천 개수 5.74→8.54, Precision 0.024→0.098)와 함께, **"interaction이 적을수록 안정적인 이웃 형성이 어렵다"는 기존 가설과 일관된 방향의 정황**으로 해석했습니다.

---

# 🔗 Self-Similarity 처리 원칙 (User-Based / Item-Based 공통)

Item-Based CF로 전환하기 전, "자기 자신과의 유사도를 계산하는 것 자체가 문제인가"를 정리했습니다.

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

Item-Based CF를 기존 `userbase.py` 기반으로 구현하면서, 두 모델이 단순히 "similarity 대상만 다른 것"이 아니라 구조적으로 다르다는 점을 정리했습니다.

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

### 핵심 정리

- **`.getrow(0)`은 User-Based/Item-Based를 결정하는 요소가 아닙니다.** 어떤 similarity(User-User vs Item-Item)가 계산되는지는 matrix의 각 행이 무엇을 의미하는지가 결정하며, `.getrow()`는 단지 결과 행을 꺼내는 연산일 뿐입니다.
- **Self-similarity 제거와 이미 플레이한 게임(Seen-item) 제거는 별개의 과정입니다.** `A↔A`를 0으로 만드는 것은 자기 자신이 trivial하게 최고 유사도를 차지하는 걸 막는 것이고, Train에서 이미 interaction한 아이템을 최종 후보에서 빼는 것(`predicted_scores[col_idx] = -np.inf`)은 다른 목적입니다. 두 과정 모두 필요합니다.
- **여러 Source Item의 similarity를 합치는 aggregation이 필요합니다.** 추천은 최종적으로 1차원 ranking을 만들어야 하므로, 여러 관계를 하나의 candidate score로 축약하는 과정 자체는 불가피합니다. 현재는 가장 기본적인 similarity sum 방식을 baseline으로 채택했으며, 평균·최대값·가중치 부여 등 다른 aggregation은 Model-Based baseline 완료 이후 개선 단계에서 검토합니다.

### 구현 완료 사항

- `evaluation.py`와 main pipeline 그대로 유지 — User-Based/Item-Based 비교 조건 통일
- 기존 `build_interaction_matrix()` 재사용, `interaction_matrix.T`로 Item 관점 추가
- Source item과 전체 item 간 sparse similarity만 계산 (전체 Item×Item dense similarity는 생성하지 않음)
- Self-similarity 0 처리 후 Source Item별 행 단위 Positive Top-K 추출 (k=30)
- 동일 candidate로 모이는 similarity를 합산하는 Candidate Aggregation 구현
- Train Item Exclusion 및 Positive Score Filtering 후 Top-N 반환
- 기존 `recommend()` 출력 형식(`DataFrame["app_id"]`) 및 `exclude_user_idx` 인터페이스 유지 (Item-Based 내부에서는 미사용, 공통 evaluation pipeline 호환을 위해 유지)

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
| 추천 입력 | 입력 게임 목록 → 유사도 계산 | `user_id` → 학습된 latent vector 조회 (게임 목록은 seen-item 제거용 보조 입력) |

이 차이로 인해 평가 시점마다 사용자별로 70:30 split을 적용하던 기존 방식은 Funk SVD에 그대로 적용할 수 없다는 점을 직접 확인했고, 아래와 같이 Train/Test 구조를 재설계했습니다.

### Global Train/Test 설계

- 평가 대상 유저(interaction 10~78개, **666,781명**, interaction 12,564,053건): 70% Train / 30% Test
- 그 외 유저: 100% Train
- 결과: 전체 41,154,794건 = Train 37,113,471건 + Test 4,041,323건 (원본과 정확히 일치, 누락 없음 확인)
- `random_state` 고정 시 interaction 개수가 같은 유저는 동일한 split position pattern을 가진다는 점을 이용해, `train_test_split()` 호출을 666,781회 → 최대 69회로 줄이는 최적화 적용 (전체 split 약 5초 소요)
- 재사용을 위해 `data/split/mf_train.parquet`, `data/split/mf_test.parquet`로 저장

### Funk SVD 모델 학습 (Surprise)

```python
from surprise import Dataset, Reader, SVD

train_df["is_recommended"] = train_df["is_recommended"].astype(int)  # False→0, True→1
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(train_df[["user_id", "app_id", "is_recommended"]], reader)
trainset = data.build_full_trainset()

model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
model.fit(trainset)
```

- `n_factors`: User/Item을 표현하는 latent 차원 수
- `n_epochs` / `lr_all` / `reg_all`: 전체 반복 횟수 / SGD learning rate / regularization
- 10만 건 샘플(유저 98,022 / 아이템 9,360)과 Train 전체(37,113,471건)에서 학습 파이프라인이 정상 작동하는 것을 확인

### Funk SVD 추천(Top-N) 구현

```python
inner_uid = self.trainset.to_inner_uid(user_id)
user_vector = self.model.pu[inner_uid]              # 40차원 latent vector
scores = self.model.qi @ user_vector                # 전체 게임과 내적

scores = (
    self.trainset.global_mean
    + self.model.bu[inner_uid]
    + self.model.bi
    + scores
)  # μ + b_u + b_i + q_i^T p_u

scores[inner_iid_list] = -np.inf   # Train에서 이미 본 게임 제외

top_idx = np.argpartition(scores, -top_n)[-top_n:]   # 상위 N개 후보 추출
top_idx = top_idx[np.argsort(-scores[top_idx])]        # 후보 내 순위 정렬
recommended_app_ids = self.raw_item_ids[top_idx]        # 내부 index → 실제 app_id 복원
```

Surprise SVD는 latent vector 내적만 쓰는 것이 아니라 $\hat r_{ui} = \mu + b_u + b_i + q_i^Tp_u$ 형태로, 전체 평균(`global_mean`) + 사용자 성향(`bu`) + 게임 성향(`bi`) + latent interaction을 함께 반영한다는 것을 코드 레벨에서 확인했습니다. 이미 본 게임은 점수를 `-inf`로 만들어 Top-N에서 자연스럽게 제외하고, `argpartition`(상위 후보 추출) → `argsort`(후보 내 정렬)로 벡터화된 방식으로 Top-N을 뽑도록 구현했습니다.

Funk SVD의 예측값은 정확한 평점이 아니라 **ranking score**로 사용하기로 했으며, rating 기반 loss가 pairwise ranking을 직접 최적화하지는 않는다는 한계도 확인했습니다(추후 BPR 탐색 여지로 남김).

---

# 📊 User-Based CF Evaluation Results

그룹당 100명 샘플링 기준 (유효 평가 유저 362명, 스킵 38명). NDCG는 순위 보존 버그 수정 반영된 최신 값입니다.

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

**핵심 발견**: 리뷰 수(review_group)가 많은 유저일수록 Precision/Hit Rate/NDCG가 함께 상승하는 패턴을 확인. 동일 구간에서 추천 후보 개수(`n_recommended`, 평균 7.28/10, 10개 완전 채움 비율 54.4%)도 함께 증가하며, 개별 유저 단위 Pearson Correlation(`n_games ↔ n_recommended` r=+0.31, `n_games ↔ precision` r=+0.25, 둘 다 p<0.0001)에서도 같은 방향의 관계를 확인. **User-Based CF는 상호작용 데이터가 풍부한 유저에게는 어느 정도 작동하지만, 데이터가 희소(sparse)한 유저에게는 이웃 매칭 자체가 어려워 성능이 급격히 저하되는 구조적 한계**를 가짐을 확인. 단일 지표가 아닌 추천 생성 안정성·그룹별 추이·상관분석이 일관된 방향을 보인다는 점을 근거로 Item-Based CF로 전환하기로 결정.

> 참고: Offline 평가의 Test set은 "사용자가 좋아할 수 있는 모든 정답"이 아니라 "숨겨둔 일부 관측된 interaction을 얼마나 복원하는가"에 가까우므로, Precision 절대값만으로 모델의 좋고 나쁨을 단정하지 않고 있습니다.

---

# 📊 Item-Based CF Evaluation Results

그룹당 100명씩 총 400명 평가 (스킵 0명).

| 구분 | Precision@10 | Recall@10 | F1@10 | Hit Rate@10 | NDCG@10 |
|---|---|---|---|---|---|
| Macro | 0.0783 | 0.0880 | - | 0.4800 | 0.1078 |
| Micro | 0.0783 | 0.0813 | 0.0797 | - | - |

| review_group | precision_mean | recall_mean | hit_rate | ndcg_mean | n_users |
|---|---|---|---|---|---|
| 10-15개 | 0.046 | 0.1147 | 0.30 | 0.0999 | 100 |
| 16-25개 | 0.048 | 0.0761 | 0.33 | 0.0771 | 100 |
| 26-45개 | 0.089 | 0.0886 | 0.58 | 0.1025 | 100 |
| 46-78개 | 0.130 | 0.0726 | 0.71 | 0.1516 | 100 |

### User-Based vs Item-Based 비교

| Model | Aggregate | Precision@10 | Recall@10 | F1@10 | Hit Rate@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| User-Based | Macro | 0.0545 | 0.0427 | - | 0.3011 | 0.0655 |
| User-Based | Micro | 0.0626 | 0.0454 | 0.0526 | - | - |
| **Item-Based** | **Macro** | **0.0783** | **0.0880** | - | **0.4800** | **0.1078** |
| **Item-Based** | **Micro** | **0.0783** | **0.0813** | **0.0797** | - | - |

**핵심 발견**:

- 모든 주요 지표가 User-Based 대비 상승 (Hit Rate@10 +0.1789, Macro Recall@10 약 2배 증가)
- Item-Based는 400명 전원이 Top-10을 전부 채움(`n_recommended` 평균 10.0/10, 100%) — User-Based에서 발견된 candidate 부족 문제가 완전히 해소됨
- `n_games ↔ precision` Pearson r=0.3387 (p<0.001)로 User-Based보다 상승. 다만 이는 Item-Based의 정보량 증가뿐 아니라 Train 크기 증가에 따른 test relevant item 수 증가(Precision@10의 metric ceiling 상승)가 함께 작용했을 가능성으로 해석 — 단일 원인으로 단정하지 않음
- 활동량이 높은 유저 그룹일수록 Precision·Hit Rate는 뚜렷이 상승하지만 Recall은 동일한 패턴으로 증가하지 않음 (`0.115 → 0.076 → 0.089 → 0.073`) — Top-10 내 정답 발견 가능성은 높아지지만 전체 relevant item 대비 회수 비율은 비례하지 않음

---

# 📊 Funk SVD (Model-Based CF) Evaluation Results

그룹당 100명씩 총 400명 평가. **주의: 이 평가는 `positive_only=True`(is_recommended=True인 Test만 정답으로 인정) 조건이며, User/Item-Based 평가는 True/False를 모두 정답으로 인정하는 조건이라 아래 지표를 위 두 모델과 직접 비교할 수 없습니다.**

| Metric | Result |
|---|---:|
| Precision@10 | 0.0003 |
| Recall@10 | 0.0006 |
| Hit Rate@10 | 0.0025 |
| NDCG@10 | 0.0004 |
| Micro Precision@10 | 0.0003 |
| Micro Recall@10 | 0.0003 |
| Micro F1@10 | 0.0003 |
| 전체 Hits | 1 |
| 전체 추천 | 4,000 |
| Test 정답 | 3,273 |

| review_group | hit_rate |
|---|---:|
| 10-15개 | 0.01 |
| 16-25개 | 0 |
| 26-45개 | 0 |
| 46-78개 | 0 |

**핵심 발견**:

- 400명에게 총 4,000개를 추천했으나 실제 Test 정답과 일치한 것은 **단 1개**로, User/Item-Based 대비 baseline이 매우 낮게 나옴
- 모든 사용자에게 정확히 10개씩 추천되어(`n_recommended` 완전 상수) `n_games ↔ n_recommended` Pearson 상관 계산 시 `ConstantInputWarning`(r=nan) 발생 — 모델 에러가 아니라 평가 대상 변수가 상수이기 때문이며, 현재 MF에서는 이 상관 분석 자체가 의미가 없음
- 가장 유력한 원인 후보로 **평가 조건 불일치**(`positive_only=True` vs 기존 평가의 True/False 통합)를 발견했으나, 그 외에도 True/False 86:14 클래스 불균형, explicit rating prediction과 ranking objective의 불일치, sparsity, item bias/popularity 영향, cold item, latent factor 수 등 여러 원인 후보가 있어 **단일 원인으로 단정하지 않고 다음 단계에서 순차적으로 진단 예정**

---

# 🔍 Item-Based CF Qualitative Experiment Findings

Item-Based CF의 정량적 우위가 실제 추천 결과에서도 납득 가능한지, 그리고 Content-Based에서 발견한 장르 혼합 쏠림 현상이 재현되는지 확인하기 위해 6개 실험(단일 취향 2건, 혼합 취향 1건, 단일 게임 확장성 3건)을 진행했습니다.

### 발견 1 — Content-Based와 정반대 방향의 쏠림 (핵심 발견)

Content-Based 실험(카드/덱빌딩 + 슈팅 혼합 입력)에서 발견했던 장르 혼합 쏠림 현상을 Item-Based에 동일하게 입력하여 재현 여부를 확인했습니다.

| 모델 | 동일한 4개 입력(카드 2 + 슈팅 2) 결과 |
|---|---|
| Content-Based | 카드/덱빌딩 쪽으로 강하게 쏠림 |
| Item-Based CF | 슈팅(CS2·L4D2) 쪽으로 강하게 쏠림 |

카드/덱빌딩 게임 2종을 추가로 넣었음에도 슈팅 단독 입력 실험의 Top-5 추천이 그대로 유지되는 현상을 확인했습니다. 두 모델 모두 서로 다른 두 취향을 균형 있게 보존하지 못했지만, **쏠리는 방향이 정반대**라는 점이 핵심입니다.

### 발견 2 — Content-Based와 Item-Based는 서로 다른 signal을 포착

- **Content-Based**가 상대적으로 잘 보는 것: "이 게임은 무엇인가?" — 장르·태그·테마 등 콘텐츠 자체의 semantic 유사성 (예: RDR2 → GTA V)
- **Item-Based**가 상대적으로 잘 보는 것: "이 게임을 플레이한 사람은 다른 무엇을 함께 소비하는가?" — 공통 interaction 기반의 인접 취향으로 확장 (예: PUBG → 경쟁 FPS·생존 PvP·배틀로얄·온라인 액션으로 확장)

### 발견 3 — 세부 의미 특징 보존의 한계

Left 4 Dead 2 입력 시 좀비/공포/협동이라는 세부 특징은 추천 결과에서 강하게 유지되지 않고 전반적인 슈팅 취향만 두드러졌습니다. RDR2 입력에서도 콘텐츠적으로 가장 직접적인 후보(GTA V)보다 넓은 유저 취향의 게임(Witcher 3, Cyberpunk 2077 등)이 우선되었습니다.

> 이 발견들을 종합해 Content-Based(콘텐츠 세부 특성) + Item-Based(실제 사용자 행동 기반 인접 취향)를 결합하는 **Hybrid 설계의 필요성**을 확인했습니다. 구체적 설계는 Model-Based baseline 완료 이후로 보류했습니다.

---

# 🐛 알려진 이슈 (To-Do)

- **Name = NaN metadata mismatch**: Item-Based CF 정성평가 과정에서 일부 추천 결과의 게임 이름이 NaN으로 조회되는 현상을 발견. 원인 분석 및 수정은 진행하지 않고 기록만 남김 (Model-Based 단계 이후 처리 예정)
- **Party Animals 입력 시 빈 DataFrame 반환**: 정성평가 대상에서 제외하고 Stardew Valley로 대체. 원인 분석 보류
- **`ModuleNotFoundError: No module named 'data_split'`**: `models/` 내부 파일을 직접 실행할 때 import 기준 경로가 달라져 발생. `python -m models.funk_svd`처럼 project root 기준으로 module을 실행하는 방식으로 해결 방향 확인
- **Funk SVD 평가 조건(`positive_only=True`)이 기존 CF 평가(True/False 모두 정답)와 다름**: 현재 Funk SVD 지표를 User/Item-Based와 직접 비교할 수 없음. `positive_only=False` 재평가가 다음 단계 최우선 작업
- **Funk SVD baseline 성능이 극단적으로 낮음 (P@10=0.0003, Hits 1/4000)**: 원인 미확정. 평가 조건 외에도 클래스 불균형, objective 불일치, sparsity, item bias 등 복수 후보를 놓고 순차 진단 예정

---

# ✅ Implemented Features

## Content-Based

- Steam Metadata Loading (with Parquet Caching)
- Data Validation
- User-based Train/Test Split
- Data Preprocessing (Name/AppID 기준 중복 제거)
- Combined Features Generation
- TF-IDF Vectorization
- Cosine Similarity Recommendation
- Multi-Game Recommendation
- AppID 기반 게임 식별 및 동명이인 게임 선택 기능
- Duplicate Recommendation Handling (AppID 기준)
- 정성적 실험 (단일/장르 혼합 입력 분석)

## User-Based Collaborative Filtering

- Sparse Interaction Matrix 구축 (+1 / -1 인코딩)
- Cosine Similarity 기반 Top-K 이웃 탐색
- 자기 자신 제외 로직 (Self-Exclusion, Data Leakage 방지)
- Train 게임 목록 기반 Query Vector 생성 (Test 누수 방지)
- 이웃 가중합 기반 예측 점수 계산

## Item-Based Collaborative Filtering

- User×Item → Item×User 구조 변환 (기존 Interaction Matrix 재사용, transpose)
- Source Item(Train App ID) 기반 Sparse Similarity 계산 (Query Vector 생성 불필요)
- Self-Similarity 0 처리
- Source Item별 행 단위 Positive Top-K 추출
- 동일 Candidate Similarity Sum Aggregation
- Train Item Exclusion 및 Positive Score Filtering
- User-Based와 동일한 `recommend()` 인터페이스 유지로 evaluation pipeline 공용화
- 정성적 실험 (단일 취향 / 혼합 취향 / 단일 게임 확장성 6종)

## Model-Based Collaborative Filtering (Funk SVD)

- Global Train/Test Split 설계 및 최적화 (interaction count별 position 재사용)
- 41,154,794건 interaction Parquet 저장/로드 (`data/split/mf_train.parquet`, `mf_test.parquet`)
- Surprise `Dataset`/`Reader` 기반 데이터 변환 (`is_recommended` → 0/1 rating)
- Funk SVD(`SVD`) 모델 학습 파이프라인 구현 및 정상 작동 확인
- **`recommend()` Top-N 추천 함수 구현** (latent vector 내적 + bias(μ, bu, bi) 반영, seen-item `-inf` 처리, `argpartition`/`argsort` 기반 벡터화 Top-N, raw item id 복원)
- **MF 전용 Evaluation 구현** (`build_mf_user_review_groups`, `evaluate_mf_user`, `run_mf_evaluation` — 기존 지표 재사용 + MF Train/Test 구조에 맞는 데이터 공급)
- **400명 정량평가 실행 및 결과 확보** (Precision@10 0.0003, Hits 1/4,000)
- [진행 중] 낮은 성능 원인 진단 (`positive_only` 조건, Top-10 overlap, item bias, sparsity 등)

## Evaluation (공통)

- 리뷰 수 구간 기반 층화 표집 평가 시스템
- Precision@K / Recall@K / Hit Rate@K / NDCG@K 계산
- Macro/Micro Precision·Recall·F1
- Pearson Correlation 기반 가설 검증
- 구간별 평가 결과 Breakdown 리포트
- Object-Oriented Recommendation Model
- Modular Project Structure

---

# 🚀 Development Roadmap

## ✅ V1. Content-Based Recommendation

### Data Processing

- [x] Steam Metadata Loading
- [x] Parquet 기반 데이터 캐싱
- [x] Data Validation
- [x] User-based Train/Test Split
- [x] Data Preprocessing
- [x] Missing Value Handling
- [x] Combined Features Generation

### Recommendation

- [x] Content-Based Recommendation
- [x] TF-IDF Vectorization
- [x] Cosine Similarity
- [x] Multi-Game Recommendation
- [x] AppID 기반 게임 식별 구조 (Name → AppID → Index)
- [x] 동명이인 게임 선택 기능
- [x] Duplicate Recommendation Handling

### Evaluation

- [x] Recommendation Evaluation Dataset
- [x] 리뷰 수 구간 기반 층화 표집
- [x] Precision@K
- [x] Recall@K
- [x] NDCG@K
- [x] 정성적 실험 (단일 입력 / 장르 혼합 입력)
- [ ] 장르 혼합 쏠림 현상 최소 검증 (단독 입력 비교)
- [ ] 그룹당 400명 규모 통계적 재검증

### Software Engineering

- [x] Function Modularization
- [x] Object-Oriented Design
- [x] Project Structure Refactoring
- [x] Data Loading Caching Strategy
- [x] pandas 버전 호환성 이슈 대응 (groupby 안전 패턴 적용)

---

## ✅ V2. Collaborative Filtering

### User-Based CF

- [x] Sparse Interaction Matrix 구축
- [x] User-Based Collaborative Filtering 구현
- [x] 자기 자신 제외(Self-Exclusion) 로직 구현
- [x] 데이터 누수 버그 진단 및 수정 (`user_to_idx` 인자 전달 누락)
- [x] 기존 evaluation.py 파이프라인 재사용하여 Content-Based와 정량 비교
- [x] Sparsity에 따른 성능 한계 분석 및 Item-Based 전환 결정
- [x] NDCG 계산 순위 보존 버그 수정 (`set` → `list`)
- [x] Micro-Average Precision/Recall/F1 계산
- [x] 유저 게임 수 - Precision 상관관계 정량 검증 (Pearson Correlation)
- [x] User-Based/Item-Based 공통 Self-Similarity 처리 원칙 정립

### Item-Based CF

- [x] `models/itembase.py` 생성 (기존 `userbase.py` 기반)
- [x] User×Item → Item×User 구조 변환 (`interaction_matrix.T`)
- [x] Source Item(Train App ID) 기반 similarity 계산 (별도 Query Vector 불필요 확인)
- [x] Source Item별 Self-Similarity 0 처리
- [x] User-Based/Item-Based 구조적 차이 정리 (query 필요 여부, output 공간과의 거리 등)
- [x] 행별(Source Item별) Top-K 추출 로직 완성
- [x] Candidate Aggregation 구현 (동일 candidate로 모이는 similarity 합산)
- [x] Train Item Exclusion 적용 (`predicted_scores[col_idx] = -np.inf`)
- [x] Positive Score Filtering 및 Top-N 반환 (기존 `recommend()` 출력 형식 유지)
- [x] 기존 evaluation.py 재사용하여 User-Based CF와 정량 비교 (400명 규모)
- [x] User-Based CF 대비 Sparsity 강건성 비교 (`n_recommended` 부족 현상 완전 해소 확인)
- [x] 콘텐츠 기반에서 발견한 쏠림 현상이 협업 필터링에서도 재현되는지 비교 (방향은 반대로 재현됨)
- [x] Item-Based CF 정성평가 (단일/혼합/단일게임 확장성 6개 실험)

### Model-Based CF 학습 및 적용 순서

#### 1. Funk SVD — 핵심 Model-Based baseline

- [x] Memory-based CF와 Model-Based CF(Global Model)의 학습 구조 차이 파악
- [x] Global Train/Test Split 설계 및 대규모 split 성능 최적화 (interaction count별 position 재사용)
- [x] Train/Test Parquet 저장·로드 구조 구축 (`data_split.py`)
- [x] Surprise 기반 Funk SVD 기본형 구현
- [x] User / Item latent factor 구조 및 prediction 흐름 이해 (bias 포함 공식까지 코드 레벨로 확인)
- [x] 10만 건 샘플 및 전체 Train(37,113,471건) 학습 파이프라인 정상 작동 확인
- [x] Top-N 추천 함수 구현 (Candidate score 계산, Seen-item Exclusion, 벡터화 Ranking)
- [x] 기존 evaluation pipeline에 연결하여 Precision@K / Recall@K / Hit Rate@K / NDCG@K 확보 (400명)
- [ ] 성능 저조 원인 진단 (`positive_only=False` 재평가, Top-10 overlap, item bias, sparsity 등)
- [ ] 원인 진단 결과에 따른 개선 실험 또는 모델 방향 재결정
- [ ] `n_factors`, regularization 등 핵심 파라미터 소규모 실험 (원인 진단 이후)
- [ ] 필요 시 간단한 정성평가

#### 2. ALS — 짧은 비교 실험

- [ ] Funk SVD와 동일한 latent factor 계열이지만 Alternating Least Squares로 학습되는 구조 확인
- [ ] Sparse interaction에서의 학습 방식, 실행 시간, 추천 성능을 SVD와 비교
- [ ] 현재 데이터의 실제 비추천(0)과 미관측 interaction을 구분하는 처리 방식 검토
- [ ] 학습 중복도가 높거나 프로젝트 기간이 길어질 경우 baseline 비교까지만 진행 후 종료

#### 3. Clustering — 사용자 집단 구조 분석

- [ ] Sparse User-Item representation을 이용한 MiniBatch K-Means baseline 실험
- [ ] Cluster별 대표 게임/선호 패턴을 확인하여 사용자 segmentation 가능성 분석
- [ ] 추천 모델로서 의미가 있을 경우에만 동일 Top-K 지표로 추가 평가
- [ ] 기존 User-KNN과 중복되는 Cluster-KNN, SVD embedding 기반 K-Means 확장은 이번 baseline 단계에서는 생략

#### 4. BPR — 선택적 Ranking 확장

- [ ] Rating prediction이 아닌 pairwise ranking 학습 구조 이해
- [ ] Top-N 추천 목적에 맞춰 BPR baseline 구현 및 기존 SVD/ALS와 비교
- [ ] 프로젝트 진행 속도, 모델 간 중복도, 앞선 실험 결과를 보고 실제 구현 여부 최종 결정
- [ ] BPR까지 진행한 경우 추가 MF 변형(SVD++, NMF, PMF 등)은 확장하지 않고 모델 추가 종료

### Model Comparison & Selection (Model-Based 이후)

- [ ] Content-Based / User-Based / Item-Based / SVD / 진행한 Model-Based 모델을 동일 조건으로 비교
- [ ] Accuracy 지표뿐 아니라 Sparsity, Coverage, Personalization, Explainability, 실행 시간 관점 비교
- [ ] 성능과 역할이 가장 뚜렷한 1~2개 모델만 소규모 튜닝
- [ ] 모든 모델을 반복적으로 튜닝하지 않고 Hybrid 단계로 전환

### MAP & Popularity Baseline (Model-Based 이후, Hybrid 이전 진행 예정)

- [ ] MAP@K 구현
- [ ] Popularity Baseline 비교
- [ ] Hit item의 Popularity Bias 분석

---

## 🚧 V3. Hybrid Recommendation

- Hybrid Recommendation
- 콘텐츠 기반 실험에서 발견한 쏠림 현상을 보완하는 장르별 가중치 설계
- Content-Based와 Item-Based CF가 정반대 방향으로 쏠린 결과를 반영한 Hybrid aggregation 설계

---

## 🚧 V4. Deployment

- FastAPI
- Streamlit

---

# 📊 Evaluation

추천 시스템은 **Train Dataset**으로 학습하고 **Test Dataset**으로 평가합니다. Memory-based CF(User/Item-Based)는 유저별 게임 수가 제각각이라 별도의 Train/Test 데이터셋 파일을 미리 만들어두지 않고, `sklearn.model_selection.train_test_split`을 유저 단위로 런타임에 적용하는 방식(Leave-N-Out, 70/30)을 사용합니다. 반면 Model-Based CF(Funk SVD)는 전체 사용자 interaction을 한 번에 학습하는 Global Model이므로, 평가 대상 유저(666,781명)는 70:30으로, 그 외 유저는 100% Train으로 분리한 별도의 Global Train/Test를 Parquet으로 저장해 재사용합니다.

### Recommendation Quality

- Precision@K
- Recall@K
- Hit Rate@K
- NDCG@K (순위 보존 검증 완료)
- Micro Precision/Recall/F1 (추천 개수 편차 보완용)
- MAP (예정)

### Sampling Strategy

- 유저별 리뷰 수(플레이 게임 수) 기준 범위 필터링 (봇/이상치 배제)
- 리뷰 수 구간(`pd.cut`)별 층화 표집으로 표본 편향 방지
- 통계적 신뢰 기준(95% 신뢰수준, ±5% 오차 → 구간당 약 385명) 고려한 표본 크기 설계
- User-Based CF는 구간당 100명 규모로 1차 검증 후, Sparsity 한계가 뚜렷이 확인되어 대규모(400명) 재검증은 생략
- Item-Based CF는 구간당 100명, 총 400명 규모로 본 평가 완료
- Funk SVD는 Global Train/Test 구조로 별도 설계 (평가 대상 유저 666,781명 70:30 + 그 외 유저 100% Train)하여 400명 규모 본 평가 완료. 단, 현재 `positive_only=True` 조건이 기존 평가와 달라 결과 비교는 재평가 이후 진행

### Qualitative Evaluation

- 실제 게임을 직접 입력하여 추천 결과를 눈으로 확인
- 단일 입력 / 장르 혼합 입력 비교를 통한 모델 편향 발견
- Content-Based ↔ Item-Based 간 동일 입력 비교를 통한 쏠림 방향 차이 분석
- Funk SVD는 랜덤 사용자 1명 대상 정성 확인 완료 (추천 결과가 실제 Test 취향과 동떨어짐을 1차 확인, 정식 정성평가는 원인 진단 이후 진행)

### Model Comparison

- Content-Based Recommendation
- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Model-Based CF / Matrix Factorization (Funk SVD 학습·추천·평가 파이프라인 구현 완료, baseline 성능 저조 원인 진단 중)
- Hybrid Recommendation (예정)
- Popularity Baseline (예정)

### Performance

- Execution Time
- Memory Usage
- Cache Hit / Miss (Parquet 캐싱 적용 후 로딩 속도 비교)
- 대규모 Split 연산 최적화 (Python loop 66만 회 → interaction count별 최대 69회, 약 5초 소요)

---

# 📈 Current Progress

| Module | Status |
| :--- | :---: |
| Data Loading | ✅ |
| Data Caching (Parquet) | ✅ |
| Data Validation | ✅ |
| Train/Test Split | ✅ |
| Data Preprocessing | ✅ |
| TF-IDF Vectorization | ✅ |
| Content-Based Recommendation | ✅ |
| Multi-Game Recommendation | ✅ |
| AppID 기반 식별 구조 | ✅ |
| Evaluation Dataset | ✅ |
| Stratified Sampling | ✅ |
| Precision@K | ✅ |
| Recall@K | ✅ |
| Hit Rate@K | ✅ |
| NDCG@K | ✅ |
| Qualitative Experiment (Content-Based) | ✅ |
| User-Based Collaborative Filtering | ✅ |
| Data Leakage 진단/수정 | ✅ |
| NDCG 순위 보존 버그 수정 | ✅ |
| Macro/Micro 평가 (Precision/Recall/F1) | ✅ |
| Sparsity 가설 정량 검증 (Pearson Correlation) | ✅ |
| Self-Similarity 처리 원칙 정립 | ✅ |
| Item-Based Collaborative Filtering | ✅ |
| Item-Based CF 정량평가 (400명) | ✅ |
| Item-Based CF 정성평가 | ✅ |
| Model-Based CF Global Train/Test Split | ✅ |
| Funk SVD 학습 파이프라인 구현 | ✅ |
| Funk SVD Top-N 추천 함수 구현 | ✅ |
| Funk SVD Evaluation 연결 및 400명 정량평가 | ✅ |
| Funk SVD 성능 저조 원인 진단 | 🚧 |
| ALS (짧은 비교 실험) | ⏳ |
| Clustering (사용자 군집화 분석) | ⏳ |
| BPR (상황에 따라 선택적 진행) | ⏳ |
| Model Comparison & Selection | ⏳ |
| MAP | 🚧 |
| Popularity Baseline | 🚧 |
| Hybrid Recommendation | 🚧 |
| Deployment | 🚧 |

---

# 📌 Project Status

**Current Version:** `V2.8 - Funk SVD Top-N 추천/Evaluation 연결 완료, baseline 성능 저조 원인 진단 중`

### Completed

- Steam Metadata Preprocessing
- Parquet 기반 데이터 로딩 캐싱
- User-based Train/Test Split
- Combined Features Generation
- TF-IDF Vectorization
- Multi-Game Recommendation
- AppID 기반 게임 식별 및 동명이인 처리 구조
- Cosine Similarity Recommendation
- 리뷰 수 구간별 층화 표집 평가 시스템
- Precision@K / Recall@K / Hit Rate@K / NDCG@K
- 정성적 실험을 통한 장르 혼합 쏠림 현상 발견 (Content-Based)
- Object-Oriented Recommendation Model
- Project Modularization
- **User-Based Collaborative Filtering 구현 (Sparse Matrix)**
- **평가 파이프라인 데이터 누수 버그 진단 및 수정**
- **Sparsity에 따른 User-Based CF 성능 한계 정량 확인**
- **NDCG 계산 순위 보존 버그 수정**
- **Macro/Micro Precision·Recall·F1 비교 평가 도입**
- **Pearson Correlation 기반 Sparsity 가설 정량 검증**
- **User-Based/Item-Based 공통 Self-Similarity 처리 원칙 정립**
- **Item-Based CF 구현 완료** (행별 Top-K, Candidate Aggregation, Train Item Exclusion, Top-N 반환)
- **Item-Based CF 정량평가 완료** (400명, User-Based 대비 전 지표 상승, Sparsity 강건성 확인)
- **Item-Based CF 정성평가 완료** (6개 실험, Content-Based와의 정반대 쏠림 방향 발견, Hybrid 설계 필요성 확인)
- **Model-Based CF Global Train/Test Split 설계 및 저장 완료** (41,154,794건, 66만 회 → 최대 69회 최적화)
- **Surprise 기반 Funk SVD 학습 파이프라인 구현 및 정상 작동 확인** (10만 건 샘플 + 전체 Train 규모)
- **Funk SVD Top-N 추천 함수 구현 완료** (bias 포함 score 계산, seen-item 제거, 벡터화 Top-N)
- **Funk SVD Evaluation 연결 및 400명 정량평가 완료** (Precision@10 0.0003, Hits 1/4,000 확인)
- **`positive_only=True`로 인한 기존 CF 평가와의 조건 불일치 발견**

### In Progress

- **Funk SVD baseline 성능 저조 원인 진단** (`positive_only=False` 재평가 최우선, Top-10 overlap, item bias, sparsity 등 순차 확인)
- **진단 결과에 따른 모델 방향 결정** (Funk SVD 유지·개선 / BPR·implicit MF 검토 / 기존 CF 유지)

### Next Milestone

➡️ `positive_only=False` 재평가로 기존 CF와 비교 가능한 조건 확보
➡️ Top-10 overlap, item bias, sparsity 등 추가 원인 진단
➡️ 진단 결과에 따라 Funk SVD 개선 실험 또는 BPR/implicit MF 검토, 혹은 기존 CF 유지 결정
➡️ ALS 짧은 비교 실험 (SVD와 학습 방식·성능·실행 시간 비교)
➡️ Clustering 사용자 군집화/세그먼트 분석
➡️ BPR은 Top-N Ranking 확장 후보로 유지하되, 프로젝트 진행 속도와 앞선 실험 결과를 보고 실제 구현 여부 결정
➡️ Model-Based 단계 종료 후 전체 모델 비교 및 최상위 1~2개 모델만 소규모 튜닝
➡️ MAP@K 구현
➡️ Popularity Baseline 및 Popularity Bias / Coverage 분석
➡️ Hybrid Recommendation (모델별 failure mode 및 쏠림 방향 차이 기반 설계)
➡️ Web Service Deployment