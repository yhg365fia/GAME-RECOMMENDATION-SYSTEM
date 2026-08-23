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

를 완료하였으며, Model-Based CF(Matrix Factorization)는 다음 baseline으로 설계를 준비 중입니다.

향후에는 Matrix Factorization, Hybrid Recommendation, 추천 성능 평가 및 웹 서비스 배포까지 확장하는 것을 목표로 합니다.

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

---

## 🚧 In Progress

- Model-Based CF (Matrix Factorization) baseline 모델 선정 및 설계

---

## 🚀 Future

- MAP@K
- Matrix Factorization (SVD)
- Hybrid Recommendation (실패 유형 기반 설계)
- Popularity Baseline 비교
- Popularity Bias / Near-Duplicate / Coverage 분석
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

### Algorithms

- TF-IDF Vectorization
- Cosine Similarity
- User-Based Collaborative Filtering (Neighborhood-based, Top-K)
- Item-Based Collaborative Filtering (Neighborhood-based, 행별 Top-K, Similarity Sum Aggregation)

## Evaluation

- Precision@K / Recall@K / Hit Rate@K / NDCG@K (Macro)
- Micro Precision / Micro Recall / Micro F1
- Stratified Sampling (리뷰 수 구간 기반)
- Qualitative Experiment (단일/혼합 입력 결과 분석)
- Leave-N-Out 유저별 Train/Test Split
- Pearson Correlation (SciPy) 기반 가설 검증

## Future Libraries

- Surprise
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
│   └── cache/              # 전처리/로딩 결과 캐싱 (Parquet)
│       ├── games.parquet
│       ├── train.parquet
│       ├── test.parquet
│       └── recommendations.parquet
│
├── docs/
│   ├── Day01.md
│   ├── Day02.md
│   ├── Day03.md
│   ├── Day04.md
│   ├── Day05.md
│   ├── Day06.md
│   ├── Day07.md
│   ├── Day08.md
│   ├── Day09.md
│   ├── Day10.md
│   ├── Day11.md
│   ├── Day12.md
│   ├── Day13.md
│   ├── Day14.md
│   └── Day15.md
│
├── models/
│   ├── content_base.py
│   ├── userbase.py
│   └── itembase.py          # 구현 완료
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_content_based_recommendation.ipynb
│   └── 04_evaluation_dataset.ipynb
│
├── preprocessing.py
├── evaluation.py
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

> Game Name은 중복될 수 있지만 AppID는 고유하므로, 내부 로직은 전부 **AppID 기준**으로 동작하도록 설계하였습니다. User-Based/Item-Based CF에서도 동일한 원칙을 적용하여, `user_to_idx` / `game_to_idx` / `idx_to_game`을 통해 ID ↔ 행렬 인덱스 변환을 일관되게 관리합니다.

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

> 참고: Offline 평가의 Test set은 "사용자가 좋아할 수 있는 모든 정답"이 아니라 "숨겨둔 일부 관측된 interaction을 얼마나 복원하는가"에 가까우므로, Precision 절대값만으로 모델의 좋고 나쁨을 단정하지 않고 있습니다. Popularity Bias, Near-Duplicate/Series Bias, Coverage/Personalization 등은 Model-Based까지 구현한 뒤 여러 모델의 failure mode를 비교하며 분석할 예정입니다.

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

# 🔍 Item-Based CF Qualitative Experiment Findings

Item-Based CF의 정량적 우위가 실제 추천 결과에서도 납득 가능한지, 그리고 Content-Based에서 발견한 장르 혼합 쏠림 현상이 재현되는지 확인하기 위해 6개 실험(단일 취향 2건, 혼합 취향 1건, 단일 게임 확장성 3건)을 진행했습니다.

### 발견 1 — Content-Based와 정반대 방향의 쏠림 (핵심 발견)

Content-Based 실험(카드/덱빌딩 + 슈팅 혼합 입력)에서 발견했던 장르 혼합 쏠림 현상을 Item-Based에 동일하게 입력하여 재현 여부를 확인했습니다.

| 모델 | 동일한 4개 입력(카드 2 + 슈팅 2) 결과 |
|---|---|
| Content-Based | 카드/덱빌딩 쪽으로 강하게 쏠림 |
| Item-Based CF | 슈팅(CS2·L4D2) 쪽으로 강하게 쏠림 |

카드/덱빌딩 게임 2종을 추가로 넣었음에도 슈팅 단독 입력 실험의 Top-5 추천이 그대로 유지되는 현상을 확인했습니다. 두 모델 모두 서로 다른 두 취향을 균형 있게 보존하지 못했지만, **쏠리는 방향이 정반대**라는 점이 핵심입니다. Content-Based의 쏠림 원인 후보가 TF-IDF feature/IDF 구조였다면, Item-Based의 쏠림 원인 후보는 interaction density·user overlap·similarity-sum aggregation 구조입니다(정량 검증은 Model-Based 단계 이후로 보류).

### 발견 2 — Content-Based와 Item-Based는 서로 다른 signal을 포착

- **Content-Based**가 상대적으로 잘 보는 것: "이 게임은 무엇인가?" — 장르·태그·테마 등 콘텐츠 자체의 semantic 유사성 (예: RDR2 → GTA V)
- **Item-Based**가 상대적으로 잘 보는 것: "이 게임을 플레이한 사람은 다른 무엇을 함께 소비하는가?" — 공통 interaction 기반의 인접 취향으로 확장 (예: PUBG → 경쟁 FPS·생존 PvP·배틀로얄·온라인 액션으로 확장, Stardew Valley → 샌드박스·제작·인디·스토리로 확장)

### 발견 3 — 세부 의미 특징 보존의 한계

Left 4 Dead 2 입력 시 좀비/공포/협동이라는 세부 특징은 추천 결과에서 강하게 유지되지 않고 전반적인 슈팅 취향만 두드러졌습니다. RDR2 입력에서도 콘텐츠적으로 가장 직접적인 후보(GTA V)보다 넓은 유저 취향의 게임(Witcher 3, Cyberpunk 2077 등)이 우선되었습니다.

> 이 발견들을 종합해 Content-Based(콘텐츠 세부 특성) + Item-Based(실제 사용자 행동 기반 인접 취향)를 결합하는 **Hybrid 설계의 필요성**을 확인했습니다. 다만 Hybrid가 다중 취향 쏠림을 자동으로 해결해주는 것은 아니며(두 모델이 서로 다른 방향으로 쏠렸으므로 단순 score 합산은 또 다른 쏠림을 만들 수 있음), 구체적 설계는 Model-Based baseline 완료 이후로 보류했습니다.

---

# 🐛 알려진 이슈 (To-Do)

- **Name = NaN metadata mismatch**: Item-Based CF 정성평가 과정에서 일부 추천 결과의 게임 이름이 NaN으로 조회되는 현상을 발견. 원인 분석 및 수정은 진행하지 않고 기록만 남김 (Model-Based 단계 이후 처리 예정)
- **Party Animals 입력 시 빈 DataFrame 반환**: 정성평가 대상에서 제외하고 Stardew Valley로 대체. 원인 분석 보류

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

### Matrix Factorization (다음 단계)

- [ ] Model-Based CF baseline 모델 선정 (SVD 등 Matrix Factorization 계열 포함)
- [ ] Matrix Factorization (SVD) 기본형 구현
- [ ] 동일 evaluation pipeline으로 정량평가
- [ ] 필요 시 간단한 정성평가

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

추천 시스템은 **Train Dataset**으로 학습하고 **Test Dataset**으로 평가합니다. 유저별 게임 수가 제각각이라 별도의 Train/Test 데이터셋 파일을 미리 만들어두지 않고, `sklearn.model_selection.train_test_split`을 유저 단위로 런타임에 적용하는 방식(Leave-N-Out, 70/30)을 사용합니다.

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
- User-Based CF는 구간당 100명 규모로 1차 검증 후, Sparsity 한계가 뚜렷이 확인되어 대규모(400명) 재검증은 생략 — 리소스를 Item-Based CF 본 평가(구간당 100명, 총 400명)에 집중
- Item-Based CF는 구간당 100명, 총 400명 규모로 본 평가 완료

### Qualitative Evaluation

- 실제 게임을 직접 입력하여 추천 결과를 눈으로 확인
- 단일 입력 / 장르 혼합 입력 비교를 통한 모델 편향 발견
- Content-Based ↔ Item-Based 간 동일 입력 비교를 통한 쏠림 방향 차이 분석

### Model Comparison

- Content-Based Recommendation
- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Model-Based CF / Matrix Factorization (설계 중)
- Hybrid Recommendation (예정)
- Popularity Baseline (예정)

### Performance

- Execution Time
- Memory Usage
- Cache Hit / Miss (Parquet 캐싱 적용 후 로딩 속도 비교)

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
| Matrix Factorization | 🚧 |
| MAP | 🚧 |
| Popularity Baseline | 🚧 |
| Hybrid Recommendation | 🚧 |
| Deployment | 🚧 |

---

# 📌 Project Status

**Current Version:** `V2.5 - Item-Based CF 구현·정량평가·정성평가 완료`

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

### In Progress

- Model-Based CF (Matrix Factorization) baseline 모델 선정 및 설계

### Next Milestone

➡️ Model-Based CF (Matrix Factorization / SVD) 구현

➡️ 동일 evaluation pipeline으로 Content-Based / User-Based / Item-Based / Model-Based 비교

➡️ MAP@K 구현

➡️ Popularity Baseline 및 Popularity Bias / Coverage 분석

➡️ Hybrid Recommendation (모델별 failure mode 및 쏠림 방향 차이 기반 설계)

➡️ Web Service Deployment