# 🎮 Steam Game Recommendation System

Steam 게임 데이터를 활용하여 다양한 추천 시스템 알고리즘을 구현하고 성능을 비교하는 프로젝트입니다.

현재는 **Content-Based Recommendation System**과 **User-Based Collaborative Filtering**을 구현하였으며,

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

를 완료하였습니다.

향후에는 Item-Based Collaborative Filtering, Matrix Factorization, Hybrid Recommendation, 추천 성능 평가 및 웹 서비스 배포까지 확장하는 것을 목표로 합니다.

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

---

## 🚀 Future

- MAP@K
- Item-Based Collaborative Filtering
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
│   └── Day12.md
│
├── models/
│   ├── content_base.py
│   └── userbase.py
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

> Game Name은 중복될 수 있지만 AppID는 고유하므로, 내부 로직은 전부 **AppID 기준**으로 동작하도록 설계하였습니다. User-Based CF에서도 동일한 원칙을 적용하여, `user_to_idx` / `game_to_idx` / `idx_to_game`을 통해 ID ↔ 행렬 인덱스 변환을 일관되게 관리합니다.

---

# 🔍 Qualitative Experiment Findings (Content-Based)

정량 지표만으로는 "왜 이런 결과가 나오는가"를 확인하기 어려워, 실제 게임을 직접 입력하고 추천 결과를 눈으로 확인하는 정성적 실험을 병행하였습니다.

### 발견 1 — 텍스트에 없는 특성은 포착 불가
`Party Animals` 입력 시 장르/인원수는 유사하게 나왔지만, 기대했던 "동물 캐릭터" 테마는 전혀 반영되지 않음을 확인. TF-IDF는 Genres/Tags 등 텍스트 메타데이터에 명시된 정보만 학습하므로, 비주얼/테마적 특성은 원천적으로 포착할 수 없다는 한계를 확인하였습니다.

### 발견 2 — 장르 혼합 시 쏠림 현상
카드/덱빌딩(`Slay the Spire`, `Balatro`) + 슈팅(`Counter-Strike 2`, `Left 4 Dead 2`)을 함께 입력했을 때, 추천 결과가 카드/덱빌딩 계열로 완전히 쏠리고 슈팅 계열은 단 하나도 포함되지 않는 현상을 확인하였습니다. 두 가지 가설을 세웠습니다:

- **가설 1 (IDF 희귀도)**: 카드/덱빌딩류 태그(`deck-building`, `roguelike` 등)는 카탈로그 내 등장 빈도가 낮아 TF-IDF 가중치가 크게 작동하는 반면, FPS류 태그(`action`, `shooter`, `multiplayer`)는 흔해서 변별력이 낮았을 가능성
- **가설 2 (벡터 응집력)**: 여러 게임 벡터를 평균 낼 때, 방향이 비슷하고 크기가 큰 쪽(카드게임군)이 평균을 지배했을 가능성

> 이 발견을 우연이 아닌 "콘텐츠 기반 + 단순 평균 방식"의 구조적 특성일 가능성으로 보고, 최소 검증(단독 입력 시 같은 장르끼리 서로 상위에 오르는지 확인 및 추가 장르 조합 실험)을 다음 단계로 남겨두었습니다. 이 발견은 향후 하이브리드 모델의 장르별 가중치 설계, 그리고 협업 필터링에서 동일한 쏠림이 재현되는지 비교하는 기준점으로 활용할 예정입니다.

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

> 참고: Offline 평가의 Test set은 "사용자가 좋아할 수 있는 모든 정답"이 아니라 "숨겨둔 일부 관측된 interaction을 얼마나 복원하는가"에 가까우므로, Precision 절대값만으로 모델의 좋고 나쁨을 단정하지 않고 있습니다. Popularity Bias, Near-Duplicate/Series Bias, Coverage/Personalization 등은 Item-Based·Model-Based까지 구현한 뒤 여러 모델의 failure mode를 비교하며 분석할 예정입니다.

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

## Evaluation (공통)

- 리뷰 수 구간 기반 층화 표집 평가 시스템
- Precision@K / Recall@K / Hit Rate@K / NDCG@K 계산
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

## 🚧 V2. Collaborative Filtering

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

### Item-Based CF (예정)

- [ ] Item-Based Collaborative Filtering 구현
- [ ] User-Based CF 대비 Sparsity 강건성 비교
- [ ] 콘텐츠 기반에서 발견한 쏠림 현상이 협업 필터링에서도 재현되는지 비교

### Matrix Factorization (예정)

- [ ] Matrix Factorization (SVD)

### MAP & Popularity Baseline (Item-Based 이후, Hybrid 이전 진행 예정)

- [ ] MAP@K 구현
- [ ] Popularity Baseline 비교
- [ ] Hit item의 Popularity Bias 분석

---

## 🚧 V3. Hybrid Recommendation

- Hybrid Recommendation
- 콘텐츠 기반 실험에서 발견한 쏠림 현상을 보완하는 장르별 가중치 설계

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
- User-Based CF는 구간당 100명 규모로 1차 검증 후, Sparsity 한계가 뚜렷이 확인되어 대규모(400명) 재검증은 생략 — 리소스를 Content-Based 재검증 및 최종 채택 모델 검증에 집중

### Qualitative Evaluation

- 실제 게임을 직접 입력하여 추천 결과를 눈으로 확인
- 단일 입력 / 장르 혼합 입력 비교를 통한 모델 편향 발견

### Model Comparison

- Content-Based Recommendation
- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering (예정)
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
| Qualitative Experiment | ✅ |
| User-Based Collaborative Filtering | ✅ |
| Data Leakage 진단/수정 | ✅ |
| NDCG 순위 보존 버그 수정 | ✅ |
| Macro/Micro 평가 (Precision/Recall/F1) | ✅ |
| Sparsity 가설 정량 검증 (Pearson Correlation) | ✅ |
| Self-Similarity 처리 원칙 정립 | ✅ |
| Item-Based Collaborative Filtering | 🚧 |
| Matrix Factorization | 🚧 |
| MAP | 🚧 |
| Popularity Baseline | 🚧 |
| Hybrid Recommendation | 🚧 |
| Deployment | 🚧 |

---

# 📌 Project Status

**Current Version:** `V2.2 - User-Based CF 평가 보완 및 Item-Based 전환 결정`

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
- 정성적 실험을 통한 장르 혼합 쏠림 현상 발견
- Object-Oriented Recommendation Model
- Project Modularization
- **User-Based Collaborative Filtering 구현 (Sparse Matrix)**
- **평가 파이프라인 데이터 누수 버그 진단 및 수정**
- **Sparsity에 따른 User-Based CF 성능 한계 정량 확인**
- **NDCG 계산 순위 보존 버그 수정**
- **Macro/Micro Precision·Recall·F1 비교 평가 도입**
- **Pearson Correlation 기반 Sparsity 가설 정량 검증**
- **User-Based/Item-Based 공통 Self-Similarity 처리 원칙 정립**

### Next Milestone

➡️ Item-Based Collaborative Filtering 구현

➡️ User-Based 대비 Sparsity 강건성 비교

➡️ Matrix Factorization (SVD)

➡️ MAP@K 구현

➡️ Popularity Baseline 및 Popularity Bias / Coverage 분석

➡️ Hybrid Recommendation (모델별 failure mode 기반 설계)

➡️ Web Service Deployment