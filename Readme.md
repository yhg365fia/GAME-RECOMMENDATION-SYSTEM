# 🎮 Steam Game Recommendation System

Steam 게임 데이터를 활용하여 다양한 추천 시스템 알고리즘을 구현하고 성능을 비교하는 프로젝트입니다.

현재는 **Content-Based Recommendation System**을 구현하였으며,

- Steam 메타데이터 전처리
- User-based Train/Test Split
- TF-IDF Vectorization
- Cosine Similarity 기반 추천
- Multi-Game Recommendation
- 리뷰 수 구간별 층화 표집 기반 평가 시스템 구축
- Precision@K, Recall@K, NDCG@K 지표 구현 및 결과 분석
- **정성적 실험을 통한 콘텐츠 기반 추천의 구조적 한계 발견** (장르 혼합 시 쏠림 현상 등)
- AppID 기반 안정적인 게임 식별 구조 설계
- 데이터 로딩 캐싱(Parquet)을 통한 성능 최적화
- 프로젝트 모듈화 및 객체지향 설계

를 완료하였습니다.

향후에는 Collaborative Filtering, Hybrid Recommendation, 추천 성능 평가 및 웹 서비스 배포까지 확장하는 것을 목표로 합니다.

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
- Precision@K / Recall@K / NDCG@K
- 정성적 실험 (단일 입력 / 장르 혼합 입력 분석)
- 데이터 로딩 캐싱 (Parquet)
- 프로젝트 구조 모듈화

---

## 🚀 Future

- MAP@K
- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Matrix Factorization (SVD)
- Hybrid Recommendation
- Popularity Baseline 비교
- FastAPI & Streamlit Deployment

---

# 🛠 Tech Stack

## Language

- Python

## Data Processing

- Pandas
- NumPy
- PyArrow (Parquet 캐싱)

## Machine Learning

- Scikit-learn

### Algorithms

- TF-IDF Vectorization
- Cosine Similarity

## Evaluation

- Precision@K / Recall@K / NDCG@K
- Stratified Sampling (리뷰 수 구간 기반)
- Qualitative Experiment (단일/혼합 입력 결과 분석)

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
│   ├── split/
│   │   ├── train.csv
│   │   └── test.csv
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
│   └── Day08.md
│
├── models/
│   └── content_base.py
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
User-based Train/Test Split
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

> Game Name은 중복될 수 있지만 AppID는 고유하므로, 내부 로직은 전부 **AppID 기준**으로 동작하도록 설계하였습니다.

---

# 🔍 Qualitative Experiment Findings

정량 지표만으로는 "왜 이런 결과가 나오는가"를 확인하기 어려워, 실제 게임을 직접 입력하고 추천 결과를 눈으로 확인하는 정성적 실험을 병행하였습니다.

### 발견 1 — 텍스트에 없는 특성은 포착 불가
`Party Animals` 입력 시 장르/인원수는 유사하게 나왔지만, 기대했던 "동물 캐릭터" 테마는 전혀 반영되지 않음을 확인. TF-IDF는 Genres/Tags 등 텍스트 메타데이터에 명시된 정보만 학습하므로, 비주얼/테마적 특성은 원천적으로 포착할 수 없다는 한계를 확인하였습니다.

### 발견 2 — 장르 혼합 시 쏠림 현상
카드/덱빌딩(`Slay the Spire`, `Balatro`) + 슈팅(`Counter-Strike 2`, `Left 4 Dead 2`)을 함께 입력했을 때, 추천 결과가 카드/덱빌딩 계열로 완전히 쏠리고 슈팅 계열은 단 하나도 포함되지 않는 현상을 확인하였습니다. 두 가지 가설을 세웠습니다:

- **가설 1 (IDF 희귀도)**: 카드/덱빌딩류 태그(`deck-building`, `roguelike` 등)는 카탈로그 내 등장 빈도가 낮아 TF-IDF 가중치가 크게 작동하는 반면, FPS류 태그(`action`, `shooter`, `multiplayer`)는 흔해서 변별력이 낮았을 가능성
- **가설 2 (벡터 응집력)**: 여러 게임 벡터를 평균 낼 때, 방향이 비슷하고 크기가 큰 쪽(카드게임군)이 평균을 지배했을 가능성

> 이 발견을 우연이 아닌 "콘텐츠 기반 + 단순 평균 방식"의 구조적 특성일 가능성으로 보고, 최소 검증(단독 입력 시 같은 장르끼리 서로 상위에 오르는지 확인 및 추가 장르 조합 실험)을 다음 단계로 남겨두었습니다. 이 발견은 향후 하이브리드 모델의 장르별 가중치 설계, 그리고 협업 필터링에서 동일한 쏠림이 재현되는지 비교하는 기준점으로 활용할 예정입니다.

---

# ✅ Implemented Features

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
- 리뷰 수 구간 기반 층화 표집 평가 시스템
- Precision@K / Recall@K / NDCG@K 계산
- 구간별 평가 결과 Breakdown 리포트
- 정성적 실험 (단일/장르 혼합 입력 분석)
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
- [ ] MAP@K
- [ ] 장르 혼합 쏠림 현상 최소 검증 (단독 입력 비교)
- [ ] Popularity Baseline 비교

### Software Engineering

- [x] Function Modularization
- [x] Object-Oriented Design
- [x] Project Structure Refactoring
- [x] Data Loading Caching Strategy
- [x] pandas 버전 호환성 이슈 대응 (groupby 안전 패턴 적용)

---

## 🚧 V2. Collaborative Filtering

- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Matrix Factorization (SVD)
- 기존 evaluation.py 파이프라인 재사용하여 Content-Based와 정량 비교
- 콘텐츠 기반에서 발견한 장르 혼합 쏠림 현상이 협업 필터링에서도 재현되는지 비교

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

추천 시스템은 **Train Dataset**으로 학습하고 **Test Dataset**으로 평가합니다.

### Recommendation Quality

- Precision@K
- Recall@K
- NDCG@K
- MAP (예정)

### Sampling Strategy

- 유저별 리뷰 수(플레이 게임 수) 기준 범위 필터링 (봇/이상치 배제)
- 리뷰 수 구간(`pd.cut`)별 층화 표집으로 표본 편향 방지
- 통계적 신뢰 기준(95% 신뢰수준, ±5% 오차 → 구간당 약 385명) 고려한 표본 크기 설계

### Qualitative Evaluation

- 실제 게임을 직접 입력하여 추천 결과를 눈으로 확인
- 단일 입력 / 장르 혼합 입력 비교를 통한 모델 편향 발견

### Model Comparison

- Content-Based Recommendation
- Collaborative Filtering
- Hybrid Recommendation
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
| NDCG@K | ✅ |
| Qualitative Experiment | ✅ |
| MAP | 🚧 |
| Popularity Baseline | 🚧 |
| Collaborative Filtering | 🚧 |
| Hybrid Recommendation | 🚧 |
| Deployment | 🚧 |

---

# 📌 Project Status

**Current Version:** `V1.3 - Content-Based Recommendation Evaluation`

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
- Precision@K
- Recall@K
- NDCG@K
- 정성적 실험을 통한 장르 혼합 쏠림 현상 발견
- Object-Oriented Recommendation Model
- Project Modularization

### Next Milestone

➡️ MAP@K 구현

➡️ 장르 혼합 쏠림 현상 최소 검증

➡️ User-Based Collaborative Filtering

➡️ Item-Based Collaborative Filtering

➡️ Matrix Factorization (SVD)

➡️ Hybrid Recommendation

➡️ Web Service Deployment