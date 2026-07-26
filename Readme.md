# 🎮 Steam Game Recommendation System

Steam 게임 데이터를 활용하여 다양한 추천 시스템 알고리즘을 구현하고 성능을 비교하는 프로젝트입니다.

현재는 **Content-Based Recommendation System**을 구현하였으며,

- Steam 메타데이터 전처리
- User-based Train/Test Split
- TF-IDF Vectorization
- Cosine Similarity 기반 추천
- Multi-Game Recommendation
- Recommendation Evaluation Dataset 구축
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
- Recommendation Evaluation Dataset
- 데이터 로딩 캐싱 (Parquet)
- 프로젝트 구조 모듈화

---

## 🚀 Future

- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Matrix Factorization (SVD)
- Hybrid Recommendation
- Recommendation Evaluation
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
│       └── test.parquet
│
├── docs/
│   ├── Day01.md
│   ├── Day02.md
│   ├── Day03.md
│   ├── Day04.md
│   ├── Day05.md
│   └── Day06.md
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
Recommendation Evaluation
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
- Recommendation Evaluation Dataset
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
- [ ] Precision@K
- [ ] Recall@K
- [ ] MAP
- [ ] NDCG

### Software Engineering

- [x] Function Modularization
- [x] Object-Oriented Design
- [x] Project Structure Refactoring
- [x] Data Loading Caching Strategy

---

## 🚧 V2. Collaborative Filtering

- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Matrix Factorization (SVD)

---

## 🚧 V3. Hybrid Recommendation

- Hybrid Recommendation

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
- MAP
- NDCG

### Model Comparison

- Content-Based Recommendation
- Collaborative Filtering
- Hybrid Recommendation

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
| Precision@K | 🚧 |
| Recall@K | 🚧 |
| MAP | 🚧 |
| NDCG | 🚧 |
| Collaborative Filtering | 🚧 |
| Hybrid Recommendation | 🚧 |
| Deployment | 🚧 |

---

# 📌 Project Status

**Current Version:** `V1.2 - Content-Based Recommendation (AppID 기반 리팩토링 & 캐싱 적용)`

### Completed

- Steam Metadata Preprocessing
- Parquet 기반 데이터 로딩 캐싱
- User-based Train/Test Split
- Recommendation Evaluation Dataset
- Combined Features Generation
- TF-IDF Vectorization
- Multi-Game Recommendation
- AppID 기반 게임 식별 및 동명이인 처리 구조
- Cosine Similarity Recommendation
- Object-Oriented Recommendation Model
- Project Modularization

### Next Milestone

➡️ Precision@K / Recall@K / MAP / NDCG

➡️ Collaborative Filtering

➡️ Hybrid Recommendation

➡️ Web Service Deployment