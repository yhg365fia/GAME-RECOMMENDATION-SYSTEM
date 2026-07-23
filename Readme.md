# 🎮 Steam Game Recommendation System

Steam 게임 메타데이터를 활용하여 다양한 추천 시스템 알고리즘을 구현하고 성능을 비교하는 프로젝트입니다.

현재는 **Content-Based Recommendation System**을 구현하였으며,

- Steam 메타데이터 전처리
- TF-IDF Vectorization
- Cosine Similarity 기반 추천
- 사용자 입력 기반 Top-N Recommendation
- 프로젝트 모듈화 및 클래스 설계

를 완료하였습니다.

향후에는 Collaborative Filtering, Hybrid Recommendation, 추천 성능 평가 및 웹 서비스 배포까지 확장하는 것을 목표로 합니다.

---

# 📌 Project Goals

## ✅ Current

- Steam 메타데이터 전처리
- Content-Based Recommendation 구현
- TF-IDF Vectorization
- Cosine Similarity 기반 추천
- User Input Recommendation
- Duplicate Recommendation Handling
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

---

## Data Processing

- Pandas
- NumPy

---

## Machine Learning

- Scikit-learn

### Algorithms

- TF-IDF Vectorization
- Cosine Similarity

---

## Future Libraries

- Surprise
- Implicit

---

## Visualization

- Matplotlib

---

## Deployment

- FastAPI
- Streamlit

---

# 📂 Project Structure

```text
Game-Recommendation-System/

│
├── data/
│
├── docs/
│   ├── Day01.md
│   ├── Day02.md
│   ├── Day03.md
│   └── Day04.md
│
├── models/
│   └── content_base.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_content_based_recommendation.ipynb
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
Steam Metadata
      │
      ▼
Data Preprocessing
      │
      ▼
Combined Features
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Cosine Similarity
      │
      ▼
Top-N Recommendation
      │
      ▼
Recommendation Result
```

---

# ✅ Implemented Features

- Steam Metadata Loading
- Data Preprocessing
- Combined Features Generation
- TF-IDF Vectorization
- Cosine Similarity Recommendation
- User Input Recommendation
- Duplicate Recommendation Handling
- Object-Oriented Recommendation Model
- Modular Project Structure

---

# 🚀 Development Roadmap

## ✅ V1. Content-Based Recommendation (Completed)

### Data Processing

- [x] Steam Metadata Loading
- [x] Data Validation
- [x] Data Preprocessing
- [x] Missing Value Handling
- [x] Combined Features Generation

### Recommendation

- [x] Content-Based Recommendation
- [x] TF-IDF Vectorization
- [x] Cosine Similarity
- [x] Top-N Recommendation
- [x] User Input Recommendation
- [x] Duplicate Recommendation Handling

### Software Engineering

- [x] Function Modularization
- [x] Object-Oriented Design
- [x] Project Structure Refactoring

---

## 🚧 V1.1 Feature Engineering

현재 Baseline(Content-Based Recommendation)을 완성하였으며,

추천 품질 향상을 위해 Feature를 점진적으로 추가할 예정입니다.

### Planned Features

- Release Year
- Developer
- Publisher
- About the Game
- Price
- User Score
- Positive / Negative Reviews
- Ranking Strategy
- Feature Weighting

---

## 🚧 V2. Collaborative Filtering

다양한 Collaborative Filtering 알고리즘을 구현하고 성능을 비교합니다.

### Algorithms

- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Matrix Factorization (SVD)
- ALS
- BPR

---

## 🚧 V3. Hybrid Recommendation

Content-Based Recommendation과

Collaborative Filtering을 결합하여

추천 성능을 향상시킵니다.

---

## 🚧 V4. Deployment

추천 시스템을 실제 서비스 형태로 배포합니다.

### Backend

- FastAPI

### Frontend

- Streamlit

---

# 📒 Development Process

프로젝트는 실험 코드와 실제 구현 코드를 분리하여 개발합니다.

## 📒 notebooks/

Notebook에서는

- 데이터 탐색(EDA)
- 데이터 검증
- 데이터 전처리
- Feature Engineering
- 추천 알고리즘 실험
- 추천 결과 분석

을 수행합니다.

---

## 🐍 Python Modules

Python Module에서는

- 함수화
- 모듈화
- 클래스 설계
- 프로젝트 구조 관리
- 재사용 가능한 코드 작성

을 수행합니다.

---

## 📝 docs/

프로젝트 진행 과정을 Day Log 형태로 기록합니다.

예시

- Day01.md
- Day02.md
- Day03.md
- Day04.md

매일

- 구현 내용
- 문제 해결 과정
- 의사결정 과정
- 배운 내용
- 회고
- 다음 계획

을 기록합니다.

---

# 📊 Evaluation

추천 시스템 구현 후 아래 항목을 비교 및 평가할 예정입니다.

### Recommendation Quality

- Recommendation Results
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
- Feature Engineering Before/After

---

# 📈 Current Progress

| Module | Status |
|----------|:------:|
| Data Loading | ✅ |
| Data Validation | ✅ |
| Data Preprocessing | ✅ |
| Feature Engineering (Baseline) | ✅ |
| TF-IDF Vectorization | ✅ |
| Cosine Similarity | ✅ |
| Content-Based Recommendation | ✅ |
| Duplicate Recommendation Handling | ✅ |
| Evaluation | 🚧 |
| Collaborative Filtering | 🚧 |
| Hybrid Recommendation | 🚧 |
| Deployment | 🚧 |

---

# 🎯 Learning Objectives

## Recommendation System

- Content-Based Recommendation
- Collaborative Filtering
- Hybrid Recommendation
- Recommendation Evaluation

---

## Data Processing

- Data Validation
- Data Preprocessing
- Feature Engineering
- Large-scale Dataset Processing

---

## Machine Learning

- TF-IDF Vectorization
- Cosine Similarity
- Ranking Strategy
- Recommendation Evaluation

---

## Software Engineering

- Function Modularization
- Object-Oriented Programming (OOP)
- Python Project Structure
- Code Reusability
- GitHub Project Management

---

## Practical Skills

- Recommendation System Design
- Recommendation Result Analysis
- Memory Optimization
- FastAPI Deployment
- Streamlit Deployment

---

# 📌 Project Status

**Current Version:** `V1 - Content-Based Recommendation`

### Completed

- Steam Metadata Preprocessing
- Combined Features Generation
- TF-IDF Vectorization
- Cosine Similarity Recommendation
- User Input Recommendation
- Duplicate Recommendation Handling
- Object-Oriented Recommendation Model
- Project Modularization

### Next Milestone

➡️ Recommendation Evaluation (`evaluation.py`)

➡️ Collaborative Filtering

➡️ Hybrid Recommendation

➡️ Web Service Deployment