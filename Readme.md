# 🎮 Steam Game Recommendation System

Steam 게임 데이터를 활용하여 다양한 추천 시스템 알고리즘을 구현하고 성능을 비교하는 프로젝트입니다.

현재는 **Content-Based Recommendation System**을 구현하였으며,

- Steam 메타데이터 전처리
- User-based Train/Test Split
- TF-IDF Vectorization
- Cosine Similarity 기반 추천
- Multi-Game Recommendation
- Recommendation Evaluation Dataset 구축
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
- Recommendation Evaluation Dataset
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
│   └── split/
│       ├── train.csv
│       └── test.csv
│
├── docs/
│   ├── Day01.md
│   ├── Day02.md
│   ├── Day03.md
│   ├── Day04.md
│   └── Day05.md
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
Data Preprocessing
      │
      ▼
Combined Features
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Content-Based Recommendation
      │
      ▼
Top-N Recommendation
      │
      ▼
Recommendation Evaluation
```

---

# ✅ Implemented Features

- Steam Metadata Loading
- Data Validation
- User-based Train/Test Split
- Data Preprocessing
- Combined Features Generation
- TF-IDF Vectorization
- Cosine Similarity Recommendation
- Multi-Game Recommendation
- Duplicate Recommendation Handling
- Recommendation Evaluation Dataset
- Object-Oriented Recommendation Model
- Modular Project Structure

---

# 🚀 Development Roadmap

## ✅ V1. Content-Based Recommendation

### Data Processing

- [x] Steam Metadata Loading
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

---

# 📈 Current Progress

| Module | Status |
| :--- | :---: |
| Data Loading | ✅ |
| Data Validation | ✅ |
| Train/Test Split | ✅ |
| Data Preprocessing | ✅ |
| TF-IDF Vectorization | ✅ |
| Content-Based Recommendation | ✅ |
| Multi-Game Recommendation | ✅ |
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

**Current Version:** `V1.1 - Content-Based Recommendation`

### Completed

- Steam Metadata Preprocessing
- User-based Train/Test Split
- Recommendation Evaluation Dataset
- Combined Features Generation
- TF-IDF Vectorization
- Multi-Game Recommendation
- Cosine Similarity Recommendation
- Object-Oriented Recommendation Model
- Project Modularization

### Next Milestone

➡️ Precision@K / Recall@K / MAP / NDCG

➡️ Collaborative Filtering

➡️ Hybrid Recommendation

➡️ Web Service Deployment