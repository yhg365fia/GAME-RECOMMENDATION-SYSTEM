# 🎮 Steam Game Recommendation System

Steam 게임 데이터를 활용하여 다양한 추천 시스템 알고리즘을 구현하고 성능을 비교하는 프로젝트입니다.

이번 프로젝트에서는 단순히 추천 알고리즘을 구현하는 것이 아니라,

- 데이터 탐색(EDA)
- 데이터 검증(Data Validation)
- 데이터 전처리(Data Preprocessing)
- Feature Engineering
- 추천 알고리즘 구현
- 추천 결과 분석 및 개선
- 함수화 및 모듈화
- 프로젝트 구조 설계
- 성능 평가
- 서비스 배포

까지 실제 추천 시스템 개발 과정을 경험하는 것을 목표로 합니다.

---

# 📌 프로젝트 목표

- Steam 게임 추천 시스템 구현
- Steam 메타데이터 기반 콘텐츠 추천 시스템(Content-Based Recommendation) 구현
- 다양한 Collaborative Filtering 알고리즘 구현
- Hybrid Recommendation 구현
- 데이터 검증(Data Validation) 및 전처리
- Feature Engineering을 통한 추천 성능 개선
- 대용량 데이터셋을 고려한 메모리 효율적인 추천 시스템 구현
- 추천 결과 분석 및 Ranking 전략 설계
- 코드 함수화 및 모듈화
- 프로젝트 구조 설계 및 유지보수성 향상
- FastAPI와 Streamlit을 이용한 추천 시스템 서비스 배포

---

# 🛠️ Tech Stack

### Language

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Surprise
- Implicit

### Visualization

- Matplotlib

### Deployment

- FastAPI
- Streamlit

---

# 📂 Project Structure

```text
Steam-Game-Recommendation-System/
│
├── data/
│   ├── raw/                    # 원본 데이터
│   └── processed/              # 전처리 완료 데이터
│
├── docs/
│   ├── Day01.md
│   ├── Day02.md
│   ├── Day03.md
│   └── ...
│
├── models/
│   ├── content_based.py
│   ├── user_cf.py
│   ├── item_cf.py
│   ├── svd.py
│   ├── als.py
│   ├── bpr.py
│   └── hybrid.py
│
├── notebook/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_vectorization.ipynb
│   ├── 04_content_based_recommendation.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_evaluation.ipynb
│
├── preprocessing.py
├── evaluation.py
├── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Development Roadmap

## V1. Content-Based Recommendation (Baseline)

- 데이터 탐색(EDA)
- 메타데이터 분석
- 데이터 검증(Data Validation)
- 데이터 전처리(Data Preprocessing)
- Metadata 생성
- Combined Features 생성
- TF-IDF Vectorization
- On-demand Cosine Similarity 계산
- Top-N Recommendation 구현
- 추천 결과 검증

---

## V1.1 Feature Engineering

Baseline 모델을 완성한 후

추천 결과를 분석하면서 Feature를 하나씩 추가하여 성능을 개선합니다.

예정 Feature

- Release Year
- Developer
- Publisher
- About the Game
- Price
- Review Score
- Missing Value Penalty
- Ranking Strategy

---

## V2. Collaborative Filtering

- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Matrix Factorization (SVD)
- ALS
- BPR

각 알고리즘을 구현하고

성능을 비교합니다.

---

## V3. Hybrid Recommendation

Content-Based Recommendation과

Collaborative Filtering을 결합하여

추천 성능을 향상시킵니다.

---

## V4. Deployment

- FastAPI
- Streamlit

추천 시스템을 웹 서비스 형태로 배포합니다.

---

# 📒 Development Process

프로젝트는 다음과 같은 역할로 구분하여 개발합니다.

## 📒 notebook/

주로 실험 및 분석을 수행합니다.

- 데이터 탐색(EDA)
- 메타데이터 분석
- 데이터 검증(Data Validation)
- 데이터 전처리
- Feature Engineering
- 알고리즘 실험
- 추천 결과 분석
- 성능 검증

---

## 🐍 Python Modules

최종 구현 코드를 작성합니다.

- 함수화
- 모듈화
- 프로젝트 구조 관리
- 재사용 가능한 코드 작성

---

## 📝 docs/

프로젝트 진행 과정을 기록합니다.

- Day01.md
- Day02.md
- Day03.md
- ...

매일

- 진행 내용
- 문제 해결 과정
- 의사결정 과정
- 배운 점
- 회고
- 다음 계획

을 기록합니다.

---

# 📊 Evaluation

추천 시스템 구현 후 다음 항목들을 비교할 예정입니다.

- 추천 결과 품질
- Feature Engineering 전/후 비교
- Content-Based vs Collaborative Filtering
- Hybrid Recommendation 성능
- 메모리 사용량
- 실행 속도

---

# 🎯 Learning Objectives

이번 프로젝트를 통해 다음 내용을 학습하는 것을 목표로 합니다.

## Recommendation System

- Content-Based Recommendation
- Collaborative Filtering
- Hybrid Recommendation

## Data Processing

- 데이터 탐색(EDA)
- 데이터 검증(Data Validation)
- 데이터 전처리
- Feature Engineering

## Machine Learning

- TF-IDF Vectorization
- Cosine Similarity
- Ranking Strategy
- Recommendation Evaluation

## Software Engineering

- 함수화 및 모듈화
- Python 프로젝트 구조 설계
- 코드 재사용성 향상
- GitHub 프로젝트 관리

## Practical Skills

- 대용량 데이터셋 처리
- 메모리 최적화
- 추천 시스템 설계
- FastAPI 서비스 구축
- Streamlit 배포