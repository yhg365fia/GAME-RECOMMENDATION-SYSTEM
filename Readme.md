# 🎮 Steam Game Recommendation System

Steam 게임 데이터를 활용하여 다양한 추천 시스템 알고리즘을 구현하고 성능을 비교하는 프로젝트입니다.

이번 프로젝트에서는 단순히 추천 알고리즘을 구현하는 것이 아니라,

- 데이터 탐색(EDA)
- 데이터 검증(Data Validation)
- 데이터 전처리
- 추천 알고리즘 구현
- 함수화 및 모듈화
- 프로젝트 구조 설계
- 성능 평가
- 서비스 배포

까지 실제 추천 시스템 개발 과정을 경험하는 것을 목표로 합니다.

---

# 📌 프로젝트 목표

- Steam 게임 추천 시스템 구현
- Steam 메타데이터 기반 콘텐츠 추천 시스템 구현
- 다양한 Model-Based Collaborative Filtering 구현
- 추천 알고리즘 성능 비교 및 평가
- 데이터 검증(Data Validation) 및 전처리
- 코드 함수화 및 모듈화
- 프로젝트 구조 설계 및 유지보수성 향상
- FastAPI와 Streamlit을 이용한 추천 시스템 서비스 배포

---

# 🛠️ 사용 기술

- Python
- Pandas
- NumPy
- Scikit-learn
- Surprise
- Implicit
- Matplotlib
- FastAPI
- Streamlit

---

# 📂 프로젝트 구조

```text
Steam-Game-Recommendation-System/
│
├── data/                       # 원본 데이터 및 전처리 데이터
│
├── docs/                       # 프로젝트 일지 및 회고
│   ├── Day01.md
│   ├── Day02.md
│   └── ...
│
├── models/                     # 추천 알고리즘
│   ├── content_based.py
│   ├── user_cf.py
│   ├── item_cf.py
│   ├── svd.py
│   ├── als.py
│   ├── bpr.py
│   └── hybrid.py
│
├── notebook/                   # EDA 및 알고리즘 실험
│   ├── 01_data_exploration.ipynb
│   └── ...
│
├── preprocessing.py            # 데이터 전처리
├── evaluation.py               # 추천 성능 평가
├── main.py                     # 프로젝트 실행
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📖 개발 로드맵

## V1. Content-Based Recommendation

- 데이터 탐색(EDA)
- 메타데이터 분석 및 선정
- 데이터 검증(Data Validation)
- 데이터 전처리
- Metadata 생성
- TF-IDF Vectorization
- Cosine Similarity 계산
- 추천 함수 구현

---

## V2. Collaborative Filtering

- User-Based CF
- Item-Based CF
- SVD
- ALS
- BPR

각 알고리즘을 구현하고 성능을 비교합니다.

---

## V3. Hybrid Recommendation

Content-Based Recommendation과 Collaborative Filtering을 결합하여 성능을 개선합니다.

---

## V4. Deployment

- FastAPI
- Streamlit

추천 시스템을 웹 서비스 형태로 배포합니다.

---

# 📚 프로젝트 진행 방식

이번 프로젝트에서는 역할을 다음과 같이 구분하여 개발합니다.

### 📒 notebook/

- 데이터 탐색(EDA)
- 메타데이터 분석
- 데이터 검증 및 전처리
- 알고리즘 실험
- 성능 검증

### 🐍 Python 파일

- 최종 구현
- 함수화
- 모듈화
- 프로젝트 구조 관리

### 📝 docs/

- Day01.md
- Day02.md
- ...

매일 진행한 내용, 문제 해결 과정, 배운 점, 회고를 기록합니다.

---

# 🎯 학습 목표

이번 프로젝트를 통해 다음 내용을 함께 학습하는 것을 목표로 합니다.

- 추천 시스템 알고리즘 이해
- 데이터 검증(Data Validation)
- 데이터 전처리
- Feature Engineering
- 함수화 및 모듈화
- Python 프로젝트 구조 설계
- 코드 재사용성 향상
- 추천 모델 성능 평가
- GitHub 프로젝트 관리
- FastAPI 및 Streamlit을 활용한 서비스 배포