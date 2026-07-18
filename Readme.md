# 🎮 Steam Game Recommendation System

> 실제 Steam 게임 데이터를 활용하여 처음부터 끝까지(End-to-End) 추천 시스템을 구축하는 프로젝트입니다.

---

# 📌 프로젝트 소개

본 프로젝트는 실제 Steam 게임 데이터를 활용하여 다양한 추천 알고리즘을 구현하고 비교하며, 추천 품질을 지속적으로 향상시키는 것을 목표로 합니다.

콘텐츠 기반 추천(Content-Based Recommendation), 협업 필터링(Collaborative Filtering), 하이브리드 추천(Hybrid Recommendation)을 단계적으로 구현하고, 각 알고리즘의 성능을 비교·분석합니다. 최종적으로는 웹 서비스로 배포하여 실제 사용 가능한 추천 시스템을 개발합니다.

---

# 🎯 프로젝트 목표

- 실제 Steam 게임 데이터를 활용한 추천 시스템 구축
- 추천시스템 이론을 실제 프로젝트에 적용
- 다양한 추천 알고리즘 구현 및 성능 비교
- 추천 품질 향상을 위한 성능 평가 및 최적화
- 웹 서비스 형태로 배포하여 실제 사용 가능한 추천 시스템 개발

---

# 🛠 기술 스택

### Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Surprise

### Deployment

- Streamlit
- FastAPI

---

# 📂 데이터셋

- Steam Game Dataset
- (추후 데이터셋 출처 추가)

---

# 🚀 프로젝트 로드맵

## ✅ V1. 콘텐츠 기반 추천 (Content-Based Recommendation)

- TF-IDF 기반 추천
- Metadata 기반 추천
- 추천 성능 평가

---

## ⏳ V2. 협업 필터링 (Collaborative Filtering)

### Memory-Based

- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering

### Model-Based

- SVD
- ALS
- BPR

### 성능 개선

- 추천 성능 평가
- 모델 성능 비교
- 추천 품질 최적화

---

## ⏳ V3. 하이브리드 추천 (Hybrid Recommendation)

- Content-Based + Collaborative Filtering 결합
- 추천 성능 평가
- 추천 품질 최적화

---

## ⏳ V4. 웹 서비스

- Streamlit
- FastAPI

---

# 📊 평가 지표

추천 시스템의 성능은 다음과 같은 추천 전용 평가 지표를 활용하여 평가합니다.

- Precision@K
- Recall@K
- MAP
- NDCG

각 추천 알고리즘의 성능을 비교하고, 추천 품질을 지속적으로 개선합니다.

---

# 📅 개발 일지

- [ ] Day 01
- [ ] Day 02
- [ ] Day 03
- ...

---

# 🔮 향후 발전 방향

- Neural Collaborative Filtering (NCF)
- Two-Tower Model
- 개인화 추천 고도화
- 실시간 추천 시스템
- MLOps 적용