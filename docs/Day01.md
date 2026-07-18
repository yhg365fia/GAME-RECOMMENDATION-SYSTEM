# Day01

## 📅 날짜

2026-07-18

---

# 🎯 오늘의 목표

- 추천 시스템 프로젝트 주제 선정
- 프로젝트 구조 설계
- 개발 환경 구축
- README 및 프로젝트 목표 작성
- Steam 데이터셋 탐색 시작

---

# ✅ 진행 내용

## 1. 프로젝트 생성 및 환경 구축

프로젝트를 새롭게 생성하고 기본 개발 환경을 구축하였다.

- Git 저장소 생성
- Python 가상환경(.venv) 생성
- requirements.txt 작성
- .gitignore 작성
- README.md 작성
- 기본 프로젝트 폴더 구조 생성

```text
Steam-Game-Recommendation-System/
│
├── data/
├── docs/
├── models/
├── notebook/
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

## 2. 프로젝트 목표 설계

이번 프로젝트는 단순히 추천 시스템을 구현하는 것이 아니라,

- 다양한 추천 알고리즘 구현
- 모델 성능 비교
- 함수화 및 모듈화
- 프로젝트 구조 설계
- 코드 재사용성 향상

까지 함께 학습하는 것을 목표로 설정하였다.

또한 README 역시 기존 프로젝트처럼 단순한 설명이 아니라, 프로젝트의 목적과 개발 계획을 직접 고민하고 수정하면서 작성하였다.

---

## 3. 프로젝트 로드맵 작성

프로젝트를 다음과 같은 단계로 진행하기로 계획하였다.

### V1

- Content-Based Recommendation

### V2

Collaborative Filtering

- User-Based CF
- Item-Based CF
- SVD
- ALS
- BPR

### V3

- Hybrid Recommendation

### V4

- FastAPI
- Streamlit
- 추천 시스템 웹 서비스 배포

---

## 4. 프로젝트 구조 설계

이번 프로젝트부터는 하나의 Python 파일에 모든 코드를 작성하지 않고,

기능별로 파일을 분리하여 프로젝트를 진행하기로 하였다.

```text
models/
    content_based.py
    user_cf.py
    item_cf.py
    svd.py
    als.py
    bpr.py
    hybrid.py

preprocessing.py

evaluation.py

main.py
```

파일과 폴더 구조는 먼저 설계하고,

세부 함수와 클래스는 구현하면서 추가하는 방식으로 진행하기로 결정하였다.

---

## 5. 데이터셋 선정

초기에는 게임 정보만 있는 데이터셋을 사용하려고 하였지만,

추천 시스템에서는 사용자와 아이템의 상호작용 데이터가 반드시 필요하다는 것을 알게 되었다.

따라서 다음 데이터를 사용하는 Steam 데이터셋을 선택하였다.

- games.csv
- recommendations.csv
- users.csv
- games_metadata.json

---

## 6. 데이터 탐색 시작

Notebook에서 Steam 데이터를 불러와 데이터 구조를 확인하기 시작하였다.

확인한 데이터

- recommendations.csv
- games.csv
- users.csv

Notebook가 `notebook/` 폴더 안에 있기 때문에

상대경로(`../data/...`)를 사용하여 데이터를 불러오는 방법도 함께 학습하였다.

---

# 📚 오늘 배운 내용

## 1. 추천 시스템에서 사용하는 데이터

추천 시스템에서는 크게 두 종류의 데이터를 사용한다.

### Content Data

- 게임 정보
- 장르
- 태그
- 설명
- 개발사

Content-Based Recommendation에서 사용된다.

---

### User Interaction Data

- user_id
- app_id
- 추천 여부
- 플레이 시간

Collaborative Filtering에서 사용된다.

---

## 2. 프로젝트 구조 설계의 중요성

이번 프로젝트부터는

- preprocessing.py
- models/
- evaluation.py
- main.py

처럼 기능별로 파일을 분리하여 개발하기로 하였다.

이를 통해

- 함수화
- 모듈화
- 유지보수
- 코드 재사용

을 함께 연습하기로 하였다.

---

## 3. Notebook와 Python 파일의 역할

Notebook는

- 데이터 탐색(EDA)
- 실험
- 알고리즘 검증

을 위한 공간으로 사용한다.

실험이 완료된 코드는

Python 파일로 옮겨 최종 프로젝트를 구성한다.

---

## 4. 프로젝트 기록 방식

이번 프로젝트부터는

### notebook/

- 데이터 탐색
- 실험
- 알고리즘 검증

### Python 파일

- 최종 구현
- 함수화
- 모듈화

### docs/

- Day01.md
- Day02.md
- ...

형태로 매일 회고를 작성하여

프로젝트 진행 과정과 배운 내용을 기록하기로 하였다.

---

## 5. 프로젝트 진행 방식

프로젝트 초반에는

Notebook의 비중이 높고,

프로젝트가 진행될수록 Python 코드의 비중이 높아지는 방식으로 개발을 진행하기로 하였다.

### 프로젝트 초반

Notebook 약 80%

- 데이터 탐색
- 전처리 실험
- 추천 알고리즘 실험

Python 약 20%

- 프로젝트 구조 작성
- 기본 함수 작성

---

### 프로젝트 중반

Notebook 약 50%

Python 약 50%

실험한 코드를 함수화하여 프로젝트 구조에 적용한다.

---

### 프로젝트 후반

Notebook 약 20%

Python 약 80%

최종 구현 및 리팩토링을 진행한다.

---

# ⚠️ 문제 및 해결

## FileNotFoundError

Notebook에서

```python
pd.read_csv("data/...")
```

를 사용하였더니

파일을 찾지 못하였다.

### 원인

현재 작업 경로가

```
Game-Recommendation-System/notebook
```

이었기 때문이다.

### 해결

상대경로를 사용하였다.

```python
pd.read_csv("../data/...")
```

---

## Windows 경로 경고

절대경로를 사용할 때

```python
"C:\projects\..."
```

를 사용하여

SyntaxWarning이 발생하였다.

### 원인

Windows의 `\`가 Escape Sequence로 해석되기 때문이다.

### 해결

프로젝트에서는 상대경로를 사용하기로 결정하였다.

---

# 💡 회고

이번 프로젝트는 기존 Titanic, House Prices 프로젝트와는 달리, 단순히 모델의 성능을 높이는 것만을 목표로 하지 않았다.

프로젝트를 직접 설계하면서 README의 내용과 프로젝트 목표를 스스로 고민하고 수정하였고, 프로젝트 구조 또한 직접 설계하면서 개발을 시작하였다.

또한 이번 프로젝트에서는 함수화와 모듈화를 적용하여 유지보수가 쉬운 프로젝트를 만드는 방법도 함께 학습하기로 하였다.

특히 **Notebook는 실험 공간**, **Python 파일은 최종 구현**, **docs는 학습 기록과 회고**라는 역할을 분리하여 관리하는 방식을 처음 알게 되었는데, 프로젝트가 커질수록 매우 효율적인 개발 방식이라는 점이 인상 깊었다.

처음 접한 개발 방식인 만큼 아직 익숙하지는 않지만, 이번 프로젝트를 진행하면서 적극적으로 활용해 보고 싶다.

또한 ALS(Alternating Least Squares)와 BPR(Bayesian Personalized Ranking)과 같은 모델 기반 협업 필터링 알고리즘도 이번에 처음 알게 되었다.

아직 알고리즘의 원리와 수학적 배경은 잘 이해하지 못하지만, 추천 시스템에서 널리 사용되는 중요한 기법이라는 점이 흥미로웠다.

이번 프로젝트에서는 단순히 구현만 하는 것이 아니라,

- ALS가 어떤 문제를 해결하기 위해 만들어졌는지
- BPR이 기존 추천 알고리즘과 어떤 차이가 있는지
- 각 모델이 어떤 상황에서 좋은 성능을 보이는지

까지 함께 공부하고 직접 구현 및 성능 비교를 진행해 보고 싶다.

이번 프로젝트를 통해 추천 시스템뿐만 아니라 프로젝트 설계 능력과 코드 구조를 설계하는 능력까지 함께 성장하는 것을 목표로 한다.

---

# 🎯 다음 목표

- games.csv 컬럼 분석
- games_metadata.json 구조 확인
- Content-Based Recommendation에 사용할 컬럼 선정
- Metadata(Soup) 설계
- Content-Based Recommendation 구현 시작