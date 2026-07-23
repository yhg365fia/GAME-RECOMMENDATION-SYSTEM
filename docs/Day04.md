# 📅 Day04 - Content-Based Recommendation System

## 🎯 Today's Goal

- 프로젝트 구조 리팩토링
- Content-Based Recommendation System 구현
- TF-IDF 기반 추천 모델 클래스화
- 사용자 입력을 통한 추천 시스템 완성

---

## 📌 What I Did

### 1. Project Structure Refactoring

기존 Notebook 중심 코드에서 프로젝트 구조로 리팩토링하였다.

```text
Game-Recommendation-System

├── main.py
├── preprocessing.py
├── evaluation.py
├── models
│   └── content_base.py
├── notebooks
└── data
```

각 파일의 역할을 분리하였다.

- `main.py` : 전체 실행 흐름 관리
- `preprocessing.py` : 데이터 로드 및 전처리
- `content_base.py` : 콘텐츠 기반 추천 모델 구현
- `evaluation.py` : 추천 성능 평가(예정)

---

### 2. Data Preprocessing (`preprocessing.py`)

#### load_games()

Steam 게임 메타데이터를 불러오는 함수를 구현하였다.

- `games_inc.csv` 로드
- Steam 데이터셋 컬럼명을 직접 지정하여 올바르게 매핑

#### preprocess()

추천 모델에 사용할 메타데이터를 생성하였다.

##### 필요한 컬럼 선택

- AppID
- Name
- Genres
- Tags
- Categories
- About the game

##### 데이터 정제

- Name이 없는 게임 제거
- Genres, Tags, Categories, About the game이 모두 존재하는 데이터만 사용 (`thresh=4`)
- 텍스트 결측치는 빈 문자열(`""`)로 대체

##### Combined Features 생성

추천 모델에서 사용할 텍스트 Feature를 생성하였다.

```python
combined_features =
Genres
+ Tags
+ Categories
```

쉼표를 공백으로 변경하여 TF-IDF 입력 문자열을 생성하였다.

---

### 3. Content-Based Recommendation (`content_base.py`)

기존의 비어있던 `fit()`과 `recommend()`를 실제 추천 시스템으로 구현하였다.

#### fit()

TF-IDF Vectorizer를 이용하여 게임 Feature를 벡터화하였다.

```python
TfidfVectorizer(
    stop_words="english",
    lowercase=True
)
```

`combined_features`를 학습하여 TF-IDF Matrix를 생성하였다.

생성 결과

```
(82411, 544)
```

- 82,411개의 게임
- 544개의 단어 Feature

---

#### recommend()

추천 기능을 구현하였다.

구현 내용

- 입력한 게임 이름 검색
- 동일한 이름의 게임(AppID) 처리
- 코사인 유사도 계산
- 자기 자신 제외
- 유사도 기준 내림차순 정렬
- 추천 결과 DataFrame 반환

또한 추천 결과에서 동일한 게임명이 반복되는 문제를 해결하여 서로 다른 게임만 추천하도록 개선하였다.

---

### 4. Main Program (`main.py`)

전체 실행 흐름을 새롭게 구성하였다.

```text
Load Data
      ↓
Preprocessing
      ↓
Create Recommender
      ↓
TF-IDF Fit
      ↓
User Input
      ↓
Recommendation
      ↓
Print Result
```

사용자가 직접 추천 기준 게임을 입력할 수 있도록 변경하였다.

```python
game_name = input("추천 기준 게임을 입력하세요: ")
```

---

### 5. Recommendation Result Improvement

추천 결과의 품질을 개선하였다.

기존에는 PUBG를 기준으로 추천했을 때

- Mini Taoism
- Paint Puzzle Quest
- Mahjong

등 장르와 관련성이 낮은 게임이 추천되었다.

데이터 로드 과정과 추천 로직을 수정한 후

- DEATH FIELD
- Tom Clancy's Rainbow Six Siege
- Z1 Battle Royale
- Darwin Project
- Battle Teams 2

등 장르와 태그가 유사한 게임이 정상적으로 추천되는 것을 확인하였다.

자세한 문제 해결 과정은 아래 Troubleshooting에 정리하였다.

---

# 🛠 Troubleshooting

### Problem 1. 추천 결과가 관련 없는 게임으로 출력되는 문제

#### Problem

PUBG를 기준으로 추천했을 때

- Mini Taoism
- Paint Puzzle Quest
- Mahjong

등 장르와 관련성이 낮은 게임들이 추천되는 문제가 발생하였다.

---

#### Analysis

추천 알고리즘 자체의 문제인지 확인하기 위해 아래 항목을 순차적으로 검증하였다.

- CSV 데이터 로드 방식
- Steam 컬럼 매핑
- Combined Features 생성
- TF-IDF Matrix 생성
- Cosine Similarity 계산
- 추천 결과 검증

디버깅 과정에서 데이터를 함수화(`load_games()`)하면서 기존 Notebook에서 사용했던 **Steam 메타데이터 컬럼 매핑(`names=correct_columns`)** 을 적용하지 않은 것을 발견하였다.

그 결과 CSV 컬럼이 잘못 매핑되어 `Genres`, `Tags`, `Categories` 등의 메타데이터가 정상적으로 로드되지 않았고, TF-IDF 입력 데이터 역시 올바르게 생성되지 않았다.

이를 통해 추천 알고리즘의 문제가 아니라 **데이터 로드 과정에서 발생한 메타데이터 매핑 문제**임을 확인하였다.

---

#### Solution

`load_games()` 함수에서 Steam 메타데이터 컬럼명을 다시 명시적으로 지정하도록 수정하였다.

```python
games = pd.read_csv(
    file_path,
    header=0,
    names=correct_columns
)
```

수정 후

- Combined Features
- TF-IDF Matrix
- Cosine Similarity

가 모두 정상적으로 생성되는 것을 확인하였다.

또한 추천 결과 생성 시 이미 추천된 게임은 제외하고 다음 후보를 선택하도록 추천 로직을 수정하였다.

최종적으로 PUBG를 기준으로

- DEATH FIELD
- Rainbow Six Siege
- Z1 Battle Royale
- Darwin Project

등 장르와 태그가 유사한 게임들이 추천되었으며, 추천 품질 저하의 원인이 데이터 로드 과정에 있었음을 확인하였다.

### Problem 2. 동일한 게임이 여러 번 추천되는 문제

#### Problem

추천 결과에서 동일한 게임명이 여러 번 출력되는 문제가 발생하였다.

예시

```
Tom Clancy's Rainbow Six Siege
Tom Clancy's Rainbow Six Siege
Tom Clancy's Rainbow Six Siege
```

---

#### Analysis

Steam 데이터셋에는 동일한 게임명이 서로 다른 AppID를 가지는 경우가 존재하였다.

기존 추천 로직은 유사도 상위 Top-N을 그대로 반환했기 때문에 동일한 게임이 여러 번 추천되는 문제가 발생하였다.

---

#### Solution

추천 결과를 순차적으로 확인하면서 이미 추천된 게임명은 제외하고 다음 후보를 추천하도록 로직을 수정하였다.

이를 통해 항상 서로 다른 게임으로 Top-N 추천 결과를 제공하도록 개선하였다.

---

### Lessons from Debugging

이번 디버깅을 통해 추천 시스템에서는 **모델뿐만 아니라 데이터 로드 과정도 추천 품질에 직접적인 영향을 준다는 점**을 학습하였다.

또한 Notebook에서 프로젝트 구조로 리팩토링할 때 기존 전처리 과정뿐 아니라 **컬럼 매핑과 같은 세부 구현도 함께 이전되어야 한다는 점**을 경험하였다.

---

# 📚 What I Learned

- 프로젝트 구조 분리 및 모듈화
- 클래스 기반 추천 시스템 설계
- `fit()`과 `recommend()`의 역할
- TF-IDF Vectorization
- Cosine Similarity 기반 추천
- Pandas를 이용한 데이터 전처리
- 추천 결과 후처리
- 추천 시스템 디버깅 과정

---

### Additional Concepts Learned

#### DataFrame Index와 `reset_index()`

`dropna()`를 사용하면 행은 삭제되지만 기존 인덱스는 그대로 유지된다는 점을 학습하였다.

예를 들어

```
0
1
5
8
10
```

처럼 인덱스가 유지될 수 있으며,

TF-IDF Matrix는

```
0
1
2
3
4
```

처럼 순차적인 행 번호를 사용하기 때문에 DataFrame과 행 번호가 일치하지 않을 수 있다는 점을 이해하였다.

이를 방지하기 위해

```python
meta = meta.reset_index(drop=True)
```

를 사용하여 DataFrame과 TF-IDF Matrix의 행 순서를 일치시키는 이유를 학습하였다.

---

#### flatten()

`cosine_similarity()`의 반환값이 `(1, n)` 형태의 2차원 배열이라는 점을 이해하였다.

```python
.flatten()
```

을 사용하여

```
(1, 82411)
```

↓

```
(82411,)
```

형태의 1차원 배열로 변환하고, 이후 정렬과 인덱싱에 사용할 수 있다는 점을 학습하였다.

---

#### argsort()

```python
sim_scores.argsort()[::-1]
```

의 동작 원리를 이해하였다.

- `argsort()` : 값을 정렬했을 때의 인덱스 반환
- `[::-1]` : 내림차순 정렬

이를 이용하여 가장 유사도가 높은 게임 순으로 추천 목록을 생성하는 과정을 학습하였다.

---

#### iloc()와 copy()

추천 결과를 생성할 때

```python
self.meta.iloc[sim_indices]
```

를 사용하는 이유를 이해하였다.

- `iloc()` : 행 번호 기반 접근
- `copy()` : 원본 DataFrame을 변경하지 않기 위해 복사본 생성

추천 결과에 `Similarity` 컬럼을 추가하더라도 원본 데이터가 변경되지 않는 이유를 학습하였다.

---

#### 객체지향 프로그래밍(OOP)

추천 시스템을 클래스 형태로 구현하면서

```python
self.meta
self.tfidf
self.tfidf_matrix
```

와 같이 객체 내부 상태를 저장하는 이유를 이해하였다.

`fit()`에서 생성한 데이터를 `recommend()`에서 재사용하기 위해 `self`를 사용하는 객체지향 설계를 학습하였다.

---

#### 프로젝트 모듈화

프로젝트를

- preprocessing
- models
- evaluation
- main

으로 분리하여 각 파일이 하나의 역할만 담당하도록 설계하는 방식을 학습하였다.

또한 함수와 클래스를 `import`하여 사용하는 프로젝트 구조를 익혔다.

---

#### 추천 시스템 디버깅

추천 결과가 예상과 다르게 나왔을 때

- 데이터 로드 확인
- 컬럼 매핑 확인
- Combined Features 확인
- TF-IDF Matrix 확인
- Cosine Similarity 확인
- 추천 결과 분석

순서대로 문제를 좁혀가며 원인을 찾는 디버깅 과정을 경험하였다.

단순히 코드를 수정하는 것이 아니라 데이터와 모델을 단계별로 검증하는 중요성을 학습하였다.

또한 추천 시스템에서는 **모델뿐 아니라 데이터와 전처리 과정도 함께 검증해야 한다는 점**을 경험하였다.

---

## 🚀 Next Step

- `evaluation.py` 구현
- 추천 성능 평가
- Collaborative Filtering 구현
- Content-Based와 Collaborative Filtering 성능 비교

---

## ✅ Result

- Project Structure Refactoring
- Data Preprocessing
- TF-IDF Vectorization
- Content-Based Recommendation System
- User Input Recommendation
- Duplicate Recommendation Handling
- Recommendation Quality Improvement
- Recommendation System Debugging
```