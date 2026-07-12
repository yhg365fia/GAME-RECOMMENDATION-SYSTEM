# DAY00 - Chapter 2 학습 완료 및 Chapter 3 환경 구축

## 목표
- Chapter 2 학습 마무리
- 추천 시스템 프로젝트 환경 구축
- Chapter 3(Simple Recommender) 시작

---

## 오늘 진행한 내용

### Chapter 2 학습 완료

#### Pandas 데이터 타입 변환

- `astype()`를 이용한 데이터 타입 변환
- `apply()`를 이용하여 문자열을 float로 변환하는 방법 학습
- 변환이 불가능한 값은 `NaN`으로 처리

#### 날짜 데이터 처리

- `pd.to_datetime()`을 이용한 날짜 데이터 변환
- 개봉일에서 연도(`year`) 추출

#### 데이터 정렬

- `sort_values()`를 이용한 오름차순/내림차순 정렬
- 개봉 연도와 매출(`revenue`) 기준으로 데이터 정렬

#### Pandas Series

- DataFrame의 하나의 컬럼은 Series 객체임을 학습
- `apply()`와 `astype()`는 Series 메서드임을 이해

---

### Chapter 3 시작

#### IMDb Top 250 Clone 프로젝트

Chapter 3에서는 IMDb Top 250을 기반으로 한 Simple Recommender를 구현한다.

추천 시스템 구현 과정은 다음과 같다.

1. 평가 기준(Metric) 선택
2. 추천 대상 조건 설정
3. 영화별 점수 계산
4. 점수 기준으로 영화 정렬

---

## 프로젝트 환경 구축

- 추천 시스템 프로젝트 생성
- Python 가상환경(.venv) 생성
- pandas, numpy 설치
- movies_metadata.csv 데이터셋 준비
- VS Code 실행 환경 설정

---

## 새롭게 알게 된 내용

- `revenue`는 영화의 총매출을 의미한다.
- `budget`은 영화 제작비를 의미한다.
- DataFrame의 한 컬럼은 Series 객체이다.
- `sort_values()`를 이용하면 원하는 기준으로 데이터를 쉽게 정렬할 수 있다.
- `pd.to_datetime()`을 이용하면 날짜 데이터를 쉽게 처리할 수 있다.

---

## 다음 목표

- Simple Recommender 구현 시작
- Metric(평가 기준) 계산 방법 학습
