# Day09 회고 — User-based CF 구현과 데이터 구조에 대한 이해

## 오늘 한 일

### 1. User-Item Interaction 행렬 설계
- `user_history` 데이터가 `userbase`와 동일 데이터임을 확인
- `is_recommended`(True/False)를 `+1`(좋아함) / `-1`(싫어함)로 매핑, interaction 없는 경우는 `0`(모름)으로 구분
  - 세 가지 상태(좋아함/싫어함/모름)를 값으로 구분해야 하는 이유: `False`를 `0`과 동일하게 처리하면 "명시적 부정"이라는 강한 정보를 버리게 됨
- `scipy.sparse.csr_matrix`로 (user_id → 인덱스, app_id → 인덱스) 매핑 후 sparse 행렬 구성
  - 결과: shape `(13,781,059, 37,610)`, nnz `41,154,773`
  - dense였다면 약 4TB, sparse라 약 500MB 수준 — 왜 sparse가 필수인지 체감

### 2. 평가 샘플링 재사용
- Content-based 때 썼던 stratified sampling 함수(`build_user_review_groups`, `stratified_sample_users`, `random_state=42`)를 그대로 재사용해 동일한 400명(구간별 100명)을 확보
- 평가 대상 필터링(`eligible_users`)과 학습용 interaction 행렬은 서로 다른 목적임을 구분 — 학습 행렬은 필터링 없이 전체 데이터로 구성

### 3. User-based CF 파이프라인 구현
1. 샘플 400명 vs 전체 유저 코사인 유사도 계산 (`dense_output=False` 필수 — dense로 하면 약 44GB)
2. 자기 자신과의 유사도 제거
3. Top-K(K=30) 이웃 추출 (`np.argpartition`으로 효율적으로)
4. 이웃들의 유사도 × interaction 값을 가중합(`neighbor_sims @ neighbor_matrix`)해 게임별 예측 점수 계산
5. 이미 상호작용한 게임 제외 후 top-N 추천

### 4. 버그 발견 및 수정
- 문제: 추천 결과 10개 중 다수가 `predicted_score = 0.0000`으로 나옴
- 원인: 이웃 30명 중 아무도 플레이하지 않은 게임까지 추천 후보에 억지로 채워 넣는 구조였음 (top-N을 무조건 다 채우는 로직)
- 수정: `predicted_scores > 0`인 후보만 남기고, 없으면 빈 리스트 반환하도록 `recommend_topn` 변경

## 오늘 배운 핵심 개념

- **왜 User-based CF가 이 데이터에서 근본적으로 약한가**: 유저 수(1,378만) 대비 게임 수(3.76만)가 극단적으로 비대칭. 유저당 평균 interaction은 3개, 반면 게임당 평균 interaction은 1,094개 → 유저 벡터가 게임 벡터보다 훨씬 sparse함. 이는 리뷰 기반 데이터(플레이 데이터보다 훨씬 희소)라는 점 때문에 더 심화됨
- **이 비대칭 때문에 Item-based CF가 구조적으로 더 적합하다는 결론**: Amazon이 item-based로 유명해진 이유와 동일한 논리
- **예측 점수 계산의 직관**: "나와 비슷한 친구들의 의견을 신뢰도(유사도)로 가중 평균" — 분자는 의견의 상쇄(정상), 분모는 절댓값 합이어야 신뢰도 자체가 상쇄되지 않음
- **Dense vs Sparse**: 값이 있는 칸만 저장하는지 여부의 차이, 메모리 사용량에 결정적 영향
- **Hybrid의 여러 형태**: Weighted(가중 평균, 지금 쓰는 방식) / Switching(상황별 전환) / Cascade(한 방식으로 거른 뒤 다른 방식으로 재정렬) / Mixed(결과를 나란히 제시) — 목적에 따라 다르게 선택됨
- **추천의 목적은 accuracy 하나가 아님**: Accuracy(유사한 것) / Novelty·Serendipity(새로운 발견) / Diversity / Social(소속감, 트렌드) — 서로 트레이드오프 관계이며, 실무에서는 여러 추천 슬롯을 목적별로 나눠 구성하는 경우가 많음
- **Model-based CF(ALS, SVD 등)의 위치**: 유사도를 실시간 계산하는 대신, 유저·아이템을 저차원 잠재 벡터로 학습해 내적으로 예측 — sparse 데이터에서 memory-based보다 강건함. 이번 프로젝트의 자연스러운 다음 확장 지점으로 남겨둠

## 심화 논의

### User-based가 이 데이터에 구조적으로 안 맞는 이유 (직접 도출)

유저당 게임 수를 10개 이상으로 제한하거나 K(이웃 수)를 늘리는 것은 증상 완화일 뿐, 근본 해결책이 아니라는 결론에 도달함.

- 근본 원인은 데이터가 **리뷰 기반**이라는 점 — 플레이 데이터보다 훨씬 희소함 (유저당 평균 3개)
- 유저-아이템 벡터가 서로 극단적으로 비대칭: 유저 수(1,378만) ≫ 게임 수(3.76만) → 게임 벡터가 유저 벡터보다 훨씬 밀도가 높음 (게임당 평균 1,094개)
- User-based는 유저 벡터끼리 비교하는 방식이라, 벡터 자체가 sparse하면 "의미 있는 이웃"을 찾을 확률 자체가 낮음 — K를 늘리거나 필터링 조건을 걸어도 원본 데이터의 밀도는 바뀌지 않으므로 한계가 그대로 남음
- 반대로 Item-based는 게임 벡터(=그 게임을 플레이한 유저들의 집합)를 비교하는데, 인기 게임일수록 이 벡터가 훨씬 촘촘해짐 → Steam처럼 유저 수가 압도적으로 많고 게임 수가 상대적으로 적은 구조에서는 Item-based가 근본적으로 더 적합함 (Amazon이 item-based로 유명해진 것과 같은 논리)
- 단, 비인기 게임(유저 벡터가 희박하거나 없는 경우)은 Item-based로도 추천이 어렵다는 한계는 남음 — 이건 Item-based 자체의 한계이지 User-based로 되돌아갈 이유는 아님

### Hybrid 전략의 종류와 선택 기준

Content-based가 실제로 Hybrid에 널리 쓰이는 이유는, <cite index="6-1">협업 필터링은 평점이 있어야만 추천 가능한 반면 content-based는 아이템의 특징을 이용하기 때문에 한 번도 평가되지 않은 아이템도 추천할 수 있기 때문</cite>. 이미지가 있으면 CNN 임베딩을, 텍스트/리뷰가 있으면 LLM 임베딩을 content feature로 쓰는 방향도 실무에서 통용됨.

Hybrid는 "가중치로 섞기"와 "따로 뽑기"가 둘 다 존재하며, 상황에 따라 다른 방식을 씀:

- **Weighted**: <cite index="4-1">협업 필터링과 content-based의 출력을 가중 평균으로 결합, 성능에 따라 비중 조절</cite> — 지금 프로젝트에서 채택한 방식
- **Switching**: <cite index="4-1">유저 행동이나 아이템 특성 등 기준에 따라 두 방식 사이를 전환</cite> (예: 신규 유저는 content-based, 상호작용이 쌓이면 협업 필터링으로 전환) — "인기 게임=item-based, 신작/비인기 게임=content-based"로 전환하는 것도 이 방식에 해당
- **Cascade**: <cite index="4-1">한 방식을 먼저 적용하고 그 결과를 다른 방식으로 다듬는 방식</cite> — 상위 추천기가 구분 못한 동점을 하위 추천기가 정리
- **Mixed**: <cite index="4-1">두 모델의 추천을 각각 보여주고 유저가 직접 고르게 함</cite>

즉 "정확도를 서로 보완하고 싶으면 weighted, 데이터 상태(신규/기존, 인기/비인기)에 따라 완전히 다른 로직이 필요하면 switching, 한쪽이 후보를 좁혀줘야 하면 cascade"로 정리됨.

### 추천의 목적은 정확도 하나가 아님

지금까지 만든 evaluation.py의 지표(Precision, Recall, NDCG)는 모두 **Accuracy**(과거에 좋아한 것과 비슷한 것을 얼마나 잘 맞추는가)를 재는 지표. 그러나 실무에서 추천의 목적은 이보다 다양함:

- **Accuracy** — 좋아하던 것과 비슷한 것
- **Novelty / Serendipity** — 새롭고 뜻밖의 발견
- **Diversity** — 추천 목록 자체의 다양성
- **Social** — 비슷한 사람들과의 공감, 트렌드 편승

이 네 가지는 서로 트레이드오프 관계임 (Accuracy를 극대화하면 필터버블 심화). 실무에서는 하나의 알고리즘이 다 담당하지 않고, 여러 추천 슬롯(예: "당신을 위한 추천" / "요즘 뜨는 콘텐츠" / "이런 것도 있어요")을 목적별로 나눠 각각 다른 알고리즘으로 채우는 방식이 일반적. 현재 프로젝트는 Accuracy만 측정하고 있다는 점을 향후 개선 방향으로 남겨둘 만함.

## 코드 분석

### 1. Sparse Matrix 생성 과정
```python
row_idx = userbase['user_id'].map(user_to_idx).values
col_idx = userbase['app_id'].map(game_to_idx).values
values = userbase['score'].values
```
- `map()`은 딕셔너리를 이용해 값을 변환함
- `.values`는 Pandas Series를 NumPy 배열로 변환함
- user_id, app_id를 0부터 시작하는 연속 인덱스로 바꾸는 이유는 Sparse Matrix를 만들기 위함
- 최종적으로 `(user_index, game_index, score)` 형태의 triplet 데이터가 만들어짐

### 2. `enumerate()`
```python
for idx, user in enumerate(unique_users):
```
- `enumerate`는 `(index, value)` 쌍을 반환함 (예: `0 userA`, `1 userB`, `2 userC`)
- 추천 시스템에서는 사용자 번호와 게임 번호를 인덱스와 함께 다뤄야 할 때 자주 사용됨

### 3. Cosine Similarity 계산
```python
user_sim = cosine_similarity(sample_matrix, interaction_matrix, dense_output=False)
```
- 샘플 사용자와 전체 사용자 간 코사인 유사도를 계산
- 결과는 (샘플 사용자 수 × 전체 사용자 수) 형태의 유사도 행렬

### 4. 자기 자신 유사도 제거
```python
for i, orig_idx in enumerate(sample_indices):
    user_sim[i, orig_idx] = 0
```
- 코사인 유사도에서 자기 자신 ↔ 자기 자신은 항상 1.0으로 가장 큼
- 하지만 추천에서는 자기 자신을 이웃으로 쓰면 안 되므로 0으로 바꿔 제외

### 5. Top-K 사용자 선택
```python
top_k_idx = np.argpartition(row_data, -k)[-k:]
```
- 유사도가 가장 큰 K명의 인덱스만 빠르게 찾음 (정렬은 하지 않음)
- `np.arange(len(row_data))`는 0부터 N-1까지 전체 인덱스를 의미
- 즉 K명만 필요할 때는 `argpartition`, 전체가 필요할 때는 `arange`

### 6. 예측 평점 계산
```python
predicted = weighted_sum / sim_sum if sim_sum > 0 else weighted_sum
```
- 정상적인 경우: 예측점수 = (유사도 × 평점)의 합 / 유사도의 합
- `sim_sum == 0`이면 0으로 나눌 수 없으므로 예외처리

### 7. 이미 플레이한 게임 제거
```python
pred_scores[already_played] = -np.inf
```
- 이미 플레이한 게임은 추천하면 안 되므로 점수를 `-∞`로 만들어 정렬 시 항상 가장 뒤로 보내 제외

### 8. Top-N 추천 생성
```python
top_n_idx = np.argpartition(predicted_scores, -n)[-n:]         # 점수 높은 N개 후보 빠르게 탐색
top_n_idx = top_n_idx[np.argsort(-predicted_scores[top_n_idx])]  # 후보를 점수 높은 순으로 재정렬
```
- 1차: 후보 찾기 (argpartition, 빠르지만 순서 없음)
- 2차: 찾은 후보만 argsort로 다시 정렬 (전체 정렬보다 훨씬 빠름)

## 학습 과정에 대한 메타 회고

- 코드를 AI에게 맡기고 해석 위주로 따라가는 방식이 정보량이 많아지면서 중간에 맥락을 놓치는 문제가 발생함 (변수/함수가 어디서 왔는지 추적이 안 되는 상태)
- 해결 방향: 한 번에 여러 함수를 이어붙여 실행하지 않고, 함수 하나 → 결과 확인 → 다음 함수 순서로 진행. 변수명과 의미를 짧게 메모해두는 방식 도입
- "시스템 전체가 한눈에 안 그려지는 느낌"은 아직 파이프라인의 조각(Item-based, Hybrid, 최종 비교)이 다 채워지지 않았기 때문이라는 진단 — 첫 프로젝트 완주 후 두 번째 유사 프로젝트에서 패턴 인식 속도가 빨라질 것으로 기대
- 추천시스템이 학부 ML/DL 커리큘럼과 이질적으로 느껴지는 지점 정리: implicit feedback 해석 문제, ranking 기반 평가, 유저-아이템 관계형 데이터 구조, sparse가 기본값이라는 점

## 다음 회차(Day10)에 할 일

1. **User-based CF 평가 마무리** (깊게 튜닝하지 않고 최소한으로)
   - 수정된 `recommend_topn`으로 400명 전체 추천 리스트 생성
   - `evaluation.py`로 Precision@K, Recall@K, Hit Rate@K, NDCG@K 계산
   - K값은 1~2개만 가볍게 비교(예: 30 vs 100)하여 "늘려도 근본적 한계는 그대로"라는 점만 확인
   - 몇 명 샘플만 정성평가 (추천 개수 편차, 이상 케이스 확인 정도)
2. **Item-based CF 구현**
   - User-based와 대칭 구조(유사도 계산 대상만 유저 → 아이템으로 전환)이므로 상대적으로 빠르게 진행 가능
   - 아이템 벡터 밀도가 유저 벡터보다 훨씬 높다는 게 오늘 확인한 핵심 근거이므로, User-based보다 지표가 개선되는지 직접 확인
3. **Hybrid(Weighted) 구현**
   - Item-based + User-based 가중 평균, α 값을 evaluation.py로 튜닝
4. **네 가지 방식(Content-based / User-based / Item-based / Hybrid) 통합 비교**
   - 같은 sampled_users, 같은 evaluation.py 기준으로 metric 정리
   - 왜 각 방식의 metric이 다르게 나오는지 오늘 배운 데이터 구조(유저-아이템 비대칭)로 설명 작성
5. (여유가 되면) Model-based CF(ALS 등) 도입을 다섯 번째 비교 대상으로 검토