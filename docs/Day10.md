# Day10 회고 — User-based CF 시스템 통합과 데이터 누수 버그 발견

## 오늘 한 일

### 1. User-based CF를 실제 파이프라인에 통합
- `models/user_cf.py`에 `build_interaction_matrix()`, `UserBasedCFRecommender` 클래스 작성
- `evaluation.py`의 `run_evaluation(recommender, user_history, sampled_users, top_n)` 인터페이스에 맞춰, `recommend(app_id_list, top_n)` 메서드로 감쌈
- Content-based recommender와 동일한 인터페이스를 따르므로, `main.py`에서 recommender 생성부만 교체하면 나머지(평가 로직)는 그대로 재사용 가능하다는 구조 확정
- `evaluate_pipeline()` 함수는 내부적으로 `build_user_review_groups()` 호출 시 인자 시그니처가 어긋나 있어(`n_groups` vs `bins`/`labels`) 사용하지 않고, `build_user_review_groups` → `stratified_sample_users` → `run_evaluation` → `print_evaluation_report`를 직접 순서대로 호출하는 방식으로 진행

### 2. 실행 중 발생한 오류들과 해결
- `ModuleNotFoundError: No module named 'user_cf'` — 파일이 `models/` 폴더 안에 있어 `from models.user_cf import ...`로 경로 수정 필요
- `UnboundLocalError: user_history` — `user_history`를 정의하기 전에 `build_interaction_matrix(user_history)`를 먼저 호출해서 발생. `user_history = load_recommendations()`를 recommender 생성부보다 위로 이동시켜 해결
- 실행 시간: 392명 평가에 674초 소요 (content-based 691초와 비슷한 수준)

### 3. 이상하게 높은 평가 결과 발견 및 원인 진단
실행 결과:

| 지표 | 값 |
|---|---|
| Precision@10 | 0.6254 |
| Recall@10 | 0.6332 |
| Hit Rate@10 | 0.8418 |
| NDCG@10 | 0.6856 |

Content-based(Precision 0.0268)와 비교해 비정상적으로 높은 수치가 나옴. "말이 안 된다"고 스스로 판단하고 원인을 추적함.

**원인 (데이터 누수)**: `interaction_matrix`는 `user_history` 전체(train+test 미분리 원본)로 만들어졌기 때문에, 평가 대상 유저 본인의 전체 데이터가 이웃 후보 풀에 그대로 남아있었음. `recommend()`가 train_ids로 쿼리 벡터를 만들어도, 이 쿼리는 원본 자기 자신과 거의 동일하므로 코사인 유사도가 1에 가깝게 나와 **자기 자신이 최상위 이웃으로 뽑히는 문제**가 발생. 그 결과 예측 점수에 test 게임 정보(자기 자신의 미래 정답)가 그대로 새어 들어감.

- 근본 원인: 이전(Day08~09)에 400명을 일괄 처리할 때는 `user_sim[i, orig_idx] = 0`으로 자기 자신 유사도를 명시적으로 제거했으나, `recommend(app_id_list, top_n)` 인터페이스로 리팩토링하면서 "이 쿼리가 원래 몇 번 유저인지" 추적하는 정보가 빠져 이 제거 로직이 통째로 누락됨

### 4. 데이터 누수 버그 수정 (오늘 내로 완료)
- `UserBasedCFRecommender.recommend()`에 `exclude_user_idx=None` 파라미터 추가, 유사도 계산 직후 `sims[0, exclude_user_idx] = 0`으로 자기 자신 제외
  - CSR sparse matrix에 개별 원소를 대입하면 `SparseEfficiencyWarning` 발생 → `tolil()`로 변환해 대입한 뒤 다시 `tocsr()`로 되돌리는 방식으로 처리
- `evaluation.py`의 `evaluate_user`, `run_evaluation`에 각각 `user_idx`, `user_to_idx` 파라미터를 추가해, 유저별 반복 시 `user_to_idx.get(user_id)`로 인덱스를 구해 `recommend()`까지 전달되도록 연결
- `NameError: name 'exclude_user_idx' is not defined` 발생 — 함수 본문에서 그 이름을 쓰면서 정작 함수 시그니처(정의부)에는 파라미터로 추가하지 않아서 발생. 파라미터 추가로 해결
- `user_to_idx=None`을 기본값으로 둬서, content-based recommender로 되돌아갈 때는 이 값을 넘기지 않아도 기존 방식대로 동작하도록 호환성 유지

## 오늘 배운 핵심 개념

- **인터페이스 통일의 힘과 위험**: `recommend(app_id_list, top_n)`이라는 동일한 인터페이스를 따르면 recommender 종류(content-based/user-based)에 상관없이 평가 로직 전체를 재사용할 수 있음. 그러나 인터페이스를 맞추기 위해 리팩토링하는 과정에서 원래 있던 로직(자기 자신 제외)이 조용히 누락될 수 있다는 것도 함께 확인함
- **결과가 "너무 좋게" 나오는 것도 버그 신호**: 에러 없이 실행되고 지표가 오히려 개선된 것처럼 보이는 버그(silent failure)는 에러 메시지가 뜨는 버그보다 발견하기 어려움. "왜 이렇게 잘 나오지?"라는 의심이 유일한 탐지 수단
- **데이터 누수(data leakage)의 실제 사례**: train_ids로만 쿼리를 만들어도, 비교 대상 행렬 자체에 본인의 test 포함 원본이 남아있으면 여전히 누수가 발생할 수 있음. "입력을 train으로 제한하는 것"과 "비교 대상 풀에서 자기 자신을 제외하는 것"은 별개의 조치이며 둘 다 필요함

## 코드 분석

### 클래스와 `build_interaction_matrix()`의 관계

**1. `build_interaction_matrix()`의 역할**

학습 데이터(`train_df`)를 받아 추천기에 필요한 데이터를 생성함:
- `interaction_matrix`
- `user_to_idx`
- `game_to_idx` (예: `{730: 0, 570: 1, 440: 2}` — 게임 ID → 인덱스 매핑)
- `idx_to_game`

이 네 가지를 `return`으로 밖에 전달함.

**2. 추천기(`UserBasedCFRecommender`) 생성**

생성된 값들을 생성자에 전달:
```python
recommender = UserBasedCFRecommender(interaction_matrix, game_to_idx, idx_to_game)
```

**3. `self`에 저장**

생성자(`__init__`)에서 `self.interaction_matrix = interaction_matrix` 등으로 저장됨 — 전달받은 딕셔너리가 클래스 내부의 `self.game_to_idx`로 옮겨 담기는 것.

**4. `recommend()`에서 재사용**

이후 `self.game_to_idx`를 이용해 게임 ID를 인덱스로 변환함:
```python
col_idx = [self.game_to_idx[a] for a in app_id_list]
```
매번 새로운 딕셔너리를 만드는 게 아니라, 생성 시점에 한 번 만들어 저장해둔 딕셔너리를 계속 재사용하는 구조.

**`self`의 의미**: 클래스가 기억하고 있는 변수. `self.game_to_idx`는 추천기가 생성될 때 저장된 딕셔너리이며, `recommend()`뿐 아니라 클래스의 다른 메서드에서도 동일하게 접근 가능함.

**왜 딕셔너리를 쓰는가**: 게임 ID(`730`, `440`, `570`처럼 불규칙한 값)를 행렬에서 쓸 수 있는 연속된 인덱스(`0, 1, 2`)로 바꿔야 하는데, 딕셔너리를 쓰면 `game_to_idx[730]`처럼 O(1)로 빠르게 조회 가능함.

**전체 흐름**
```
train_df → build_interaction_matrix()
    ├── interaction_matrix 생성
    ├── user_to_idx 생성
    ├── game_to_idx 생성
    └── idx_to_game 생성
  → return
  → UserBasedCFRecommender() 생성자 호출
  → self.game_to_idx 등으로 저장
  → recommend() 호출 시 self.game_to_idx로 ID → 인덱스 변환
```

### 오늘 이해한 핵심
- `build_interaction_matrix()`는 추천기에 필요한 데이터를 생성하는 함수
- `game_to_idx`, `user_to_idx`는 ID를 행렬 인덱스로 바꾸기 위한 딕셔너리
- 생성된 딕셔너리는 `return`으로 전달되고, `__init__()`에서 `self`에 저장됨
- `recommend()`는 `self.game_to_idx`를 사용해 저장된 딕셔너리를 재사용함 (매번 새로 안 만듦)
- `self`는 클래스가 계속 기억하는 변수이므로, 다른 메서드에서도 자유롭게 접근 가능

## 학습 과정에 대한 메타 회고

- 오늘 유독 힘들었던 이유를 스스로 분석함: 지금까지는 개념 하나를 순서대로 쌓는 작업이었다면, 오늘은 알고리즘 이해·클래스 설계·인터페이스 통합·환경 설정(경로)·실행 순서 디버깅·성능 예측·숨은 버그 추적이 한 번에 섞인 하루였음. 서로 다른 종류의 사고를 오가는 것 자체가 피로도를 높인다는 것을 체감
- **소프트웨어 엔지니어링 vs AI 리서치의 역량 차이**를 스스로 정리함: 캐글처럼 정제된 데이터에서 모델 성능만 올리는 작업과, 오늘처럼 여러 컴포넌트를 통합하고 시스템으로 굴러가게 만드는 작업은 요구되는 역량이 다르다는 결론. 후자가 ML 엔지니어링에 가깝고, 본인이 지향하는 방향(recsys + LLM 연계)과 부합한다는 것도 함께 인식
- **역할 분담에 대한 원칙을 명확히 함**: "코드를 직접 타이핑하느냐"가 기준이 아니라, "이 시스템에서 무엇이 정상이고 무엇이 비정상인지 판단할 수 있는 모델이 스스로에게 있느냐"가 기준. 오늘 리팩토링과 인터페이스 설계는 AI가 수행했지만, 결과값(비정상적으로 높은 지표)을 보고 이상함을 감지하고 원인을 함께 추적한 것은 본인이 수행함 — 이 구분이 실제로 작동한 첫 사례로 기록할 만함

## 다음 회차(Day11)에 할 일

1. **수정 후 재평가**
   - 400명(또는 스킵 제외 인원) 재실행, 지표가 content-based 대비 현실적인 수준(비슷하거나 다소 낮은 수준)으로 돌아오는지 확인
   - 이전에 예측했던 "User-based가 리뷰 기반 sparse 데이터에서 구조적으로 약할 것"이라는 가설이 숫자로 실제 뒷받침되는지 확인
2. **K값 가벼운 비교**
   - K=30 vs K=100 정도만 비교해 개선 여지가 근본적으로 제한적인지 확인 (깊은 튜닝은 하지 않음)
3. **Item-based CF로 이동**
   - User-based와 대칭 구조(유사도 비교 축을 유저→아이템으로 전환)이므로 빠르게 구현 가능
   - 동일한 자기 자신 제외 로직이 아이템 쪽에서는 애초에 문제가 되는지(아이템은 자기 자신과 비교할 이유가 없으므로) 미리 점검
4. **네 가지 방식(Content-based / User-based / Item-based / Hybrid) 비교로 이어가기**