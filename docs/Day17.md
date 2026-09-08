# Day17 — Funk SVD 추천 구조 이해 및 Evaluation 연결

## 1. 오늘 한 일 요약

새로운 모델을 구현하기보다는 **저번에 구현해둔 Funk SVD를 직접 이해하고, 기존 추천시스템 평가 구조까지 연결하는 작업**을 했다. 전체 MF Train 데이터(약 3,711만 건)로 학습·저장까지는 저번에 완료된 상태였고, 저번 코드 해석의 기준점은 `# 3. Top-N 추천` 직전까지였다.

```text
저번 학습된 Funk SVD 모델 확인
        ↓
Funk_SVD.py 코드 해석 재개 (recommend() 전체 구조 이해)
        ↓
main.py에서 SVD가 호출되는 과정 이해
        ↓
랜덤 사용자 Top-10 추천 테스트
        ↓
기존 evaluation.py와 MF 구조의 차이 확인
        ↓
MF용 evaluation 함수 추가 및 연결
        ↓
400명 평가 실행
        ↓
매우 낮은 baseline 성능 확인 (원인 분석은 다음 작업으로 보류)
```

오늘은 특히 **"GPT가 만들어준 코드를 그냥 돌리는 상태"에서 벗어나, 코드가 왜 이렇게 구성돼 있는지 따라가며 이해하는 것**에 시간을 많이 썼다.

---

## 2. Funk SVD 추천 구조 이해

### 전체 흐름

처음에는 코드가 기존 User-based/Item-based CF와 많이 달라 보여 낯설었지만, 뜯어보니 핵심은 단순했다.

```text
user_id
↓ Surprise 내부 user index
↓ 해당 사용자의 latent vector
↓ 모든 게임 latent vector와 점수 계산
↓ bias 추가
↓ 이미 본 게임 제거
↓ 점수가 가장 높은 Top-N 추출
↓ 내부 item id를 실제 app_id로 복원
```

### 핵심 코드 — latent vector 내적

```python
inner_uid = self.trainset.to_inner_uid(user_id)   # 실제 user_id → Surprise 내부 index
user_vector = self.model.pu[inner_uid]             # 학습된 사용자 latent vector
scores = self.model.qi @ user_vector               # 모든 게임 벡터와 내적
```

`n_factors=40`이므로 사용자 한 명은 **40차원 latent vector**로 표현된다는 것을 확인했고, 개념으로만 알던 $p_u^Tq_i$가 실제 코드에서 어떻게 쓰이는지 여기서 연결됐다.

### Bias까지 포함한 Surprise SVD 구조

단순 내적만 쓰는 게 아니라, Surprise SVD는 기본적으로 다음 형태를 사용한다.

$$ \hat r_{ui} = \mu + b_u + b_i + q_i^Tp_u $$

```python
scores = (
    self.trainset.global_mean
    + self.model.bu[inner_uid]
    + self.model.bi
    + scores
)
```

- `global_mean`: 전체 데이터의 평균 rating
- `bu`: 특정 사용자가 전체적으로 점수를 높게/낮게 주는 경향
- `bi`: 특정 게임이 전체적으로 높게/낮게 평가되는 경향
- `qi @ pu`: 그 사용자와 게임의 latent 취향 궁합

즉 MF 추천 점수는 단순히 "취향 벡터가 비슷한가"만 보는 게 아니라 **전체 평균 + 사용자 성향 + 게임 성향 + 개인화된 latent interaction**으로 만들어진다는 것을 코드로 확인했다.

### 이미 본 게임 제외

```python
scores[inner_iid] = -np.inf
```

왜 굳이 `-inf`를 넣는지 살펴보니, Top-N을 고를 때 절대 선택되지 않도록 만드는 간단한 방법이었다. `이미 본 게임 → score = -∞ → Top-N에서 자연스럽게 제외`되는 구조다.

### Top-N 추출 — argpartition vs argsort

```python
np.argpartition(...)   # 상위 N개 후보를 빠르게 찾음
np.argsort(...)        # 그 N개 안에서 실제 1등~N등 순서로 정렬
self.raw_item_ids[top_n_idx]   # Surprise 내부 item index → 실제 app_id 복원
```

두 함수의 역할이 다르다는 것(전체 정렬 없이 상위 후보만 빠르게 추출 → 그 안에서만 정확히 정렬)을 이해했다. 여기까지 `recommend()` 함수 전체(구현부 3-1~3-11)를 이해했다.

### main.py 연결 이해

전체 흐름은 `load_games() → load_mf_split() → FunkSVDRecommender 객체 생성 → fit(train_df) → user 선정 → Train interaction 추출 → recommend() → 게임 이름 merge → 추천 결과 확인 → Test 데이터 확인` 순서였다. `recommender.fit(train_df)`는 저장된 모델이 이미 있어 실제로는 재학습하지 않고 `저장된 Funk SVD 모델 발견 → 모델 불러오는 중... → 로드 완료` 과정을 수행한다는 것을 확인했다.

### `main()` 안/밖 scope 차이

평가 코드를 붙이다가 변수에 노란 밑줄이 생겼는데, 원인은 붙인 평가 코드가 `main()` 밖에 있었기 때문이었다. `main()` 내부에서 생성한 `train_df`, `test_df`, `recommender`는 `main()` 내부 변수이므로, 이를 사용하는 `run_mf_evaluation()`도 같은 `main()` 내부에 있어야 한다는 점에서 **함수 내부 변수의 scope**를 다시 확인했다.

### 랜덤 사용자 추천 테스트로 변경

기존에는 `user_id = int(input(...))`으로 직접 ID를 입력했지만, 실제 평가용 사용자 ID를 미리 알 필요는 없으므로 Test 사용자 중 하나를 랜덤으로 뽑도록 변경했다.

```python
user_id = (
    test_df["user_id"]
    .drop_duplicates()
    .sample(n=1)
    .iloc[0]
)
```

`drop_duplicates()`가 필요한 이유도 확인했다. Test는 interaction 단위라 같은 사용자가 여러 번 등장하는데(`user 100, user 100, user 100, user 200, user 300`), 중복 제거 없이 랜덤으로 뽑으면 interaction이 많은 유저가 뽑힐 확률이 높아진다. `drop_duplicates() → 사용자당 한 행 → 모든 사용자가 같은 확률로 선택`되도록 만들었다.

### 실제 추천 테스트 결과

랜덤으로 선택된 사용자는 `user_id: 4057666, Train 상호작용 게임 수: 9`였다. 추천 결과(Pixel Puzzle Makeout League, Templar Battleforce, Romance of Rome, Cleo - a pirate's tale, Letters - a written adventure, Escape Lala, Hyper Bounce Blast, Master of Magic Classic, The Legend of Bear-Truck Trucker, Loading Story)는 실제 Test positive(Unturned, Far Cry 3, Killing Floor, Trine Enchanted Edition)와 상당히 동떨어져 보였다. 다만 이 시점에서는 한 사람만 보고 모델 성능을 판단하지 않고 정량 평가로 넘어갔다.

---

## 3. Evaluation 연결 및 실제 결과

### 기존 Evaluation을 그대로 사용할 수 없는 이유

기존 CF 평가에서는 `run_evaluation()` 안에서 `train_test_split(...)`으로 다시 데이터를 분리했다. 하지만 MF는 이미 `mf_train.parquet` / `mf_test.parquet`을 별도로 생성했고 실제 SVD도 `mf_train`으로 학습했다. 따라서 MF 평가에서 다시 데이터를 나누면 **모델이 학습한 Train/Test와 평가에서 사용하는 Train/Test가 달라지는 문제**가 생긴다는 것을 발견했다. 그래서 기존 evaluation 전체를 버리는 대신 **기존 평가 지표는 유지하되, 이미 만들어진 MF Train/Test 구조에 맞게 데이터를 공급하는 부분만** 새로 만들었다.

### MF Evaluation 구조

새로 추가한 핵심 함수는 `build_mf_user_review_groups()`, `evaluate_mf_user()`, `run_mf_evaluation()`이었다. 기존 `stratified_sample_users()`, `print_evaluation_report()`는 그대로 재사용했다.

- **`build_mf_user_review_groups()`**: 기존에는 전체 `user_history`가 하나였으므로 interaction 수를 바로 계산할 수 있었지만, MF는 Train/Test가 이미 분리되어 있어 `train_counts + test_counts`로 원래 interaction 수를 복원했다(예: Train 14 + Test 6 = 원래 20). 이후 기존과 동일한 구간(10-15 / 16-25 / 26-45 / 46-78)으로 분류했다.
- **`run_mf_evaluation()`**: 400명 평가 전체를 관리하는 함수. 평가 대상 user ID만 `sampled_ids = set(sampled_users["user_id"])`로 모으고, 전체 3,700만 Train/400만 Test에서 해당 사용자들만 추출한 뒤 사용자별 게임 목록(`{user_id: [app1, app2, ...]}`)을 만들어 400명을 한 명씩 `evaluate_mf_user()`로 넘긴다. 즉 **400명을 관리하는 관리자 역할**로 이해했다.
- **`evaluate_mf_user()`**: 사용자 한 명을 평가하는 함수. 핵심은 `recommender.recommend(user_id=user_id, app_id_list=train_app_ids, top_n=10)`이며, 오늘 앞부분에서 이해한 `recommend()`가 여기서 실제 Evaluation 시스템과 연결된다. `사용자의 Train 기록 → Funk SVD → Top-10 → Test와 비교` 흐름이 된다.

### Hits와 기존 평가 지표 연결

```python
hits = len(recommended_ids & test_ids)
```

이 값이 기존부터 계속 쓰던 "이 유저에게 추천한 것 중 실제 정답을 몇 개 맞혔는가?"라는 평가의 핵심이며, 이를 바탕으로 Precision@10 / Recall@10 / Hit Rate@10 / NDCG@10을 계산한다. 즉 MF라고 해서 평가 철학이 완전히 달라진 것은 아니었다 — **추천을 만드는 방법만 달라지고, 최종적으로 Top-10과 정답을 비교한다는 구조는 그대로였다.**

### 실제 평가 결과 (400명, 그룹당 100명)

| Metric | Result |
|---|---:|
| Precision@10 | 0.0003 |
| Recall@10 | 0.0006 |
| Hit Rate@10 | 0.0025 |
| NDCG@10 | 0.0004 |
| Micro Precision@10 | 0.0003 |
| Micro Recall@10 | 0.0003 |
| Micro F1@10 | 0.0003 |
| 전체 Hits | 1 |
| 전체 추천 | 4,000 |
| Test 정답 | 3,273 |

400명에게 총 4,000개를 추천했는데 실제 Test 정답과 일치한 것은 **단 1개**였다. Review Group별로도 `10-15개 → Hit Rate 0.01`, `16-25개/26-45개/46-78개 → 0`으로, 유일한 Hit도 10~15 interaction 그룹에서 발생했다. 표본 400명에 단 1 hit라는 극단적인 결과이므로 지금 이 그룹 차이를 의미 있게 해석하지 않기로 했다.

### `ConstantInputWarning` 확인

평가 중 `ConstantInputWarning`(Pearson r: nan)이 발생했다. 원인은 모든 사용자에게 SVD가 정확히 10개씩 추천했기 때문이다(`n_recommended`가 10, 10, 10, ...으로 완전히 상수라 `n_games`와 상관계수를 계산할 수 없음). 이는 모델 에러가 아니며, 기존 User-based에서는 추천 근거가 없으면 10개를 못 채우는 경우가 있어 이 분석이 의미가 있었지만, 현재 MF에서는 이 분석의 의미가 거의 없다는 것도 정리했다.

### 오늘 발견한 중요한 평가 설계 문제

현재 MF 평가에는 `positive_only=True`가 들어가 있어 `is_recommended=True`인 Test 게임만 정답으로 취급한다. 반면 기존 User/Item CF 평가는 `test_ids = set(test_app_ids)`로 처리해 True/False 둘 다 Test 정답으로 들어갔다 — 즉 사용자가 싫다고 표시한 게임을 추천해도 기존 평가에서는 Hit가 될 수 있었다. 따라서 기존 결과(User-based P@10≈0.0545, Item-based P@10≈0.078)와 오늘의 Funk SVD P@10=0.0003을 **그대로 비교할 수 없다**는 것을 발견했다. 이 부분은 오늘 바로 수정하지 않고 다음 실험의 주요 문제로 남겼다.

---

## 4. 내가 직접 판단한 부분

- **"SVD도 기존 추천 함수처럼 만들면 되는 것 아닌가?"**: 처음엔 User/Item CF와 비슷한 입력 구조(입력 게임들 → 유사도 → 추천)를 기대했지만, MF는 이미 학습된 user/item latent vector를 쓰므로 `user_id → 학습된 latent vector → item vector와 점수 계산 → 추천` 구조로 근본적으로 다르다는 것을 이해
- **"추천할 때 게임 목록까지 내가 입력해야 하나?"**: 처음엔 `recommend(user_id, app_id_list)` 형태라 게임 목록도 알아야 하는 것처럼 느껴졌지만, 실제로 `app_id_list`는 취향을 새로 계산하기 위한 입력이 아니라 **이미 본 게임을 제거하기 위한 Train history**일 뿐이라는 것을 확인 — 추천 함수의 논리적 입력과 사용자 입력은 다를 수 있다는 점 정리
- **"기존 Evaluation을 그냥 불러오면 되는 것 아닌가?"**: 기존 함수 내부에서 다시 `train_test_split()`을 수행하기 때문에 이미 만들어놓은 MF split과 충돌한다는 것을 직접 발견 → 평가 지표 자체는 재사용하되 데이터를 공급하는 부분만 MF용으로 변경하는 쪽으로 정리
- **`positive_only=True`가 기존 평가와 조건이 다르다는 것**을 발견하고, 현재 수치를 기존 모델과 바로 비교하면 안 된다고 판단
- 결과가 매우 낮게 나왔음에도 **원인 분석과 튜닝을 오늘 바로 진행하지 않기로 결정**: `n_factors 튜닝, epoch 증가, learning rate 변경, regularization 변경, negative sampling, True/False 비율 조정, BPR 도입, implicit MF 도입, 다른 모델로 교체, 기존 CF 재평가` 등을 의도적으로 보류하고, 오늘의 목표를 "구현과 Evaluation 완료 + 결과 확보"까지로 한정 — 원인 분석과 모델 의사결정을 같은 날 섞지 않기로 함
- "Matrix Factorization은 이 프로젝트에 안 맞는다"고 성급히 결론 내리지 않고, 가능한 원인 후보(평가 조건 변화, True/False 86:14 불균형, 0/1 rating으로 SVD를 쓰는 방식, explicit rating prediction과 ranking 목적의 불일치, sparsity, item bias/popularity 영향, cold item, latent factor 수, epoch 수 등)를 나열만 해두고 검증은 다음으로 미룸

## 5. GPT(Claude)가 주로 담당한 부분

`FunkSVDRecommender` 클래스 구조 설계, 모델 저장/로드 구조 작성, `recommend()`의 vectorized Top-N 구현, 기존 `main.py`를 보존하면서 Funk SVD 연결, 랜덤 사용자 테스트 코드 작성, 기존 Evaluation을 분석해 MF와 충돌하는 지점 발견, `build_mf_user_review_groups()` / `evaluate_mf_user()` / `run_mf_evaluation()` 작성, `main.py`에 MF Evaluation 연결, scope 문제 수정, 평가 결과의 1차 해석, 기존 CF 평가와 MF 평가 조건이 다르다는 문제 발견 — 오늘도 코드 작성 자체에서는 GPT의 도움을 많이 받았다. 하지만 **각 코드의 의미를 질문하고 이해하는 과정과, 어떤 설계를 유지할지 결정하는 부분**은 계속 직접 확인하며 진행했다.

---

## 6. 오늘의 회고

오늘 가장 중요한 개념적 성장은 단순히 "SVD 코드를 이해했다"가 아니었다. **모델이 복잡하다고 추천이 좋아지는 것은 아니다.** User/Item CF보다 Matrix Factorization이 더 발전된 방식처럼 보여도, 데이터 표현·학습 objective·평가 기준·추천 문제의 형태가 맞지 않으면 성능이 오히려 훨씬 낮을 수 있다. 특히 현재 Surprise SVD는 본질적으로 **user-item rating 값을 예측**하는 모델인데, 이 시스템은 **수많은 unseen 게임 중 실제 선호할 게임을 Top-10 위로 올리는 ranking**이 목적이다. 이 차이가 실제 실패 원인인지는 다음 실험에서 확인해야 한다.

오늘 결과만 보면 "몇 시간 동안 SVD 구현하고 코드 이해했는데 P@10이 0.0003?"이라 허탈할 만하지만, 프로젝트 관점에서는 오히려 **모델 개발에서 중요한 단계가 처음 발생했다**고 정리했다. 지금까지는 `모델 구현 → 숫자 얻기`였다면, 이제는 `모델 구현 → 평가 → 실패 → 평가 설계 검토 → 데이터와 objective 검토 → 원인 가설 → 다음 모델/개선 전략 결정`이라는 실제 추천시스템 실험의 형태로 넘어왔다. 모델이 실패했다는 사실 자체보다 **왜 실패했는지를 구분해서 설명할 수 있느냐**가 다음 단계에서 더 중요하다.

> **오늘은 Funk SVD 코드를 복붙해 실행하는 단계에서 벗어나 latent vector가 실제 Top-N 추천으로 만들어지는 과정을 끝까지 이해하고 기존 평가 시스템에 MF를 연결했으며, 매우 낮은 baseline 결과를 통해 다음 단계가 단순 튜닝이 아니라 평가 공정성과 데이터·학습 objective·모델 적합성을 진단하는 과정이어야 한다는 문제의식을 얻었다.**

---

## 7. 현재 프로젝트 상태

```
[완료] Funk SVD recommend() 구조 전체 이해 (latent vector, bias 공식, seen-item 제거, argpartition/argsort, id 복원)
[완료] main.py 연결 이해 및 main() 안/밖 scope 확인
[완료] 랜덤 사용자 추천 테스트 (drop_duplicates 기반 균등 샘플링)
[완료] MF 전용 Evaluation 함수 구현 (build_mf_user_review_groups / evaluate_mf_user / run_mf_evaluation)
[완료] 400명 규모 MF 정량평가 실행
[발견] positive_only=True로 인한 기존 CF 평가와의 조건 불일치
[발견] n_recommended가 전원 10으로 고정되어 Pearson 상관 분석이 무의미해짐 (ConstantInputWarning)
[미완료] 낮은 성능(P@10=0.0003)의 원인 진단
[미완료] positive_only=False 재평가
[미완료] 모델 개선/교체 방향 결정 (SVD 유지 / BPR·implicit MF 검토 / 기존 CF 유지)
```

## 8. 다음 시간에 가장 먼저 해야 할 일

다음에는 **코드 추가부터 하면 안 된다.** 가장 먼저 해야 하는 것은 현재 결과가 왜 이렇게 낮았는지 진단하는 것이다. 확인할 항목:

```text
1. positive_only=True의 영향
2. False까지 정답에 포함하면 MF 성능이 얼마나 변하는지
3. 여러 유저에게 거의 같은 게임을 추천하고 있는지 (Top-10 overlap)
4. item bias가 추천을 지배하고 있는지
5. Test positive 중 Train vocabulary에 없는 게임의 비율
6. sparse interaction이 latent vector 학습에 미치는 영향
```

가장 간단한 첫 실험은 `positive_only=False`로 동일 SVD를 재평가해 기존 평가 방식과 가까운 조건에서 성능을 확인하는 것이다 — **재학습은 필요 없다.**

이 sanity check 이후 세 방향 중 하나로 의사결정한다.

- **A. Funk SVD 유지**: 현재 문제는 구현/평가/파라미터 문제로 보고 개선 실험 진행
- **B. Funk SVD는 baseline으로만 유지**: ranking에는 다른 MF 방식이 필요하다고 보고 BPR / implicit MF 검토
- **C. 현재 CF 계열이 더 적합**: 모델 간 공정 평가 후 프로젝트 시스템 관점에서 선택

어느 쪽이든 모델을 비교하려면 반드시 **같은 Train/Test, 같은 평가 사용자, 같은 True-only ground truth, 같은 Top-K** 조건으로 다시 평가해야 한다.