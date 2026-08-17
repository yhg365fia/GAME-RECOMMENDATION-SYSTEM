# Day12 회고 — User-Based CF 평가 보완, Sparsity 정량 검증 및 Item-Based CF 전환 결정

## 1. 오늘 한 일 요약

- Day11에서 확정한 User-Based CF 평가 코드를 재점검하며 `evaluate_user()`의 NDCG 계산에서 추천 순위가 보존되지 않는 구현 문제(`set` 사용)를 발견하고 수정
- 기존 Macro Average 중심 평가에 Micro Precision / Micro Recall / Micro F1을 추가하고, Hit Rate·NDCG는 Micro로 확장하지 않기로 판단한 근거를 정리
- 유저의 상호작용 수(`n_games`)와 `n_recommended`, Precision 사이의 관계를 Pearson correlation으로 정량 검증
- Macro/Micro, 그룹별 지표, 추천 개수 편차, 상관분석을 종합해 "User-Based CF가 Steam 데이터의 sparsity에 구조적으로 민감하다"는 Day11의 가설을 정량적으로 보강
- User-Based CF에 대한 세부 튜닝 대신 **Item-Based CF로 전환**하기로 최종 결정
- User-Based/Item-Based CF에서 **자기 자신과의 유사도 계산 자체**와 **자기 자신을 neighbor로 사용하는 것**이 서로 다른 문제라는 점을 정리하고, 두 모델 모두 "유사도 계산 직후 self-similarity를 0으로 만든 뒤 Top-K 선정"하는 구현 원칙으로 통일
- 오늘 분석 과정에서 GPT 피드백을 통해 스스로 과했던 해석 3가지를 확인하고 수정

## 2. 분석 과정 (시간순 — 가설과 깨달음)

**1) NDCG 구현 문제 발견**
Micro 지표를 추가하려고 `evaluate_user()` 코드를 다시 열어보다가, 추천 결과를 `set(result["app_id"])`로 바로 변환하고 있다는 걸 확인함. Precision/Recall은 포함 여부와 개수만 필요해 순서 무관이지만, NDCG는 "정답이 몇 번째 순위에 있는가"를 평가하는 지표라 순서가 반드시 보존되어야 한다는 걸 짚음. `recommended_list`(순서 보존, NDCG용)와 `recommended_ids`(집합, Precision/Recall용)로 역할을 분리해 수정.

**2) Macro만으로는 부족하다는 문제의식**
Day11에서 확인한 `n_recommended` 평균 7.28/10, Top-10 완전 채움 비율 54.4%라는 수치를 다시 보면서, "추천 후보를 1~2개만 생성한 유저와 10개를 다 채운 유저가 Macro 평균에서 동일한 가중치를 갖는 게 맞나"라는 의문을 제기함.

**3) Micro Precision/Recall/F1 설계**
`evaluate_user()`가 이미 반환하던 `hits`, `n_recommended`, `n_test`를 활용해, 개별 유저 평가 함수에 전체 통계를 섞지 않고 `eval_df`에서 raw count를 합산하는 방식으로 Micro Precision(`ΣHits/Σn_recommended`), Micro Recall(`ΣHits/Σn_test`)을 계산. Precision과 Recall의 균형을 하나의 값으로 보기 위해 Micro F1도 추가.

**4) Hit Rate·NDCG는 Micro로 확장하지 않기로 판단**
Hit Rate는 애초에 "유저 단위 성공 여부"의 평균이라 별도 Micro 버전이 의미가 크지 않다고 판단. NDCG는 유저별 DCG/IDCG로 이미 정규화된 값이라, 전체 유저의 DCG·IDCG를 그냥 합치면 test item이 많은 유저에게 가중치가 쏠리는 문제가 생겨 기존처럼 유저별 평균을 유지하기로 결정.

**5) 상관분석으로 sparsity 가설 정량화**
review_group별 평균 비교(Day11)만으로는 "패턴이 있어 보인다" 수준이었던 것을, `n_games ↔ n_recommended`, `n_games ↔ precision`에 대해 Pearson correlation을 계산해 개별 유저 단위에서도 같은 방향의 관계가 나타나는지 확인함.

**6) 해석 검증 — GPT 피드백을 통해 스스로 수정한 지점들**
결과를 해석하는 과정에서 아래 세 가지를 과하게 단정했다가, GPT 피드백을 받고 표현을 수정함 (자세한 내용은 4-2 참고):
   - "Micro가 sparsity 문제를 완화했다" → 모델 자체는 그대로, 집계 방식의 영향을 다른 관점에서 본 것일 뿐
   - "p<0.0001이니 sparsity가 원인임이 증명됐다" → r은 약~중간 수준이고, 상관관계는 인과관계를 증명하지 않음
   - "상관계수 하나로 User-Based 부적합을 판단" → 여러 독립적 관찰이 같은 방향을 가리킨다는 점을 근거로 삼는 쪽으로 재구성

**7) Self-Similarity 개념 재정리**
Item-Based CF로 넘어가기 전, "자기 자신과의 유사도를 애초에 계산하면 안 되는 것 아닌가"라는 의문에서 출발해, User-Based와 Item-Based 각각에서 self-similarity가 왜/어떻게 문제가 되는지 구조적으로 정리함 (5번 항목 참고).

## 3. 최종 평가 결과

### 3-1. 전체 평가 결과

| 구분 | Precision@10 | Recall@10 | F1@10 | Hit Rate@10 | NDCG@10 |
|---|---|---|---|---|---|
| Macro | 0.0545 | 0.0427 | - | 0.3011 | 0.0655 |
| Micro | 0.0626 | 0.0454 | 0.0526 | - | - |

### 3-2. 전체 Count

| 항목 | 결과 |
|---|---|
| 평가 성공 유저 | 362명 |
| 스킵 | 38명 |
| 전체 Hits | 165개 |
| 전체 추천 개수 | 2,637개 |
| 전체 Test 개수 | 3,634개 |
| 평균 추천 개수 | 7.28 / 10 |
| Top-10을 모두 채운 유저 비율 | 54.4% |
| 평가 시간 | 1,319.7초 (약 22분) |

### 3-3. Review Group별 결과

| Review Group | Precision | Recall | Hit Rate | NDCG | n_users |
|---|---|---|---|---|---|
| 10-15개 | 0.0242 | 0.0303 | 0.1039 | 0.0346 | 77 |
| 16-25개 | 0.0388 | 0.0451 | 0.2151 | 0.0530 | 93 |
| 26-45개 | 0.0482 | 0.0445 | 0.3478 | 0.0599 | 92 |
| 46-78개 | 0.0982 | 0.0483 | 0.4900 | 0.1061 | 100 |

### 3-4. Review Group별 추천 생성 개수

| Review Group | 평균 n_recommended | 최소 | 최대 | n_users |
|---|---|---|---|---|
| 10-15개 | 5.74 | 1 | 10 | 77 |
| 16-25개 | 6.47 | 1 | 10 | 93 |
| 26-45개 | 8.03 | 1 | 10 | 92 |
| 46-78개 | 8.54 | 2 | 10 | 100 |

### 3-5. Sparsity 가설 정량 검증 (Pearson Correlation)

| 비교 변수 | Pearson r | p-value |
|---|---|---|
| n_games ↔ n_recommended | +0.3096 | < 0.0001 |
| n_games ↔ precision | +0.2504 | < 0.0001 |

## 4. 지표 결과 분석

### 4-1. 내가 직접 분석/판단한 부분

- Macro→Micro 전환 시 Precision(0.0545→0.0626), Recall(0.0427→0.0454)이 모두 상승했고, 특히 Precision 상승폭이 더 컸다는 걸 확인하고, 추천을 1~2개만 생성한 유저가 Macro에서 유저 1명 몫의 동일 가중치를 갖던 것이 Micro에서는 실제 추천 item 수 기준으로 재분배되면서 발생한 차이라고 해석함.
- Pearson correlation 결과(r=0.31, r=0.25, 둘 다 p<0.0001)와 review_group별 추이(평균 추천 개수 5.74→8.54, Precision 0.024→0.098)가 같은 방향을 가리킨다는 걸 근거로, "interaction이 적을수록 안정적인 이웃 형성이 어려워 추천 후보가 부족해지고, 이것이 성능 저하로 이어진다"는 구조를 정리함.
- 여러 세부 튜닝(neighbor 수 변경, threshold 조정, fallback, smoothing 등)으로 User-Based를 계속 보완할 수도 있지만, 이 프로젝트의 목적이 "User-Based를 최고 성능으로 만드는 것"이 아니라 "이 데이터 구조에 어떤 방식이 적합한지 비교/이해하는 것"이라는 판단 하에, Item-Based CF로 전환하기로 최종 결정함.
- User-Based와 Item-Based 모두에서 self-similarity 계산 자체는 문제가 아니며, 문제는 자기 자신이 Top-K neighbor로 실제 추천 계산에 쓰이는 순간부터 발생한다는 점, 그리고 두 모델에서 그 문제의 성격이 다르다는 점(User-Based는 데이터 누수, Item-Based는 trivial self-similarity)을 스스로 구조화해서 정리함.
- Precision@10이 0.05 수준으로 낮아 보이는 것에 대해, Offline 평가에서 Test set이 "사용자가 좋아할 수 있는 모든 정답"이 아니라 "숨겨둔 일부 관측된 interaction을 얼마나 복원하는가"에 가깝다는 점을 재확인하고, 절대적인 숫자만으로 모델의 좋고 나쁨을 단정하면 안 된다고 판단함.

### 4-2. GPT 도움을 받아 확인/수정한 부분

- **NDCG 구현 버그 자체**: `set(result["app_id"])`로 순서 정보가 사라진다는 걸 스스로 먼저 발견한 게 아니라, Micro 지표를 추가하려고 기존 코드를 다시 리뷰하는 과정에서 GPT가 짚어줌. Precision/Recall과 NDCG가 요구하는 자료구조(집합 vs 순서 있는 리스트)가 다르다는 걸 이 피드백으로 명확히 함.
- **"Micro가 sparsity 문제를 완화했다"는 해석 수정**: 처음에는 Micro Precision/Recall이 오른 것을 보고 "Micro 방식이 sparsity 문제를 완화했다"고 표현하려 했으나, 추천 모델의 결과 자체는 전혀 바뀌지 않았다는 지적을 받고 "모델의 문제가 완화된 것이 아니라, 평가 집계에서 n_recommended 편차가 미치는 영향을 다른 관점으로 확인한 것"으로 표현을 수정함.
- **"p<0.0001이므로 증명됐다"는 표현 수정**: r(관계의 방향·크기)과 p-value(귀무가설에 대한 통계적 증거)의 역할이 다르다는 지적을 받음. r=0.31, r=0.25는 강한 상관이 아니라 약~중간 수준이며, 상관관계 자체가 인과관계를 증명하지 않는다는 점을 반영해 "sparsity가 원인임이 증명되었다" → "sparsity 취약성 가설과 일관된, 통계적으로 유의한 양의 관계가 확인되었다"로 수정함.
- **단일 지표로 결론 내리려던 관점 수정**: 상관계수 하나만으로 User-Based CF를 종료할 근거로 삼으려던 것에서, 추천 생성 안정성·그룹별 추이·Precision/Hit Rate/NDCG 추이·상관분석까지 여러 독립적 관찰이 같은 방향을 반복적으로 가리킨다는 것 자체를 근거로 삼는 방식으로 재구성함.

## 5. 오늘 정리한 핵심 개념 — Self-Similarity

### 5-1. User-Based CF

```
평가 대상 유저의 Train Vector
        ↓
전체 User와 cosine similarity 계산 (자기 자신도 포함되어 계산됨)
        ↓
자기 자신 similarity를 0으로 변경
        ↓
Top-K User 선정
```

자기 자신과의 유사도를 계산하는 것 자체는 문제가 아니다. 문제는 자기 자신을 Top-K neighbor로 사용하는 순간 발생한다. Interaction matrix의 본인 row에는 Test interaction까지 포함되어 있어서, 본인을 neighbor로 사용하면 Test 정보가 그대로 추천 점수에 유출된다. Day11에서 발견한 비정상적인 `Precision@10 = 0.6254`가 정확히 이 문제였다. 즉 User-Based에서 self-exclusion의 의미는 **본인 interaction이 이웃 정보로 사용되며 Test 정보가 유출되는 것을 막는 것**이다.

### 5-2. Item-Based CF

```
게임 A
        ↓
모든 Item과 cosine similarity 계산
        ↓
A ↔ A = 1.0
```

"A를 좋아했으니 A 같은 게임을 추천받고 싶다"와 `A↔A=1`은 다른 이야기다. `A↔A=1`은 단순히 "A는 A와 동일하다"는 자명한 값이며, 새로운 추천 정보를 만들지 못한다. 이를 제거하지 않으면 자기 자신이 항상 최고 유사도로 Top-K 한 자리를 차지해 자기 자신에게 다시 높은 가중치를 주게 된다. 즉 Item-Based에서 self-exclusion의 의미는 **trivial한 최고 유사도 item이 candidate neighbor에 포함되는 것을 막는 것**이다.

### 5-3. 공통점과 차이, 구현 원칙

두 모델 모두 "유사도 계산 → 자기 자신도 포함해 계산됨 → Top-K 선정 전 자기 자신을 0으로 제거 → 나머지로 Top-K 선정"이라는 순서를 따른다. 다만 자기 자신을 제거하지 않았을 때의 핵심 문제가 다르다.

| 모델 | 자기 자신을 제거하지 않았을 때 핵심 문제 |
|---|---|
| User-Based CF | 본인 interaction이 neighbor로 쓰이며 Test 정보가 유출되는 데이터 누수 |
| Item-Based CF | 자기 자신을 가장 유사한 item으로 선택해 trivial하게 자기 자신에 재가중치 부여 |

앞으로 두 모델 모두 다음 구현 원칙으로 통일하기로 결정:
```python
similarities = cosine_similarity(...)
similarities[self_idx] = 0
# 이후 Top-K 선정
```

## 6. 오늘 배운 핵심 정리

**평가 지표**
- Macro와 Micro는 단순 계산 방식 차이가 아니라 "어떤 단위에 동일한 가중치를 주는가"의 차이다.
- Micro 수치가 높아졌다고 모델 자체가 좋아진 게 아니다 — 동일한 결과를 다른 방식으로 집계한 것뿐이다.
- Micro Precision/Recall은 유저별 추천 개수 편차가 클 때 유용한 추가 관점을 제공한다.
- Hit Rate는 원래 유저 단위 성공률이라 별도 Micro가 현재 목적상 필요하지 않다.
- NDCG는 유저별로 이미 정규화된 지표이므로 기존 방식(유저별 평균)을 유지한다.
- NDCG 계산에서는 반드시 추천 순서를 보존해야 한다.

**통계**
- Pearson r → 관계의 방향과 크기, p-value → 상관이 0이라는 귀무가설에 대한 통계적 증거. 이 둘의 역할은 다르다.
- 작은 p-value가 곧 강한 상관관계를 의미하지 않는다.
- 상관관계는 인과관계를 증명하지 않는다.

**Collaborative Filtering**
- User-Based의 핵심 약점 중 하나는 sparse user에서 안정적인 neighborhood를 만들기 어렵다는 것이다.
- User-Based/Item-Based 모두 자기 자신과의 유사도를 계산하는 것 자체는 문제없다. 문제는 그것이 Top-K neighbor로 실제 계산에 쓰이는 순간부터다.
- User-Based는 이게 Test leakage로 이어지고, Item-Based는 trivial self-similarity 문제로 이어진다.

**모델 선택**
- 낮은 지표 하나만으로 모델을 버려서는 안 된다.
- 데이터 구조 → 모델 특성 → failure mode → 평가 결과를 연결해서 판단해야 한다.
- 반대로, 구조적 부적합이 충분히 확인된 모델을 끝없이 미세 튜닝하는 것도 좋은 의사결정이 아니다.

## 7. 향후 의심할 문제 (지금은 보류)

- **Popularity Bias**: 맞힌 165개 Hit가 실제로 Counter-Strike, PUBG, GTA, Apex 같은 인기 게임에 집중되어 있다면, 이게 "개인화를 잘해서" 맞힌 것인지 "많은 사람이 하는 게임이라 그냥 맞힌 것"인지 구분이 안 됨.
- **Near-Duplicate / Series Bias**: 본편-Edition-Beta-Test Server-후속작 등이 Hit에 집중되어 있다면 높은 지표의 의미를 다시 봐야 함.
- **Coverage / Personalization**: 몇몇 인기 게임에만 추천이 집중되는지, 서로 다른 유저에게 충분히 다른 추천을 주는지, catalog 활용 비율은 어느 정도인지.
- 이 문제들은 User-Based 하나만 붙잡고 깊게 분석하기보다, Content-Based → User-Based → Item-Based → Model-Based까지 구현한 뒤 각 모델의 성공/실패 패턴을 비교하면서 분석하는 것이 더 가치 있다고 판단해 지금은 보류함. 이 비교 결과를 이후 Hybrid 설계(여러 모델을 단순히 섞는 게 아니라, 각 모델에서 실제로 발견한 서로 다른 failure mode를 보완하는 모델)의 근거로 연결할 계획.

## 8. 성찰 및 느낀 점

- 오늘 진행한 분석(NDCG 버그 수정, Macro/Micro 비교, 상관분석, self-similarity 개념 정리)은 사실 프로젝트 진행을 위해 반드시 필요한 과정은 아니었다. "User-Based 성능이 별로다 → 이 데이터에 안 맞는 것 같다 → Item-Based로 넘어간다"로 바로 진행했다면 Item-Based, Matrix Factorization, Hybrid까지 훨씬 빠르게 갈 수도 있었다.
- 하지만 이번에는 처음 겪는 문제였기 때문에 이상 결과 발견 → 누수 디버깅(Day11) → 추천 개수 편차 발견 → Macro 방식 의심 → Micro 도입 → 상관분석 → 데이터 구조와 모델 구조 연결 → 다음 모델 선택까지 깊게 진행했고, 이 과정에서 "모델을 많이 구현하는 것"과 "추천시스템을 개발하는 것"이 같은 능력이 아니라는 걸 느꼈다. 왜 이 숫자가 나왔는지, 평가 자체가 올바른지, 데이터 누수가 없는지, 어떤 유저에게 실패하는지, 데이터 구조와 모델의 가정이 맞는지, 지금 모델을 더 튜닝할 가치가 있는지를 판단하는 능력이 핵심이라는 걸 체감함.
- 앞으로 모든 모델을 이렇게 깊게 분석하지는 않기로 함. 매번 처음부터 끝까지 통계적으로 깊게 검증하면 진행 속도가 지나치게 느려질 수 있기 때문에, 앞으로는 데이터 구조 파악 → 예상되는 failure mode 설정 → 최소 baseline → 결과 비교 순으로 먼저 진행하고, 예상과 일치하면 핵심 검증만 하고 이유를 기록한 뒤 다음 모델로, 예상과 다르면 추가 실험으로, 말이 안 되는 결과라면 진행을 멈추고 누수/버그/평가 자체부터 재검증하는 식으로 판단 기준을 세움.
- 경험이 쌓인다는 것은 실험을 안 하게 되는 게 아니라, **어떤 실험이 실제 의사결정에 필요한지 더 빠르게 판단하게 되는 것**이라고 정리하게 됨.

## 9. 다음에 해야 할 것 (To-Do)

### Item-Based Collaborative Filtering
- [ ] `models/itembase.py` 생성
- [ ] 기존 User × Item interaction matrix를 Item 관점으로 변환하는 구조 설계
- [ ] Item ↔ Item cosine similarity 구현
- [ ] similarity 계산 이후 각 기준 Item의 self-similarity를 0으로 변경
- [ ] self-exclusion 이후 Top-K similar item 선정
- [ ] 사용자의 Train item들을 추천 source로 사용
- [ ] 각 source item의 similarity를 이용한 candidate score 계산 방식 결정
- [ ] 이미 interaction한 Train item을 최종 추천에서 제거
- [ ] 기존 User-Based와 최대한 동일한 `recommend()` 출력 인터페이스 유지
- [ ] 기존 `evaluation.py` 그대로 재사용
- [ ] 동일한 Macro / Micro / Hit Rate / NDCG 기준으로 User-Based와 비교
- [ ] User-Based에서 발견된 `n_recommended` 부족 현상이 Item-Based에서 완화되는지 확인

### 이후
- [ ] Matrix Factorization / Model-Based Recommendation
- [ ] 주요 모델 구현 완료 후 동일한 평가 조건으로 비교
- [ ] Popularity Baseline
- [ ] Hit item의 Popularity Bias 분석
- [ ] 동일 시리즈 / Edition / Near-Duplicate 분석
- [ ] Catalog Coverage / Personalization 분석
- [ ] 모델별 failure mode 비교
- [ ] 최종 후보 모델은 충분한 표본 + 신뢰구간/유의성 검정으로 재검증
- [ ] failure mode를 기반으로 Hybrid Recommendation 목표 정의

## 10. 오늘의 한 문장 회고

Day11이 비정상적으로 높은 성능을 의심해 데이터 누수를 찾아낸 날이었다면, Day12는 정상화된 결과를 평가 방식과 데이터 구조의 관점에서 다시 해석하고, 스스로 과했던 해석을 수정하면서 User-Based CF의 한계를 설명 가능한 수준까지 이해한 뒤, 근거를 갖고 Item-Based CF로 전환하기로 결정한 날이었다.