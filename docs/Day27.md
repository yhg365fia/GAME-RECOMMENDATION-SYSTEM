# Day27 — 최종 E_HR 확정과 Steam 게임 추천시스템 프로젝트 종료

> Day27은 이 프로젝트의 마지막 작업일이다.  
> 4-Retriever 기반 Hybrid 구조를 최적화하고, 평가 방식의 문제를 수정한 뒤 Candidate / Retriever Ratio / Feature / XGBoost를 체계적으로 검증했다. 이후 Feature Ablation, 3-Seed Stability, LTR1000 전체 재학습, Final400 정량평가, 신규 사용자 정성평가까지 마무리하면서 **최종 모델 E_HR을 확정하고 프로젝트를 종료했다.**

---

## 1. 최종 결과 한눈에 보기

### 최종 추천 구조

```text
User History
    ↓
┌─────────────────────────────┐
│ Item-CF        46 candidates │
│ BPR            20 candidates │
│ Content-Based  15 candidates │
│ User-CF        20 candidates │
└─────────────────────────────┘
    ↓
Candidate Union
(max 101)
    ↓
Full15 Feature Engineering
    ↓
XGBoost LambdaMART
    ↓
Top-10 Recommendation
```

| 항목 | 최종 값 |
|---|---:|
| Final Model | **E_HR** |
| Candidate Size | **101** |
| Item-CF | **46** |
| BPR | **20** |
| Content-Based | **15** |
| User-CF | **20** |
| Features | **15** |
| Ranker | **XGBoost LambdaMART** |
| Final Training | **LTR1000 전체** |
| Trees | **54** |
| Final Evaluation | **고정 Final400 1회** |
| 프로젝트 상태 | **완료** |

### Final400 최종 성능

| P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Candidate Recall |
|---:|---:|---:|---:|---:|---:|
| **0.096250** | **0.127890** | **0.570000** | **0.139857** | **0.069833** | **0.272015** |

| Mean Log Popularity@10 | Novelty@10 | Hits | Runtime |
|---:|---:|---:|---:|
| **10.252428** | **10.354832** | **385** | **약 6.51분** |

최종 모델은 모델 탐색에 사용하지 않은 고정 Final400에서 평가했다. 이 결과를 마지막 공식 정량 결과로 기록하고, 이후 모델을 다시 튜닝하지 않았다.

---

## 2. Day27에서 끝낸 핵심 작업

Day26까지는 XGBoost Learning-to-Rank를 Hybrid Pipeline에 연결하고 Runtime Prototype을 만든 상태였다. Day27에서는 그 구조를 실제로 반복 실험할 수 있도록 최적화한 뒤, 최종 Architecture 선택과 검증까지 한 번에 마무리했다.

```text
4-Retriever Feature Generation 병목
        ↓
Item/User CF Batch + Cache 최적화
        ↓
4-Retriever + XGBoost 정상 실행
        ↓
Candidate / Ratio / Feature / XGB 탐색
        ↓
평가 방식 문제 발견
        ↓
Actual Top-10 기준으로 평가 수정
        ↓
Constrained Optuna
        ↓
Candidate Upper-Bound Sweep
        ↓
Final Joint Optuna
        ↓
Feature Ablation
        ↓
3-Seed Stability
        ↓
E_HR 최종 확정
        ↓
LTR1000 전체 Final Training
        ↓
Final400 Leakage Guard
        ↓
Final400 정량평가
        ↓
신규 사용자 정성평가
        ↓
프로젝트 종료
```

오늘의 핵심은 단순히 XGBoost 점수를 더 올린 것이 아니라, **평가 → 탐색 → 검증 → 최종 학습 → 최종 평가**를 하나의 일관된 실험 과정으로 완성한 것이다.

---

## 3. 4-Retriever Pipeline 최적화

Day26 마지막에는 다음 구조를 시도하고 있었다.

```text
Item 50
+ BPR 20
+ Content 15
+ User 15
    ↓
Candidate ≈ 100
    ↓
XGBoost Full14
    ↓
Top-10
```

문제는 Item-CF를 Retriever로 추가하면서 Feature Generation 시간이 급격하게 증가했다는 점이었다. 당시에는 사용자 1명당 약 90~117초까지 걸려, LTR 사용자와 평가 사용자를 반복 처리하기 어려웠다.

### Item-CF

평가 사용자마다 같은 Item similarity를 반복 계산하던 구조를 다음과 같이 바꿨다.

```text
평가 사용자 Train History
        ↓
Unique Source Item 수집
        ↓
필요한 Item만 Batch Neighbor Precompute
        ↓
Cache
```

필요한 Unique Source Item 약 **5,239개**만 대상으로 Neighbor를 미리 계산하고, Sparse Matrix 연산을 Batch 단위로 수행했다.

### User-CF

기존 구조:

```text
사용자 1명
→ 전체 User Matrix와 Similarity
→ 다음 사용자
→ 다시 전체 User Matrix와 Similarity
```

개선 구조:

```text
8 Users
   ↓
Sparse Batch Similarity
   ↓
Top-K User Neighbor
```

그 결과 User-CF 계산은 대략 **0.4~0.6초/user** 수준까지 감소했다.

### Checkpoint / Resume

장시간 Feature Generation이 중단되어도 처음부터 다시 하지 않도록 다음 결과를 저장했다.

- 이미 처리한 사용자
- Feature 결과
- Item Neighbor Cache
- User-CF Batch 결과

프로젝트 후반의 핵심 병목은 XGBoost 학습 자체보다 **Retrieval, Similarity 계산, Feature Generation**이라는 점을 확인했다.

---

## 4. 4-Retriever + XGBoost 성능 확인

최적화 후 다음 구조를 실제로 끝까지 실행할 수 있었다.

```text
Item50 + BPR20 + Content15 + User15
        ↓
Candidate Union
        ↓
Full14
        ↓
XGBoost Ranker
        ↓
Top-10
```

당시 대표 성능은 다음과 같았다.

| Metric | 결과 |
|---|---:|
| Precision@10 | **0.0938** |
| Recall@10 | **0.1270** |
| Hit Rate@10 | **0.5775** |
| NDCG@10 | **0.1326** |
| MAP@10 | **0.0635** |
| Mean Log Popularity@10 | **10.3590** |
| Novelty@10 | **10.2011** |
| Hits | **375** |

기존 Item-Based Ranker 기준선은 P@10 0.0872, R@10 0.1110, HR@10 0.5400, NDCG@10 0.1216, Hits 348이었다. 즉 Item-CF까지 Retriever로 포함한 4-Retriever 구조가 실제로 성능 향상에 기여했다.

### 당시 주요 Feature Importance

| 순위 | Feature | Importance |
|---:|---|---:|
| 1 | `bpr_rank` | **0.2651** |
| 2 | `retriever_count` | **0.1858** |
| 3 | `bpr_score_norm` | 0.1254 |
| 4 | `item_score_norm` | 0.1061 |
| 5 | `item_popularity` | 0.0846 |
| 6 | `item_rank` | 0.0674 |
| 7 | `content_rank` | 0.0596 |
| 8 | `content_score_norm` | 0.0397 |
| 9 | `user_rank` | 0.0264 |
| 10 | `user_score_norm` | 0.0258 |
| 11 | `user_interaction_count` | 0.0143 |

특히 `bpr_rank`와 `retriever_count`가 강하게 나타났다. 여러 Retriever가 같은 아이템을 동시에 선택했다는 정보 자체가 Ranking Signal로 작동하고 있었다.

---

## 5. 가장 중요한 문제 발견 — Candidate 내부 NDCG

Candidate Size / Retriever Ratio / Feature / XGBoost Parameter를 함께 탐색하기 위해 Optuna를 도입했다.

초기 Joint Optuna에서는 Candidate Size가 작아질수록 Validation NDCG가 계속 높아지는 이상한 결과가 나타났다.

예를 들어 한 초기 설정은 다음과 같았다.

```text
Candidate Size = 60
Item    = 27
BPR     = 12
Content = 15
User    = 6

Candidate Recall ≈ 0.1965
Candidate 내부 NDCG ≈ 0.5125
```

Candidate를 40 근처까지 줄이면 NDCG가 0.60대까지 올라갔다. 하지만 Candidate Recall은 계속 감소했다.

처음에는 Candidate가 적으면 Ranker의 문제가 쉬워져서 그런 것이라고 생각할 수 있었지만, 평가 정의를 다시 확인하면서 실제 원인을 찾았다.

### 문제

기존 평가는 사실상 다음 질문에 가까웠다.

> Retriever가 Candidate 안에 넣은 아이템만 놓고 XGBoost가 얼마나 잘 정렬했는가?

예를 들어 실제 Positive가 10개인데 Candidate 40이 그중 2개만 가져오고 그 2개를 1~2위에 배치하면 Candidate 내부 NDCG는 매우 높아질 수 있다.

반대로 Candidate 100이 Positive 5개를 가져오면 더 많은 정답을 확보했음에도 Ranking 문제가 어려워져 내부 NDCG가 낮아질 수 있다.

즉 **Retriever가 놓친 실제 Test Positive가 충분히 패널티되지 않는 평가 artifact**가 있었다.

### 평가 원칙 수정

> **Candidate 내부만 평가하지 않고, 전체 Test Positive를 정답으로 두고 실제 최종 Top-10 결과를 평가한다.**

이후 Candidate Size, Retriever Ratio, Optuna 실험은 모두 이 기준으로 통일했다.

이번 프로젝트에서 가장 중요한 방법론적 수정 중 하나였다.

---

## 6. Actual Top-10 Candidate Trade-off

평가 방식을 수정한 뒤 Candidate Size를 다시 비교했다.

| Size | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Candidate Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.0987 | 0.0974 | 0.5645 | 0.1313 | 0.0623 | 0.1342 |
| 40 | 0.1020 | 0.1027 | 0.5910 | 0.1356 | 0.0637 | 0.1577 |
| 50 | 0.1043 | 0.1050 | 0.5995 | 0.1378 | 0.0646 | 0.1789 |
| 55 | 0.1057 | 0.1060 | 0.6025 | 0.1392 | 0.0656 | 0.1888 |
| 65 | 0.1068 | 0.1073 | 0.6065 | **0.1398** | **0.0660** | 0.2061 |
| 70 | **0.1072** | **0.1077** | **0.6090** | 0.1395 | 0.0657 | **0.2142** |

평가를 고치자 이전의 “Candidate가 작을수록 좋다”는 패턴이 사라졌다.

```text
Candidate Size 증가
        ↓
Candidate Recall 증가
        ↓
P / R / NDCG도 일정 구간까지 개선
        ↓
이후 Plateau
```

여기서 명확해진 것은 다음과 같다.

> **Candidate Recall 최대화와 Final Top-10 성능 최대화는 같은 문제가 아니다.**

Retriever는 충분한 정답을 가져와야 하지만, Candidate가 지나치게 커지면 Ranker가 더 많은 Negative와 유사 Candidate를 구분해야 한다.

---

## 7. Constrained Optuna와 Upper-Bound 확인

평가 기준을 수정한 뒤 Optuna도 다시 설계했다.

### Guardrail

Search 단계:

```text
Precision@10 >= 0.1065
Recall@10    >= 0.1065
```

정식 5-Fold 재평가:

```text
Precision@10 >= 0.1070
Recall@10    >= 0.1070
```

Guardrail을 통과한 후보 안에서 **Macro NDCG@10을 Primary Objective**로 사용했다.

주요 5-Fold 결과:

| Trial | Size | Ratio | P@10 | R@10 | NDCG@10 | MAP@10 | Candidate Recall |
|---:|---:|---|---:|---:|---:|---:|---:|
| **64** | 86 | user20 | 0.1090 | 0.1087 | **0.1445** | **0.0688** | 0.2376 |
| 32 | 84 | user20 | **0.1104** | **0.1115** | 0.1439 | 0.0679 | 0.2349 |
| 52 | 85 | user20 | 0.1085 | 0.1087 | 0.1433 | 0.0679 | 0.2359 |
| 0 | 55 | baseline | 0.1083 | 0.1085 | 0.1432 | 0.0679 | 0.1903 |
| 73 | 89 | user20 | 0.1101 | 0.1105 | 0.1428 | 0.0671 | **0.2421** |

상위 Trial의 Candidate Size가 84~89에 몰렸기 때문에 Search Upper Bound 90 자체가 결과를 제한하고 있을 가능성을 확인했다.

### Candidate Upper-Bound Sweep — 80~120

| Size | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Candidate Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 80 | 0.1067 | 0.1077 | 0.6080 | 0.1398 | 0.0661 | 0.2312 |
| 90 | 0.1076 | 0.1085 | 0.6075 | 0.1397 | 0.0659 | 0.2442 |
| 100 | 0.1084 | 0.1094 | 0.6120 | 0.1419 | **0.0671** | 0.2574 |
| **110** | **0.1087** | **0.1107** | **0.6173** | **0.1422** | 0.0669 | 0.2697 |
| 120 | 0.1081 | 0.1098 | 0.6090 | 0.1406 | 0.0663 | **0.2798** |

Candidate Recall은 120까지 계속 증가했지만 Ranking Metric은 약 100~110에서 정점을 형성했다.

이 결과를 바탕으로 최종 Joint Search 범위를 **90~130**으로 확장했다.

---

## 8. Final Joint Optuna — 최종 후보 A~E

최종 탐색 조건:

| 항목 | 설정 |
|---|---|
| Candidate Size | 90~130, step 1 |
| Trials | 80 |
| Search | 3-Fold + Pruning |
| Search Guardrail | P≥0.1065 / R≥0.1065 |
| Recheck | 상위 후보 5-Fold |
| Final Guardrail | P≥0.1070 / R≥0.1070 |
| Primary Objective | Macro NDCG@10 |
| LTR Users | 1,000 |
| Final400 | 탐색에 사용하지 않음 |

최종적으로 서로 다른 강점을 가진 후보 A~E를 남겼다.

| 후보 | Trial | Size | I/B/C/U | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | NDCG Std | Candidate Recall |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **A** | 2 | 100 | 45/20/15/20 | 0.1097 | 0.1111 | 0.617 | **0.1460** | **0.0697** | 0.00762 | 0.2557 |
| **B** | 57 | 92 | 42/18/23/9 | 0.1106 | **0.1119** | 0.613 | 0.1451 | 0.0691 | 0.00775 | 0.2445 |
| **E** | 60 | 101 | 46/20/15/20 | 0.1096 | 0.1110 | **0.622** | 0.1447 | 0.0684 | 0.00576 | 0.2571 |
| **D** | 67 | 93 | 42/19/23/9 | 0.1100 | 0.1117 | 0.621 | 0.1446 | 0.0681 | 0.00475 | 0.2466 |
| **C** | 3 | 105 | 47/21/16/21 | **0.1110** | 0.1115 | 0.620 | 0.1446 | 0.0681 | **0.00387** | **0.2616** |

후보 이름은 각각의 상대적 특성을 나타내기 위해 붙였다.

- A: NDCG / MAP
- B: Recall
- C: Precision / Candidate Recall / Fold Stability
- D: Stability
- E: Hit Rate

여기서 중요한 점은 **Optuna의 best trial 하나를 그대로 최종 모델로 선택하지 않았다는 것**이다.

---

## 9. Feature Ablation

최종 후보별 Candidate Size, Retriever Ratio, XGBoost Parameter, LTR1000, 5-Fold 조건을 고정하고 **Feature만 변경**했다.

### A — Full16 유지

| 설정 | NDCG | MAP | P | R |
|---|---:|---:|---:|---:|
| **Full16** | **0.1460** | **0.0697** | 0.1097 | 0.1111 |
| - Item Flag | 0.1438 | 0.0679 | 0.1097 | 0.1113 |
| - PopAffinity | 0.1448 | 0.0689 | 0.1095 | 0.1112 |
| - RetrieverCount | 0.1444 | 0.0680 | 0.1101 | 0.1116 |
| Full14 | 0.1449 | 0.0691 | 0.1090 | 0.1102 |

A는 Ranking 품질을 대표하는 후보였기 때문에 Full16을 유지했다.

### B / C / D — `retriever_count` 유지

| 후보 | Full NDCG | -RetrieverCount | ΔNDCG | ΔHits |
|---|---:|---:|---:|---:|
| B | 0.1451 | 0.1434 | -0.00164 | -17 |
| C | 0.1446 | 0.1414 | **-0.00321** | **-27** |
| D | 0.1446 | 0.1432 | -0.00140 | -4 |

세 후보 모두 `retriever_count` 제거 시 성능이 하락했다.

### E — PopAffinity 제거

| 설정 | P | R | HR | NDCG | MAP | Hits |
|---|---:|---:|---:|---:|---:|---:|
| Full16 | 0.1096 | 0.1110 | 0.622 | 0.1447 | 0.0684 | 1096 |
| - Item Flag | 0.1086 | 0.1109 | 0.615 | 0.1439 | 0.0683 | 1086 |
| **- PopAffinity** | **0.1105** | 0.1113 | **0.623** | **0.1450** | **0.0692** | **1105** |
| - RetrieverCount | 0.1095 | **0.1115** | 0.617 | 0.1436 | 0.0677 | 1095 |
| Full14 | 0.1101 | **0.1115** | 0.620 | 0.1435 | 0.0677 | 1101 |

`user_popularity_affinity`는 제거했을 때 P, HR, NDCG, MAP, Hits가 모두 개선됐다.

따라서 E는 최종적으로 **Full15**를 사용했다.

---

## 10. 3-Seed Stability

Feature Ablation 이후에는 Architecture를 더 바꾸지 않고 Split Seed에 따른 변동만 확인했다.

```text
Seed 1 = 42
Seed 2 = 20260926
Seed 3 = 20260927
```

### NDCG@10

| 후보 | Seed 1 | Seed 2 | Seed 3 | 3-Seed 평균 | Seed Std |
|---|---:|---:|---:|---:|---:|
| **E** | 0.145021 | 0.146414 | 0.144031 | **0.145155** | 0.000977 |
| **A** | 0.146001 | 0.144614 | 0.144561 | 0.145059 | **0.000667** |
| D | 0.144648 | 0.147546 | 0.142834 | 0.145009 | 0.001941 |
| C | 0.144629 | 0.147526 | 0.142488 | 0.144881 | 0.002064 |
| B | 0.145058 | 0.144670 | 0.143299 | 0.144342 | 0.000755 |

### 3-Seed 평균 성능

| 후보 | Avg P@10 | Avg R@10 | Avg HR@10 | Avg NDCG | Avg MAP |
|---|---:|---:|---:|---:|---:|
| **E** | **0.11053** | 0.11139 | **0.62467** | **0.14516** | **0.06911** |
| A | 0.10977 | 0.11098 | 0.61800 | 0.14506 | 0.06909 |
| D | 0.11013 | **0.11153** | 0.62133 | 0.14501 | 0.06867 |
| C | 0.11013 | 0.11112 | 0.62167 | 0.14488 | 0.06877 |
| B | 0.11007 | 0.11141 | 0.61700 | 0.14434 | 0.06875 |

E는 **3-Seed 평균 Macro NDCG@10 1위**였고 평균 HR@10도 가장 높았다. A는 가장 낮은 Seed Std를 보였지만 Primary Metric의 평균값은 E가 근소하게 앞섰다.

따라서 최종 모델은 **E_HR**로 확정했다.

---

## 11. 최종 E_HR Architecture

### Candidate Retriever

```text
Item-CF        46
BPR            20
Content-Based  15
User-CF        20

Total Candidate Budget = 101
```

### Full15 Features

```text
# Model Scores
item_score_norm
bpr_score_norm
content_score_norm
user_score_norm

# Model Ranks
item_rank
bpr_rank
content_rank
user_rank

# Retriever Agreement
retriever_count

# Candidate Source
is_bpr_candidate
is_content_candidate
is_user_candidate
is_item_candidate

# Context
item_popularity
user_interaction_count
```

`user_popularity_affinity`는 Ablation 결과 제거했다.

### Final XGBoost

Cross Validation에서 `best_iteration`의 Median이 53이었기 때문에 최종 모델은 54개 Tree로 학습했다.

```python
max_depth = 5
min_child_weight = 2
learning_rate = 0.042841786327162734
subsample = 0.8621568879089554
colsample_bytree = 0.7010833568732288
reg_lambda = 0.8133078460477237
reg_alpha = 0.49527026495550774
n_estimators = 54
```

CV는 모델 선택과 Tree 수 결정에 사용하고, 실제 최종 Ranker는 **LTR1000 전체로 한 번 다시 학습**했다.

```text
LTR1000 전체
    ↓
E_HR Candidate
    ↓
Full15
    ↓
XGBRanker
    ↓
xgb_ranker_final_e_hr.json
```

---

## 12. Final400 Leakage Guard

최종 평가 데이터가 학습 과정에 섞이지 않도록 코드 수준에서 확인했다.

```text
LTR1000 User IDs
∩
Final400 User IDs
=
0
```

Overlap이 한 명이라도 있으면 실행을 중단하도록 만들었다.

Final400은 Candidate / Feature / XGBoost 탐색과 모델 선택에 사용하지 않았고, **최종 구조가 완전히 확정된 뒤 한 번만 공식 평가**했다.

---

## 13. Final400 최종 정량평가

최종 E_HR은 모델 탐색에 사용하지 않은 **고정 Final400 사용자 400명**에서 한 번만 공식 평가했다.

평가 사용자는 Train Review Count를 기준으로 네 그룹에서 각각 100명씩 구성했다.

| Review Group | Users |
|---|---:|
| 10~15개 | 100 |
| 16~25개 | 100 |
| 26~45개 | 100 |
| 46~78개 | 100 |
| **Total** | **400** |

### 13.1 최종 전체 결과

| Metric | Final E_HR |
|---|---:|
| Precision@10 | **0.096250** |
| Recall@10 | **0.127890** |
| Hit Rate@10 | **0.570000** |
| NDCG@10 | **0.139857** |
| MAP@10 | **0.069833** |
| Candidate Recall | **0.272015** |
| Mean Log Popularity@10 | **10.252428** |
| Novelty@10 | **10.354832** |
| Total Hits | **385** |

최종 정량평가 Runtime은 Batch / Cache 최적화 이후 약 **6.51분**이었다.

### 13.2 Review Group별 결과

| Review Group | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Candidate Recall | Mean Log Pop. | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10~15개 | 0.054 | 0.143333 | 0.34 | 0.126247 | **0.080001** | **0.292667** | 10.450322 | 10.069126 |
| 16~25개 | 0.072 | 0.137786 | 0.50 | 0.121104 | 0.060314 | 0.275440 | 10.233539 | 10.382083 |
| 26~45개 | 0.107 | 0.129295 | 0.67 | 0.142498 | 0.064291 | 0.274840 | 10.288286 | 10.302900 |
| 46~78개 | **0.152** | 0.101148 | **0.77** | **0.169579** | 0.074725 | 0.245115 | **10.037565** | **10.665218** |

활동량이 많은 사용자일수록 **Precision / HR / NDCG가 높아지는 경향**이 나타났다. 반면 Recall과 Candidate Recall은 감소했는데, 활동량이 많은 사용자일수록 Test Positive 수 자체가 많아 Top-10만으로 전체 정답을 회수하기 어려워지는 영향도 함께 존재한다.

특히 46~78개 그룹은 `P@10 = 0.152`, `HR@10 = 0.77`, `NDCG@10 = 0.169579`로 가장 높은 Ranking 성능을 보였다. 동시에 Mean Log Popularity는 가장 낮고 Novelty는 가장 높아, 활동량이 많은 사용자에게 상대적으로 더 다양한 아이템을 추천하는 경향도 확인됐다.

### 13.3 이전 구조와 비교

| 모델 | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Hits |
|---|---:|---:|---:|---:|---:|---:|
| Item-Based Ranker Hybrid | 0.0872 | 0.1110 | 0.5400 | 0.1216 | - | 348 |
| 초기 4-Retriever + Full14 XGBoost | 0.0938 | 0.1270 | **0.5775** | 0.1326 | 0.0635 | 375 |
| **Final E_HR** | **0.096250** | **0.127890** | 0.570000 | **0.139857** | **0.069833** | **385** |

Item-Based Ranker Hybrid와 비교하면 Final E_HR은:

```text
Precision@10   +0.00905
Recall@10      +0.01689
HR@10          +0.03000
NDCG@10        +0.018257
Hits           +37
```

만큼 상승했다.

초기 Full14 XGBoost와 비교하면 Final E_HR은 HR@10이 `0.5775 → 0.5700`으로 소폭 낮아졌지만, 최종 선택의 Primary Metric인 NDCG@10은 `0.1326 → 0.139857`, MAP@10은 `0.0635 → 0.069833`, Hits는 `375 → 385`로 개선됐다.

즉 최종 모델은 단순히 Hit 여부만 최대화한 구조가 아니라, **전체 정답을 고려하면서 Top-10 내부의 Ranking 품질을 더 높인 구조**로 정리할 수 있다.

### 13.4 Popularity / Novelty 해석

최종 전체 결과는:

```text
Mean Log Popularity@10 = 10.252428
Novelty@10             = 10.354832
```

였다.

이 두 지표는 모델 선택을 다시 수행하기 위한 Objective가 아니라, **최종 E_HR이 어떤 추천 성향을 보이는지 확인하기 위한 보조 분석 지표**로 사용했다.

최종 모델과 Final400 결과를 확인한 뒤에는 이 수치를 보고 다시 튜닝하지 않았다. 이 결과를 프로젝트의 공식 최종 정량평가로 확정했다.

---
## 14. Final400 평가 속도 문제와 해결

처음에는 실제 Runtime과 동일하게 사용자마다 `engine.recommend(...)`를 호출했다.

```python
for user in final400:
    engine.recommend(...)
```

25/400 사용자까지 처리하는 데 10분 이상 걸렸다.

문제는 E_HR Architecture 자체가 아니라, 매 사용자마다 User-CF / Item-CF / Content / BPR / Feature를 반복 계산하는 **Evaluation Implementation**이었다.

### 최종 평가 구조

```text
Fixed Final400
    ↓
400명의 Train History 추출
    ↓
Item-CF Unique Source Item Batch Precompute
    ↓
User-CF Batch Sparse Similarity
    ↓
BPR User Vector
    ↓
Content Profile
    ↓
E_HR Candidate
    ↓
Full15 Feature Cache
    ↓
XGBoost Batch Prediction
    ↓
Top-10
    ↓
Final Metrics
```

주요 Cache:

```text
models/saved_model/results/final400_batch_cache_e_hr/

final400_e_hr_i46_b20_c15_u20_ft3_features.parquet
final400_feature_checkpoint.parquet
final400_processed_users.npy
```

최적화 후 Final400 전체 평가를 약 **6.51분**에 완료했다.

---

## 15. 신규 사용자 Cold-Start

신규 사용자는 기존 BPR User ID가 없기 때문에 플레이한 게임의 Item Embedding을 이용해 초기 User Vector를 만든다.

```text
Played Games
    ↓
BPR Item Embedding Mean
    ↓
Initial User Vector
    ↓
User-only BPR Fine-Tuning
(Item Factors Fixed)
    ↓
Item46 + BPR20 + Content15 + User20
    ↓
Full15
    ↓
E_HR XGBoost
    ↓
Top-10
```

기존 Item Factors는 고정하고 신규 User Factor만 Fine-Tuning한다.

이 방식으로 **신규 사용자도 기존 최종 E_HR Pipeline에 연결**했다.

---

## 16. 신규 사용자 최종 정성평가

신규 사용자는 실제 Test Ground Truth가 없기 때문에 Precision / Recall / NDCG / MAP 같은 Offline Metric을 계산할 수 없다.

따라서 정성평가에서는 실제 추천 Top-10과 함께 **Input / Output Mean Log Popularity와 Novelty**를 기록해 다음을 확인했다.

- 서로 다른 취향이 함께 들어왔을 때 어떤 신호가 유지되는가
- 입력 신호를 강화하면 약한 취향이 살아나는가
- 인기 게임과 Long-tail 게임이 함께 들어왔을 때 어느 쪽이 더 강하게 작용하는가
- 매우 비인기인 niche 취향도 추천 결과에 유지될 수 있는가

### 16.1 Q1~Q8 최종 결과

| ID | 시나리오 | Input Mean Log Pop. | Input Novelty | Output Mean Log Pop.@10 | Output Novelty@10 | 핵심 관찰 |
|---|---|---:|---:|---:|---:|---|
| **Q1** | 현실적 혼합 취향 | 11.1557 | 9.0512 | 11.4322 | 8.6523 | 액션 / 오픈월드 / 스토리 취향은 잘 잡았지만 약한 Cute / Animal 신호는 묻힘 |
| **Q2** | 완전히 다른 2장르 | 11.6154 | 8.3880 | 12.0015 | 7.8310 | Shooter 쪽으로 강하게 기울고 Cozy 신호는 약함 |
| **Q3** | 완전히 다른 3장르 | 11.7046 | 8.2593 | 11.8263 | 8.0837 | Shooter와 Strategy는 보존됐지만 Cozy는 약함 |
| **Q4** | 3장르 강화 2+2+2 | 11.3063 | 8.8339 | 11.7858 | 8.1421 | 각 취향을 여러 게임으로 강화하자 Strategy 신호가 더 명확해짐 |
| **Q5** | Popular Only | 12.3299 | 7.3571 | 11.7232 | 8.2325 | 입력보다 약간 덜 인기 있고 더 참신한 추천으로 이동 |
| **Q6** | Long-Tail Only | 3.1355 | 20.6860 | 4.1095 | 19.2614 | Geneforge 계열을 상위에 배치하며 명확한 niche signal을 포착 |
| **Q7** | Popular 3 + Long-Tail 3 | 7.7899 | 13.9390 | 11.8511 | 8.0479 | 같은 개수로 섞어도 Popular signal이 Long-tail signal을 강하게 압도 |
| **Q8** | 테마를 부분 통제한 Popular / Long-Tail 혼합 | 7.9331 | 13.7245 | 11.6286 | 8.3689 | 장르 차이만으로 설명하기 어려운 Popularity 자체의 강한 영향이 확인됨 |

### 16.2 Q1 — 실제 취향과 가까운 입력

Q1은 실제 사용자 취향과 가까운 혼합 입력으로 구성했다.

추천 Top-10 중 다음 네 게임은 실제로 이미 플레이한 게임이었다.

```text
God of War
Marvel's Spider-Man Remastered
Grand Theft Auto V Legacy
Cyberpunk 2077
```

이 사례에서는 **Action / Open World / Story 중심의 주 취향을 상당히 잘 포착**했다.

반면 Cute / Animal 계열은 상대적으로 약하게 나타났다. 즉 여러 취향이 동시에 들어왔을 때 모든 취향을 균등하게 보존하는 구조라기보다는, 현재 데이터와 Retriever / Ranker에서 강한 신호를 가진 취향이 더 크게 반영됐다.

이 결과는 한 명의 실제 취향 사례에 대한 확인이므로 전체 사용자 정확도로 일반화하지는 않았다.

### 16.3 Q2~Q4 — Multi-Interest 반응

Q2에서는 `Counter-Strike 2 + Stardew Valley`처럼 거의 다른 두 취향을 넣었을 때 Shooter 쪽 신호가 더 강했다.

Q3에서는 Shooter / Cozy / Strategy를 한 게임씩 넣었고, Shooter와 Strategy는 어느 정도 유지됐지만 Cozy는 약했다.

이를 확인하기 위해 Q4에서는 각 취향을 2개씩 넣어 입력 Signal을 강화했다.

```text
Shooter
- Counter-Strike 2
- PUBG

Cozy
- Stardew Valley
- Slime Rancher

Strategy
- Civilization VI
- Total War: WARHAMMER III
```

그 결과 Strategy 계열이 Q3보다 더 명확하게 추천에 나타났다.

이 실험을 통해 특정 취향이 결과에 나타나지 않는 이유가 항상 “모델이 그 취향을 이해하지 못해서”인 것은 아니며, **입력 Signal 자체가 다른 취향보다 약하기 때문일 수도 있음**을 확인했다.

### 16.4 Q5~Q8 — Popularity Bias / Long-Tail

Q5에서는 Popular Only 입력의 Mean Log Popularity가 `12.3299`였지만 출력은 `11.7232`로 낮아졌고, Novelty는 `7.3571 → 8.2325`로 상승했다.

즉 인기작만 입력하더라도 완전히 더 인기 있는 게임으로만 수렴하지는 않았다.

반대로 Q6 Long-Tail Only에서는 입력 Novelty가 `20.6860`, 출력 Novelty가 `19.2614`로 매우 높은 수준을 유지했다.

실제 Top-10은 다음과 같았다.

| Rank | Recommendation |
|---:|---|
| 1 | Geneforge 2 |
| 2 | Geneforge 5: Overthrow |
| 3 | Geneforge 4: Rebellion |
| 4 | Geneforge 1 |
| 5 | The Wizard's Pen |
| 6 | Talismania Deluxe |
| 7 | AstroPop Deluxe |
| 8 | The Clockwork Man |
| 9 | Trials 2: Second Edition |
| 10 | The Room |

특히 Geneforge 계열이 1~4위를 차지하면서 **입력 취향이 충분히 명확하면 비인기 / niche 취향도 유지할 수 있음**을 확인했다.

하지만 Q7에서 Popular 3개와 Long-tail 3개를 동일한 개수로 섞자:

```text
Input Mean Log Popularity  = 7.7899
Output Mean Log Popularity = 11.8511

Input Novelty              = 13.9390
Output Novelty             = 8.0479
```

으로 크게 인기 쪽으로 이동했다.

Q8에서도 장르 차이를 어느 정도 통제했지만:

```text
Input Mean Log Popularity  = 7.9331
Output Mean Log Popularity = 11.6286

Input Novelty              = 13.7245
Output Novelty             = 8.3689
```

가 나타났다.

따라서 최종 정성평가에서는 다음 두 특성이 동시에 확인됐다.

> **명확한 Long-tail 단일 취향은 유지할 수 있지만, Popular 신호와 함께 들어오면 Ranking 단계에서 Popular 신호가 훨씬 강하게 작용한다.**

### 16.5 추가 Animal Theme 실험

세부적인 `Animal` 테마가 얼마나 반영되는지도 추가로 확인했다.

| ID | 입력 | 상태 | 결과 |
|---|---|---|---|
| Q9 | ANIMAL WELL | 실패 | Metadata에는 있으나 BPR item mapping에 없어 신규 사용자 벡터 생성 불가 |
| Q10 | Stray | 성공 | Animal 자체보다 Story / Adventure 유사성이 더 강하게 나타남 |
| Q11 | ANIMAL WELL + Stray | 참고용 | ANIMAL WELL이 BPR universe 밖이라 결과가 사실상 Stray 단독과 동일 |

Stray 단독 추천에서는 `Ori and the Will of the Wisps`, `Cult of the Lamb`처럼 동물/생물 캐릭터와 관련된 결과가 일부 있었지만, 전체 Top-10은 주로 Story / Adventure 계열이었다.

이 추가 실험에서는 두 가지 한계를 확인했다.

1. 세부적인 시각적·테마적 속성보다 **협업 신호와 장르 / 게임플레이 유사성**이 더 강하게 작동했다.
2. **BPR 학습 universe 밖의 아이템은 현재 신규 사용자 representation에 직접 사용할 수 없다.**

Q9 / Q11은 이 데이터 한계 때문에 Animal 취향 강화 여부를 검증한 유효한 실험으로 보지 않았다.

### 16.6 정성평가 최종 해석

최종 정성평가를 통해 E_HR은 다음과 같은 성격을 보였다.

```text
강한 주 취향
→ 비교적 잘 유지

여러 취향
→ 동일 비중으로 보존되지 않음

약한 취향
→ 입력 Signal을 강화하면 일부 복구 가능

명확한 Long-tail 단일 취향
→ niche 추천 유지 가능

Popular + Long-tail 혼합
→ Popular signal이 강하게 우세

BPR universe 밖 아이템
→ 현재 Cold-Start BPR representation에 직접 사용 불가
```

즉 이 모델은 “모든 취향을 균등하게 섞는 Diversity 모델”이라기보다, **현재 관측된 플레이 기록에서 강한 Relevance Signal을 찾아 Top-10을 구성하는 Hybrid Ranker**라고 보는 것이 정확하다.

정량평가에서는 Final400으로 전체 Ranking 성능을 확인했고, 정성평가에서는 신규 사용자에 대해 복합 취향과 Popularity / Long-tail 반응을 확인했다. 두 평가를 모두 완료한 뒤 모델을 추가로 수정하지 않았으며, 이 결과를 프로젝트의 최종 평가로 확정했다.

---
## 17. 실행 파일 역할 분리

프로젝트 후반에는 하나의 `main.py`에 모든 기능을 넣지 않고 역할을 분리했다.

| 파일 | 역할 |
|---|---|
| `main.py` | 실제 신규 사용자 Runtime 추천 |
| `main_final_e_hr_with_qualitative.py` | 신규 사용자 Q1~Q8 정성평가 |
| `main_final_e_hr_with_quantitative.py` | Final400 정량평가 |

실험용 코드와 최종 Runtime / Evaluation 코드를 분리하면서 프로젝트의 최종 구조를 명확하게 만들었다.

---

## 18. 이번 프로젝트에서 가장 크게 배운 것

### 1. Retriever와 Ranker는 다른 문제다

Retriever는 Relevant Item을 Candidate 안으로 가져오는 역할이고, Ranker는 Candidate의 최종 순서를 정한다.

Candidate Recall이 높아도 Final Top-10이 반드시 좋아지는 것은 아니고, 반대로 Ranker가 좋아도 정답이 Candidate 안에 없다면 복구할 수 없다.

### 2. Candidate Recall은 Final Objective가 아니다

Size 120은 Size 110보다 Candidate Recall이 높았지만 NDCG는 낮았다.

> **더 많은 정답을 Candidate에 넣는 것 ≠ 더 좋은 Top-10 추천을 만드는 것**

Candidate Recall은 Retriever를 진단하는 Metric으로 사용하고, 최종 선택은 실제 Top-K Ranking Metric과 함께 해야 한다.

### 3. 평가 정의가 모델 선택보다 먼저다

Candidate 내부 NDCG를 그대로 믿었다면 Candidate 40~60 근처의 구조를 최종 모델로 선택했을 수도 있다.

> **잘못 정의된 Metric을 정확하게 최적화하면 잘못된 모델을 매우 효율적으로 찾을 수 있다.**

이번 프로젝트에서 가장 중요한 경험 중 하나였다.

### 4. Optuna Best Trial은 최종 답이 아니다

실제 과정은 다음과 같았다.

```text
Cheap CV Search
    ↓
Top Candidates
    ↓
Stronger CV Recheck
    ↓
Feature Ablation
    ↓
Multi-Seed Stability
    ↓
Final Model
```

Optuna는 답을 대신 결정하는 모델이 아니라, 좋은 탐색 영역과 후보를 빠르게 찾는 도구에 가까웠다.

### 5. Feature는 많다고 항상 좋은 것이 아니다

`user_popularity_affinity`는 이론적으로 유용해 보였지만 E에서는 제거했을 때 성능이 좋아졌다.

반대로 `retriever_count`는 여러 후보에서 제거할 때 성능이 반복적으로 하락했다.

```text
Feature Idea
    ↓
Implementation
    ↓
Ablation
    ↓
Contribution Check
    ↓
Keep / Remove
```

### 6. Seed 하나만 보면 판단이 달라질 수 있다

Seed1에서는 A가 가장 좋았지만 Seed2에서는 D/C가 강했고, Seed3에서는 다시 A/E가 올라왔다.

그래서 단일 Split 결과가 아니라 3-Seed 평균과 변동성까지 확인했다.

### 7. 추천시스템에서는 Feature Generation 비용이 매우 크다

XGBoost 학습 자체보다 다음 작업이 더 비쌌다.

- Item Similarity
- User Similarity
- Content Profile
- Candidate Retrieval
- Feature Generation

프로젝트 후반에는 **Cache, Batch, Sparse Operation, Precomputation, Checkpoint**가 실험 가능성을 결정했다.

---

## 19. 내가 직접 판단하고 결정한 부분

이번 프로젝트에서 단순히 코드를 구현하는 것보다 다음 판단을 직접 내리는 과정이 중요했다.

- 느린 4-Retriever 구조를 바로 버리지 않고 Batch / Cache로 최적화해 검증했다.
- Candidate Size / Retriever Ratio를 수동으로 몇 개만 비교하지 않고 범위 탐색했다.
- 작은 Candidate의 비정상적인 NDCG 상승을 그대로 믿지 않고 Evaluation 정의를 다시 확인했다.
- Candidate 내부 NDCG를 버리고 전체 Test Positive 기준 Actual Top-10 평가로 수정했다.
- Precision / Recall Guardrail을 둔 뒤 Macro NDCG@10을 Primary Objective로 유지했다.
- Candidate Recall을 최종 목표가 아니라 Diagnostic Metric으로 사용했다.
- Search 결과가 Upper Bound에 몰리자 범위를 더 넓혀 확인했다.
- Optuna 1위 하나를 바로 선택하지 않고 A~E 후보를 남겨 5-Fold / Ablation / Stability를 진행했다.
- Feature Ablation에서는 다른 조건을 고정하고 Feature만 바꿨다.
- E에서 `user_popularity_affinity`를 제거하고 Full15를 채택했다.
- 단일 Seed가 아니라 3-Seed Stability까지 확인했다.
- 기존에 정한 Primary Metric인 Macro NDCG@10 기준을 마지막까지 유지했다.
- 3-Seed 평균 NDCG 1위인 E_HR을 최종 모델로 확정했다.
- Final400에 들어간 뒤에는 모델을 더 변경하지 않았다.
- Runtime / Qualitative / Quantitative 코드를 분리했다.
- Final400의 느린 평가 방식도 기다리는 대신 Batch / Cache 구조로 다시 설계했다.

---

## 20. GPT(AI)가 도움을 준 부분

반복 구현과 실험 자동화에서는 AI 도움을 적극적으로 사용했다.

주요 지원 영역:

- 4-Retriever Item/User CF Batch Optimization
- Cache / Checkpoint / Resume 구조
- Candidate Size / Ratio 탐색 자동화
- Optuna Objective 코드 구성
- Precision / Recall Guardrail
- Actual Top-10 Metric 계산
- Candidate Upper-Bound Sweep
- Final Joint Optuna
- A~E 후보 결과 정리
- Feature Ablation
- 3-Seed Stability 평가
- Final E_HR XGB Training
- Leakage Guard
- Runtime / Qualitative / Quantitative 코드 분리
- Q1~Q8 정성평가 자동화
- Final400 Batch / Cache Evaluation

다만 다음과 같은 핵심 판단은 실험 결과를 직접 확인하면서 결정했다.

```text
이 결과가 이상한가?
무엇을 통제해야 하는가?
어떤 Metric을 Primary로 볼 것인가?
어떤 실험이 추가로 필요한가?
언제 탐색을 멈출 것인가?
최종 모델을 어떤 기준으로 선택할 것인가?
```

---

## 21. 최종 회고

이번 프로젝트는 처음에는 User-Based, Item-Based, SVD, BPR 같은 개별 추천 알고리즘의 성능을 비교하는 프로젝트로 시작했다.

하지만 진행하면서 질문이 바뀌었다.

```text
초기
"어떤 추천 알고리즘이 가장 좋은가?"
```

에서

```text
후반
"어떤 Retriever 조합이 Candidate를 만드는가?"
"어떤 Feature가 실제 Ranking에 도움이 되는가?"
"평가 방식이 모델을 올바르게 평가하고 있는가?"
"Validation 결과가 Split이 바뀌어도 유지되는가?"
"이 Pipeline을 반복 실행할 수 있는가?"
```

로 바뀌었다.

단일 모델 비교에서 시작해 Reranking, Multi-Retriever, Rank Fusion, Item-Based Ranker를 거쳤고, 최종적으로 **4-Retriever + XGBoost Learning-to-Rank** 구조까지 도달했다.

특히 가장 기억에 남는 부분은 높은 점수가 나왔을 때 그 숫자를 그대로 받아들이지 않고, 왜 Candidate Size가 작아질수록 NDCG가 비정상적으로 높아지는지 다시 확인한 것이다. 평가 정의의 문제를 수정한 뒤 최적점이 달라졌고, 그 이후에는 단순히 최고 점수를 찾는 것보다 **통제된 실험과 재현 가능한 검증**을 더 중요하게 보게 됐다.

또한 Feature를 많이 넣는다고 모델이 자동으로 좋아지는 것이 아니었고, Optuna가 최종 답을 대신 정해주는 것도 아니었다. Candidate Retrieval, Ranking, Evaluation, Feature Engineering, Stability, Runtime Optimization이 서로 연결되어 있다는 것을 실제 실험으로 확인했다.

최종적으로 선택한 모델은 다음과 같다.

```text
Item-CF 46
+ BPR 20
+ Content 15
+ User-CF 20
        ↓
Candidate ≤ 101
        ↓
Full15
        ↓
XGBoost LambdaMART
        ↓
Top-10
```

이 구조는 한 번의 최고 점수 때문에 선택한 것이 아니다.

- Actual Top-10 기준 평가
- 5-Fold Recheck
- Feature Ablation
- 3-Seed Stability
- LTR1000 전체 Final Training
- Final400 Leakage Guard
- Final400 One-Time Evaluation
- 신규 사용자 정성평가
- Runtime / Evaluation 최적화

까지 거쳐 최종적으로 확정했다.

이번 프로젝트를 통해 추천시스템은 단순히 “추천 알고리즘 하나를 잘 고르는 문제”가 아니라 **Candidate를 만들고, Ranking하고, 올바르게 평가하고, 반복 가능한 속도로 실험하며, 신규 사용자까지 연결하는 전체 시스템 문제**라는 것을 배웠다.

---

## 22. 프로젝트 종료

이 프로젝트에서 계획한 핵심 작업은 모두 마무리했다.

- [x] 단일 추천 모델 비교
- [x] Reranking / Multi-Retriever / Rank Fusion 실험
- [x] Item-Based Ranker Hybrid
- [x] XGBoost Learning-to-Rank 도입
- [x] 4-Retriever Pipeline 최적화
- [x] Candidate Size / Retriever Ratio 탐색
- [x] 평가 방식 문제 발견 및 Actual Top-10 기준으로 수정
- [x] Constrained Optuna
- [x] Candidate Upper-Bound Sweep
- [x] Final Joint Optuna
- [x] A~E 후보 비교
- [x] Feature Ablation
- [x] 3-Seed Stability
- [x] E_HR 최종 Architecture 확정
- [x] LTR1000 전체 Final XGBoost 학습
- [x] Final400 Leakage Guard
- [x] Final400 최종 정량평가 및 Review Group 분석
- [x] Popularity / Novelty 최종 분석
- [x] 신규 사용자 Cold-Start 연결
- [x] 신규 사용자 Q1~Q8 최종 정성평가
- [x] Q1 실제 취향 확인 / Q6 Long-tail 분석 / Animal Theme 추가 확인
- [x] Runtime / Qualitative / Quantitative 코드 분리
- [x] Final400 Batch / Cache 최적화
- [x] 최종 문서 및 프로젝트 정리

> **Steam Game Recommendation System 프로젝트는 Day27을 마지막으로 종료한다.**
>
> 추가 모델 탐색이나 튜닝을 남겨둔 상태가 아니라, 현재 E_HR과 Final400 결과를 이 프로젝트의 최종 결과로 확정한다. 이후 새로운 아이디어가 생기더라도 그것은 이 프로젝트의 미완성 작업이 아니라 별도의 후속 프로젝트나 새로운 실험으로 다룬다.

---

## Final

```text
Final Model
= E_HR

Candidate
= Item46 + BPR20 + Content15 + User20

Features
= Full15

Ranker
= XGBoost LambdaMART

Final400
P@10       = 0.096250
R@10       = 0.127890
HR@10      = 0.570000
NDCG@10    = 0.139857
MAP@10     = 0.069833
CandRecall = 0.272015
Popularity = 10.252428
Novelty    = 10.354832
Hits       = 385

Status
= PROJECT COMPLETED
```
