# Experiment Summary

> Steam Game Recommendation System의 핵심 실험과 의사결정을 정리한 문서다.

> 모든 실행 로그를 나열하기보다 **무엇을 실험했고, 무엇을 확인했으며, 그 결과 다음 구조를 왜 선택했는지**에 초점을 맞췄다.

> **빠르게 보려면:** `0. Executive Summary` → `17. 최종 모델 확정` → `19. Final400 정량 평가` → `21. 신규 사용자 정성 평가` → `23. 최종 결론` 순서로 보면 된다.

---

## 0. Executive Summary

### 최종 추천 구조

```text
New / Existing User History
        ↓
Item / BPR / Content / User Retriever
        ↓
Candidate Union
I46 + B20 + C15 + U20
(max 101)
        ↓
Full15 Feature Engineering
        ↓
XGBoost LambdaMART Ranker
        ↓
Top-10 Recommendation
```

프로젝트는 단일 모델 비교에서 시작해 Reranking, Multi-Retriever, Rank Fusion, Item-Based Ranker를 거쳐 최종적으로 **4개 Retriever + XGBoost Learning-to-Rank** 구조로 발전했다.

가장 중요한 방법론적 변화는 다음 세 가지였다.

1. **Retriever와 Ranker의 역할을 분리했다.**
2. Candidate 내부 지표가 아니라 **전체 Test Positive 기준 Actual Top-10**으로 평가 방식을 수정했다.
3. Candidate Size / Retriever Ratio / Feature / XGBoost 설정을 탐색한 뒤 **3-Seed Stability + Feature Ablation**으로 최종 모델을 확정했다.

### 최종 후보 5개와 최종 선택

| 후보 | 역할 | Candidate Size | Item / BPR / Content / User | Features |
|---|---|---:|---|---:|
| A | Ranking / NDCG | 100 | 45 / 20 / 15 / 20 | 16 |
| B | Recall | 92 | 42 / 18 / 23 / 9 | 14 |
| C | Precision | 105 | 47 / 21 / 16 / 21 | 14 |
| D | Stability | 93 | 42 / 19 / 23 / 9 | 14 |
| **E_HR** | **Final** | **101** | **46 / 20 / 15 / 20** | **15** |

E는 Feature Ablation에서 `user_popularity_affinity`를 제거했을 때 P / HR / NDCG / MAP / Hits가 모두 개선되었고, 3-Seed 평균 **Macro NDCG@10 = 0.145155**로 최종 선택 기준 1위를 기록했다.

### Final E_HR 설정

| 항목 | 최종 값 |
|---|---|
| Candidate Size | 101 |
| Retriever | Item 46 / BPR 20 / Content 15 / User 20 |
| Ranker | XGBoost LambdaMART |
| Feature | Full15 |
| Final Training | LTR1000 전체 |
| Trees | 54 |
| Final evaluation | 기존 고정 Final400 1회 |
| Selection200 | 공식 최종 방법론에서 사용하지 않음 |

### Final400 최종 결과

| Metric | Value | Metric | Value |
|---|---:|---|---:|
| Precision@10 | **0.096250** | Recall@10 | **0.127890** |
| Hit Rate@10 | **0.570000** | NDCG@10 | **0.139857** |
| MAP@10 | **0.069833** | Candidate Recall | **0.272015** |
| Mean Log Popularity@10 | **10.252428** | Novelty@10 | **10.354832** |
| Total Hits | **385** | Users | **400** |

현재 모델 구조 탐색, Feature 실험, 안정성 검사, 최종 학습, Final400 평가, 신규유저 정성평가까지 완료했다.

---

## 1. 평가 환경

### 데이터 규모

| 항목 | 값 |
|---|---:|
| Recommendations | 약 41.15M |
| Train interactions | 37,113,471 |
| Test interactions | 4,041,323 |
| BPR interaction matrix | 13,781,059 users × 37,567 items |
| BPR nnz | 37,113,455 |
| LTR users | **1,000** |
| Final evaluation users | **400** |
| Final400 | 모델 탐색/튜닝 중 **미사용** |

LTR 실험은 동일한 1,000명을 대상으로 수행했다.

Final400은 모델 탐색과 Feature 튜닝에 사용하지 않고 최종 평가 단계용으로 보존했다.

### 주요 평가 지표

| 지표 | 의미 |
|---|---|
| Precision@10 | 추천한 10개 중 실제 정답의 비율 |
| Recall@10 | 사용자의 실제 정답 중 Top-10이 회수한 비율 |
| HR@10 | 사용자별 Top-10에 정답이 하나 이상 존재하는 비율 |
| NDCG@10 | 정답을 Top-10 앞쪽에 얼마나 잘 배치했는지 |
| MAP@10 | 여러 정답을 순위 전반에서 얼마나 잘 배치했는지 |
| Candidate Recall | Ranker에 들어가기 전 후보군이 실제 정답을 얼마나 확보했는지 |
| Total Hits | 전체 Top-10에서 맞힌 정답 개수 |
| NDCG Std | Fold별 NDCG 변동성 |

#### 최종 LTR 선택 기준

- Search Guardrail: `P@10 >= 0.1065`, `R@10 >= 0.1065`
- Final Guardrail: `P@10 >= 0.1070`, `R@10 >= 0.1070`
- Guardrail 통과 후보 중 **Macro NDCG@10 최대화**
- HR, MAP, Hits, Candidate Recall은 보조 판단 지표로 사용

---

## 2. 단일 모델 Baseline

| 모델 | P@10 | R@10 | HR@10 | NDCG@10 | 핵심 해석 |
|---|---:|---:|---:|---:|---|
| User-Based CF | 0.0545 | 0.0427 | 0.3011 | 0.0655 | 개인화 가능하지만 하이브리드 기여도는 낮음 |
| Item-Based CF | 약 **0.078** | - | - | - | 단일 모델 중 강했고 이후에도 핵심 signal로 유지 |
| Funk-SVD | 0.0003 | 0.0006 | 0.0025 | 0.0004 | 현재 implicit Top-N 문제와 부적합 |
| Final BPR | 0.05225 | 0.06870 | 0.3850 | 0.06968 | 단독 성능보다 후보 확장 Retriever로 활용 가치가 높음 |

### 결정

- **Funk-SVD 제외**
- **Item-Based는 강한 ranking signal**
- **BPR은 단독 Ranker보다 Retriever 역할에 적합**
- User-Based는 보조 신호로 유지

---

## 3. Hybrid 구조 탐색

### 주요 구조 실험

| 실험 | 구조 | 결과 | 다음 결정 |
|---|---|---|---|
| Case 1 | Item Top-30 → BPR Rerank | 후보가 Item에 제한되어 Recall 확장 한계 | Retriever가 놓친 정답은 Ranker가 복구할 수 없음 |
| Case 1 Reverse | BPR Top-30 → Item Rerank | BPR 후보를 Item이 정렬하는 편이 더 유효 | 후보 확장과 Ranking 역할 분리 |
| Case 2 | Item + User + Content → BPR Ranker | Candidate Recall은 증가했지만 BPR Ranking 한계 | Ranker 자체를 개선할 필요 |
| Case 3 | Item + User + BPR + Content Rank Fusion | 당시 최고 성능 | 모델별 기여도 분석 |
| Item Ranker Hybrid | BPR + Content + User → Item Rerank | Size100에서 좋은 성능 | 학습형 Ranker로 확장 |

### 4-Model Fusion Ablation

초기 Fusion weight:

| Retriever | Weight |
|---|---:|
| Item | 0.50 |
| BPR | 0.20 |
| User | 0.15 |
| Content | 0.15 |

Ablation 결과 모델 중요도는 대략 다음 순서였다.

```text
Item >> BPR > Content >> User
```

Item 제거 시 성능 하락이 가장 컸고, User 제거 시 일부 P/R은 오히려 개선됐지만 NDCG가 감소했다.

### 결정

User-Based를 완전히 제거하기보다 **작은 비율의 보조 Retriever**로 유지했다.

---

## 4. Item-Based Ranker 단계

다음 구조를 한동안 기준 구조로 사용했다.

```text
BPR 56
Content 39
User 5
= Candidate 100
      ↓
Item-Based Reranking
      ↓
Top-10
```

### Candidate Size Sweep

| Candidate Size | 해석 |
|---:|---|
| 50 | 후보 부족 |
| **100** | 최종 Top-10 성능 가장 좋음 |
| 150 | Candidate Recall은 증가하지만 Top-10 개선 제한 |
| 200 | Candidate Recall은 증가하지만 Top-10 개선 제한 |

#### Size100 결과

| P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---:|---:|---:|---:|---:|
| **0.0872** | **0.1110** | **0.5400** | **0.1216** | **348** |

### 의미

Candidate를 많이 확보하는 것과 최종 Top-10 성능을 높이는 것은 같은 문제가 아니었다.

이 결과가 이후 **Candidate Generation과 Ranking을 별도로 최적화**하는 방향으로 이어졌다.

---

## 5. XGBoost Learning-to-Rank 도입

### 초기 구조

```text
Item 50 + BPR 20 + Content 15 + User 15
                  ↓
             Candidate Union
                  ↓
            Full14 Features
                  ↓
             XGBoost Ranker
                  ↓
                Top-10
```

### 초기 대표 성능

| P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 |
|---:|---:|---:|---:|---:|
| 0.0938 | 0.1270 | 0.5775 | 0.1326 | 0.0635 |

추가 분석:

| 지표 | 결과 |
|---|---:|
| Mean Log Popularity@10 | 10.3590 |
| Novelty@10 | 10.2011 |
| Hits | 375 |

### 주요 Feature Importance

| Feature | Importance |
|---|---:|
| **bpr_rank** | **0.2651** |
| **retriever_count** | **0.1858** |
| bpr_score | 0.1254 |
| item_score | 0.1061 |
| item_popularity | 0.0846 |
| item_rank | 0.0674 |
| content_rank | 0.0596 |
| content_score | 0.0397 |
| user_rank | 0.0264 |
| user_score | 0.0258 |
| user_interaction | 0.0143 |

### 의미

단순 Score Fusion보다 XGBoost가 여러 Retriever의 신호를 조합하는 데 유리했다.

특히:

- `bpr_rank`
- `retriever_count`
- `item_score`

가 강한 signal로 나타났다.

---

## 6. Candidate Size 탐색과 평가 방식 수정

이 구간은 프로젝트에서 가장 중요한 방법론적 수정이 일어난 부분이다.

### 초기 K2 / K3 탐색

#### K2 — Size 90~150

| Size | Mean NDCG | Best NDCG | Candidate Recall |
|---:|---:|---:|---:|
| 90 | 0.4313 | 0.4433 | 0.2479 |
| 100 | 0.4130 | 0.4272 | 0.2609 |
| 110 | 0.3950 | 0.4103 | 0.2738 |
| 125 | 0.3731 | 0.3850 | 0.2900 |
| 150 | 0.3476 | 0.3597 | 0.3156 |

#### K3 — Size 70~90

| Size | Mean NDCG | Best NDCG | Candidate Recall |
|---:|---:|---:|---:|
| 70 | 0.4886 | 0.4900 | 0.2142 |
| 75 | 0.4730 | 0.4757 | 0.2234 |
| 80 | 0.4598 | 0.4644 | 0.2312 |
| 85 | 0.4496 | 0.4554 | 0.2372 |
| 90 | 0.4396 | 0.4433 | 0.2442 |

#### 문제 발견

결과만 보면 Candidate Size가 작아질수록 NDCG가 계속 좋아졌다.

하지만 동시에 Candidate Recall은 계속 떨어졌다.

```text
Candidate Size ↓
→ 내부 NDCG ↑
→ Candidate Recall ↓
```

이는 기존 평가가 **Retriever 단계에서 놓친 정답을 충분히 패널티하지 못하고 있다는 신호**였다.

---

## 7. Actual Top-10 기준으로 평가 재설계

후보군 내부에서만 NDCG를 계산하는 대신,

**전체 Test Positive를 기준으로 최종 Top-10 성능을 계산**하도록 평가 방식을 수정했다.

### Candidate Trade-off Sweep

| Size | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Candidate Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.0987 | 0.0974 | 0.5645 | 0.1313 | 0.0623 | 0.1342 |
| 40 | 0.1020 | 0.1027 | 0.5910 | 0.1356 | 0.0637 | 0.1577 |
| 50 | 0.1043 | 0.1050 | 0.5995 | 0.1378 | 0.0646 | 0.1789 |
| 55 | 0.1057 | 0.1060 | 0.6025 | 0.1392 | 0.0656 | 0.1888 |
| 65 | 0.1068 | 0.1073 | 0.6065 | **0.1398** | **0.0660** | 0.2061 |
| 70 | **0.1072** | **0.1077** | **0.6090** | 0.1395 | 0.0657 | **0.2142** |

#### 결론

평가 방식을 수정하자 작은 Candidate가 일방적으로 유리하던 현상이 사라졌다.

실제 Top-10에서는 Candidate Size가 커질수록 일정 구간까지 성능이 개선된 뒤 완만해졌다.

> **Candidate Recall 최대화 ≠ Top-10 성능 최대화**

이 결론을 이후 모든 LTR 실험의 기준으로 사용했다.

---

## 8. Constrained Optuna — Candidate 50~90

Candidate Size, Retriever Ratio, Feature Group, XGBoost Hyperparameter를 함께 탐색했다.

```text
Search
  ↓
P/R Guardrail
  ↓
Guardrail 통과 후보
  ↓
NDCG 최대화
```

### 주요 5-Fold 결과

| Trial | Size | Ratio | P@10 | R@10 | NDCG@10 | MAP@10 | Candidate Recall |
|---:|---:|---|---:|---:|---:|---:|---:|
| **64** | 86 | user20 | 0.1090 | 0.1087 | **0.1445** | **0.0688** | 0.2376 |
| 32 | 84 | user20 | **0.1104** | **0.1115** | 0.1439 | 0.0679 | 0.2349 |
| 52 | 85 | user20 | 0.1085 | 0.1087 | 0.1433 | 0.0679 | 0.2359 |
| 0 | 55 | baseline | 0.1083 | 0.1085 | 0.1432 | 0.0679 | 0.1903 |
| 73 | 89 | user20 | 0.1101 | 0.1105 | 0.1428 | 0.0671 | **0.2421** |

### 의미

상위 Trial들이 Size **84~89**, 즉 탐색 상단에 몰렸다.

따라서 이 결과만으로 Size를 확정하지 않고 **더 큰 Candidate 영역을 확인**했다.

---

## 9. Upper-Bound Sweep — Size 80~120

Trial64의 Feature/XGB 설정을 고정하고 Candidate Size와 Ratio만 다시 비교했다.

### Size Aggregate

| Size | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Candidate Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 80 | 0.1067 | 0.1077 | 0.6080 | 0.1398 | 0.0661 | 0.2312 |
| 90 | 0.1076 | 0.1085 | 0.6075 | 0.1397 | 0.0659 | 0.2442 |
| 100 | 0.1084 | 0.1094 | 0.6120 | 0.1419 | **0.0671** | 0.2574 |
| **110** | **0.1087** | **0.1107** | **0.6173** | **0.1422** | 0.0669 | 0.2697 |
| 120 | 0.1081 | 0.1098 | 0.6090 | 0.1406 | 0.0663 | **0.2798** |

#### Pareto 후보

| Candidate | P@10 | R@10 | NDCG@10 | MAP@10 | Candidate Recall |
|---|---:|---:|---:|---:|---:|
| **110 / user20** | 0.1091 | 0.1110 | **0.1433** | **0.0675** | 0.2680 |
| **120 / content20_user10** | **0.1094** | **0.1114** | 0.1418 | 0.0667 | **0.2828** |

### 결정

- 110 부근에서 Ranking 성능이 가장 좋음
- 120에서는 Candidate Recall은 증가하지만 NDCG는 다시 하락

따라서 마지막 Joint Optuna 범위를 **90~130**으로 설정했다.

---

## 10. Final Joint Optuna — Size 90~130

### 설정

| 항목 | 설정 |
|---|---|
| Candidate Size | 90~130, step=1 |
| Trials | 80 |
| Search | 3-Fold + Pruning |
| Search Guardrail | P ≥ 0.1065 / R ≥ 0.1065 |
| Recheck | Search-feasible Top10 → 5-Fold |
| Final Guardrail | P ≥ 0.1070 / R ≥ 0.1070 |
| Primary Objective | Macro NDCG@10 |
| LTR Users | 1,000 |
| Final400 | **미사용** |

### 최종 5-Fold 상위 후보

| Trial | Size | Ratio | I/B/C/U | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | NDCG Std | Cand. Recall | F |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **2** | 100 | user20 | 45/20/15/20 | 0.1097 | 0.1111 | 0.617 | **0.1460** | **0.0697** | 0.00762 | 0.2557 | 16 |
| **57** | 92 | content25 | 42/18/23/9 | 0.1106 | **0.1119** | 0.613 | 0.1451 | 0.0691 | 0.00775 | 0.2445 | 14 |
| **60** | 101 | user20 | 46/20/15/20 | 0.1096 | 0.1110 | **0.622** | 0.1447 | 0.0684 | 0.00576 | 0.2571 | 16 |
| **67** | 93 | content25 | 42/19/23/9 | 0.1100 | 0.1117 | 0.621 | 0.1446 | 0.0681 | 0.00475 | 0.2466 | 14 |
| **3** | 105 | user20 | 47/21/16/21 | **0.1110** | 0.1115 | 0.620 | 0.1446 | 0.0681 | **0.00387** | **0.2616** | 14 |

#### 후보를 5개 남긴 이유

단순히 NDCG Top5를 그대로 선택한 것이 아니라 서로 다른 장점을 가진 모델을 남겼다.

| 후보 | Trial | 선택 이유 |
|---|---:|---|
| **A** | 2 | NDCG / MAP 최고 |
| **B** | 57 | Recall 중심 |
| **C** | 3 | Precision 최고 + Candidate Recall 최고 + 가장 낮은 NDCG Std |
| **D** | 67 | 높은 성능 + 낮은 Fold 변동성 |
| **E** | 60 | HR 최고 |

---

## 11. Feature 구조

### Full14

기본 14개 Feature는 다음 계열로 구성된다.

- Retriever score × 4
- Retriever rank × 4
- `retriever_count`
- BPR / Content / User candidate flag
- `item_popularity`
- `user_interaction_count`

추가 Feature 후보:

- `is_item_candidate`
- Item 중심 agreement features
- `rrf_score`
- `rank_std`
- `user_popularity_affinity`

Final Joint Optuna 결과는 크게 두 계열로 나뉘었다.

| 계열 | 후보 | 구성 |
|---|---|---|
| 16-feature | A, E | Full14 + Item Flag + PopAffinity |
| 14-feature | B, C, D | Full14 |

이 차이가 실제로 필요한지 확인하기 위해 최종 Feature Ablation을 수행했다.

---

## 12. Final Feature Ablation

모든 실험에서 다음을 고정했다.

- Candidate Size
- Retriever Ratio
- XGBoost Hyperparameter
- 동일 LTR 1,000명 / 5-Fold

즉 **Feature만 제거**했다.

### A — NDCG형

| 설정 | P@10 | R@10 | NDCG@10 | MAP@10 | Hits | ΔNDCG |
|---|---:|---:|---:|---:|---:|---:|
| **Full16** | 0.1097 | 0.1111 | **0.1460** | **0.0697** | 1097 | - |
| - Item Flag | 0.1097 | 0.1113 | 0.1438 | 0.0679 | 1097 | **-0.00221** |
| - PopAffinity | 0.1095 | 0.1112 | 0.1448 | 0.0689 | 1095 | -0.00118 |
| - RetrieverCount | **0.1101** | **0.1116** | 0.1444 | 0.0680 | **1101** | -0.00164 |
| Full14 | 0.1090 | 0.1102 | 0.1449 | 0.0691 | 1090 | -0.00113 |

**결정: Full16 유지**

A는 일부 P/R보다 **Ranking quality**가 목적이므로 Full16의 NDCG/MAP 우위를 유지했다.

---

### B / C / D — RetrieverCount Ablation

| 후보 | Full NDCG | -RetrieverCount | ΔNDCG | ΔHits | 결정 |
|---|---:|---:|---:|---:|---|
| **B** | 0.1451 | 0.1434 | -0.00164 | -17 | Full14 유지 |
| **C** | 0.1446 | 0.1414 | **-0.00321** | **-27** | Full14 유지 |
| **D** | 0.1446 | 0.1432 | -0.00140 | -4 | Full14 유지 |

특히 C에서 `retriever_count` 제거 영향이 가장 컸다.

### 의미

여러 Retriever가 동시에 선택한 후보라는 정보 자체가 **후보의 신뢰도 signal**로 작동했다.

---

### E — HR형

| 설정 | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Hits |
|---|---:|---:|---:|---:|---:|---:|
| Full16 | 0.1096 | 0.1110 | 0.622 | 0.1447 | 0.0684 | 1096 |
| - Item Flag | 0.1086 | 0.1109 | 0.615 | 0.1439 | 0.0683 | 1086 |
| **- PopAffinity** | **0.1105** | 0.1113 | **0.623** | **0.1450** | **0.0692** | **1105** |
| - RetrieverCount | 0.1095 | **0.1115** | 0.617 | 0.1436 | 0.0677 | 1095 |
| Full14 | 0.1101 | **0.1115** | 0.620 | 0.1435 | 0.0677 | 1101 |

`PopAffinity` 제거 시 P, HR, NDCG, MAP, Hits가 모두 개선됐다.

**결정: E = 15 Features (`PopAffinity` 제거)**

---

## 13. Ablation 이후 최종 후보

| 후보 | 역할 | Size | I/B/C/U | Features | 대표 강점 |
|---|---|---:|---|---:|---|
| **A** | NDCG | 100 | 45/20/15/20 | **16** | NDCG / MAP |
| **B** | Recall | 92 | 42/18/23/9 | **14** | Recall |
| **C** | Precision | 105 | 47/21/16/21 | **14** | Precision / Candidate Recall / Stability |
| **D** | Stability | 93 | 42/19/23/9 | **14** | Fold 안정성 |
| **E** | Hit Rate | 101 | 46/20/15/20 | **15** | HR, Ablation 후 전반적 개선 |

---

## 14. 프로젝트에서 확인한 핵심 결론

## 1. Retriever와 Ranker는 분리하는 편이 좋았다

```text
여러 Retriever
→ 다양한 후보 확보
XGBoost Ranker
→ 후보의 최종 순서 학습
```

단일 모델이 두 역할을 모두 맡는 구조보다 역할을 분리했을 때 실험 설계와 성능 개선이 더 명확했다.

## 2. Candidate Recall만 높다고 좋은 추천은 아니다

Candidate Size를 늘리면 Candidate Recall은 꾸준히 증가했지만,

최종 P/R/NDCG는 일정 구간 이후 정체되거나 하락했다.

따라서 목표는 **최대한 많은 후보**가 아니라 **Ranker가 효과적으로 처리할 수 있는 후보군**이다.

## 3. 평가 방식이 결과를 왜곡할 수 있다

초기 Candidate 내부 NDCG에서는 작은 Candidate Size가 과도하게 유리했다.

전체 Test Positive 기준 Actual Top-10 평가로 수정하면서 Retriever miss도 평가에 반영했고,

Candidate Size와 Ranking 성능의 실제 trade-off를 볼 수 있게 됐다.

## 4. Item-Based는 강했지만 최종 구조는 학습형 Ranker가 더 적합했다

Item-Based는 가장 강한 단일 signal 중 하나였지만,

여러 Retriever의 score/rank/metadata를 동시에 활용하는 XGBoost Ranker가 더 유연했다.

## 5. RetrieverCount는 반복적으로 중요한 Feature였다

특히 C에서 제거 시:

```text
NDCG -0.00321
Hits -27
```

로 큰 하락이 발생했다.

여러 Retriever가 같은 아이템을 동시에 선택했다는 사실 자체가 유용한 signal이었다.

## 6. Feature는 많다고 항상 좋은 것이 아니다

A는 Full16이 가장 좋았지만 E는 PopAffinity를 제거한 15-feature 모델이 더 좋았다.

```text
Feature 추가
→ 성능 검증
→ Ablation
→ 불필요 Feature 제거
```

과정이 필요했다.

---

## 15. 전체 실험 흐름

```text
Single Models
(User / Item / Funk-SVD / BPR)
        ↓
Reranking
(Item→BPR / BPR→Item)
        ↓
Multi-Retriever
        ↓
4-Model Rank Fusion
        ↓
Item-Based Ranker Hybrid
        ↓
XGBoost Learning-to-Rank
        ↓
Candidate Size / Ratio Search
        ↓
평가 방식 문제 발견
        ↓
Actual Top-10 Evaluation
        ↓
Constrained Optuna 50~90
        ↓
Upper-Bound Sweep 80~120
        ↓
Final Joint Optuna 90~130
        ↓
Final Candidate A~E
        ↓
Feature Ablation
        ↓
A16 / B14 / C14 / D14 / E15
        ↓
3-Seed Stability Check
        ↓
E_HR Final Selection
        ↓
LTR1000 전체 Final Training
        ↓
Final400 One-Time Evaluation
        ↓
New-User Qualitative Evaluation
        ↓
Runtime Batch / Cache Optimization
```

---

## 16. Final Stability Check

Feature Ablation 이후 확정된 5개 후보를 대상으로 **Candidate / Feature / XGBoost 설정은 모두 고정**하고,

5-Fold 사용자 분할 seed만 바꿔 총 3회 안정성 검사를 수행했다.

```text
Seed 1 = 42
Seed 2 = 20260926
Seed 3 = 20260927
```

각 seed마다 동일한 LTR 1,000명을 5-fold로 다시 나눴으며, Final400은 전혀 사용하지 않았다.

### 16.1 Seed 1

| 순위 | 후보 | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Hits | NDCG Std |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **1** | **A** | 0.1097 | 0.1111 | 0.617 | **0.1460** | **0.0697** | 1097 | 0.00762 |
| 2 | B | 0.1106 | **0.1119** | 0.613 | 0.1451 | 0.0691 | 1106 | 0.00775 |
| 3 | E | 0.1105 | 0.1113 | **0.623** | 0.1450 | 0.0692 | 1105 | 0.00713 |
| 4 | D | 0.1100 | 0.1117 | 0.621 | 0.1446 | 0.0681 | 1100 | 0.00475 |
| 5 | C | **0.1110** | 0.1115 | 0.620 | 0.1446 | 0.0681 | **1110** | **0.00387** |

**해석:** A가 Ranking/NDCG 측면에서 가장 강했고, C는 Precision/Hits와 fold 안정성이 좋았다.

---

### 16.2 Seed 2

| 순위 | 후보 | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Hits | NDCG Std |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **1** | **D** | **0.1122** | **0.1139** | 0.626 | **0.1475** | 0.0705 | **1122** | **0.00478** |
| 2 | C | 0.1120 | 0.1135 | 0.626 | **0.1475** | **0.0707** | 1120 | 0.00671 |
| 3 | E | 0.1114 | 0.1125 | **0.629** | 0.1464 | 0.0699 | 1114 | 0.00640 |
| 4 | B | 0.1111 | 0.1119 | 0.620 | 0.1447 | 0.0687 | 1111 | 0.00550 |
| 5 | A | 0.1094 | 0.1103 | 0.615 | 0.1446 | 0.0689 | 1094 | 0.00650 |

**해석:** D/C가 크게 올라왔고, E도 상위권을 유지했다. A는 상대 순위가 1위에서 5위로 내려갔다.

---

### 16.3 Seed 3

| 순위 | 후보 | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Hits | NDCG Std |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **1** | **A** | **0.1102** | **0.1116** | **0.622** | **0.1446** | **0.0687** | **1102** | 0.00915 |
| 2 | E | 0.1097 | 0.1104 | **0.622** | 0.1440 | 0.0683 | 1097 | 0.00829 |
| 3 | B | 0.1085 | 0.1104 | 0.618 | 0.1433 | 0.0684 | 1085 | **0.00646** |
| 4 | D | 0.1082 | 0.1090 | 0.617 | 0.1428 | 0.0674 | 1082 | 0.00823 |
| 5 | C | 0.1074 | 0.1084 | 0.619 | 0.1425 | 0.0676 | 1074 | 0.00828 |

**해석:** A가 다시 1위로 올라왔고 E가 2위였다. Seed2에서 강했던 C/D는 하락했다.

---

### 16.4 Seed별 NDCG 비교

| 후보 | Seed 1 | Seed 2 | Seed 3 | 3-Seed 평균 | Seed 간 Std | 범위 |
|---|---:|---:|---:|---:|---:|---:|
| **E** | 0.145021 | 0.146414 | 0.144031 | **0.145155** | 0.000977 | 0.002383 |
| **A** | **0.146001** | 0.144614 | **0.144561** | **0.145059** | **0.000667** | **0.001440** |
| **D** | 0.144648 | **0.147546** | 0.142834 | 0.145009 | 0.001941 | 0.004712 |
| **C** | 0.144629 | 0.147526 | 0.142488 | 0.144881 | 0.002064 | 0.005038 |
| **B** | 0.145058 | 0.144670 | 0.143299 | 0.144342 | 0.000755 | 0.001759 |

#### 핵심

- **E**: 평균 NDCG 1위
- **A**: 평균 NDCG 2위지만 seed 간 NDCG Std가 가장 낮음
- **D / C**: 평균은 높지만 seed별 변동이 상대적으로 큼
- **B**: 변동은 작지만 평균 NDCG가 가장 낮음

---

### 16.5 3-Seed 평균 성능

| 순위 | 후보 | Avg P@10 | Avg R@10 | Avg HR@10 | Avg NDCG | Avg MAP | Avg Hits |
|---:|---|---:|---:|---:|---:|---:|---:|
| **1** | **E** | **0.11053** | 0.11139 | **0.62467** | **0.14516** | **0.06911** | **1105.3** |
| **2** | **A** | 0.10977 | 0.11098 | 0.61800 | **0.14506** | 0.06909 | 1097.7 |
| **3** | **D** | 0.11013 | **0.11153** | 0.62133 | 0.14501 | 0.06867 | 1101.3 |
| **4** | **C** | 0.11013 | 0.11112 | 0.62167 | 0.14488 | 0.06877 | 1101.3 |
| **5** | **B** | 0.11007 | 0.11141 | 0.61700 | 0.14434 | 0.06875 | 1100.7 |

---

### 16.6 Seed별 NDCG 순위 변화

| 후보 | Seed 1 | Seed 2 | Seed 3 | 3-Seed 평균 |
|---|---:|---:|---:|---:|
| **A** | **1위** | 5위 | **1위** | **2위** |
| **B** | 2위 | 4위 | 3위 | 5위 |
| **C** | 5위 | **2위** | 5위 | 4위 |
| **D** | 4위 | **1위** | 4위 | 3위 |
| **E** | 3위 | 3위 | **2위** | **1위** |

이 표는 단순 평균뿐 아니라 각 후보가 사용자 Fold 구성에 얼마나 민감한지도 보여준다.

---

### 16.7 왜 C와 B가 평균 NDCG 4·5위인가?

후보 이름인 `Precision형`, `Recall형`은 **해당 지표에서 상대적으로 강한 성격을 가진 후보**라는 뜻이지,

최종 Stability 순위를 Precision이나 Recall로 매긴다는 뜻은 아니다.

Stability 비교의 중심은 기존 최종 선택 기준과 동일하게 **NDCG@10**이다.

#### C — Precision형인데 4위인 이유

C는 Top-10 안에 정답을 넣는 능력과 Precision/Hits는 강하지만,

정답을 **상위 몇 번째 위치에 배치하는가**는 seed에 따라 크게 흔들렸다.

```text
C NDCG
Seed1 = 0.144629
Seed2 = 0.147526
Seed3 = 0.142488
```

Seed 간 NDCG 범위는 **0.005038**로 5개 중 가장 컸다.

즉:

> **정답을 Top-10 안에 넣는 능력은 좋지만, 그 정답을 매우 앞쪽에 배치하는 Ranking 품질은 fold 구성에 민감했다.**

따라서 Precision형이라는 강점과 별개로 평균 NDCG 기준에서는 4위가 되었다.

#### B — Recall형인데 5위인 이유

B의 3-seed 평균 Recall은 높은 편이다.

| 지표 | B의 상대 위치 |
|---|---:|
| Precision 평균 | 4위 |
| **Recall 평균** | **2위** |
| HR 평균 | 5위 |
| **NDCG 평균** | **5위** |

Recall은 정답이 Top-10에 포함됐는지를 중요하게 보지만,

NDCG는 **정답이 얼마나 앞쪽에 위치했는가**까지 반영한다.

예를 들어:

```text
B          : 정답이 7위, 9위
다른 후보 : 정답이 2위, 4위
```

라면 Recall은 비슷할 수 있지만 NDCG는 B가 더 낮아질 수 있다.

즉 B는:

> **관련 게임을 회수하는 능력은 괜찮지만, 회수한 정답을 최상단으로 끌어올리는 Ranking 성능이 상대적으로 약했다.**

그래서 Recall형임에도 평균 NDCG 순위에서는 5위가 되었다.

---

### 16.8 Stability Check 최종 해석

| 후보 | 해석 |
|---|---|
| **E** | 세 seed 모두 상위권을 유지했고 평균 NDCG/HR이 가장 높음 |
| **A** | 평균 NDCG 2위이며 NDCG 변동성이 가장 낮아 Ranking 성능이 안정적 |
| **D** | Recall 평균이 가장 좋지만 seed별 변동이 큼 |
| **C** | Precision/Hits 강점이 있지만 NDCG가 seed에 민감 |
| **B** | Recall은 좋고 변동도 작지만 평균 Ranking 품질은 상대적으로 낮음 |

#### 중요한 해석

A는 순위만 보면 `1 → 5 → 1`로 흔들려 보이지만 실제 NDCG 값은:

```text
0.146001
0.144614
0.144561
```

로 차이가 작고, **seed 간 NDCG Std = 0.000667**로 가장 낮았다.

즉 Seed2에서 A가 실제로 크게 무너졌다기보다 **D/C/E가 일시적으로 크게 상승하면서 상대 순위가 내려간 것**에 가깝다.

반면:

```text
D: 4 → 1 → 4
C: 5 → 2 → 5
```

는 실제 NDCG 변동폭도 더 커서 사용자 Fold 구성에 민감한 편이었다.

#### Stability 단계 결론

Stability 결과만 보면:

```text
E = 평균 성능과 HR이 가장 좋고 꾸준함
A = Ranking 성능 자체가 가장 안정적
D = Recall 강점, seed 민감도 존재
C = Precision 강점, seed 민감도 큼
B = Recall 강점, 평균 NDCG는 상대적으로 낮음
```

최종 선택 기준은 프로젝트 전체에서 미리 사용하던 **Macro NDCG@10**이다.

3-Seed 평균에서 E가 **0.145155**로 가장 높았고, HR 평균도 가장 높았다. 따라서 Stability Check 이후 **E의 15-feature 버전을 최종 모델 E_HR로 확정**했다.

A는 Seed 간 NDCG 변동성이 더 낮았지만, 최종 선택 기준 자체를 사후에 바꾸지 않고 평균 NDCG 기준을 유지했다.

---

## 17. 최종 모델 확정 — E_HR

### 17.1 최종 선택 근거

3-Seed Stability에서 각 후보의 평균 성능은 다음과 같았다.

| 후보 | Avg P@10 | Avg R@10 | Avg HR@10 | Avg NDCG | Avg MAP | Seed 간 NDCG Std |
|---|---:|---:|---:|---:|---:|---:|
| **E** | **0.110533** | 0.111385 | **0.624667** | **0.145155** | **0.069112** | 0.000977 |
| A | 0.109767 | 0.110979 | 0.618000 | 0.145059 | 0.069087 | **0.000667** |
| D | 0.110133 | **0.111525** | 0.621333 | 0.145009 | 0.068666 | 0.001941 |
| C | 0.110133 | 0.111121 | 0.621667 | 0.144881 | 0.068767 | 0.002064 |
| B | 0.110067 | 0.111407 | 0.617000 | 0.144342 | 0.068749 | 0.000755 |

기존 Primary Objective인 **Macro NDCG@10**을 유지하여 E를 최종 선택했다.

### 17.2 Selection200 분기 폐기

중간에 Final400을 `Selection200 + Final Test200`으로 분할하여 후보를 고르는 분기를 한 번 실행했지만, 이 방식은 원래 최종 평가용으로 보존한 Final400을 모델 선택에 사용하게 된다.

따라서 해당 분기의 결과는 **공식 최종 모델 선택과 최종 성능 산정에서 모두 폐기**했다.

최종 방법론은 다음으로 확정했다.

```text
LTR1000
   ↓
Candidate / Feature / XGB Search
   ↓
Feature Ablation
   ↓
3-Seed Stability
   ↓
E_HR 선택
   ↓
LTR1000 전체로 E_HR 학습
   ↓
원본 Final400에서 1회 최종 평가
```

Final400은 최종 모델 선택, Feature 선택, Candidate Size 탐색, XGBoost tuning에 사용하지 않았다.

---

## 18. Final E_HR 구조

### 18.1 Candidate Retriever

| Retriever | Top-N |
|---|---:|
| Item-Based CF | **46** |
| BPR | **20** |
| Content-Based | **15** |
| User-Based CF | **20** |
| Candidate Union 최대 | **101** |

### 18.2 Final Full15 Features

```text
item_score_norm
bpr_score_norm
content_score_norm
user_score_norm
item_rank
bpr_rank
content_rank
user_rank
retriever_count
is_bpr_candidate
is_content_candidate
is_user_candidate
item_popularity
user_interaction_count
is_item_candidate
```

`user_popularity_affinity`는 E Ablation에서 제거 시 성능이 개선되어 최종 Feature에서 제외했다.

### 18.3 Final XGBoost

| Parameter | Value |
|---|---:|
| n_estimators | 54 |
| max_depth | 5 |
| min_child_weight | 2 |
| learning_rate | 0.0428417863 |
| subsample | 0.8621568879 |
| colsample_bytree | 0.7010833569 |
| reg_lambda | 0.8133078460 |
| reg_alpha | 0.4952702650 |
| objective | rank:ndcg |
| eval_metric | ndcg@10 |
| tree_method | hist |

최종 모델은 **LTR1000 전체**로 학습했으며 Final400은 학습에 사용하지 않았다.

---

## 19. Final400 정량 평가

고정된 원본 Final400 사용자 400명에서 최종 E_HR을 한 번 평가했다.

### 19.1 전체 결과

| Metric | Final E_HR |
|---|---:|
| Users | 400 |
| Precision@10 | **0.096250** |
| Recall@10 | **0.127890** |
| Hit Rate@10 | **0.570000** |
| NDCG@10 | **0.139857** |
| MAP@10 | **0.069833** |
| Candidate Recall | **0.272015** |
| Mean Log Popularity@10 | **10.252428** |
| Novelty@10 | **10.354832** |
| Total Hits | **385** |

### 19.2 사용자 활동량 그룹별 결과

| Review Group | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Candidate Recall | Mean Log Pop. | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10-15개 | 0.054 | 0.143333 | 0.34 | 0.126247 | 0.080001 | 0.292667 | 10.450322 | 10.069126 |
| 16-25개 | 0.072 | 0.137786 | 0.50 | 0.121104 | 0.060314 | 0.275440 | 10.233539 | 10.382083 |
| 26-45개 | 0.107 | 0.129295 | 0.67 | 0.142498 | 0.064291 | 0.274840 | 10.288286 | 10.302900 |
| 46-78개 | **0.152** | 0.101148 | **0.77** | **0.169579** | 0.074725 | 0.245115 | 10.037565 | **10.665218** |

활동량이 많은 사용자일수록 Precision / HR / NDCG가 높아지는 경향이 나타났다.

반대로 Recall과 Candidate Recall은 감소했는데, 활동량이 많은 그룹일수록 Test Positive 자체가 많아져 Top-10으로 전체 정답을 회수하기 더 어려운 영향이 함께 존재한다.

---

## 20. 신규 사용자 처리

신규 사용자는 기존 BPR user factor가 없으므로 다음 방식으로 처리했다.

```text
Played Games
    ↓
각 게임의 BPR Item Embedding
    ↓
Embedding 평균
    ↓
Initial New-User Vector
    ↓
Item Factors 고정
    ↓
해당 User Vector만 BPR Fine-Tuning
    ↓
Retriever / Ranker Pipeline
```

Fine-Tuning 설정:

| 항목 | 값 |
|---|---:|
| Epochs | 3 |
| Learning Rate | 0.01 |
| Regularization | 0.001 |

이 방식으로 신규 사용자의 플레이 기록만으로 BPR representation을 생성한다.

---

## 21. 신규 사용자 정성 평가

정량 지표만으로 확인하기 어려운 **복합 취향, 인기 편향, Long-tail 대응, 세부 테마 반영**을 보기 위해 신규 사용자 시나리오를 직접 구성했다.

Ground Truth가 없는 신규 사용자이므로 P/R/NDCG/MAP는 계산하지 않고, 실제 Top-10 추천과 Popularity / Novelty를 확인했다.

### 21.1 Q1~Q8 결과 요약

| ID | 시나리오 | Input Mean Log Pop. | Input Novelty | Output Mean Log Pop.@10 | Output Novelty@10 | 핵심 관찰 |
|---|---|---:|---:|---:|---:|---|
| Q1 | 현실적 혼합 취향 | 11.1557 | 9.0512 | 11.4322 | 8.6523 | 주 취향인 액션/오픈월드/스토리는 잘 잡았지만 약한 Cute/Animal 신호는 묻힘 |
| Q2 | 완전히 다른 2장르 | 11.6154 | 8.3880 | 12.0015 | 7.8310 | Shooter 쪽으로 강하게 기울고 Cozy 신호는 약함 |
| Q3 | 완전히 다른 3장르 | 11.7046 | 8.2593 | 11.8263 | 8.0837 | Shooter와 Strategy는 보존, Cozy는 약함 |
| Q4 | 3장르 강화 2+2+2 | 11.3063 | 8.8339 | 11.7858 | 8.1421 | 각 취향을 여러 게임으로 강화하자 Strategy 신호가 더 명확해짐 |
| Q5 | 인기게임만 | 12.3299 | 7.3571 | 11.7232 | 8.2325 | 입력보다 약간 덜 인기 있고 더 참신한 추천으로 이동 |
| Q6 | 비인기/Long-tail만 | 3.1355 | 20.6860 | 4.1095 | 19.2614 | Geneforge 계열을 1~4위에 배치해 명확한 niche signal을 포착 |
| Q7 | 인기3 + Long-tail3 | 7.7899 | 13.9390 | 11.8511 | 8.0479 | 같은 개수로 섞어도 인기게임 signal이 Long-tail signal을 압도 |
| Q8 | 테마를 부분 통제한 인기/Long-tail 혼합 | 7.9331 | 13.7245 | 11.6286 | 8.3689 | 장르 차이만이 아니라 popularity 자체도 추천에 강하게 작용하는 정황 |

### 21.2 Q1 실제 사용자 취향 확인

Q1은 실제 사용자 취향과 가까운 입력으로 구성했다.

추천 Top-10 중:

```text
God of War
Marvel's Spider-Man Remastered
Grand Theft Auto V Legacy
Cyberpunk 2077
```

4개는 실제로 이미 플레이한 게임이었다.

따라서 이 한 사례에서는 **주된 Action / Open World / Story 취향을 타당하게 잡는 모습**이 확인됐다. 다만 단일 사용자 사례이므로 이를 전체 사용자 정확도로 일반화하지 않는다.

### 21.3 Q6 Long-tail 실험

Long-tail 게임만 입력했을 때 추천 상위가 다음처럼 나타났다.

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

즉 모델이 무조건 인기작만 추천하는 것은 아니며, **입력 signal이 충분히 명확하면 비인기 / niche 취향도 유지할 수 있음**을 확인했다.

### 21.4 Animal Theme 추가 실험

마지막으로 세부적인 `동물` 테마가 얼마나 반영되는지 확인하려 했다.

| ID | 입력 | 상태 | 결과 |
|---|---|---|---|
| Q9 | ANIMAL WELL | 실패 | metadata에는 있으나 BPR item mapping에 없어 신규 사용자 벡터 생성 불가 |
| Q10 | Stray | 성공 | 동물 자체보다 Story / Adventure 유사성이 더 강하게 나타남 |
| Q11 | ANIMAL WELL + Stray | 참고용 | ANIMAL WELL이 BPR universe 밖이어서 결과가 Stray 단독과 동일 |

Stray 단독 추천에서는 `Ori and the Will of the Wisps`, `Cult of the Lamb`처럼 동물/생물 캐릭터와 관련된 결과가 일부 있었지만, 전체 Top-10은 주로 Story / Adventure 계열이었다.

따라서 이 실험에서 확인된 것은 두 가지다.

1. **세부적인 시각적/테마적 속성보다 협업 신호와 장르/게임플레이 유사성이 더 강하게 작동했다.**
2. **BPR 학습 universe에 없는 아이템은 현재 신규 사용자 representation에 직접 사용할 수 없다.**

Q9/Q11은 이 데이터 한계 때문에 동물 취향 강화 여부를 검증한 유효 실험으로 보지 않는다.

---

## 22. Runtime / Evaluation 최적화

Final400을 사용자별로 그대로 계산했을 때 Feature Generation이 매우 느렸기 때문에 다음 최적화를 적용했다.

- Item-CF source item neighbor batch precompute
- User-CF sparse batch scoring
- Candidate Feature cache
- XGBoost candidate 전체 batch prediction
- 기존 결과 재사용을 위한 checkpoint / cache

결과:

| 실행 | 시간 |
|---|---:|
| Final400 첫 optimized 실행 | 약 **6.51분** |
| Feature cache 재사용 실행 | 약 **0.03분** |

XGBoost 예측 자체보다 **Retriever / Feature Generation이 실제 계산 병목**이라는 점을 확인했다.

이 최적화는 Candidate quota, Feature, XGB model, ranking 조건을 변경하지 않고 계산 방식을 batch/cache화한 것이다.

---

## 23. 최종적으로 확인한 프로젝트 결론

1. **Retriever와 Ranker를 분리한 구조가 가장 효과적이었다.**
2. **Candidate Recall과 최종 Top-10 성능은 같은 목표가 아니었다.**
3. **평가 정의가 잘못되면 Optuna가 잘못된 방향을 매우 효율적으로 최적화할 수 있다.**
4. **Candidate-internal NDCG 대신 전체 Test Positive 기준 Actual Top-10을 사용해야 했다.**
5. **Item-Based signal은 프로젝트 전반에서 강했지만, 최종 Ranking은 여러 Retriever를 함께 학습하는 XGBoost가 더 유연했다.**
6. **RetrieverCount는 여러 모델의 동의를 나타내는 유효한 Feature였다.**
7. **Feature는 추가한다고 항상 좋아지지 않으며 Ablation이 필요했다.**
8. **Long-tail 취향 자체는 포착할 수 있지만, 인기 신호와 섞이면 인기 쪽이 압도할 수 있었다.**
9. **여러 취향이 섞이면 강한 취향은 살아남고 약한 취향은 Top-10에서 묻힐 수 있었다.**
10. **최종 시스템의 실제 병목은 XGBoost 학습보다 Retriever / Feature Generation이었다.**

---

## 24. Final Experiment Status

- [x] Baseline CF models
- [x] Funk-SVD
- [x] Implicit BPR
- [x] Hybrid Reranking
- [x] Multi-Retriever
- [x] Rank Fusion
- [x] Item-Based Ranker Hybrid
- [x] Candidate Size Sweep
- [x] XGBoost Learning-to-Rank
- [x] Candidate Ratio Search
- [x] Feature Search
- [x] Constrained Optuna
- [x] Actual Top-10 Evaluation redesign
- [x] Upper-Bound Sweep
- [x] Final Joint Optuna
- [x] Feature Ablation
- [x] 3-Seed Stability Check
- [x] **E_HR Final Model Selection**
- [x] **LTR1000 Final Training**
- [x] **Final400 One-Time Evaluation**
- [x] **MAP / Popularity / Novelty Evaluation**
- [x] **New-User BPR Initialization + Fine-Tuning**
- [x] **Qualitative Evaluation**
- [x] **Runtime Batch / Cache Optimization**
