# 🎮 Steam Game Recommendation System

Steam 게임 데이터를 활용해 **Content-Based / Collaborative Filtering / Matrix Factorization / Pairwise Ranking / Hybrid Recommendation / Learning-to-Rank**를 단계적으로 구현하고, 동일한 데이터 파이프라인 안에서 추천 구조와 성능을 비교한 프로젝트입니다.

이 프로젝트의 핵심은 단순히 여러 추천 모델을 구현하는 데 그치지 않고, **각 모델이 어떤 signal을 학습하는지 분석하고 Retrieval과 Ranking의 역할을 분리해 최종 Hybrid Architecture를 설계하는 것**입니다.

또한 프로젝트 후반에는 단순 성능 비교를 넘어 **평가 방식 검증, Candidate Search, Feature Ablation, Multi-Seed Stability, Cold-Start, Popularity / Novelty 분석, Runtime 최적화**까지 확장했습니다.

---

## 🚀 At a Glance

| 항목 | 최종 상태 |
|---|---|
| 데이터 | Steam 약 **4,115만 interactions** |
| Train / Test | **37,113,471 / 4,041,323** |
| LTR 학습 사용자 | **1,000명** |
| 최종 평가 사용자 | **고정 Final400** |
| 최종 출력 | **Top-10 Recommendation** |
| 최종 Architecture | **4-Retriever + XGBoost LambdaMART** |
| Candidate | **Item 46 + BPR 20 + Content 15 + User 20** |
| Candidate Budget | **최대 101** |
| Feature | **Full15** |
| Final Ranker | **XGBoost `rank:ndcg`** |
| 최종 NDCG@10 | **0.139857** |
| 최종 MAP@10 | **0.069833** |
| 프로젝트 상태 | **완료** |

### Final Recommendation Architecture

```text
                         User History
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
   Item-CF Retriever     BPR Retriever      Content Retriever
        Top-46              Top-20              Top-15
          │                   │                   │
          └──────────────┬────┴──────────────┬────┘
                         │                   │
                         │             User-CF Retriever
                         │                  Top-20
                         │                   │
                         └──────────┬────────┘
                                    ▼
                          Candidate UNION
                              max 101
                                    │
                                    ▼
                         Full15 Feature Set
                                    │
                                    ▼
                      XGBoost LambdaMART Ranker
                                    │
                                    ▼
                           Top-10 Recommendation
```

> 최종 구조는 **Retriever가 다양한 후보를 확보하고, XGBoost Ranker가 여러 모델의 score / rank / context를 함께 학습해 최종 순서를 결정하는 2-stage recommendation architecture**입니다.

---

# 📊 Final Result

최종 모델 `E_HR`은 **LTR1000 내부에서 Candidate / Feature / XGBoost를 탐색하고, Feature Ablation과 3-Seed Stability를 거쳐 확정**했습니다.

그 뒤 **LTR1000 전체로 Final Ranker를 다시 학습**하고, 모델 선택에 사용하지 않은 **원본 Final400에서 한 번 최종 평가**했습니다.

## Final400

| Metric | Final E_HR |
|---|---:|
| Users | **400** |
| Precision@10 | **0.096250** |
| Recall@10 | **0.127890** |
| Hit Rate@10 | **0.570000** |
| NDCG@10 | **0.139857** |
| MAP@10 | **0.069833** |
| Candidate Recall | **0.272015** |
| Mean Log Popularity@10 | **10.252428** |
| Novelty@10 | **10.354832** |
| Total Hits | **385** |

### Review Group별 결과

| Review Group | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Cand. Recall | Mean Log Pop. | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10–15개 | 0.054 | **0.143333** | 0.34 | 0.126247 | **0.080001** | **0.292667** | 10.450322 | 10.069126 |
| 16–25개 | 0.072 | 0.137786 | 0.50 | 0.121104 | 0.060314 | 0.275440 | 10.233539 | 10.382083 |
| 26–45개 | 0.107 | 0.129295 | 0.67 | 0.142498 | 0.064291 | 0.274840 | 10.288286 | 10.302900 |
| 46–78개 | **0.152** | 0.101148 | **0.77** | **0.169579** | 0.074725 | 0.245115 | **10.037565** | **10.665218** |

활동량이 많은 사용자일수록 **Precision / HR / NDCG가 상승**하는 경향을 보였습니다.

반면 Recall과 Candidate Recall은 감소했는데, 활동량이 많은 사용자는 Test Positive 자체가 많아 **Top-10만으로 전체 정답을 회수하기 더 어려운 영향**도 함께 존재합니다.

---

# 🧭 Project Evolution

프로젝트는 단일 모델 구현에서 시작해 최종적으로 **Multi-Retriever + Learning-to-Rank 시스템**으로 확장되었습니다.

| 단계 | 핵심 작업 | 결정 |
|---|---|---|
| 1 | Content / User / Item CF 구현 | Item-CF가 강한 co-consumption signal을 보임 |
| 2 | Funk-SVD 실험 | Rating objective와 Top-N ranking 목적 불일치 확인 |
| 3 | Implicit BPR + Explicit False FT | BPR을 강한 Candidate Retriever로 사용 |
| 4 | Item→BPR / BPR→Item Reranking | Retrieval과 Ranking 역할 분리 필요성 확인 |
| 5 | Multi-Retriever / Rank Fusion | 서로 다른 모델이 서로 다른 정답을 보완 |
| 6 | Item-Based Ranker Hybrid | `BPR56 + Content39 + User5 → Item Ranker` 확보 |
| 7 | XGBoost LTR 도입 | 고정 rule보다 여러 model signal을 함께 학습 |
| 8 | 4-Retriever 확장 | Item-CF도 Candidate Retriever로 포함 |
| 9 | Candidate / Ratio / Feature / XGB Search | Optuna + Sweep 기반 탐색 |
| 10 | Evaluation 재설계 | Candidate 내부 NDCG → Actual Top-10 평가 |
| 11 | Ablation / Stability | Feature 기여와 Seed 민감도 검증 |
| 12 | Final E_HR | `I46+B20+C15+U20 → Full15 → XGB` 확정 |
| 13 | Final400 / Qualitative | 정확도 + Popularity / Novelty + Cold-Start 분석 |
| 14 | Runtime Optimization | Batch / Cache / Sparse 연산으로 반복 평가 최적화 |

---

# 🧠 Key Conclusions

1. **Retrieval과 Ranking은 별개의 문제였습니다.**  
   Candidate Recall이 높아져도 최종 Top-10 성능이 반드시 좋아지지는 않았습니다.

2. **Candidate Recall은 Final Objective가 아니라 Diagnostic Metric입니다.**  
   Retriever가 충분한 정답 후보를 확보해야 하지만, 최종 선택은 P/R/NDCG/MAP와 함께 판단해야 했습니다.

3. **평가 정의가 모델 선택보다 먼저였습니다.**  
   Candidate 내부에서만 NDCG를 계산했을 때 작은 Candidate가 비정상적으로 유리해지는 artifact를 발견했습니다.

4. **Optuna의 Best Trial 하나가 곧 최종 모델은 아니었습니다.**  
   Cheap CV → Stronger CV → Ablation → Multi-Seed Stability 순서로 다시 검증했습니다.

5. **Feature는 많다고 항상 좋은 것이 아니었습니다.**  
   `user_popularity_affinity`는 제거했을 때 성능이 개선됐고, `retriever_count`는 여러 후보에서 반복적으로 중요한 feature였습니다.

6. **Long-tail signal도 학습할 수 있지만 Popular signal과 섞이면 밀릴 수 있었습니다.**  
   Long-tail만 입력했을 때는 niche 취향을 잘 유지했지만, Popular + Long-tail 혼합에서는 인기 signal이 강하게 우세했습니다.

7. **실제 병목은 마지막 XGBoost보다 Retrieval / Feature Generation이었습니다.**  
   대규모 sparse similarity 계산을 Batch / Cache / Precompute하는 것이 반복 실험 가능성을 크게 좌우했습니다.

---

# 📈 Model Performance

## Standalone Model Comparison

| 모델 | P@10 | R@10 | HR@10 | NDCG@10 | 핵심 해석 |
|---|---:|---:|---:|---:|---|
| Content-Based (TF-IDF) | 0.0268 | 0.0238 | 0.2250 | 0.0286 | Metadata semantic similarity |
| User-Based CF | 0.0545 | 0.0427 | 0.3011 | 0.0655 | 개인화 가능하지만 sparsity 영향 큼 |
| **Item-Based CF** | **0.0783** | **0.0880** | **0.4800** | **0.1078** | Standalone에서 가장 강한 signal |
| Funk-SVD (biased) | 0.0003 | 0.0006 | 0.0025 | 0.0004 | Item bias가 ranking 지배 |
| Funk-SVD (unbiased) | 0.0037 | 0.0030 | 0.0350 | 0.0042 | Bias 제거 후 개선, 절대 성능은 낮음 |
| Final BPR | 0.05225 | 0.06870 | 0.3850 | 0.06968 | Retriever로 높은 활용 가치 |

> Hybrid 단계에서는 standalone score만이 아니라 **각 모델이 다른 모델이 놓친 정답을 보완하는지**를 함께 봤습니다.

## Hybrid Architecture Progression

| 단계 | 구조 | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Hits |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Item-Based | 0.0633 | 0.0814 | 0.3925 | 0.0913 | - | 253 |
| Case 1 | Item Top-30 → BPR Rerank | 0.0695 | 0.0888 | 0.4525 | 0.0978 | - | 278 |
| Case 1 Reverse | BPR Top-30 → Item Rerank | 0.0695 | 0.0874 | 0.4675 | 0.0977 | - | 278 |
| Case 2 | Multi-Retriever → BPR Ranker | 0.0673 | 0.0862 | 0.4425 | 0.0953 | - | 269 |
| Case 3 | 4-Model Weighted Fusion | 0.0775 | 0.1001 | 0.4675 | 0.1134 | - | 310 |
| Fusion User 5% | Weighted Fusion | 0.0800 | 0.1027 | 0.4825 | 0.1148 | - | 320 |
| Cross-Ranker | BPR59 + Content41 → Item Ranker | 0.0827 | 0.1061 | 0.5225 | 0.1168 | - | 330 |
| Item-Ranker Hybrid | BPR56 + Content39 + User5 → Item | 0.0872 | 0.1110 | 0.5400 | 0.1216 | - | 348 |
| XGB Full14 | 4-Retriever → XGBoost | 0.0938 | 0.1270 | **0.5775** | 0.1326 | 0.0635 | 375 |
| **Final E_HR** | **I46+B20+C15+U20 → XGBoost** | **0.09625** | **0.12789** | 0.5700 | **0.139857** | **0.069833** | **385** |

> 앞 단계의 실험 결과는 구조 탐색 과정의 historical result이고, **최종 공식 성능은 마지막 Final400 평가 결과**입니다.

---

# 🧪 Learning-to-Rank & Final Model Selection

## 1. Why Learning-to-Rank?

Item-Based Ranker는 Candidate 100에서 강했지만 Candidate를 150 / 200으로 늘리면 **Candidate Recall은 증가해도 추가 정답을 Top-10으로 충분히 끌어올리지 못했습니다.**

```text
Candidate Recall ↑
        ≠
Final Ranking Performance ↑
```

그래서 Candidate를 확보하는 Retriever와 최종 순서를 학습하는 Ranker를 분리하고, 여러 Retriever의 정보를 동시에 사용할 수 있는 **XGBoost LambdaMART**를 도입했습니다.

---

## 2. Evaluation Redesign

초기 Joint Search에서는 Candidate Size가 작을수록 내부 NDCG가 비정상적으로 증가했습니다.

```text
Candidate Size ↓
→ Candidate Recall ↓
→ Candidate 내부 NDCG ↑
```

원인은 **Candidate에 들어오지 못한 실제 Test Positive가 평가에서 충분히 패널티되지 않았기 때문**이었습니다.

따라서 이후 모든 Search / Optuna / Ablation에서는:

```text
전체 Test Positive
        ↓
최종 Top-10 Recommendation
        ↓
Actual Precision / Recall / HR / NDCG / MAP
```

기준을 사용했습니다.

> **잘못 정의된 Metric을 정확하게 최적화하면 잘못된 모델을 매우 효율적으로 찾을 수 있다**는 것이 프로젝트에서 가장 중요한 학습 중 하나였습니다.

---

## 3. Search Process

```text
Candidate Trade-off Sweep
        ↓
Constrained Optuna (50~90)
        ↓
Upper-Bound Sweep (80~120)
        ↓
Final Joint Optuna (90~130)
        ↓
Top Candidates A~E
        ↓
Feature Ablation
        ↓
3-Seed Stability
        ↓
Final E_HR
```

### Guardrail

Search 단계:

```text
Precision@10 >= 0.1065
Recall@10    >= 0.1065
```

Final 5-Fold 재검증:

```text
Precision@10 >= 0.1070
Recall@10    >= 0.1070
```

Guardrail을 통과한 설정 중 **Macro NDCG@10**을 Primary Objective로 사용했습니다.

---

## 4. Final Candidates

| 후보 | Size | I/B/C/U | Features | 역할 / 강점 |
|---|---:|---|---:|---|
| A | 100 | 45/20/15/20 | 16 | NDCG / MAP |
| B | 92 | 42/18/23/9 | 14 | Recall |
| C | 105 | 47/21/16/21 | 14 | Precision / Candidate Recall |
| D | 93 | 42/19/23/9 | 14 | Fold Stability |
| **E** | **101** | **46/20/15/20** | **15** | **HR + Ablation 후 균형** |

### Feature Ablation

E의 초기 Full16에서 `user_popularity_affinity`를 제거했을 때:

| 설정 | P@10 | R@10 | HR@10 | NDCG@10 | MAP@10 | Hits |
|---|---:|---:|---:|---:|---:|---:|
| Full16 | 0.1096 | 0.1110 | 0.622 | 0.1447 | 0.0684 | 1096 |
| **- PopAffinity** | **0.1105** | 0.1113 | **0.623** | **0.1450** | **0.0692** | **1105** |
| - RetrieverCount | 0.1095 | **0.1115** | 0.617 | 0.1436 | 0.0677 | 1095 |
| Full14 | 0.1101 | **0.1115** | 0.620 | 0.1435 | 0.0677 | 1101 |

따라서 **E = Full15**로 확정했습니다.

---

## 5. 3-Seed Stability

Candidate / Feature / XGBoost 조건은 고정하고, LTR1000의 5-Fold split seed만 바꿔 다시 평가했습니다.

| 후보 | Avg P@10 | Avg R@10 | Avg HR@10 | Avg NDCG | Avg MAP | Seed NDCG Std |
|---|---:|---:|---:|---:|---:|---:|
| **E** | **0.110533** | 0.111385 | **0.624667** | **0.145155** | **0.069112** | 0.000977 |
| A | 0.109767 | 0.110979 | 0.618000 | 0.145059 | 0.069087 | **0.000667** |
| D | 0.110133 | **0.111525** | 0.621333 | 0.145009 | 0.068666 | 0.001941 |
| C | 0.110133 | 0.111121 | 0.621667 | 0.144881 | 0.068767 | 0.002064 |
| B | 0.110067 | 0.111407 | 0.617000 | 0.144342 | 0.068749 | 0.000755 |

Primary Metric을 사후 변경하지 않고 **3-Seed 평균 Macro NDCG@10** 기준을 유지해 E를 최종 선택했습니다.

최종 모델명은 **`E_HR`**입니다.

---

# 🧩 Final E_HR Design

## Candidate Retriever

| Retriever | Top-N |
|---|---:|
| Item-Based CF | **46** |
| BPR | **20** |
| Content-Based | **15** |
| User-Based CF | **20** |
| Candidate Union | **max 101** |

## Full15 Features

```text
# Retriever Scores
item_score_norm
bpr_score_norm
content_score_norm
user_score_norm

# Retriever Ranks
item_rank
bpr_rank
content_rank
user_rank

# Agreement
retriever_count

# Candidate Source
is_item_candidate
is_bpr_candidate
is_content_candidate
is_user_candidate

# Context
item_popularity
user_interaction_count
```

`user_popularity_affinity`는 Ablation에서 제거 시 성능이 개선되어 최종 Feature에서 제외했습니다.

## Final XGBoost

| Parameter | Value |
|---|---:|
| objective | `rank:ndcg` |
| eval_metric | `ndcg@10` |
| n_estimators | **54** |
| max_depth | 5 |
| min_child_weight | 2 |
| learning_rate | 0.0428417863 |
| subsample | 0.8621568879 |
| colsample_bytree | 0.7010833569 |
| reg_lambda | 0.8133078460 |
| reg_alpha | 0.4952702650 |
| tree_method | `hist` |
| random_state | 42 |

Cross Validation에서 확인한 `best_iteration` median을 기준으로 최종 `n_estimators=54`를 사용했습니다.

---

# 👤 New User / Cold-Start

기존 BPR에는 신규 사용자의 user factor가 없기 때문에 플레이한 게임의 item embedding을 이용해 새로운 user vector를 초기화합니다.

```text
Played Games
      ↓
BPR Item Embeddings
      ↓
Embedding Mean
      ↓
Initial User Vector
      ↓
Freeze Existing Item Factors
      ↓
User-only BPR Fine-Tuning
      ↓
4-Retriever Candidate Generation
      ↓
Full15
      ↓
E_HR XGBoost
      ↓
Top-10
```

Fine-Tuning:

| Parameter | Value |
|---|---:|
| Epochs | **3** |
| Learning Rate | **0.01** |
| Regularization | **0.001** |

이 방식은 **기존 item representation은 유지하면서 새로운 사용자 vector만 빠르게 적응**시키는 구조입니다.

---

# 🎯 Qualitative Evaluation

신규 사용자는 held-out ground truth가 없기 때문에 Precision / Recall / NDCG를 억지로 계산하지 않고, **실제 추천 결과 + Popularity / Novelty 변화**를 함께 확인했습니다.

## Q1~Q8 Summary

| ID | 시나리오 | Input Log Pop. | Input Novelty | Output Log Pop.@10 | Output Novelty@10 | 핵심 관찰 |
|---|---|---:|---:|---:|---:|---|
| Q1 | 현실적 혼합 취향 | 11.1557 | 9.0512 | 11.4322 | 8.6523 | Action / Open World / Story 주 취향이 강하게 반영 |
| Q2 | 완전히 다른 2장르 | 11.6154 | 8.3880 | 12.0015 | 7.8310 | Shooter가 Cozy보다 강하게 반영 |
| Q3 | 완전히 다른 3장르 | 11.7046 | 8.2593 | 11.8263 | 8.0837 | Shooter / Strategy는 보존, Cozy는 약함 |
| Q4 | 3장르 강화 2+2+2 | 11.3063 | 8.8339 | 11.7858 | 8.1421 | 각 취향 signal을 강화하자 Strategy가 더 선명 |
| Q5 | Popular Only | 12.3299 | 7.3571 | 11.7232 | 8.2325 | 입력보다 약간 덜 인기 있고 더 Novel |
| Q6 | Low-interaction / Long-tail | 3.1355 | 20.6860 | 4.1095 | 19.2614 | niche signal이 명확하면 Long-tail 유지 가능 |
| Q7 | Popular 3 + Long-tail 3 | 7.7899 | 13.9390 | 11.8511 | 8.0479 | Popular signal이 Long-tail을 크게 압도 |
| Q8 | Theme partially controlled | 7.9331 | 13.7245 | 11.6286 | 8.3689 | 장르뿐 아니라 popularity 자체의 영향도 큼 |

### Q1 — 실제 취향에 가까운 사례

Q1 추천 Top-10에는:

```text
God of War
Marvel's Spider-Man Remastered
Grand Theft Auto V Legacy
Cyberpunk 2077
```

처럼 실제로 이미 플레이한 게임 4개가 포함됐습니다.

단일 사용자 사례이므로 전체 정확도로 일반화할 수는 없지만, **주된 Action / Open World / Story 취향은 자연스럽게 포착**했습니다.

### Q6 — Long-tail Signal

Low-interaction 게임만 입력했을 때:

```text
1. Geneforge 2
2. Geneforge 5: Overthrow
3. Geneforge 4: Rebellion
4. Geneforge 1
```

처럼 niche series를 상위에 배치했습니다.

즉 시스템이 무조건 인기작만 추천하는 것은 아니며, **취향 signal이 충분히 명확하면 Long-tail preference도 유지할 수 있음**을 확인했습니다.

### Animal Theme 추가 실험

`Stray` 단독 입력에서는 `Ori and the Will of the Wisps`, `Cult of the Lamb`처럼 동물/생물 character와 관련된 결과가 일부 있었지만, 전체적으로는 Story / Adventure 계열이 더 강했습니다.

`ANIMAL WELL`은 metadata에는 존재했지만 현재 BPR item mapping에는 없어 신규 user vector 생성에 활용할 수 없었습니다.

이 결과는 **세부 테마 signal보다 collaborative / genre / gameplay signal이 더 강하게 작동**하는 동시에, **모델별 item universe 불일치가 Cold-Start 처리의 한계가 될 수 있음**을 보여줬습니다.

---

# 🧱 Model Design

## Content-Based

```text
Steam Metadata
    ↓
Genres / Tags / Categories / Developers / Publishers
    ↓
Combined Features
    ↓
TF-IDF
    ↓
User Profile
    ↓
Cosine Similarity
    ↓
Top-N
```

현재 Content feature에는 게임 제목 자체를 사용하지 않습니다.

따라서 title에 특정 단어가 들어간다는 사실보다 **Tags / Genres / Categories / Developer / Publisher metadata**가 content similarity에 직접 반영됩니다.

---

## User-Based CF

```text
User × Item Sparse Matrix
        ↓
Target User Vector
        ↓
User ↔ User Cosine Similarity
        ↓
Top-K Neighbor
        ↓
Similarity Weighted Score
        ↓
Seen Item 제거
        ↓
Top-N
```

User-Based CF는 독립 성능은 Item-Based보다 낮았지만 Hybrid에서는 **보조 Retriever**로 활용할 가치가 있었습니다.

---

## Item-Based CF

```text
User × Item Matrix
        ↓ transpose
Item × User Matrix
        ↓
Source Items
        ↓
Item ↔ Item Cosine Similarity
        ↓
Positive Top-K Neighbor
        ↓
Score Aggregation
        ↓
Top-N
```

프로젝트 전반에서 가장 강한 signal 중 하나였고, 최종 E_HR에서는 **Top-46 Candidate Retriever**로 사용합니다.

---

## Funk-SVD

<details>
<summary><b>Funk-SVD 실험 요약 보기</b></summary>

초기 biased SVD:

```text
P@10    = 0.0003
R@10    = 0.0006
HR@10   = 0.0025
NDCG@10 = 0.0004
```

Bias를 제거했을 때 일부 개선:

```text
P@10    = 0.0037
R@10    = 0.0030
HR@10   = 0.0350
NDCG@10 = 0.0042
```

현재 데이터의 implicit Top-N ranking 문제에서는 **rating prediction objective와 추천 목적의 불일치**가 컸습니다.

</details>

---

## BPR

BPR은 positive item이 negative item보다 높은 score를 갖도록 pairwise ranking을 학습합니다.

```text
Stage 1
True Interaction > Unseen

Stage 2
True Interaction > Explicit False
```

Final BPR:

```python
iterations = 15
factors = 60
learning_rate = 0.05
regularization = 0.006

explicit_false_epochs = 1
explicit_false_learning_rate = 0.01
explicit_false_regularization = 0.001
```

Final standalone result:

| Metric | Score |
|---|---:|
| P@10 | 0.05225 |
| R@10 | 0.06870 |
| HR@10 | 0.3850 |
| NDCG@10 | 0.06968 |

최종 Hybrid에서는 **Top-20 Candidate Retriever**로 사용합니다.

---

# 🐛 Important Debugging & Methodology Findings

<details>
<summary><b>1. User-Based CF Data Leakage</b></summary>

초기 Precision@10이 비정상적으로 높게 나왔고, 평가 matrix에 Test interaction이 남아 있는 문제를 발견했습니다.

호출 chain과 argument 전달까지 추적해 수정한 뒤:

```text
Precision@10 = 0.0545
```

수준의 정상적인 결과를 확인했습니다.

**교훈:** 여러 파일로 분리된 Pipeline에서는 내부 함수만 보는 것보다 **데이터가 어떤 경로로 전달되는지 Outside-In 방식으로 확인**해야 합니다.

</details>

<details>
<summary><b>2. NDCG Ranking Order Loss</b></summary>

추천 결과를 너무 일찍 `set`으로 변환해 ranking order가 사라지는 문제를 발견했습니다.

```text
recommended_list → NDCG
recommended_set  → Precision / Recall
```

으로 역할을 분리했습니다.

</details>

<details>
<summary><b>3. Candidate 내부 NDCG Artifact</b></summary>

Retriever가 Candidate에 넣은 item만 대상으로 NDCG를 계산하면 Candidate에서 놓친 실제 positive가 충분히 반영되지 않았습니다.

이를 **전체 Test Positive 기준 Actual Top-10 평가**로 수정했습니다.

</details>

<details>
<summary><b>4. Model Item Universe Mismatch</b></summary>

일부 게임은 metadata에는 존재하지만 BPR item mapping에는 존재하지 않았습니다.

이 때문에 신규 사용자 입력에서 특정 게임을 representation에 사용할 수 없는 문제가 발생했습니다.

향후에는 전처리 단계에서 **모든 모델의 item universe / ID integrity를 먼저 검사**할 필요가 있습니다.

</details>

---

# 📐 Evaluation

## Dataset

| 항목 | 값 |
|---|---:|
| Total Recommendations | **41,154,794** |
| Train | **37,113,471** |
| Test | **4,041,323** |
| BPR Matrix | **13,781,059 × 37,567** |
| BPR nnz | **37,113,455** |
| Filtered evaluation users | **666,781** |
| LTR Search Users | **1,000** |
| Final Evaluation Users | **400** |

평가 사용자는 interaction 10~78개 구간에서:

```text
10~15개   100명
16~25개   100명
26~45개   100명
46~78개   100명
```

으로 구성했습니다.

## Final Data Protocol

```text
LTR1000
    ↓
Candidate / Feature / XGB Search
    ↓
Feature Ablation
    ↓
3-Seed Stability
    ↓
E_HR Final Selection
    ↓
LTR1000 전체로 Final Ranker Training
    ↓
Original Final400 One-Time Evaluation
```

`Final400`은 Candidate Size, Feature, XGB parameter, 최종 모델 선택에 사용하지 않았습니다.

중간에 `Selection200 + Final Test200` 분기를 한 번 시도했지만, 원래 최종 평가용으로 보존한 Final400 일부를 모델 선택에 사용하게 되므로 **공식 최종 방법론에서는 폐기**했습니다.

## Metrics

### Accuracy / Ranking

- Precision@10
- Recall@10
- Hit Rate@10
- NDCG@10
- MAP@10
- Candidate Recall
- Total Hits

### Recommendation Characteristics

Mean Log Popularity:

```text
mean(log(1 + interaction_count))
```

Novelty:

```text
mean(-log2(interaction_count / total_interactions))
```

정확도뿐 아니라 **추천이 얼마나 인기 item 중심인지, 상대적으로 얼마나 novel한지**도 함께 봤습니다.

---

# ⚡ Performance Optimization

대규모 데이터에서 반복 실험이 가능하도록 Sparse 연산, Batch, Cache, Checkpoint를 적극적으로 사용했습니다.

| 작업 | Before / Problem | Optimization / Result |
|---|---|---|
| Global Split | 사용자별 split 반복 | interaction count별 position 재사용 → 약 5초 |
| Item-CF Reranking | 약 1811~1848초 | neighbor precompute + cache |
| Reverse Reranking | 매우 느린 item similarity | cache 구축 포함 234.7초 |
| Cached Reranking | 반복 계산 | 약 **0.3초** |
| Case 3 Fusion | 반복 retrieval | cached 약 **3초** |
| Ratio Sweep | 7개 구조 반복 | 약 **1.75분** |
| Candidate Size Sweep | 다수 size 비교 | 약 **7.81초** |
| 4-Retriever Feature | 최대 약 90~117초/user | Item batch + User sparse batch |
| User-CF Batch | full user matrix 반복 | 약 **0.4~0.6초/user** 수준 |
| Final400 첫 optimized run | Runtime 호출식 평가 병목 | 약 **6.51분** |
| Final400 cache reuse | 동일 feature 반복 계산 | 약 **0.03분** |

Final400 최적화 구조:

```text
Fixed Final400
      ↓
Train History Preload
      ↓
Item-CF Unique Source Batch Precompute
      ↓
User-CF Sparse Batch
      ↓
BPR / Content User-level Scoring
      ↓
Candidate Feature Cache
      ↓
XGBoost Batch Prediction
      ↓
Final Metrics
```

> 최종적으로 **XGBoost 예측 자체보다 Retriever / Feature Generation이 실제 계산 병목**이라는 것을 확인했습니다.

---

# 🔗 Identifier Flow

```text
Game Name
    ↓
Resolve Candidate
    ↓
Steam AppID
    ↓
Internal Mapping / Matrix Index
    ↓
Recommendation
    ↓
AppID
    ↓
Game Name
```

게임 이름은 중복될 수 있기 때문에 내부 로직은 가능한 한 **Steam AppID 기준**으로 처리합니다.

---

# 🛠 Tech Stack

| Category | Stack |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy, PyArrow |
| Sparse Matrix | SciPy CSR / LIL |
| ML | Scikit-learn |
| Matrix Factorization | Surprise |
| Pairwise Ranking | `implicit` BPR |
| Learning-to-Rank | XGBoost Ranker / LambdaMART |
| Hyperparameter Search | Optuna |
| Content | TF-IDF, Cosine Similarity |
| Evaluation | Precision / Recall / HR / NDCG / MAP / Popularity / Novelty |
| Visualization | Matplotlib |
| Storage | CSV / Parquet / NPZ / JSON |
| Future Deployment | Database + API + Web / Cloud |

### Implemented / Explored

- TF-IDF Content-Based Recommendation
- User-Based CF
- Item-Based CF
- Funk-SVD
- Bayesian Personalized Ranking
- Explicit Negative Fine-Tuning
- Reranking
- Multi-Retriever Architecture
- Weighted Rank Fusion
- Cross-Ranker
- XGBoost LambdaMART
- Candidate Size / Retriever Ratio Sweep
- Optuna Joint Search
- Feature Ablation
- Multi-Seed Stability
- Cold-Start User Vector Adaptation
- Popularity / Novelty Evaluation
- Batch / Cache Runtime Optimization

---

# 📂 Project Structure

현재 Git의 기본 구조를 유지하면서, 최종 정리 후에는 아래처럼 **Runtime / Evaluation / Experiment / Docs 역할을 구분**하는 형태를 목표로 합니다.

```text
GAME-RECOMMENDATION-SYSTEM/
│
├── main.py
│   └── Final E_HR Runtime Recommendation
│
├── main_final_e_hr_with_quantitative.py
│   └── Fixed Final400 Quantitative Evaluation
│
├── main_final_e_hr_with_qualitative.py
│   └── New-user Qualitative Evaluation
│
├── preprocessing.py
├── data_split.py
├── evaluation.py
├── requirements.txt
├── Readme.md
├── .gitignore
│
├── models/
│   ├── content_base.py
│   ├── userbase.py
│   ├── itembase.py
│   ├── Funk_SVD.py
│   ├── bpr.py
│   ├── hybrid.py
│   │
│   └── saved_model/                 # Git 제외
│       ├── bpr_grid/
│       ├── ltr_cache/
│       ├── runtime_cache/
│       ├── case3_cache/
│       ├── results/
│       ├── xgb_ranker_final_e_hr.json
│       └── xgb_ranker_final_e_hr_metadata.json
│
├── hybrid_arctech_experiment/
│   ├── baseline_item.py
│   ├── exp_a_reranking_*.py
│   ├── exp_b_multi_retriever.py
│   ├── exp_b_ranker_comparison.py
│   ├── exp_b_candidate_size_sweep.py
│   ├── exp_b_item_ranker_ratio_sweep.py
│   ├── exp_b_retriever_ratio_finetune.py
│   ├── exp_b_bpr_grid_search.py
│   ├── exp_c_fusion.py
│   ├── exp_c_fusion_ablation.py
│   ├── exp_c_user_weight_sweep.py
│   ├── exp_d_xgb_ranker_exp1.py
│   └── ...                          # LTR / Optuna / Ablation / Stability experiments
│
├── docs/
│   ├── Day01.md
│   ├── ...
│   ├── Day25.md
│   ├── Day26.md
│   ├── Day27.md
│   └── experiment_summary.md
│
├── notebook/
│
└── data/                            # Git 제외
    ├── raw/
    ├── cache/
    └── split/
        ├── mf_train.parquet
        └── mf_test.parquet
```

> 현재 Git에는 Final Runtime / Quantitative / Qualitative 실행 파일과 Day26 / Day27 / Experiment Summary까지 반영되어 있습니다.

---

# 🚧 Known Limitations

## 1. Popularity / Novelty Imbalance

Long-tail 취향만 주어졌을 때는 niche recommendation이 가능했지만, Popular + Long-tail 취향이 함께 들어오면 인기 signal이 Top-10을 크게 지배했습니다.

다음 프로젝트에서는:

```text
Relevance
+ Novelty
+ Diversity
+ Popularity Bias Control
```

을 함께 고려하는 **Multi-Objective Ranking / Reranking**을 다뤄보고 싶습니다.

## 2. Item Universe / Data Integrity

일부 item은 metadata에는 존재하지만 특정 model mapping에는 없었습니다.

다음에는 모델 개발 전에:

```text
Raw Data
→ ID / Type 정리
→ Duplicate / Missing 검사
→ Model별 Item Universe 비교
→ Train / Validation / Test 고정
→ Data Integrity Test
```

를 먼저 끝내는 방향으로 개선할 계획입니다.

## 3. Multi-Interest Competition

여러 취향이 동시에 들어오면 강한 취향이나 인기 signal이 약한 취향을 덮을 수 있습니다.

현재 Ranker는 Diversity 자체를 최적화하는 모델이 아니므로, 향후에는 multi-interest representation / diversification도 추가 실험 대상입니다.

## 4. Experiment File Organization

프로젝트가 길어지면서 experiment script가 많아지고 이름도 복잡해졌습니다.

다음 프로젝트에서는 시작부터:

```text
experiments/
    01_baseline/
    02_candidate_search/
    03_ranker_search/
    04_feature_ablation/
    05_stability/
    06_final_evaluation/

configs/
results/
artifacts/
src/
```

처럼 실험 목적과 결과 저장 구조를 분리할 계획입니다.

## 5. Local / Offline 중심

현재 프로젝트는 Python / DataFrame / Local Cache 중심입니다.

다음에는 **Database + SQL + API + Web + Cloud Deployment**까지 포함한 실제 서비스 형태를 목표로 합니다.

---

# 🗺 Roadmap

## ✅ Completed

- [x] Steam metadata preprocessing / Parquet caching
- [x] Global Train/Test Split
- [x] Content-Based Recommendation
- [x] User-Based CF
- [x] Item-Based CF
- [x] Funk-SVD implementation / diagnosis
- [x] Implicit BPR full training
- [x] Explicit False Fine-Tuning
- [x] BPR parameter search
- [x] Hybrid Reranking
- [x] Multi-Retriever
- [x] Weighted Rank Fusion / Ablation
- [x] Cross-Ranker
- [x] Candidate Size / Retriever Ratio Search
- [x] XGBoost Learning-to-Rank
- [x] 4-Retriever Feature Generation Optimization
- [x] Actual Top-10 Evaluation redesign
- [x] Constrained Optuna
- [x] Candidate Upper-Bound Sweep
- [x] Final Joint Optuna
- [x] Feature Ablation
- [x] 3-Seed Stability
- [x] **Final E_HR Selection**
- [x] **LTR1000 Final Training**
- [x] **Final400 One-Time Evaluation**
- [x] MAP / Popularity / Novelty
- [x] New User BPR Adaptation
- [x] Qualitative Evaluation
- [x] Final400 Batch / Cache Optimization

## 🔜 Future Work

- [ ] Popularity / Novelty Multi-Objective Ranking
- [ ] More rigorous item-universe preprocessing
- [ ] Experiment Config / Result directory refactoring
- [ ] Pre-designed Grid + Optuna search protocol
- [ ] SQL / Database-based interaction pipeline
- [ ] API / Web UI
- [ ] Cloud deployment
- [ ] Large-scale retrieval model such as Two-Tower exploration

---

# 📌 Final Status

```text
Version
Final — Day27

Final Candidate Retrieval
Item-CF 46
BPR 20
Content 15
User-CF 20
        ↓
Candidate ≤ 101

Final Feature
Full15

Final Ranker
XGBoost LambdaMART
n_estimators = 54

Final400
P@10      = 0.096250
R@10      = 0.127890
HR@10     = 0.570000
NDCG@10   = 0.139857
MAP@10    = 0.069833
Cand.R    = 0.272015
Novelty   = 10.354832
Hits      = 385

Project Status
Completed
```

---

## Final Project Direction

이 프로젝트는 **단일 추천 알고리즘의 성능 비교**에서 시작했지만, 최종적으로는 다음 문제를 함께 다루는 프로젝트가 되었습니다.

```text
Data Preprocessing
        ↓
Standalone Recommenders
        ↓
Candidate Retrieval
        ↓
Learning-to-Rank
        ↓
Evaluation Design
        ↓
Hyperparameter / Architecture Search
        ↓
Feature Ablation
        ↓
Stability Validation
        ↓
Cold-Start
        ↓
Popularity / Novelty
        ↓
Runtime Optimization
```

최종 시스템은 사용자가 플레이한 Steam 게임을 입력하면:

```text
Played Games
      ↓
New User Representation
      ↓
4-Retriever Candidate Generation
      ↓
Full15 Feature Engineering
      ↓
E_HR XGBoost Ranker
      ↓
Top-10 Steam Game Recommendation
```

형태로 동작합니다.

프로젝트에서 가장 크게 얻은 것은 특정 모델 하나의 구현보다, **추천시스템에서 데이터·Retrieval·Ranking·Evaluation·Experiment Design·Runtime Engineering이 서로 연결되어 있다는 경험**입니다.
