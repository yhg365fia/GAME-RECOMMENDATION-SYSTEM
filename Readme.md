# 🎮 Steam Game Recommendation System

Steam 게임 데이터를 활용해 **Content-Based / Collaborative Filtering / Matrix Factorization / Pairwise Ranking / Hybrid Recommendation**을 단계적으로 구현하고, 동일한 평가 환경에서 추천 구조와 성능을 비교하는 프로젝트입니다.

이 프로젝트의 핵심은 단순히 여러 추천 모델을 구현하는 데 그치지 않고, **각 모델이 어떤 signal을 학습하는지 분석하고 Retrieval과 Ranking의 역할을 분리해 최종 Hybrid Architecture를 설계하는 것**입니다.

---

## 🚀 At a Glance

| 항목 | 현재 상태 |
|---|---|
| 데이터 | Steam 약 4,115만 interaction |
| 평가 방식 | Global Train/Test Split + 고정 평가 사용자 400명 |
| 최종 출력 | Top-10 Recommendation |
| 현재 Architecture | **BPR 56 + Content 39 + User-Based 5 → Item-Based Ranker** |
| Candidate Budget | **100** |
| 현재 최고 P@10 | **0.08722** |
| 현재 최고 R@10 | **0.11102** |
| 현재 최고 HR@10 | **0.5400** |
| 현재 최고 NDCG@10 | **0.12155** |
| 현재 단계 | **Hybrid 미세조정 완료 → Learning-to-Rank 실험 및 최종 시스템 통합 단계** |

### Current Recommendation Architecture

```text
User History
    │
    ├── BPR Retriever ───────────── Top-56
    ├── Content-Based Retriever ─── Top-39
    └── User-Based CF Retriever ─── Top-5
                    │
                    ▼
             Candidate UNION
             Budget ≈ 100
                    │
                    ▼
            Item-Based Ranker
                    │
                    ▼
               Top-10
```

현재 구조는 **semi-confirmed Final Baseline Architecture**입니다.  
Retriever 비율 Fine-Tuning과 BPR Hyperparameter Search까지 완료했으며, 이제 Candidate 구조는 유지한 채 **Learning-to-Rank(XGBoost Ranker / LambdaMART)가 Item-Based Ranker보다 더 나은 최종 순서를 학습할 수 있는지 검증**하는 단계입니다.

---

# 📊 Current Best Result

Hybrid 실험은 모두 **같은 Global Split / 같은 평가 사용자 400명 / 같은 Positive 정의 / 같은 Top-K / 같은 Evaluator**를 사용합니다.

| Metric | Item-Based Hybrid Baseline | **현재 반확정 구조** | 개선폭 |
|---|---:|---:|---:|
| Precision@10 | 0.06325 | **0.08722** | +37.9% |
| Recall@10 | 0.08144 | **0.11102** | +36.3% |
| Hit Rate@10 | 0.3925 | **0.5400** | +37.6% |
| NDCG@10 | 0.09129 | **0.12155** | +33.2% |
| Hits | 253 | **348** | +95 |

---


# 🧪 Day25 — Final Fine-Tuning & Learning-to-Rank Transition

Day25에서는 기존 Hybrid 구조를 더 이상 크게 바꾸지 않고, **현재 구조 내부의 마지막 미세조정과 Ranker 개선 가능성**을 검토했습니다.

## Retriever Ratio Fine-Tuning

Candidate Size=100, User-Based=5를 고정하고 BPR / Content 비율을 미세하게 조정했습니다.

| BPR / Content / User | Avg UNION | Scoreable | UNION Recall | Scoreable Recall | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 / 45 / 5 | 95.38 | 32.79 | 0.2453 | 0.2038 | 0.08656 | **0.11111** | 0.5300 | **0.12157** | 345 |
| 53 / 42 / 5 | 95.38 | 33.27 | 0.2470 | 0.2050 | 0.08672 | 0.11072 | 0.5325 | 0.12120 | 346 |
| **56 / 39 / 5** | 95.36 | 33.70 | 0.2488 | 0.2055 | **0.08722** | 0.11102 | **0.5400** | 0.12155 | **348** |
| 59 / 36 / 5 | 95.44 | 34.15 | 0.2538 | 0.2087 | 0.08666 | 0.10997 | 0.5350 | 0.12139 | 346 |
| 62 / 33 / 5 | 95.49 | 34.66 | **0.2540** | **0.2097** | 0.08612 | 0.10957 | 0.5275 | 0.12087 | 344 |

### Result

BPR 비율이 커질수록 Candidate Recall / Scoreable Recall은 증가했지만 최종 Top-10 성능은 개선되지 않았습니다.

```text
Candidate Recall ↑
        ≠
Final Ranking Performance ↑
```

따라서 Retriever 비율은 기존 **BPR56 + Content39 + User5**를 유지합니다.

---

## BPR 27-Case Grid Search

Hybrid Candidate 구조를 고정하고 BPR의:

```text
factors        = [40, 60, 80]
regularization = [0.002, 0.006, 0.010]
iterations     = [10, 15, 20]
```

을 조합해 총 **27개 모델**을 비교했습니다.

### Best BPR

| Parameter / Metric | Result |
|---|---:|
| Factors | **60** |
| Regularization | **0.006** |
| Iterations | **15** |
| BPR Candidate Recall | **0.183040** |
| UNION Candidate Recall | **0.248277** |
| Item Scoreable Recall | **0.205562** |
| Precision@10 | **0.086599** |
| Recall@10 | **0.109385** |
| HR@10 | **0.522500** |
| NDCG@10 | **0.120657** |
| Hits | **345** |

2위 조합인 `factors=40 / reg=0.01 / iterations=20`은 UNION Candidate Recall이 **0.252776**으로 더 높았지만, NDCG@10은 **0.117489**로 낮았습니다.

이 실험에서도 다시 **Retriever 품질과 Final Ranking 품질은 별개의 문제**임을 확인했습니다.

Final BPR:

```python
iterations = 15
factors = 60
learning_rate = 0.05
regularization = 0.006

explicit_false = True
explicit_false_epochs = 1
explicit_false_learning_rate = 0.01
explicit_false_regularization = 0.001
```

---

## Why Learning-to-Rank Next?

현재 Item-Based Ranker는 Candidate 100에서는 강하지만 Candidate Size를 150 / 200으로 늘렸을 때 추가 정답을 충분히 Top-10으로 끌어올리지 못했습니다.

따라서 다음 실험은 Candidate Retriever를 바꾸지 않고:

```text
BPR56 + Content39 + User5
        ↓
Candidate UNION
        ↓
[기존] Item-Based Ranker

vs

BPR56 + Content39 + User5
        ↓
Candidate UNION
        ↓
[실험] XGBoost Ranker (LambdaMART)
```

로 **Ranker만 교체**해서 비교합니다.

---

## Learning-to-Rank Feature Plan

Feature를 한 번에 많이 넣지 않고, **Ablation 형태로 단계적으로 추가**합니다.

### Experiment 1 — Model Signal Only

각 모델의 normalized score + candidate set 내부 rank:

```text
item_score_norm
bpr_score_norm
content_score_norm
user_score_norm

item_rank
bpr_rank
content_rank
user_rank
```

**총 8개 Feature**

목적:

> 기존 네 추천 모델의 출력만으로 XGBoost Ranker가 Item-Based Ranker를 이길 수 있는가?

### Experiment 2 — Candidate / User Context

Exp1 +

```text
source_count
user_interaction_count
```

**총 10개 Feature**

- `source_count`: Candidate를 몇 Retriever가 동시에 추천했는가
- `user_interaction_count`: 사용자의 Train interaction 수

### Experiment 3 — Item Statistics

Exp2 +

```text
game_popularity
game_positive_ratio
```

**총 12개 Feature**

- `game_popularity`: Train interaction count
- `game_positive_ratio`: Train에서 positive feedback 비율

모든 Item 통계는 **Train data에서만 계산**해 Test Leakage를 방지합니다.

### Optional

Exp3까지 확인한 뒤 필요할 때만:

```text
genre_similarity
tag_similarity
developer_similarity
publisher_similarity
```

등을 추가합니다.

프로젝트 막바지이므로 Feature를 무작정 늘리지 않고 **Exp1 → Exp2 → Exp3의 증가분을 비교**해 최종 Feature Set을 결정합니다.

---

## LTR Train / Validation / Final Test

기존 Hybrid 평가 사용자 400명은 계속 **Final Evaluation 전용**으로 유지합니다.

```text
LTR Users ≈ 1,000
├── Train      ≈ 800
└── Validation ≈ 200

Final Test
└── 기존 고정 400명
```

한 사용자의 Candidate Set이 하나의 Ranking Query가 됩니다.

XGBoost 입력 한 행은:

```text
(user_id, candidate_app_id)
```

이며 `user_id`, `app_id`는 identifier이고 실제 Feature에는 포함하지 않습니다.

---

## LTR Runtime Strategy

첫 실행의 가장 큰 병목은 XGBoost 자체가 아니라 **BPR / Content / User-CF / Item-CF score를 Candidate별로 생성하는 Feature Engineering**입니다.

예상 최초 실행시간:

| 단계 | 예상 |
|---|---:|
| Train / Interaction Matrix 준비 | 2~8분 |
| LTR 1,000명 Feature 생성 | 30~90분 |
| XGBRanker 학습 | 수십 초~2분 |
| Final 400명 Feature 생성 + 평가 | 10~35분 |
| **전체 최초 실행** | **약 45분~2시간** |

한 번 Feature Cache를 만들고 나면 이후 Exp1/2/3과 XGBoost Hyperparameter 실험은 훨씬 빠르게 반복할 수 있습니다.

---


# 🧠 Key Conclusions

프로젝트를 진행하며 얻은 핵심 결론입니다.

1. **모델 복잡도보다 데이터 구조와 추천 목적이 더 중요했습니다.**
   - Funk SVD는 rating prediction에는 적합하지만 Top-N ranking 목적과 맞지 않아 성능이 매우 낮았습니다.
   - 반대로 Item-Based CF는 Steam 데이터의 강한 co-consumption signal을 잘 활용했습니다.

2. **Retrieval과 Ranking은 별개의 문제였습니다.**
   - Candidate Recall이 높아져도 최종 Top-10 성능이 반드시 좋아지지는 않았습니다.
   - Candidate를 많이 확보하는 모델과 최종 순서를 잘 정하는 모델의 역할을 분리했습니다.

3. **같은 모델을 Retriever와 Ranker에 동시에 쓰면 self-ranking 문제가 발생할 수 있습니다.**
   - BPR Retriever → BPR Ranker에서는 이미 BPR이 높게 평가한 후보를 다시 BPR로 정렬해 Pure BPR Top-10과 동일해지는 현상을 확인했습니다.

4. **Item-Based CF는 최종 Ranker 역할에서 가장 강했습니다.**
   - Cross-Ranker 실험에서 Candidate Recall이 더 높은 조합보다 Item-Based Ranker를 쓴 구조가 최종 성능에서 우세했습니다.

5. **User-Based CF는 약하지만 보완적인 signal이 있었습니다.**
   - 메인 모델로는 약했지만 5% 수준의 Retriever로 추가했을 때 Unique Hit을 확보하며 최종 성능을 개선했습니다.

6. **Candidate Size는 클수록 좋은 것이 아니었습니다.**
   - 50 / 100 / 150 / 200을 비교한 결과 Candidate Recall은 계속 증가했지만 최종 Top-10은 **100**에서 가장 좋았습니다.

---

# 📈 Model Performance

## Standalone Model Comparison

| 모델 | P@10 | R@10 | HR@10 | NDCG@10 | 비고 |
|---|---:|---:|---:|---:|---|
| Content-Based (TF-IDF) | 0.0268 | 0.0238 | 0.2250 | 0.0286 | 콘텐츠 semantic 유사성 |
| User-Based CF | 0.0545 | 0.0427 | 0.3011 | 0.0655 | User neighborhood, Sparsity에 취약 |
| **Item-Based CF** | **0.0783** | **0.0880** | **0.4800** | **0.1078** | **Standalone 최고 성능**, 강한 co-consumption signal |
| Funk SVD (biased) | 0.0003 | 0.0006 | 0.0025 | 0.0004 | item bias가 랭킹 지배 |
| Funk SVD (unbiased) | 0.0037 | 0.0030 | 0.0350 | 0.0042 | bias 제거로 개선되나 절대 성능 낮음 |
| **Final BPR** | **0.05225** | **0.06870** | **0.3850** | **0.06968** | Explicit False 활용 + 최종 튜닝 |

> Standalone 성능 자체보다 Hybrid 단계에서는 **각 모델이 다른 모델이 놓친 정답을 보완하는가**를 더 중요하게 평가했습니다.

## Hybrid Architecture Comparison

| 단계 | 구조 | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Item-Based 단독 | 0.0633 | 0.0814 | 0.3925 | 0.0913 | 253 |
| Case 1 | Item Top-30 → BPR Rerank | 0.0695 | 0.0888 | 0.4525 | 0.0978 | 278 |
| Case 1 Reverse | BPR Top-30 → Item Rerank | 0.0695 | 0.0874 | 0.4675 | 0.0977 | 278 |
| Case 2 (1차) | Multi-Retriever → BPR Ranking | 0.0673 | 0.0862 | 0.4425 | 0.0953 | 269 |
| Case 3 (초기) | 4-Model Weighted Rank Fusion | 0.0775 | 0.1001 | 0.4675 | 0.1134 | 310 |
| Case 3 (User 5%) | Item .5588 + BPR .2235 + Content .1676 + User .05 | 0.0800 | 0.1027 | 0.4825 | 0.1148 | 320 |
| Cross-Ranker | BPR59 + Content41 → Item Ranker | 0.0827 | 0.1061 | 0.5225 | 0.1168 | 330 |
| **Case 2 최종** | **BPR56 + Content39 + User5 → Item Ranker** | **0.0872** | **0.1110** | **0.5400** | **0.1216** | **348** |

---

# 🧩 Hybrid Architecture Experiments

## 1. Case 1 — Reranking

```text
Item-Based Top-30 → BPR Reranking → Top-10
BPR Top-30        → Item-Based Reranking → Top-10
```

| 실험 | Hits |
|---|---:|
| Item-Based Baseline | 253 |
| Item → BPR | 278 |
| BPR → Item | 278 |

두 방향 모두 자체 Top-10이 놓친 정답을 더 넓은 Candidate 안에서 재발견했습니다.

Candidate Generation만 보면:

- Item-Based Recall@30: **0.1491**, Hits **468**
- BPR Recall@30: **0.1350**, Hits **422**

Reverse 실험에서는 Item-Based scoring 특성상 각 source item의 Top-K neighbor를 전체 item 공간에서 찾아야 해 느렸습니다.

최적화 과정:

```text
초기 평가 약 1847.7초
        ↓
unique source item만 계산
+ norm 사전 계산
+ batch sparse multiplication
+ Item→Item Top-K cache
        ↓
cache 구축 포함 234.7초
        ↓
cached reranking 자체 약 0.3초
```

---

## 2. Case 2 — Multi-Retriever → Ranker

초기 구조:

```text
Item-Based + User-Based + Content
              ↓
            UNION
              ↓
          BPR Ranking
              ↓
            Top-10
```

Candidate Recall은 증가했지만 최종 Hits는 **269**에 그쳤습니다.

### Self-Ranking 문제

BPR을 Retriever에도 넣은 뒤 동일한 BPR score로 다시 Ranking하면:

```text
BPR Retriever
    ↓
이미 BPR 고득점 후보 확보
    ↓
BPR Ranker
    ↓
Pure BPR Top-10과 사실상 동일
```

즉 **같은 모델을 Retriever와 Ranker에 동시에 쓰면 두 단계가 중복될 수 있음**을 확인했습니다.

---

## 3. Cross-Ranker Experiment

이 문제를 해결하기 위해 **Ranker는 Candidate Retrieval에 참여하지 않는 구조**를 비교했습니다.

| Retriever | Ranker | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---|---:|---:|---:|---:|---:|
| **BPR59 + Content41** | **Item-Based** | **0.0827** | **0.1061** | **0.5225** | **0.1168** | **330** |
| Item71 + BPR29 | Content-Based | 0.0600 | 0.0799 | 0.4275 | 0.0841 | 240 |
| Item78 + Content22 | BPR | 0.0705 | 0.0880 | 0.4750 | 0.0963 | 282 |

Candidate Recall이 가장 높은 구조가 최종 Top-10도 가장 좋은 것은 아니었습니다.

이 실험을 통해 **Item-Based를 최종 Ranker로 확정**했습니다.

---

## 4. Retriever Ratio Sweep

Item-Based Ranker를 고정하고 BPR / Content / User-Based 후보 비율을 비교했습니다.

| 후보 구성 | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---:|---:|---:|---:|---:|
| BPR59 + Content41 | 0.0827 | 0.1061 | 0.5225 | 0.1168 | 330 |
| BPR70 + Content30 | 0.0801 | 0.1028 | 0.5025 | 0.1143 | 320 |
| BPR50 + Content50 | 0.0819 | 0.1052 | 0.5175 | 0.1168 | 326 |
| BPR40 + Content60 | 0.0809 | 0.1032 | 0.5050 | 0.1135 | 322 |
| **BPR56 + Content39 + User5** | **0.0872** | **0.1110** | **0.5400** | **0.1216** | **348** |
| BPR67 + Content28 + User5 | 0.0849 | 0.1079 | 0.5275 | 0.1196 | 339 |
| BPR48 + Content47 + User5 | 0.0850 | 0.1094 | 0.5250 | 0.1199 | 339 |

User-Based는 독립 성능은 상대적으로 낮지만, 약 **5%의 weak auxiliary retriever**로 사용할 때 BPR/Content가 놓친 후보를 보완했습니다.

---

## 5. Candidate Size Sweep

Retriever 비율 **56 : 39 : 5**를 고정한 뒤 Candidate Size를 비교했습니다.

| Size | Candidate Recall | Scoreable Recall | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.1710 | 0.1449 | 0.0780 | 0.0976 | 0.4875 | 0.1104 | 308 |
| **100** | 0.2488 | 0.2055 | **0.0872** | **0.1110** | **0.5400** | **0.1216** | **348** |
| 150 | 0.3005 | 0.2413 | 0.0835 | 0.1078 | 0.5300 | 0.1188 | 334 |
| 200 | **0.3386** | **0.2669** | 0.0840 | 0.1081 | 0.5225 | 0.1197 | 336 |

Candidate Recall은 계속 증가했지만 최종 성능은 100에서 최고였습니다.

이 결과는:

> **현재 Item-Based Ranker가 100개를 초과해 추가된 정답 후보를 Top-10으로 충분히 끌어올리지 못하고 있을 가능성**

을 보여주며, Learning-to-Rank를 검토하게 된 핵심 근거입니다.

---

## 6. Weighted Rank Fusion / Ablation

Case 3 초기 가중치:

```text
Item    0.50
User    0.15
BPR     0.20
Content 0.15
```

각 모델을 하나씩 제거한 결과:

| 실험 | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---:|---:|---:|---:|---:|
| Full | 0.0775 | 0.1001 | 0.4675 | 0.1134 | 310 |
| - Item | 0.0598 | 0.0774 | 0.4200 | 0.0881 | 239 |
| - User | 0.0793 | 0.1019 | 0.4800 | 0.1125 | 317 |
| - BPR | 0.0683 | 0.0874 | 0.4275 | 0.1023 | 273 |
| - Content | 0.0753 | 0.0963 | 0.4575 | 0.1087 | 301 |

기여도는 다음과 같은 경향을 보였습니다.

```text
Item-Based >> BPR > Content-Based >> User-Based
```

User weight를 0.05로 줄인 뒤 Fusion Hits는 **310 → 320**으로 개선되었습니다.

하지만 동일 조건에서 Retriever→Ranker 구조가 Hits **348**을 기록해 최종 방향은 Weighted Fusion보다 **Multi-Retriever + Single Ranker**로 결정했습니다.

---

## 7. Complementarity / Unique Hit

| 모델 | Standalone Hits | Unique Hits |
|---|---:|---:|
| Item-Based | 253 | 130 |
| User-Based | 144 | 70 |
| BPR | 209 | 124 |
| Content-Based | 112 | 61 |
| 4개 모델 모두 공통 | - | 8 |

각 모델이 상당히 다른 정답을 잡고 있어 Hybrid 구성의 근거가 되었습니다.

Case 3 Fusion에서는 Item-Based 대비:

- 새로 얻은 정답: **115**
- 손실한 정답: **58**
- 순증가: **+57**

---

# 🧱 Model Design

## Content-Based

```text
Steam Metadata
    ↓
Genres / Tags / Categories / Developers / Publishers
    ↓
Combined Text Features
    ↓
TF-IDF Vectorization
    ↓
User Profile
    ↓
Cosine Similarity
    ↓
Top-N
```

### Qualitative Findings

**텍스트에 없는 특성은 포착할 수 없음**

`Party Animals` 입력에서 장르와 인원수는 유사했지만 "동물 캐릭터"라는 시각적 테마는 반영되지 않았습니다.

**장르 혼합 시 쏠림 현상**

카드/덱빌딩 + 슈팅 게임을 함께 입력했을 때 카드/덱빌딩 계열로 추천이 강하게 쏠리는 현상을 확인했습니다.

---

## User-Based CF

```text
User History
    ↓
User × Item Sparse Matrix (+1 / -1)
    ↓
Train App IDs로 Query Vector 생성
    ↓
User ↔ User Cosine Similarity
    ↓
자기 자신 제외
    ↓
Top-K Neighbor (k=30)
    ↓
Σ(similarity × interaction) / Σ|similarity|
    ↓
Seen Item 제거 + Positive Score Filtering
    ↓
Top-N
```

### Sparsity Analysis

| 비교 변수 | Pearson r | p-value |
|---|---:|---:|
| n_games ↔ n_recommended | +0.3096 | < 0.0001 |
| n_games ↔ precision | +0.2504 | < 0.0001 |

interaction이 적을수록 안정적인 neighbor 형성이 어려워지는 방향과 일관된 결과를 확인했습니다.

---

## Item-Based CF

```text
User × Item Matrix
    ↓ transpose
Item × User Matrix
    ↓
User Train Items = Source Items
    ↓
Source Item ↔ All Items Cosine Similarity
    ↓
Self-Similarity 제거
    ↓
Source Item별 Positive Top-K (k=30)
    ↓
Candidate Similarity Sum Aggregation
    ↓
Seen Item 제거
    ↓
Top-N
```

| 구분 | User-Based CF | Item-Based CF |
|---|---|---|
| Matrix 방향 | User × Item | Item × User |
| Query | 필요 | 불필요 |
| Similarity | User ↔ User | Item ↔ Item |
| Candidate Score | similarity × interaction weighted prediction | 동일 candidate로 들어오는 item similarity 합 |

Hybrid에서는 **Candidate Retriever가 아니라 Final Ranker**로 사용합니다.

---

## Funk SVD — Completed

<details>
<summary><b>Funk SVD 실험 과정과 실패 원인 보기</b></summary>

### Pipeline

```text
Global Train/Test Split
    ↓
mf_train.parquet 37,113,471
mf_test.parquet   4,041,323
    ↓
Surprise SVD
n_factors=100
n_epochs=20
lr_all=0.005
reg_all=0.02
    ↓
Top-N Recommendation
    ↓
400명 Evaluation
```

초기 결과:

```text
P@10    0.0003
R@10    0.0006
HR@10   0.0025
NDCG@10 0.0004
```

### 가설 검증

1. `positive_only=False` 재평가  
   → Hits 변화 없음 → 가설 기각

2. 추천 게임 Train 통계 분석  
   → 추천 게임 대부분 True Ratio 0.94~1.00  
   → item bias가 ranking을 지배하는 패턴 발견

3. `biased=False` Ablation  
   → HR@10 0.0025 → 0.0350으로 개선

하지만 절대 성능은 여전히 매우 낮았습니다.

### Final Conclusion

**rating prediction objective와 Top-N ranking objective의 불일치**를 근본 원인으로 판단하고 Funk SVD 단계를 종료했습니다.

</details>

---

## BPR — Completed

<details open>
<summary><b>Final BPR 구조 및 핵심 실험</b></summary>

BPR은 `(u, i, j)` triplet을 사용해 positive item이 negative item보다 높은 score를 갖도록 학습합니다.

\[
x_{uij}=p_u^T(q_i-q_j)
\]

\[
L=-\log\sigma(x_{uij})+\lambda(\|p_u\|^2+\|q_i\|^2+\|q_j\|^2)
\]

### Training Strategy

```text
1단계
True Interaction
    ↓
implicit BPR
True > Unseen

2단계
Explicit Negative Fine-Tuning
True > False
```

`implicit`은 non-zero interaction을 positive signal처럼 다루기 때문에 `is_recommended=False`를 그대로 넣으면 의미가 왜곡될 수 있었습니다.

따라서:

```text
True > Unseen
      ↓
True > False Fine-Tuning
```

으로 분리했습니다.

### Parameter Search

| 실험 | 핵심 결과 |
|---|---|
| Explicit False 효과 | P@10 0.02525 → 0.03650, Hits 101 → 146 |
| Regularization | 0.0001~0.020 중 0.005 우수 |
| Factors | 20/40/80에서 metric별 trade-off |
| Factors × Reg | 60 / 0.005 우수 |
| Local Search | **60 / 0.006 최종 선택** |

### Final BPR

```python
iterations = 15
factors = 60
learning_rate = 0.05
regularization = 0.006

explicit_false = True
explicit_false_epochs = 1
explicit_false_learning_rate = 0.01
explicit_false_regularization = 0.001

positive_only = True
```

Final Result:

| Metric | Score |
|---|---:|
| P@10 | 0.05225 |
| R@10 | 0.06870 |
| HR@10 | 0.3850 |
| NDCG@10 | 0.06968 |
| Hits | 209 |

BPR 단계에서 얻은 핵심 결론:

> **Problem Definition > Data Signal > Hyperparameter**

Hybrid에서는 BPR을 **Candidate Retriever**로 사용합니다.

</details>

---

# 🐛 Important Debugging Findings

<details>
<summary><b>User-Based CF 데이터 누수</b></summary>

초기 평가에서 Precision@10이 **0.6254**로 비정상적으로 높게 나왔습니다.

원인:

- 평가용 interaction matrix가 Train/Test 미분리 원본으로 생성됨
- 평가 대상 사용자의 전체 interaction이 neighbor pool에 남음
- Test 정답이 예측 score에 직접 유입됨

수정 과정:

```text
exclude_user_idx 추가
    ↓
지표 변화 없음
    ↓
호출 chain 추적
    ↓
run_evaluation()에서 user_to_idx 전달 누락 발견
    ↓
수정
    ↓
P@10 = 0.0545
```

이 경험을 통해 여러 파일로 분리된 파이프라인에서는 내부 구현만 보는 것보다 **호출 chain과 인자 전달을 먼저 검증하는 Outside-In debugging 원칙**을 정리했습니다.

</details>

<details>
<summary><b>NDCG 순위 정보 소실 버그</b></summary>

추천 결과를 바로 `set(result["app_id"])`로 변환하면서 추천 순서가 사라지고 있었습니다.

수정:

```text
recommended_list
→ 순서 유지, NDCG 계산

recommended_ids
→ set 변환, Precision / Recall 계산
```

수정 후 NDCG@10은 **0.0655**로 확인했습니다.

</details>

<details>
<summary><b>Self-Similarity 처리</b></summary>

User-Based와 Item-Based에서 self-similarity는 서로 다른 문제를 만듭니다.

- User-Based: 자기 interaction이 neighbor로 들어오면 **data leakage**
- Item-Based: `A ↔ A = 1.0`이 항상 Top-K를 차지하는 **trivial similarity**

공통적으로 Top-K 이전에 self-similarity를 제거합니다.

```python
similarities = cosine_similarity(...)
similarities[self_idx] = 0
```

</details>

---

# 📐 Evaluation

## Dataset Split

Model-Based CF와 Hybrid는 저장된 Global Train/Test Split을 사용합니다.

```text
Total interactions  : 41,154,794
Train               : 37,113,471
Test                :  4,041,323
```

평가 대상 사용자:

- interaction 10~78개
- 총 **666,781명**
- 사용자별 70% Train / 30% Test
- 나머지 사용자는 100% Train

기존 `train_test_split()` 666,781회 호출 구조를 interaction count별 position을 재사용하도록 바꿔 **최대 69회 수준**으로 줄였고 split 생성 시간은 약 5초까지 최적화했습니다.

## Hybrid Evaluation Users

Hybrid Architecture 비교에는 동일한 400명을 고정해 사용합니다.

| Review Group | Users |
|---|---:|
| 10–15 | 100 |
| 16–25 | 100 |
| 26–45 | 100 |
| 46–78 | 100 |
| **Total** | **400** |

## Metrics

- Precision@K
- Recall@K
- Hit Rate@K
- NDCG@K
- Micro Precision / Recall / F1
- Candidate Recall
- Scoreable Recall
- Common / Unique Hit
- Recovered / Lost Hit
- Pearson Correlation
- 그룹별 Breakdown
- Qualitative Evaluation
- MAP@K 예정
- Popularity / Coverage 분석 예정

---

# ⚡ Performance Optimization

대규모 데이터를 반복 실험하기 위해 계산 결과를 적극적으로 캐싱했습니다.

| 작업 | 최적화 결과 |
|---|---|
| Global Split | 66만 회 split → 최대 69회 |
| BPR | `implicit` Cython/C + CSR + multicore |
| Item-Based 400명 평가 | 약 1811~1848초 |
| Reverse cache 구축 포함 | 234.7초 |
| Cached Reranking | 약 0.3초 |
| Case 3 Cached Fusion | 약 3초 |
| Cross-Ranker 초기 | 약 18.1분 |
| Ratio Sweep 7종 | 약 1.75분 |
| Candidate Size Sweep 전체 | 약 7.81초 |

주요 cache:

```text
Item→Item Neighbor Cache
BPR Candidate Cache
Case3 Retrieval Cache
Evaluation Subset Cache
Content TF-IDF Cache
```

---

# 🔗 Identifier Flow

```text
Game Name
    ↓
동명이인 시 후보 선택
    ↓
AppID
    ↓
game_to_idx
    ↓
Internal Matrix Index
    ↓
Recommendation
    ↓
AppID
    ↓
Game Name
```

Game Name은 중복될 수 있지만 **AppID는 고유**하므로 내부 추천 로직은 AppID 기준으로 통일했습니다.

---

# 🛠 Tech Stack

| Category | Stack |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy, PyArrow |
| Sparse Matrix | SciPy CSR / LIL |
| ML | Scikit-learn |
| Matrix Factorization | Surprise |
| Pairwise Ranking | implicit BPR |
| Learning-to-Rank | XGBoost Ranker / LambdaMART (`rank:ndcg`) |
| Content | TF-IDF, Cosine Similarity |
| Visualization | Matplotlib |
| Deployment | FastAPI / Streamlit (planned) |
| Learning-to-Rank | XGBoost Ranker / LambdaMART (`rank:ndcg`, experiment in progress) |

### Implemented Algorithms

- TF-IDF Content-Based Recommendation
- User-Based CF
- Item-Based CF
- Funk SVD
- Bayesian Personalized Ranking
- Explicit Negative Fine-Tuning
- Reranking
- Multi-Retriever Architecture
- Weighted Rank Fusion
- Cross-Ranker
- Candidate Size / Retriever Ratio Sweep
- Hybrid Candidate Diagnostics

---

# 📂 Project Structure

```text
Game-Recommendation-System/
│
├── data/                              # Git 제외
│   ├── raw/
│   ├── cache/                         # Parquet cache
│   └── split/
│       ├── mf_train.parquet
│       └── mf_test.parquet
│
├── docs/
│   ├── Day01.md
│   ├── ...
│   ├── Day23.md
│   ├── Day24.md
│   └── Day25.md
│
├── hybrid_arctech_experiment/
│   ├── baseline_item.py               # Hybrid Item-Based baseline
│   ├── exp_a_reranking.py             # Case 1 / Reverse
│   ├── exp_b_multi_retriever.py       # Case 2 / Cross-Ranker
│   ├── exp_c_fusion.py                # Case 3 / Ablation / Weight Sweep
│   ├── exp_d_ltr_dataset.py           # LTR Feature Dataset
│   ├── exp_d_ltr_train.py             # XGBoost Ranker Training
│   └── exp_d_ltr_evaluation.py        # Item Ranker vs LTR Evaluation
│
├── models/
│   ├── content_base.py
│   ├── userbase.py
│   ├── itembase.py
│   ├── Funk_SVD.py
│   ├── bpr.py
│   ├── hybrid.py                      # 최종 Hybrid 구현용
│   │
│   ├── run_model/
│   │   ├── runsvd.py
│   │   └── run_bpr.py
│   │
│   └── saved_model/                   # Git 제외
│       ├── bpr_grid/
│       ├── hybrid_cache/
│       ├── case3_cache/
│       └── results/
│           ├── hybrid_sampled_users.csv
│           ├── item_top200_candidates.csv
│           └── Architecture별 평가 CSV
│
├── notebook/
├── preprocessing.py
├── data_split.py
├── evaluation.py
├── main.py
├── Readme.md
├── requirements.txt
└── .gitignore
```

`models/saved_model/`의 대형 모델과 cache는 Git에서 제외하고 로컬에서 재사용합니다.

`models/hybrid.py`는 Architecture 확정 후 최종 추천 구조를 정식 구현하기 위한 파일입니다.

---

# 🚧 Known Issues / To-Do

- **Name = NaN metadata mismatch**
  - Item-Based CF 정성평가 중 일부 게임 이름이 NaN으로 조회됨
  - 원인 분석 보류

- **Party Animals 입력 시 빈 DataFrame**
  - 정성평가 대상에서 제외
  - 원인 분석 보류

- **하위 폴더 직접 실행 시 import path**
  - `ModuleNotFoundError: No module named 'data_split'`
  - Project Root import 구조로 해결 방향 정리

- **User-Based Candidate Coverage**
  - Hybrid 실험에서 최대 400명 중 362명만 후보 생성
  - 평균 후보 약 14.6개
  - Sparsity 영향 지속

- **Retriever + 동일 Ranker 중복**
  - BPR Retriever + BPR Ranker에서 확인
  - Cross-Ranker 구조로 해결

- **Candidate Size > 100**
  - Candidate Recall은 증가하지만 최종 성능 하락
  - Item-Based Ranker 한계 가능성
  - Learning-to-Rank로 개선 여부 검토

- **실험 저장 경로 통일**
  - `models/saved_model/`
  - `results/`
  - `case3_cache/`
  - `hybrid_cache/`
  - 최종 리팩토링에서 Config Source 단일화 예정

- **XGBoost Environment**
  - `.venv`에 `xgboost` 설치 필요
  - LTR 실험 전 `requirements.txt` 최종 동기화 예정

- **Dependency 동기화**
  - `implicit`, Surprise, PyArrow 등과 `requirements.txt` 최종 동기화 필요

---

# 🗺 Roadmap

## ✅ Completed

- [x] Steam metadata preprocessing / Parquet caching
- [x] User-based / Global Train-Test Split
- [x] Content-Based Recommendation
- [x] User-Based CF
- [x] Item-Based CF
- [x] Precision / Recall / HR / NDCG + Macro / Micro evaluator
- [x] User-Based data leakage diagnosis and fix
- [x] NDCG implementation bug fix
- [x] Funk SVD implementation / evaluation / ablation / 종료
- [x] BPR 직접 구현
- [x] `implicit` 기반 37M full training
- [x] Explicit False Fine-Tuning
- [x] BPR 27-case Grid Search
- [x] **Final BPR = factors 60 / reg 0.006 / iter 15**
- [x] Hybrid common evaluation environment
- [x] Item-Based Hybrid Baseline
- [x] Case 1 / Case 1 Reverse
- [x] Case 2
- [x] Case 3 Weighted Rank Fusion
- [x] Case 3 Ablation
- [x] User-Based Weight Sweep
- [x] Self-Ranking 문제 발견
- [x] Cross-Ranker
- [x] Item-Based Ranker 확정
- [x] Retriever Ratio Sweep
- [x] **Retriever Ratio Fine-Tuning → 56 / 39 / 5 유지**
- [x] Candidate Size Sweep → **100 최적**
- [x] Hybrid Architecture 반확정
- [x] Retrieval / Neighbor caching optimization
- [x] LTR 개념 / Feature / Dataset 구조 설계
- [x] LTR Exp1 / Exp2 / Exp3 계획 수립
- [x] 신규 사용자 BPR 처리 방식 설계
- [x] Final evaluation metric 확장 계획 수립

## 🚧 In Progress — Learning-to-Rank

### Exp1 — 8 Features

- [ ] 4 normalized model scores
- [ ] 4 model ranks
- [ ] XGBRanker (`rank:ndcg`) 학습
- [ ] Item-Based Ranker와 동일 Candidate Set에서 비교

### Exp2 — 10 Features

- [ ] `source_count`
- [ ] `user_interaction_count`

### Exp3 — 12 Features

- [ ] `game_popularity`
- [ ] `game_positive_ratio`

### Optional

- [ ] genre / tag / developer / publisher similarity
- [ ] 필요 최소 XGBoost parameter tuning
- [ ] Final LTR Feature Set 결정

## ⏳ Final System Integration

- [ ] Item-Based vs Final XGBoost Ranker 최종 비교
- [ ] 최종 Architecture 확정
- [ ] Final Ranker 저장
- [ ] `models/hybrid.py`에 최종 recommender class 통합
- [ ] `main.py` 실제 사용자 입력 흐름 연결

### New User BPR Adaptation

```text
Played Games
    ↓
BPR Item Embedding lookup
    ↓
Item Embedding Mean
    ↓
Initial User Vector
    ↓
Freeze existing Item Factors
    ↓
User-only BPR Fine-Tuning
    ↓
New User BPR Scores
```

- [ ] 신규 User Factor 구현
- [ ] BPR56 + Content39 + User5 Candidate Retrieval 연결
- [ ] Final Ranker 연결
- [ ] Top-10 Game Name / Score 출력
- [ ] End-to-End 실제 사용 테스트

## ⏳ Final Evaluation

### Accuracy / Ranking Quality

- [ ] Precision@10
- [ ] Recall@10
- [ ] Hit Rate@10
- [ ] NDCG@10
- [ ] **MAP@10**

### Recommendation Characteristics

- [ ] **Mean Log Popularity@10**
- [ ] **Novelty@10**
- [ ] Popularity Bias 분석
- [ ] Coverage
- [ ] Sparsity Group별 성능
- [ ] Personalization / Explainability
- [ ] 실행시간 비교
- [ ] Candidate / Ranking 역할 분석
- [ ] 모델별 기여 분석

## ⏳ Project Close

- [ ] Final quantitative / qualitative evaluation
- [ ] 시스템 한계 분석
- [ ] README / Docs 최신화
- [ ] Day 회고 정리
- [ ] 프로젝트 공식 종료
- [ ] FastAPI / Streamlit Deployment (선택)

---

# 📌 Current Status

```text
Version
V3.5 — Day25

Stage
Hybrid Fine-Tuning Complete
→ Learning-to-Rank Experiment
→ Final System Integration

Current Candidate Retrieval
BPR 56
+ Content 39
+ User-Based 5
        ↓
Candidate ≈ 100

Current Baseline Ranker
Item-Based

Current Best Hybrid Result
P@10    0.08722
R@10    0.11102
HR@10   0.5400
NDCG@10 0.12155
Hits    348

Final BPR
factors = 60
reg     = 0.006
iter    = 15

LTR Plan
Exp1: 8 features
Exp2: 10 features
Exp3: 12 features

Next
XGBRanker Exp1
→ Feature Ablation
→ Item Ranker vs XGB Ranker
→ Final Architecture
→ main.py / recommender integration
→ MAP / Popularity / Novelty
→ Final Evaluation
→ Project Close
```

---

## Final Project Direction

프로젝트의 남은 목표는 **새로운 Retriever나 새로운 Hybrid 구조를 계속 추가하는 것**이 아닙니다.

```text
1. 현재 Candidate 구조 고정
2. Learning-to-Rank로 Ranking 개선 가능성 검증
3. 최종 Ranker 결정
4. 실제 사용자 입력이 가능한 recommender로 통합
5. 정확도 + Popularity / Novelty까지 최종 평가
6. 문서화 후 프로젝트 종료
```

최종적으로는:

```text
사용자가 플레이한 Steam 게임 입력
        ↓
신규 사용자 Feature / BPR User Factor 생성
        ↓
BPR56 + Content39 + User5 Candidate Retrieval
        ↓
Final Ranker
(Item-Based 또는 XGBoost Ranker)
        ↓
Top-10 Steam Game Recommendation
```

형태의 실제 사용 가능한 추천 파이프라인을 목표로 합니다.
