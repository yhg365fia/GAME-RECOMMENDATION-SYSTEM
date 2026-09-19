# Day22 — Hybrid 설계 구체화, Collaborative 모델 관계 재해석 및 Item-Based Baseline 구축

## 1. 오늘 한 일 요약

BPR까지 Standalone 모델 실험을 마치고 『Hands-On Recommendation』의 Hybrid 파트까지 읽은 뒤, 오늘부터 본격적으로 **Hybrid Recommendation System 설계 단계**에 들어갔다.

바로 Hybrid 코드를 만들기보다 지금까지 구현한 Content-Based / User-Based / Item-Based / Funk-SVD / BPR을 다시 펼쳐놓고, **각 모델이 같은 Steam interaction 데이터를 어떻게 다르게 해석하는지**, 그 차이가 Hybrid에서 어떤 역할로 이어질 수 있는지를 먼저 분석했다.

```text
전체 모델 성능표 재확인
        ↓
Steam 데이터 특성 분석
        ↓
Item-Based를 Hybrid 중심 모델로 설정
        ↓
Item/User/BPR이 같은 interaction을 어떻게 다르게 사용하는지 재정리
        ↓
BPR의 latent factor / preference space 재이해
        ↓
BPR을 초기 Candidate Filter로 사용하는 가설
        ↓
단독 filtering의 Recall Ceiling 문제 발견
        ↓
Multi-Retriever / Reranking / Fusion으로 사고 확장
        ↓
Hybrid Architecture A/B/C 실험안 설정
        ↓
프로젝트 폴더 및 저장 구조 정리
        ↓
동일 평가환경의 Item-Based Baseline 구축 (400명 평가 완료)
        ↓
Candidate Recall@20/50/100/200 실험 설계
        ↓
반복 연산 방지를 위해 Top-200 Candidate 저장 결정
```

---

## 2. 전체 모델 성능표 재분석과 Steam 데이터 해석

### 2-1. 단일 모델 성능표

| 모델 | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Content-Based | 0.0268 | 0.0238 | 0.2250 | 0.0286 |
| User-Based CF | 0.0545 | 0.0427 | 0.3011 | 0.0655 |
| **Item-Based CF** | **0.0783** | **0.0880** | **0.4800** | **0.1078** |
| Funk-SVD biased | 0.0003 | 0.0006 | 0.0025 | 0.0004 |
| Funk-SVD unbiased | 0.0037 | 0.0030 | 0.0350 | 0.0042 |
| BPR Final | 0.05225 | 0.06870 | 0.3850 | 0.06968 |

이 표를 단순한 성능 순위가 아니라 **"왜 Steam 데이터에서는 이런 순서가 나왔을까?"** 라는 질문으로 보기 시작했다.

Item-Based가 Precision뿐 아니라 Recall / HR / NDCG 모두 가장 높았다는 점에서, Steam 데이터에는 **게임 사이의 공동 소비 패턴이 강하게 존재할 가능성**이 있다고 생각했다.

```text
이 게임을 한 사람들이 어떤 다른 게임도 같이 했는가?
```

그래서 첫 번째 Hybrid 가설은 **Item-Based를 가장 큰 축이자 Anchor Model로 둔다**로 잡았다. 가장 높은 숫자를 가진 모델이기 때문만이 아니라, Steam이라는 도메인의 interaction 구조와 실험 결과가 맞물린다고 보았기 때문이다.

### 2-2. Steam 데이터는 완전한 Implicit Dataset인가?

처음에는 꽤 강한 implicit 성향의 데이터라고 표현했지만, 논의하면서 더 정확하게 수정했다.

```text
게임과 interaction함              → implicit collaborative signal
is_recommended = True / False    → 사용자가 직접 남긴 binary preference
```

즉 **interaction 기반 collaborative signal이 강하고, 그 안에 explicit binary preference도 포함된 데이터**로 보는 것이 더 정확하다.

이 구분은 BPR에서도 중요했다. `False`를 "단순히 Positive가 아닌 Interaction"으로 볼지, "실제 사용자가 명시한 Negative Preference"로 볼지에 따라 모델의 problem definition 자체가 달라진다.

---

## 3. Collaborative 모델 관계 재해석 (Item-Based / User-Based / BPR)

### 3-1. 왜 추천이 겹칠 수 있는가?

> BPR도 결국 collaborative interaction을 이용한다면 Item-Based나 User-Based와 뭐가 다른가?

세 모델 모두 같은 **User-Item Interaction 구조**에서 패턴을 찾으므로 추천이 일정 부분 겹치는 것은 당연하다. 다른 것은 **같은 interaction에서 무엇을 직접적인 관계로 정의하느냐**이다.

| 모델 | 직접적으로 보는 관계 | 핵심 질문 |
| --- | --- | --- |
| **Item-Based** | Item ↔ Item | 이 게임과 같이 소비되는 게임은? |
| **User-Based** | User ↔ User | 이 사용자와 비슷한 사용자는? |
| **BPR** | User ↔ Item latent preference | 이 사용자에게 어떤 item을 더 높은 순위에 둬야 하는가? |

### 3-2. Item-Based가 학습하는 관계

Terraria + Stardew Valley, Hades + Dead Cells처럼 반복적으로 같이 등장하는 게임 쌍의 관계가 강해지고, 사용자가 Terraria를 가지고 있다면 함께 소비되는 게임(Stardew Valley, Starbound, Don't Starve ...)을 추천한다. 즉 **사용자가 소비한 게임 주변에서 실제 공동소비 관계가 강한 게임을 찾아가는 local neighborhood 방식**이다.

Steam에서 Item-Based가 강했던 것은 이런 item-item 공동소비 구조가 실제로 유효한 signal이었기 때문일 가능성이 있다. 다만 이것은 현재 데이터를 보고 세운 **가설**이며 도메인 일반 법칙으로 단정하지 않는다.

### 3-3. User-Based가 보는 관계

게임끼리 직접 비교하지 않고, interaction이 많이 겹치는 사용자를 유사하다고 판단한 뒤 그 사용자가 소비했지만 나는 소비하지 않은 게임을 추천한다.

```text
나 → 나와 비슷한 사용자 → 그 사용자가 좋아한 게임
```

Item-Based와 User-Based는 모두 neighborhood CF지만 Item neighborhood냐 User neighborhood냐가 다르다.

### 3-4. BPR이 보는 관계

BPR은 Item-Item / User-User similarity를 직접 만들지 않고 **User Vector와 Item Vector**를 학습한다. 목표는 사용자가 좋아한 Item이 좋아하지 않았거나 Unseen인 Item보다 높은 점수를 받게 하는 것이다.

```text
User A: Positive = Hades, Negative = FIFA
→ score(User A, Hades) > score(User A, FIFA)
```

이런 pairwise comparison을 반복하며 **좋은 순위를 만들기 위해 사용자와 게임이 latent preference 공간의 어디에 위치해야 하는지**를 학습한다.

### 3-5. "Interaction을 Latent Preference Space로 압축한다"는 의미

사용자 한 명의 interaction을 그대로 표현하면 게임 수만큼 큰 차원이 필요하지만, BPR은 이를 `factors=60`, 즉 60개의 값으로 표현한다.

```text
수많은 User-Item Interaction → 60차원 User / Item Latent Vector
```

그래서 `factors=60`을 **사용자와 게임을 각각 60개의 숨은 취향 좌표를 가진 벡터로 표현한다**는 의미로 이해했다.

Latent Factor는 `Factor 1 = RPG` 같은 사람이 정한 장르 feature가 아니라, ranking objective에 유리하도록 자동으로 형성되는 축이다. 하나의 factor 안에 인디 성향, 싱글 플레이 성향, 특정 사용자 집단, 공동 소비 구조, 가격대 등이 섞일 수 있어 **latent(숨겨진 표현)** 라고 부른다.

### 3-6. 같은 데이터인데 Item-Based와 BPR이 겹치는 이유, 그리고 다른 점

"Terraria를 좋아한 사용자들이 Stardew Valley도 많이 좋아한다"는 패턴이 있다면,

- Item-Based: `Terraria ↔ Stardew Valley`라는 높은 similarity로 직접 표현
- User-Based: 이 둘을 같이 소비하는 사용자 집단끼리 유사하다고 표현
- BPR: 수많은 ranking pair를 학습하며 두 게임이 비슷한 사용자에게 높은 점수를 받도록 latent vector 위치가 형성

즉 같은 collaborative structure를 서로 다른 방식으로 포착하므로 결과가 겹칠 수 있다. 그래도 완전히 같은 모델은 아니다.

> **Item-Based = 직접적인 Local Item-Item 관계**
> **BPR = Interaction 전체를 압축한 간접적인 Global/Latent Preference**

다만 이를 "BPR은 무조건 Global, Item-Based는 무조건 Local"이라는 절대적 분류로 받아들이지는 않고, 모델 행동을 이해하기 위한 **직관적인 해석**으로만 사용한다.

### 3-7. PCA, Funk-SVD와의 비교

**PCA와 BPR**: 둘 다 고차원 데이터를 저차원 latent representation으로 바꾸지만 목적이 다르다.

```text
PCA → 원본 데이터의 분산/정보를 잘 보존하는 공간
BPR → Positive Item을 Negative/Unseen Item보다 높게 Ranking하기 좋은 공간
```

따라서 BPR의 latent factor와 PCA의 principal component는 같은 개념이 아니다.

**Funk-SVD와 BPR**: 둘 다 `User Latent Vector × Item Latent Vector` 구조를 쓰지만 최적화 대상이 다르다. Funk-SVD는 예측 score와 실제 interaction/rating의 차이를 줄이는 prediction objective, BPR은 Positive > Negative/Unseen이라는 상대적 ranking 관계를 최적화한다.

```text
Funk-SVD  P@10 ≈ 0
BPR       P@10 ≈ 0.052
```

이 큰 차이는 "BPR이 더 좋은 모델"이라기보다 **Top-N Recommendation이라는 문제 정의와 BPR의 학습 objective가 더 잘 맞았다**는 관점으로 해석했다. BPR 단계에서 얻은 **Problem Definition > Data Signal > Hyperparameter**라는 교훈과 다시 연결된다.

---

## 4. Hybrid 가설의 진화 — Candidate Filter에서 Multi-Retriever로

### 4-1. 첫 가설: BPR을 Candidate Filter로

```text
BPR (Broad preference filter)
 ↓ 사용자의 Latent Preference에 맞는 게임 Candidate Filtering
Item-Based / Content-Based (Detailed recommendation)
 ↓
최종 Ranking
```

BPR이 전체적인 취향으로 후보를 좁히고, Item-Based/Content-Based가 그 안에서 정교하게 추천하는 역할 분리를 생각했다.

### 4-2. BPR 단독 Candidate Filter의 문제 발견 (Recall Ceiling)

BPR이 후보에서 어떤 게임을 제거하면 뒤의 모델은 그 게임을 다시 살릴 수 없다.

```text
실제 정답 X
BPR Top-100 → X 없음
Item-Based → X를 높은 순위에 놓을 수 있음
Content-Based → X와 매우 유사한 게임 존재
그래도 BPR Candidate Filtering에서 X 제거 → 후속 모델이 X를 평가할 방법 없음
```

> **BPR을 유일한 Candidate Generator로 두면 전체 시스템 Recall의 상한이 BPR Candidate Recall에 묶일 수 있다.**

좋은 Ranking Model이 뒤에 있어도 정답이 Candidate Pool에 없으면 할 수 있는 일이 없다.

### 4-3. "BPR은 Ranking에 안 맞는다"는 최초 해석의 수정

처음에는 "BPR은 filtering에는 어울리지만 최종 Ranking에는 덜 어울린다"고 생각했지만, BPR의 objective 자체가 **pairwise ranking**이므로 이 주장은 맞지 않는다. 정확한 표현은:

> **현재 Steam 데이터와 현재 BPR 구현에서는 Item-Based보다 Standalone Top-N Ranking 성능이 낮다.**

**모델의 일반적 원리 ≠ 내 데이터에서 관찰된 모델 행동**임을 구분해야 한다는 것을 다시 배웠다.

### 4-4. Multi-Retriever로 확장

"왜 후보를 하나의 모델에서만 받아야 하지?"라는 질문에서 다음 구조를 떠올렸다.

```text
Item-Based Candidate ─┐
BPR Candidate ────────┼→ UNION → Candidate Pool → Ranking
Content Candidate ────┘
```

BPR 실패는 Item-Based가, Item-Based 실패는 BPR이, Collaborative 모두 실패한 경우는 다른 정보원인 Content가 복구할 수 있다. Candidate Generation 단계부터 모델들의 **보완성**을 활용하는 구조다.

### 4-5. Hybrid를 보는 질문의 변화

```text
기존: 어떤 모델이 가장 좋은가?
현재: 이 모델은 다른 모델이 모르는 무엇을 알고 있는가?
```

BPR Hit가 많아도 전부 Item-Based가 이미 맞힌 정답이라면 Hybrid에서 새로운 정보는 거의 없다. 반대로 Content-Based 성능이 낮아도 Item-Based와 BPR이 모두 실패한 정답을 Content만 맞히는 경우가 일정하게 있다면 중요한 역할을 가질 수 있다. 앞으로 분석할 항목:

```text
Common Hit / Item-Based Unique Hit / BPR Unique Hit / Content Unique Hit
Hybrid가 새로 살린 Hit / Hybrid가 오히려 잃은 Hit
```

특히 "Item-Based 실패 & BPR 성공" 정답이 상당수 존재한다면 **Item-Based의 직접적인 neighborhood로 포착하지 못한 latent preference를 BPR이 보완하고 있을 가능성**이 생긴다. Hit Overlap / Unique Hit 분석은 단순 추가 지표가 아니라 **Local Item-Item 관계와 Latent Preference가 실제로 보완적인가?** 를 검증하는 실험이다.

---

## 5. Hybrid Architecture 실험 설계

### 5-1. 현재 모델별 Hybrid 역할 가설

| 모델 | 현재 Hybrid 역할 가설 |
| --- | --- |
| **Item-Based** | Hybrid의 중심. 가장 강한 Candidate / Ranking 기본 signal |
| **BPR** | User-Item latent preference로 Item-Based가 놓치는 패턴 보완 가능 |
| **Content-Based** | Collaborative 데이터와 다른 semantic information 제공 |
| **User-Based** | 비슷한 사용자 집단 행동을 활용하는 추가 collaborative signal |
| Funk-SVD | 현재 성능상 핵심 Hybrid 후보에서 우선 제외 |

어디까지나 **실험 전 가설**이며 다음 순서로 확인한다.

```text
Model Principle + Standalone Performance
      ↓
Role Hypothesis → Hybrid Experiment → Unique Hit / Overlap → Qualitative Analysis
      ↓
실제 역할 결정
```

### 5-2. Item-Based를 Hybrid Anchor로 확정

Precision / Recall / Hit Rate / NDCG 모두 최고라는 일관된 결과와 Steam interaction 구조에 대한 현재 가설 때문이다. Hybrid의 기본 방향은 **Item-Based의 강한 성능을 최대한 보존하면서 Item-Based가 모르는 부분을 다른 모델로 보완**하는 것이다. 다만 `Item-Based가 중심 ≠ Item-Based가 모든 단계를 독점`이라는 점을 구분했다.

### 5-3. Architecture 3개 실험안 (최종 구조는 직접 비교 후 결정)

**Baseline — Item-Based**: `User → Item-Based → Top-10`

**Architecture A — Item-Based Candidate + Reranking**

```text
Item-Based Candidate → Item Score + BPR Score + Content Score → Reranking → Top-10
```

> Item-Based가 Candidate 자체는 충분히 잘 찾고 있으며 다른 모델은 순위만 보완하면 되는가?

**Architecture B — Multi-Retriever**

```text
Item-Based + BPR + Content → Candidate UNION → Ranking → Top-10
```

> Item-Based가 놓친 정답 Candidate를 다른 모델이 실제로 복구해야 하는가?

**Architecture C — Score / Rank Fusion**

```text
Item-Based / BPR / Content Recommendation → Rank / Score Fusion → Top-10
```

> 복잡한 Retrieval-Reranking Pipeline 없이 단순 결합만으로도 개선되는가?

### 5-4. 지금 하지 않는 것

```text
정밀 Weight Search / 상황별 Model Weight / User Interaction 수별 Switching
Confidence-Based Routing / Rule-Based Reranking / Learning-to-Rank / BPR 재튜닝
```

"Sparse User → Content 강화, Dense User → BPR 강화" 같은 상황별 Hybrid는 매력적이지만 가장 마지막의 확장 실험 옵션으로 남겼다. 지금의 원칙은 **Architecture → 역할 확인 → 최적화** 순서다.

---

## 6. 구현 및 실험 결과 (프로젝트 구조, Item-Based Baseline)

### 6-1. 프로젝트 구조 정리

```text
Game-Recommendation-System/
├─ data_split.py
├─ evaluation.py
├─ main.py
├─ preprocessing.py
├─ Readme.md
│
├─ docs/
│  └─ Day01.md ~ Day22.md
│
├─ hybrid_arctech_experiment/
│  ├─ baseline_item.py
│  ├─ exp_a_reranking.py
│  ├─ exp_b_multi_retriever.py
│  └─ exp_c_fusion.py
│
└─ models/
   ├─ bpr.py
   ├─ content_base.py
   ├─ Funk_SVD.py
   ├─ hybrid.py
   ├─ itembase.py
   ├─ userbase.py
   │
   ├─ run_model/
   │  ├─ runsvd.py
   │  └─ run_bpr.py
   │
   └─ saved_model/
      ├─ 저장 모델
      └─ results/
```

`models/`는 모델 구현, `models/run_model/`은 개별 모델 실행, `models/saved_model/`은 모델 저장, `results/`는 실험 결과, `hybrid_arctech_experiment/`는 Hybrid Architecture 실험, `docs/`는 개발 기록으로 역할을 나눴다.

### 6-2. Item-Based Baseline 구현

`models/itembase.py`를 `baseline_item.py`에서 import해 재사용했다(복붙 X). 모든 Hybrid 실험이 **동일한 Item-Based 구현**을 쓰게 하기 위해서다. 기존 `mf_train.parquet` / `mf_test.parquet`을 그대로 사용했고 새로운 split은 만들지 않았다.

### 6-3. 평가 사용자 400명 고정

| Review Group | Users |
| --- | ---: |
| 10–15 | 100 |
| 16–25 | 100 |
| 26–45 | 100 |
| 46–78 | 100 |
| **Total** | **400** |

`models/saved_model/results/hybrid_sampled_users.csv`로 저장했고, 앞으로 모든 Hybrid 실험은 같은 사용자를 사용한다.

### 6-4. Baseline 결과

| 항목 | 값 |
| --- | ---: |
| Train | 37,113,471 |
| Test | 4,041,323 |
| Users | 13,781,059 |
| Games | 37,567 |
| 평가 Users | 400 |
| 평가 시간 | 1811.1초 (≈ 30분 11초) |

| Metric | Result |
| --- | ---: |
| **Precision@10** | **0.06325** |
| **Recall@10** | **0.08144** |
| **Hit Rate@10** | **0.3925** |
| **NDCG@10** | **0.09129** |
| Micro Precision / Recall / F1 | 0.0633 / 0.0773 / 0.0696 |
| **Hits** | **253** |
| 추천 수 / Test Relevant Items | 4,000 / 3,273 |

400명 모두 정확히 10개 추천을 받았다.

**Review Group별 결과**

| Review Group | P@10 | R@10 | HR@10 | NDCG |
| --- | ---: | ---: | ---: | ---: |
| 10–15 | 0.041 | **0.1110** | 0.26 | 0.09657 |
| 16–25 | 0.040 | 0.06986 | 0.28 | 0.06752 |
| 26–45 | 0.064 | 0.07468 | 0.43 | 0.07999 |
| 46–78 | **0.108** | 0.07021 | **0.60** | **0.12109** |

Interaction 수와 Precision 사이에 `Pearson r = 0.2893, p < 0.0001`의 양의 관계가 나타나, interaction이 많은 사용자에서 Item-Based Precision이 어느 정도 좋아지는 경향이 관찰됐다.

### 6-5. 기존 Item-Based와의 차이

| Metric | 기존 Item-Based | Hybrid Baseline |
| --- | ---: | ---: |
| P@10 | 0.0783 | 0.06325 |
| R@10 | 0.0880 | 0.08144 |
| HR@10 | 0.4800 | 0.3925 |
| NDCG | 0.1078 | 0.09129 |

기존보다 낮아졌지만, 앞으로는 **같은 Global MF Split / 같은 평가 사용자 / 같은 Positive 정의 / 같은 Top-K / 같은 Evaluator**에서 Hybrid Architecture만 바뀌므로 현재 결과를 **실제 Hybrid 기준선**으로 사용한다 (P@10 0.06325, R@10 0.08144, HR@10 0.3925, NDCG 0.09129, Hits 253).

---

## 7. 문제 발견과 실험 구조 개선

### 7-1. Item-Based 평가가 오래 걸리는 이유

모델을 저장했다고 추천 결과가 미리 계산되어 있는 것은 아니었다. `recommend()`가 호출될 때마다 다음이 실행된다.

```python
cosine_similarity(source_items, self.item_matrix, dense_output=False)
```

```text
User History → 각 Source Item → 전체 Item과 Cosine Similarity
→ Top-K Neighbor → Score Aggregation → Top-N   (400번 반복)
```

`1811초 / 400 ≈ User당 4.53초`. **학습된 모델을 저장했다는 것과 추천 결과 또는 Item-Item Neighbor를 저장했다는 것은 다르다**는 점을 이해했다.

### 7-2. 전체 Item Similarity Precomputation은 보류

37,567개 전체 Item의 similarity/Top-K neighbors를 미리 계산하면 추론은 빨라지지만, 현재 목적은 **Serving 최적화가 아니라 Hybrid Architecture 검증**이므로 과하다고 판단했다. 필요하면 프로젝트 후반의 Inference Optimization / Serving 관점에서 다시 다룬다.

### 7-3. 다음 실험: Candidate Recall@20/50/100/200

Architecture A를 실험하려면 먼저 **Item-Based Candidate를 몇 개 뽑아야 하는가**를 알아야 한다. 20 / 50 / 100 / 200개를 비교하며 Macro/Micro Candidate Recall, Candidate Hit Rate, Total Hits를 보고, 후보 수 증가에 따라 **정답 포함률이 어디서 포화되는지** 확인한다.

```text
예) 20 → 0.20 / 50 → 0.31 / 100 → 0.36 / 200 → 0.37
    → 100→200 증가가 매우 작으므로 100개를 Architecture A의 후보 수로 사용할 근거
```

### 7-4. Top-200 Candidate Cache 결정

20/50/100/200을 각각 다시 추천하면 같은 비싼 연산을 네 번 반복한다. 그래서 **Top-200을 1회 생성해 저장**하고, Top-20/50/100은 앞에서부터 잘라 쓴다.

```text
저장 파일: models/saved_model/results/item_top200_candidates.csv
형식: user_id | rank | app_id   (400명 × 200 ≈ 최대 80,000행)
```

이후 Candidate Recall 분석, Architecture A, Hit 분석, Item-Based Candidate 분석에서 재사용한다.

---

## 8. 오늘의 배움, 협업 기록, 현재 상태 및 다음 시작점

### 8-1. 내가 직접 판단한 부분

* 전체 모델 성능표를 단순 순위로 보지 않고 **Steam 데이터의 어떤 signal이 강한지를 해석하려고 함**
* Item-Based가 모든 주요 지표에서 강하다는 점과 Steam의 공동소비 구조를 연결해 **Hybrid Anchor로 선택**
* Item-Based / User-Based / BPR이 모두 collaborative interaction을 쓰지만 **직접 표현하는 관계가 다르다는 점을 파고듦**
* BPR의 latent vector를 기술적 개념으로 넘기지 않고 **interaction을 latent preference space로 압축한다는 의미를 질문하고 이해**
* Latent Factor가 사람이 정의한 장르 feature가 아니라 **ranking objective로 자동 학습되는 숨은 축**이라는 점을 정리
* PCA와 무엇이 비슷하고 다른지 질문해 **단순 차원축소와 ranking-oriented latent learning을 구분**
* Funk-SVD와 BPR의 공통된 latent factor 구조와 서로 다른 objective를 연결해 이전 실패 결과를 다시 해석
* BPR의 특성을 보고 **Candidate Filter로 사용하자는 최초 Hybrid 가설을 직접 세움**
* 그 가설에서 **Candidate Recall Ceiling 문제를 발견**하고 **Multi-Retriever 아이디어로 확장**
* "BPR은 ranking에 덜 적합하다"는 최초 해석이 과하다는 점을 받아들이고 수정
* Hybrid에서 중요한 것은 Standalone 성능이 아니라 **Unique Hit / Complementarity**라는 관점으로 이동
* Weight Tuning보다 Architecture 실험을 먼저 하기로 결정
* 상황별 Hybrid는 첫 Hybrid 프로젝트에는 복잡성이 크다고 판단해 후순위로 보류
* Candidate Size를 감으로 정하지 않고 **Recall Saturation 실험으로 결정**
* 반복되는 Item-Based 계산 비용을 발견하고 Top-200을 한 번 저장하는 방식으로 실험 구조 개선

### 8-2. GPT가 보완·정정한 부분

GPT는 내가 세운 가설을 구조화하고 과한 일반화를 수정하는 역할을 주로 했다.

* Steam 데이터에 대해 interaction signal과 explicit binary feedback을 구분했고, BPR을 단순히 "거시적인 모델" 또는 "ranking에 덜 적합한 모델"이라고 일반화하지 않도록 정정
* Item-Based / User-Based / BPR이 같은 collaborative data를 쓰지만 `Item ↔ Item`, `User ↔ User`, `User ↔ Item latent preference`라는 서로 다른 구조로 관계를 표현한다는 점을 정리
* BPR latent factor를 PCA와 비교하며 `PCA → 분산 보존`, `BPR → Ranking Objective`라는 학습 목적의 차이를 설명하고, Funk-SVD와 BPR이 같은 구조여도 objective 차이로 Top-N 성능이 크게 달라질 수 있음을 연결
* Standalone 순위보다 Unique Hit / Overlap / Complementarity가 더 중요할 수 있다는 관점을 제시하고, 내가 생각한 Hybrid 후보를 A(Item Candidate + Reranking) / B(Multi-Retriever) / C(Score / Rank Fusion)로 나눠 각각 하나의 실험 질문을 갖도록 정리
* 코드 측면에서 Hybrid 평가환경, 동일 사용자 저장, 프로젝트 구조, Item-Based cache, Candidate Recall 실험 구조 등을 지원

### 8-3. 오늘의 핵심 배움

오늘은 단순히 Hybrid 구조를 몇 개 생각한 날이 아니라 **기존에 따로 공부했던 모델들이 서로 어떤 관계에 있는지를 다시 연결한 날**이었다.

```text
Item-Based → Item 관계 / User-Based → User 관계 / BPR → Latent User-Item Preference
```

추천 결과는 겹칠 수 있지만 **완전히 같은 정보를 가지고 있다고 볼 수는 없다.** Hybrid에서 중요한 것은 서로 다른 모델이 같은 것을 얼마나 잘 맞히느냐가 아니라 **다른 모델이 못 보는 것을 실제로 보고 있는가**이다.

추천시스템을 바라보는 단위도 `Model → Recommendation`에서 `Retriever → Candidate Pool → Scoring → Ranking → Recommendation`으로 확장됐다. BPR 단계의 **Problem Definition > Data Signal > Hyperparameter**는 Hybrid에서 **Architecture > Model Role > Weight Tuning**으로 이어지고 있다.

### 8-4. 현재 프로젝트 상태

```text
[완료] Content-Based / User-Based CF / Item-Based CF / Funk-SVD / BPR / BPR Final Tuning
[완료] Hands-On Recommendation Hybrid 학습
[완료] 전체 모델 성능 재분석, Steam 데이터 특성 재해석
[완료] Item/User/BPR 관계 비교, Neighborhood CF와 Latent CF 차이 이해
[완료] BPR Latent Factor 의미 재정리, PCA와 BPR 차원축소 차이 이해, MF/Funk-SVD와 BPR objective 차이 연결
[완료] Item-Based를 Hybrid Anchor로 설정, BPR Candidate Filtering 가설, Candidate Recall Ceiling 문제 발견, Multi-Retriever 아이디어 도출
[완료] Hybrid Architecture A/B/C 설계, Situation-Aware Hybrid 후순위 결정
[완료] 프로젝트 폴더 구조 정리, Hybrid 실험 폴더 생성, 평가 사용자 400명 고정
[완료] Hybrid Item-Based Baseline
       P@10 = 0.06325 / R@10 = 0.08144 / HR@10 = 0.3925 / NDCG = 0.09129 / Hits = 253

[다음]
Item-Based Top-200 Candidate 생성 + 저장
   → Candidate Recall@20/50/100/200 → Recall Saturation 확인 → Candidate Size 결정
   → Architecture A → B → C
   → Hit Overlap / Unique Hit → 정량 + 정성 비교 → Hybrid Architecture 선택
   → Weight / 모델 역할 최적화 → 필요 시 Situation-Aware Hybrid
   → 최종 평가 / 한계 분석 → README / GitHub / 최종 회고
```

### 8-5. 다음 시작점

다음 시작 시점에서는 **Hybrid A 코드를 바로 작성하지 않는다.** 먼저 동일한 400명의 Item-Based Top-200을 CSV로 저장하고 Recall@20/50/100/200을 확인한다.

* Candidate Recall이 비교적 빠르게 높은 수준으로 포화된다면 → `Item-Based Retriever → BPR / Content Reranking`, 즉 **Architecture A**에 강한 근거
* 200개까지 늘려도 많은 정답을 놓친다면 → `Item-Based + BPR + Content → Multi-Retriever`, 즉 **Architecture B**의 필요성 증가

그 이후에는 Item-Based / BPR / Content Unique Hit, Common Hit, Recovered Hit, Lost Hit를 확인해 **Item-Based의 local collaborative signal과 BPR의 latent preference가 실제로 얼마나 보완적인지 검증**한다.

### 8-6. 한 문장 회고

> **오늘은 Hybrid 모델을 바로 만드는 대신, 지금까지 각각 따로 구현했던 Item-Based, User-Based, BPR이 사실 같은 User-Item Interaction에서 출발하면서도 Item-Item 관계, User-User 관계, Latent User-Item Preference라는 서로 다른 형태로 취향을 표현한다는 것을 다시 연결해서 이해한 날이었다. 처음에는 BPR의 latent preference를 이용해 후보를 먼저 걸러내는 구조를 생각했지만, 그 모델의 실패가 전체 시스템의 Recall 상한이 될 수 있다는 문제를 발견하면서 Multi-Retriever와 Reranking으로 사고가 확장됐다. 이제 관심은 '어떤 모델의 점수가 가장 높은가'에서 '다른 모델이 못 맞히는 것을 이 모델이 실제로 맞히는가'로 이동했고, 앞으로의 Hybrid 실험은 바로 그 보완성을 검증하는 과정이 될 것이다.**