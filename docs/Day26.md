# Day26 — XGBoost LTR 고도화부터 최종 Runtime 통합까지

## 1. 오늘 한 일 요약

Day25에서 설계한 XGBoost Learning-to-Rank를 실제로 실행하고 개선하는 데서 시작해, 최종적으로 **신규 사용자가 실제로 추천을 받는 Runtime Pipeline 전체를 통합**하는 데까지 나아간 날이었다.

```text
XGBoost LTR Exp1 (8 Feature) 실행 → 기존 Item Ranker보다 낮은 성능 확인
        ↓
Feature Generation 병목 발견 (전체 실행 약 9.7시간)
        ↓
Early Stopping 적용 → 성능 회복
        ↓
Rank Tie 문제 발견 및 수정
        ↓
XGBoost 규제/Tree Parameter 실험
        ↓
Feature 8개 → 14개 확장 (retriever_count, source flag, item_popularity, user_interaction_count)
        ↓
Full14 최종 평가 (Item Ranker 상회) 및 Feature Ablation (Popularity가 가장 중요)
        ↓
4-Retriever 구조 시도 → Feature Generation 병목 재확인 → 구조 확정 보류
        ↓
신규 사용자 BPR 처리 확정 (Item Embedding 평균 + User-only Fine-Tuning)
        ↓
Ranker를 Item/XGB로 교체 가능하게 설계
        ↓
첫 통합 main.py Prototype 작성 (약 2,758줄)
        ↓
MAP@10 / Mean Log Popularity@10 / Novelty@10 추가, Runtime/Offline Mode 분리
        ↓
두 번째 통합본 (약 3,881줄) 및 향후 리팩토링 방향 정리
```

시작 시점 기준선은 `BPR56+Content39+User5 → Candidate 100 → Item-Based Ranker`(Precision@10 0.0872, Recall@10 0.1110, HR@10 0.5400, NDCG@10 0.1216, Hits 348)였고, 이 결과를 XGBoost Ranker가 넘어서는지가 오늘의 핵심 실험 기준이었다.

---

## 2. XGBoost LTR Experiment 1 — 첫 실행과 사용자 분리

### 1차 Feature (8개)와 사용자 분리 구조

```text
Score: item_score_norm, bpr_score_norm, content_score_norm, user_score_norm
Rank:  item_rank, bpr_rank, content_rank, user_rank
```

LTR 학습 데이터와 최종 평가 사용자가 섞이지 않도록 **LTR Users 1,000명(Train 800/Validation 200)**과 **Final Evaluation 400명**(기존과 동일한 층화 표집: 10~15개 100명 / 16~25개 100명 / 26~45개 100명 / 46~78개 100명)을 분리했다 — XGBoost가 최종 평가 사용자의 정답을 미리 보고 학습하지 않도록 하는 오염 방지 구조였다.

### 첫 결과 — 기존보다 낮음

| Model | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---:|---:|---:|---:|---:|
| Item Ranker | 0.0872 | 0.1110 | 0.5400 | 0.1216 | 348 |
| XGB Exp1 | 0.0828 | 0.1043 | 0.5100 | 0.1168 | 331 |

"여러 모델의 score/rank를 XGBoost에 넣으면 무조건 Item-Based보다 좋아질 것"이라는 가정은 성립하지 않았다. 여기서 **Ranking 문제에서는 모델 자체의 복잡도보다 Candidate 품질·Feature 품질·Feature 정의·Label 구성·학습 objective가 더 중요하다**는 것을 확인했다 — XGBoost가 강한 ML 모델이어도 입력 Feature가 충분한 정보를 못 주면 기존 heuristic ranker를 이기지 못한다.

### Feature Generation 병목

첫 실행 전체가 약 **35,084.5초(≈9.7시간)**까지 늘어났다. XGBoost 학습 자체는 오래 걸리지 않았고, 병목은 **Item-Based/User-Based/Content score 계산 및 Candidate별 feature 생성** 쪽에 있었다 — **"ML Ranker의 학습 비용보다 Ranker에 넣을 Feature 생성 비용이 더 클 수 있다"**는 추천 시스템의 실제적인 문제를 경험했다.

---

## 3. Early Stopping, Rank Tie 수정, Regularization

### Early Stopping 적용

고정된 많은 boosting iteration 대신 validation NDCG 기준 Early Stopping을 적용했다 — Best Iteration이 약 27 부근에서 선택되었고 성능이 회복되었다.

| Metric | 결과 |
|---|---:|
| Precision@10 | 0.0878 |
| Recall@10 | 0.1138 |
| Hit Rate@10 | 0.5425 |
| NDCG@10 | 0.1250 |
| Hits | 351 |

NDCG는 기존 Item Ranker보다 높아졌다 — **boosting round를 무작정 늘리기보다 validation ranking metric을 기준으로 학습을 멈추는 것이 중요하다**는 것을 확인했다.

### Rank Tie 문제 발견과 수정

Feature를 다시 분석하다 특정 모델에서 score가 없거나 동일한 candidate에도 단순 정렬 과정에서 서로 다른 rank가 부여되는 문제(예: `score=0 → rank 41/42/43`)를 발견했다 — 실제로는 의미 없는 순위 차이가 XGBoost에 마치 의미 있는 정보처럼 전달될 수 있었다. User rank 문제를 먼저 확인·수정(NDCG@10≈0.1268로 개선)하고, 전체 모델의 rank 계산 방식까지 점검했다.

| Metric | Rank Tie 수정 후 |
|---|---:|
| Precision@10 | 0.0878 |
| Recall@10 | 0.1148 |
| Hit Rate@10 | 0.5525 |
| NDCG@10 | 약 0.125 |
| Hits | 351 |

**Feature를 추가하는 것 못지않게, Feature가 실제로 의미 있는 방식으로 계산되고 있는지 검증하는 것 자체가 중요하다**는 것을 다시 확인했다.

### 규제 및 Tree Parameter 실험

과적합 가능성을 줄이기 위해 파라미터를 조정했다.

```python
max_depth = 4
min_child_weight = 1
reg_lambda = 1.0
reg_alpha = 0.0
early_stopping_rounds = 30
```

`Precision@10≈0.0890, Recall@10≈0.1141, HR@10≈0.5475, NDCG@10≈0.129`까지 개선되었다 — 무조건 깊은 Tree보다 **상대적으로 작은 Tree + L2 regularization**이 현재 데이터에 적절했다. 이 시점부터 XGBoost가 기존 Item-Based Ranker보다 실제로 경쟁력 있는 결과를 내기 시작했다.

---

## 4. Feature 8개 → 14개 확장, Full14 결과, Feature Ablation

### 추가 Feature 6개

```text
retriever_count             # 이 Candidate를 몇 개 Retriever가 동시에 추천했는지
source_bpr / source_content / source_user   # 어느 Retriever에서 왔는지 flag
item_popularity              # 해당 게임의 Train interaction 수
user_interaction_count       # 해당 사용자의 Train interaction 수
```

8+6=14개 Feature(Full14)로 확장했다.

### Validation 결과 — XGBoost 학습 자체는 1초 수준

| | Validation NDCG@10 | Best iteration | Training time |
|---|---:|---:|---:|
| 8 Feature | 0.470909 | 59 | ≈1.01초 |
| Full14 | 0.473648 | 56 | ≈0.95초 |

여기서 확실히 확인된 것: **XGBoost 학습 자체는 1초 수준이며, 앞서 발생한 몇 시간 단위 실행시간 문제는 XGBoost가 아니라 추천 Feature 생성 Pipeline의 문제였다.**

### Full14 최종 평가 — Item Ranker를 상회

| Metric | Full14 | Item Ranker |
|---|---:|---:|
| Precision@10 | **0.0890** | 0.0872 |
| Recall@10 | **0.1145** | 0.1110 |
| Hit Rate@10 | **0.5450** | 0.5400 |
| NDCG@10 | **0.1302** | 0.1216 |
| Hits | **356** | 348 |

특히 **NDCG@10에서 의미 있는 개선**을 보였다 — Top-10에 무엇을 넣는지뿐 아니라 **관련 게임을 위쪽에 배치하는 능력** 자체가 개선되었다는 뜻이다.

### Feature Group Ablation

| 제거 Feature | Validation NDCG | Full14 대비 |
|---|---:|---:|
| Full14 | 0.473648 | - |
| Popularity 제거 | 0.467191 | **-0.006457** |
| User Activity 제거 | 0.471143 | -0.002505 |
| Source 관련 제거 | 0.473319 | -0.000329 |

**Item Popularity가 가장 강한 Ranking signal**이었고, User Interaction Count는 그보다 작지만 실제로 도움이 되었으며, Source/Retriever Agreement 효과는 존재하나 매우 작았다(`Popularity > User Activity >> Source Agreement`). 이를 통해 **여러 모델의 score만이 아니라 아이템/사용자의 통계적 특성을 함께 쓰는 것이 효과적**이라는 것을 확인했고, ML Ranker를 "기존 추천 모델 중 하나를 대체하는 것"이 아니라 **"여러 Retriever가 만들어낸 정보를 Feature로 받아 최종 의사결정을 수행하는 모델"**로 다시 이해했다.

---

## 5. 4-Retriever 실험 시도와 병목 재확인

Item-Based 자체도 Candidate Retriever에 포함하는 구조(`Item50+BPR20+Content15+User15 → Candidate 100 → XGBoost Full14`)를 시도했다. LTR Users 1,000명 + Final Users 400명 = 총 1,400명에 대한 Feature Generation 도중 다음 로그를 확인했다.

```text
800/1400 users, 116.98 sec/user, ETA ≈ 1169.8분
850/1400 users, 92.61 sec/user, ETA ≈ 849.0분
```

사용자 1명당 수십~100초 수준의 Feature Generation이 발생해 정상적으로 반복 가능한 실험 Pipeline이 아니라고 판단했다 — 여기서도 **"XGBoost 학습 속도 ≠ 전체 LTR 실험 속도"**가 다시 확인되었고, 실제 병목은 Item-Based를 4번째 Retriever로 넣으면서 늘어난 사용자별 Item similarity 계산량이었다. 이 실험의 최종 성능은 확정하지 않고, **Feature Generation 구조를 Cache/Vectorization 중심으로 다시 최적화해야 할 대상**으로 남겨뒀다.

이미 안정적인 3-Retriever 구조(BPR56+Content39+User5)와 두 Ranker(Item Ranker, XGBoost Full14) 모두 평가 가능한 상태였으므로, **새로운 Retriever 조합을 무한히 늘리는 대신 기존 구조를 기준으로 프로젝트를 정리하고 실제 Runtime으로 연결**하는 작업으로 넘어가기로 했다.

---

## 6. main.py 재정의와 신규 사용자 Runtime 설계

### main.py 역할 재정의

기존 `main.py`는 Sparse Matrix 생성/BPR 학습/Fine-Tuning/모델 저장/랜덤 테스트/400명 평가/추천이 한꺼번에 들어가 있어 실제 진입점이라기보다 BPR 실험 코드에 가까웠다. 역할을 `experiment scripts(모델 실험) / evaluation.py(Offline Evaluation) / models/(실제 추천 엔진) / main.py(저장된 모델을 불러와 실제 추천만 수행)`로 분리하기로 했다.

### 최종 Runtime Pipeline

```text
사용자가 플레이한 게임 입력
      ↓
각 게임의 BPR Item Embedding 조회 → 평균 → 신규 User Vector 초기값
      ↓
User-only BPR Fine-Tuning
      ↓
Candidate Retrieval (BPR56 + Content39 + User-CF5) → Candidate Union ≈ 100
      ↓
Feature 계산 → Ranker → Top-10
```

### 신규 사용자 BPR 처리 확정

신규 사용자는 기존 BPR의 `user_id` mapping에 없어 `model.user_factors[user_idx]`를 쓸 수 없다는 문제를 다시 확인하고, **① 플레이 게임들의 Item Embedding 평균을 초기값으로 사용 → ② 기존 Item Factor/전체 모델은 고정한 채 신규 User Vector만 BPR objective(`score(user,positive) > score(user,negative)`)로 업데이트**하는 2단계 방식을 확정했다.

초기 Fine-Tuning 설정: `epochs=3, learning_rate=0.01, regularization=0.001`. 다만 `epoch=0(Item Embedding 평균만 사용)/1/3/5`을 비교할 수 있도록 설계해, "평균만 쓰는 것보다 fine-tuning이 정말 도움이 되는가?"를 추후 실험으로 검증할 수 있게 했다.

### Ranker 교체 가능 구조

```python
RANKER_MODE = "item"  # 또는 "xgb"
```

XGBoost 결과가 좋아졌지만 Runtime에서 바로 강제 고정하지 않고, Candidate Retrieval 코드는 그대로 두고 **Ranker만 교체 가능한 구조**로 설계했다.

### 통합 main.py 초안

BPR model/mapping load, CF matrix cache, Content TF-IDF cache, 신규 User BPR embedding, User-only fine-tuning, BPR/Content/User-CF Retriever, Item-CF scoring, XGBoost feature 생성, Item/XGB ranker switch, 게임 이름/app_id 입력, Top-10 출력까지 하나로 연결한 초안(약 2,758줄)을 작성했고 Python syntax check를 통과시켰다. 다만 2,700줄이 넘는 `main.py` 자체를 최종 구조로 유지하는 것은 적절하지 않다고 판단해, 최종적으로는 `models/hybrid.py` 등으로 실제 엔진을 옮기고 `main.py`는 `engine.recommend(played_games)` 수준의 얇은 실행 코드로 만들기로 방향을 잡았다 — 현재 통합 파일은 **전체 Pipeline이 실제로 연결되는지 검증하기 위한 Runtime Prototype**이다.

---

## 7. 평가 지표 확장(MAP/Popularity/Novelty)과 Runtime/Offline Mode 분리

### MAP@10

사용자별 AP@10(관련 게임이 등장한 rank에서의 Precision을 누적)을 계산해 전체 평균을 MAP@10으로 사용하기로 했다. 다만 **실제 신규 사용자가 게임을 입력해 추천받는 Runtime에서는 미래 정답이 없으므로 MAP@10을 계산할 수 없다** — `Precision/Recall/NDCG/MAP`은 Offline Evaluation Mode 전용, `Popularity/Novelty`는 두 모드 모두에서 계산 가능하도록 명확히 분리했다.

### Mean Log Popularity@10 / Novelty@10

```text
Mean Log Popularity@10 = mean(log(1 + interaction_count))
Novelty@10 = mean(-log2(interaction_count / total_train_interactions))
```

값이 높을수록 인기 게임 중심(Popularity) 또는 덜 흔한 아이템(Novelty) 추천이라는 뜻이며, 두 지표를 함께 보면 "정확도는 좋아졌지만 인기 편향도 함께 증가했다" 같은 해석이 가능해진다.

### Runtime/Offline Evaluation Mode 분리

```python
RUN_MODE = "recommend"  # 게임 입력 → 신규 User Vector → Candidate → Rank → Top-10 → Popularity/Novelty
RUN_MODE = "evaluate"   # Train/Test → 400명 샘플링 → 추천 → Test 비교 → 전체 Metric 계산
```

최종 Offline Evaluation Metric은 `Precision@10, Recall@10, Hit Rate@10, NDCG@10, MAP@10, Mean Log Popularity@10, Novelty@10, Hits`로 확장했다. 평가 사용자는 기존과 동일하게 400명(10~15개 100/16~25개 100/26~45개 100/46~78개 100) 층화 표집을 유지해 기존 결과와 비교 가능하게 했다. 결과는 `models/saved_model/results/integrated_hybrid_eval.csv`(사용자별)와 `integrated_hybrid_eval_summary.csv`(전체 요약)로 저장하도록 구성했다. MAP/Popularity/Novelty/Evaluation Mode까지 추가한 두 번째 통합본은 약 3,881줄까지 늘었고 syntax 검사를 통과했다.

---

## 8. 내가 직접 판단하고 결정한 부분

- 기존 Hybrid 구조에서 **XGBoost Ranker를 실제로 실험하기로 결정**하고, LTR이 Candidate와 각 모델 score를 Feature로 학습하는 구조인지 직접 확인
- Feature를 4개(score만)가 아닌 **score+rank 8개로 확대**
- XGBoost가 기존 Item Ranker보다 낮게 나온 결과를 보고 **추가 개선이 필요하다고 판단**
- Ranking Feature를 단계적으로 추가하는 방향 선택
- Candidate Retriever와 Ranker의 역할 차이를 계속 재확인하며 실험 설계에 반영
- 신규 사용자 처리에서 **단순 평균뿐 아니라 플레이 게임 기반 추가 Fine-Tuning 방식을 선택**
- 평균 Item Embedding을 초기값으로 쓰고 Item Factor는 고정하는 구조 선택
- `main.py`에 최종 Hybrid Pipeline을 통합하기로 결정
- 향후 구조 변경을 고려해 **Ranker를 교체 가능하게 만들 것을 요구**
- **MAP@10과 Popularity metric을 최종 평가에 추가**하기로 결정, Novelty도 함께 확인하기로 결정
- 4-Retriever 실험에서 비정상적으로 긴 ETA를 발견하고 문제를 제기해 구조 확정을 보류시킴

## 9. GPT(AI)가 담당한 부분

XGBoost LTR Pipeline 코드 작성, score/rank Feature 구성, Train/Validation/Final 사용자 분리 구조 구현, Early Stopping 적용, Rank Tie 문제 분석 및 코드 수정, XGBoost parameter/regularization 실험 코드 구성, Context Feature 14개 확장, Ablation 실험 구성, 실행 결과 비교·해석, Candidate/Ranker 구조 정리, 신규 User Vector 생성 로직 구현, user-only BPR gradient update 코드 작성, 통합 Runtime main.py 초안 작성, Item/XGB Ranker switch 구현, MAP@10/Mean Log Popularity@10/Novelty@10 구현, Offline Evaluation Mode 구현, 코드 문법 검사 — 구현량이 큰 부분과 반복 실험 코드는 AI 도움이 컸고, 결과를 보고 다음 가설과 구조를 선택하는 것은 직접 판단했다.

---

## 10. 오늘의 회고

오늘 얻은 핵심 이해:

- **Retriever와 Ranker는 역할이 다르다**: Retriever는 정답일 가능성이 있는 Candidate를 최대한 가져오는 역할, Ranker는 그 Candidate 안에서 실제 Top-K 순서를 정하는 역할 — 처음부터 모든 게임을 XGBoost로 랭킹하지 않고 Retrieval→Ranking 2-stage 구조를 쓰는 이유를 다시 이해했다.
- **XGBoost가 기존 모델보다 항상 좋은 것은 아니다**: 초기 P@10 0.0828은 Item Ranker(0.0872)보다 낮았지만, Early Stopping·Rank 수정·Regularization·Context/Popularity Feature를 적용하며 Full14에서 P@10 0.0890/NDCG@10 0.1302까지 개선됐다 — **모델 이름 자체보다 Feature와 학습 설계가 더 중요했다.**
- **Ranking에서는 Popularity도 중요한 Feature가 될 수 있다**: Ablation에서 Popularity 제거가 가장 큰 NDCG 하락을 일으켜, interaction 기반 추천에서는 협업 필터링 점수뿐 아니라 아이템의 통계적 특성도 큰 도움이 될 수 있음을 확인했다.
- **Offline Metric과 실제 사용자 Metric은 다르다**: 신규 사용자가 직접 추천받는 순간에는 정답이 없어 Precision/Recall/NDCG/MAP을 계산할 수 없지만, Popularity/Novelty 같은 추천 결과 자체의 통계는 계산할 수 있다 — 이를 Runtime과 Offline Evaluation 코드에서 분리했다.
- **Feature Generation이 실제 시스템 병목이 될 수 있다**: XGBoost 학습은 약 1초였지만 Feature Generation은 몇 시간 이상 걸렸고, 4-Retriever 실험에서는 사용자당 90~117초까지 늘었다 — 실제 추천 시스템에서는 Cache/Precomputation/Vectorization/Batch processing/ANN 같은 엔지니어링이 모델 선택만큼 중요하다는 것을 체감했다.

초기 단계에서는 User-Based/Item-Based/Content-Based/MF/BPR을 하나씩 구현하는 데 초점이 있었다면, 지금은 Candidate Retrieval / Feature Engineering / Learning-to-Rank / Ablation / Cold Start / Offline Evaluation / Runtime Architecture / Caching·Latency를 다루고 있다 — 프로젝트가 단순한 추천 알고리즘 비교에서 **실제 Multi-Stage Recommendation System 설계** 단계로 이동했다.

> **오늘은 XGBoost Learning-to-Rank를 실제로 실행하고 개선하는 과정에서 시작해, 신규 사용자를 위한 Cold-Start 처리와 Popularity/Novelty를 포함한 최종 평가 체계, 그리고 실제 추천을 수행하는 Runtime Pipeline까지 하나로 통합한 날이었다. 모델 구현 중심의 프로젝트가 Candidate Retrieval → Feature Engineering → Learning-to-Rank → Cold Start → Offline Evaluation → Runtime으로 이어지는 실제 추천 시스템 파이프라인으로 완성되어 가는 것을 확인했다.**

---

## 11. 현재 기준 구조와 프로젝트 상태

### 현재 안정적인 Candidate 구조

```text
BPR 56 + Content 39 + User 5  →  Candidate ≈ 100
```

### 최종 Ranker 후보

| Ranker | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---:|---:|---:|---:|---:|
| Item-Based (기준선) | 0.0872 | 0.1110 | 0.5400 | 0.1216 | 348 |
| **XGBoost Full14** | **0.0890** | **0.1145** | **0.5450** | **0.1302** | **356** |

```
[완료] XGBoost LTR Exp1 (8 Feature) 실행 및 초기 성능 확인 (기존보다 낮음)
[완료] LTR 학습(1,000명)/평가(400명) 사용자 분리 구조 구현
[완료] Feature Generation 병목 발견 (9.7시간) 및 원인 분석
[완료] Early Stopping 적용 → 성능 회복
[완료] Rank Tie 문제 발견 및 수정
[완료] XGBoost 규제/Tree Parameter 튜닝
[완료] Feature 8→14 확장 (retriever_count, source flag, item_popularity, user_interaction_count)
[완료] Full14 최종 평가 — Item Ranker 상회 (Hits 348→356, NDCG 0.1216→0.1302)
[완료] Feature Group Ablation — Popularity > User Activity >> Source Agreement
[시도/보류] 4-Retriever(Item 포함) 구조 — Feature Generation 병목으로 구조 확정 보류
[완료] main.py 역할 재정의 및 최종 Runtime Pipeline 설계
[완료] 신규 사용자 BPR 처리 확정 (Item Embedding 평균 + User-only Fine-Tuning)
[완료] Ranker 교체 가능 구조(Item/XGB) 설계
[완료] 통합 main.py 1차 Prototype (2,758줄) 및 2차 통합본(3,881줄, MAP/Popularity/Novelty 포함)
[완료] MAP@10, Mean Log Popularity@10, Novelty@10 설계 및 구현
[완료] Runtime(recommend) / Offline Evaluation(evaluate) Mode 분리
[미완료] Item Ranker vs XGBoost Full14 최종 선택
[미완료] 4-Retriever 구조 최적화 여부 결정
[미완료] 신규 사용자 Fine-Tuning epoch(0/1/3/5) Ablation
[미완료] 최종 Offline Evaluation 전체 지표 실행
[미완료] main.py 최종 리팩토링 (models/hybrid.py로 로직 이관)
[미완료] README 최종 반영, 프로젝트 최종 회고
```

## 12. 다음 계획

1. **Ranker 최종 선택**: Item Ranker vs XGBoost Full14 — 성능 개선폭과 시스템 복잡도를 함께 보고 결정
2. **4-Retriever 실험 최적화 여부 결정**: 필요하다면 Item score cache / Retriever cache / Candidate cache / Batch sparse operation 중심으로 재최적화하되, 프로젝트 마무리를 위해 반드시 해야 하는 실험은 아니라고 판단
3. **신규 사용자 Fine-Tuning Ablation**: epoch 0/1/3/5 비교 — "Item Embedding 평균만 사용 vs BPR user-only fine-tuning"의 실제 효과 검증
4. **최종 Offline Evaluation**: 최종 모델에서 Precision@10/Recall@10/HR@10/NDCG@10/MAP@10/Mean Log Popularity@10/Novelty@10을 모두 계산
5. **main.py 최종 리팩토링**: 통합 Prototype의 실제 모델 로직을 `models/hybrid.py` 등으로 이동, 최종 `main.py`는 모델 로드 → 입력 → `recommend()` → 결과 출력 역할만 남김
6. **README 최종 반영**: 최종 Architecture, Candidate Retriever 비율, XGBoost LTR 구조, 14 Feature, Feature Ablation, Cold-start 신규 사용자 전략, 최종 Metric, Popularity/Novelty, Runtime Architecture, 최종 프로젝트 구조를 모두 반영