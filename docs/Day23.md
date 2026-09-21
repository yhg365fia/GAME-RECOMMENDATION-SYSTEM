# Day23 — Hybrid Architecture 실험: Reranking, Multi-Retriever, Rank Fusion

## 1. 오늘 한 일 요약

오늘의 핵심은 단순 모델 성능 확인이 아니라, **하이브리드 추천 구조에서 각 모델이 Candidate Generator / Ranker / Fusion 역할을 했을 때 어떤 차이가 생기는지 실험한 것**이었다. Item-Based Baseline 이후 처음으로 실제 Hybrid Architecture를 코드로 구현하고 평가까지 마쳤다.

```text
Case 1 (Item-Based Candidate → BPR Reranking) 구현 및 분석
        ↓
Candidate@30의 의미, Test 누수 여부 등 개념 정리
        ↓
Case 1 Reverse (BPR Candidate → Item-Based Reranking) 추가 결정
        ↓
Reverse 실행시간 문제 발견 → Item-Based 4가지 최적화 적용
        ↓
Case 1 Reverse 결과 도출 및 Item-Based/BPR 역할 재해석
        ↓
Case 3 (Multi-Retriever + Weighted Rank Fusion) 구현 및 평가
        ↓
Unique Hit 분석, 코드 오류 수정, Case 2와의 구조적 차이 정리
        ↓
전체 구조 비교 → Ablation·Weight Grid 등 다음 단계 계획 수립
```

오늘 종료 시점 기준으로 **"어떤 모델을 만들지" 단계는 거의 끝났고, "어떤 하이브리드 구조를 최종 architecture로 선택할지" 단계**에 들어왔다.

---

## 2. Case 1 — Item-Based Candidate → BPR Reranking

### 구조와 결과

```text
전체 게임 → Item-Based Top-30 → BPR Reranking → Top-10
```

| 지표 | 결과 |
|---|---:|
| Precision@10 | 0.0695 |
| Recall@10 | 0.0888 |
| Hit Rate@10 | 0.4525 |
| NDCG@10 | 0.0978 |
| Hits | 278 |
| Candidate@30 Recall | 0.1491 |
| Candidate@30 HR | 0.5750 |
| Candidate Hits | 468 |

기존 순수 Item-Based Top-10(253 Hits) 대비 **253→278, +25 Hits**로 BPR reranking이 실제로 성능을 개선했다.

### Candidate@30 개념 정리

`Candidate@30`은 최종 추천이 아니라 **최종 Top-10을 만들기 전에 Candidate Generator가 뽑아놓은 30개 게임**이라는 것을 정리했다. `Candidate Recall@30 = 0.1491`은 "실제 Test 정답 중 약 14.9%가 Item-Based가 만든 후보 30개 안에 존재했다"는 뜻이며, **Candidate에 정답이 없으면 뒤의 BPR이 아무리 좋아도 그 정답은 절대 추천할 수 없다**는 것을 확인했다.

### Test 데이터 사용 여부 확인

"Case 1에서 BPR이 후보군을 Test로 가지고 판단한 거냐?"는 의문을 직접 확인했다. 정상 구조는 `Train → Item-Based 후보 생성 → Train 학습 BPR로 재정렬 → 최종 Top-10 → Test와 비교`이며, Test는 오직 평가할 때만 사용된다 — **Train 기반 추천 → Test로 성능 측정**이므로 데이터 누수는 없는 구조임을 확인했다.

### 첫 해석

Item-Based는 후보 30개 안에 정답 468개를 갖고 있었지만 자체 Top-10은 253개만 맞혔다 — **후보 자체는 많이 찾아놓고 일부 정답을 11~30위에 묻어놓고 있었다.** BPR이 이 순위를 바꾸며 253→278로 개선되었고, 이를 통해 **BPR을 Candidate Generator가 아니라 Reranker로 사용하는 것도 가치가 있다**는 것을 처음 확인했다.

---

## 3. Case 1 Reverse — BPR Candidate → Item-Based Reranking, 그리고 최적화

### 역방향 실험을 추가한 이유

"Item-Based와 BPR 역할을 바꿔서도 실험하고 싶다"는 생각에서 완전히 반대 구조를 추가했다.

```text
전체 게임 → BPR Top-30 → Item-Based Reranking → Top-10
```

순방향(Item→BPR 후보/랭킹)과 역방향(BPR→Item 후보/랭킹)을 비교하면 **누가 후보 생성기로 적합하고 누가 Ranker로 적합한지**를 더 직접적으로 볼 수 있다고 판단했다.

### 실험 코드 구현 범위에 대한 논의

"실험 코드도 내가 짜는 게 맞나, 아니면 조건만 정하고 나머지는 AI에게 맡겨도 되나?"를 확인했다. 결론은 **실험 설계와 해석은 직접 잡고, 반복 구현은 AI를 적극 활용해도 된다**는 것이었다. 직접 잡아야 하는 영역(필수)은 문제 정의, 실험 가설, Train/Test 구조, Candidate/Ranker 역할, 평가 지표, 결과 해석이고, 반복 루프 구현·파일 저장/로드·캐싱/최적화는 AI 활용이 가능한 영역으로 구분했다. 기준은 "입력이 뭔지 / Train·Test가 어디 쓰이는지 / Candidate가 어떻게 만들어지는지 / 점수가 어떻게 계산되는지 / 출력 지표가 무엇을 비교하는지" 다섯 가지를 직접 설명할 수 있는 상태인가였다.

### 실행시간 문제와 원인 파악

역방향 코드의 실행이 느려서 "Item-Based는 어차피 30개 후보만 ranking하는데 왜 오래 걸리냐"는 의문이 들었다. 확인해보니 Item-Based가 **최종적으로 재정렬하는 대상은 BPR 후보 30개가 맞지만**, 기존 Item-Based 정의상 각 source item마다 **전체 37,567개 게임과 similarity를 계산한 뒤 그 중 Top-K neighbor를 선택**해야 했다 — 즉 최종 Rank 대상은 30개지만 **점수 산출 공간은 전체 Item 공간**이었다는 구조적 원인을 파악했다.

### Item-Based 최적화 (실험 의미는 유지하면서 속도만 개선)

| 최적화 | 의미 |
|---|---|
| Item→Item Top-K 캐싱 | 동일 Item의 이웃을 반복 계산하지 않음 |
| Unique source item만 계산 | 400명에게 실제 등장한 게임만 계산 |
| Item norm 사전 계산 | cosine normalization 반복 제거 |
| Batch sparse multiplication | 여러 Item similarity를 한꺼번에 계산 |

추가로 BPR Top-30 Candidate도 캐싱했다. 평가 사용자 400명의 Train에서 실제 서로 다른 source item을 추출하니 **3,321개**였고, 전체 37,567개가 아니라 이번 평가에 실제 필요한 3,321개만 계산하도록 좁혔다. source item을 하나씩 처리하는 대신 64개씩 묶어 **sparse matrix multiplication을 배치로 수행**해 Python 반복 호출 비용도 줄였다.

### Case 1 Reverse 결과

| 지표 | 결과 |
|---|---:|
| Precision@10 | 0.0695 |
| Recall@10 | 0.0874 |
| HR@10 | 0.4675 |
| NDCG@10 | 0.0977 |
| Hits | 278 |

BPR Candidate: Recall@30 0.1350 / HR 0.5750 / Hits 422.

순수 BPR Top-10(209 Hits) 대비 Item-Based reranking 후 **209→278, +69 Hits** — 사용자별로는 개선 88명, 악화 37명, 동일 275명이었다.

**속도 개선**: 기존 forward Case 1 reranking 평가가 1847.7초(≈30분 48초)였던 것에 비해, 최적화된 Reverse 실행은 Interaction Matrix 생성 27.3초 + Item norm 0.4초 + 3,321개 Item neighbor 계산 187.2초 + BPR Top-30 생성 1.1초 + 실제 reranking 평가 0.3초 = **전체 234.7초(≈3분 55초)**로 단축됐고, 캐시는 이후 재사용되어 다음 실행은 더 빨라진다.

### Item-Based와 BPR의 역할 재해석 (스스로 수정)

"BPR이 후보를 넓혔다"는 초기 표현에 대해 "Item-Based 후보에도 놓친 게 많았고, Reverse에서는 BPR이 놓친 걸 Item-Based가 메운 건데 왜 BPR이 넓혀준다는 거냐"는 지적을 직접 제기했고, 이 지적이 맞아 표현을 수정했다.

| Candidate Generator | Recall@30 | Hits |
|---|---:|---:|
| **Item-Based** | **0.1491** | **468** |
| BPR | 0.1350 | 422 |

후보 생성 자체는 **Item-Based가 더 강했다.** 두 모델 모두 Candidate 안에는 정답을 갖고 있지만 자체 Ranking 과정에서 일부를 아래로 묻는다는 공통 패턴을 재확인했다 — Item-Based는 Top-30 정답 468→자체 Top-10 253→BPR Rerank 후 278, BPR은 Top-30 정답 422→자체 Top-10 209→Item Rerank 후 278. **Item-Based는 Candidate Generator로 상대적으로 강하고, BPR은 독립적인 개인화 신호를 제공하지만 Candidate·자체 Ranking은 현재 Item-Based보다 약하다**는 것으로 정리했다.

---

## 4. Case 3 — Multi-Retriever + Weighted Rank Fusion

### 구조와 가중치

```text
Item-Based Top-100 / User-Based Top-100 / BPR Top-100 / Content Top-100
        ↓
Weighted Rank Fusion → Top-10
```

| 모델 | Weight |
|---|---:|
| Item | 0.50 |
| User | 0.15 |
| BPR | 0.20 |
| Content | 0.15 |

### Retrieval 상황

| 모델 | 사용자 Coverage | 평균 후보 |
|---|---:|---:|
| Item | 400/400 | 100 |
| User | 362/400 | 14.6 |
| BPR | 400/400 | 100 |
| Content | 400/400 | 100 |

**User-Based만 sparse user 때문에 충분한 후보를 못 만든 사용자들이 존재**한다는 특이점을 발견했다.

### 최종 성능과 단독 모델 Hits

| 지표 | Case 3 |
|---|---:|
| Precision@10 | 0.0775 |
| Recall@10 | 0.1001 |
| HR@10 | 0.4675 |
| NDCG@10 | 0.1134 |
| Hits | 310 |
| Micro F1 | 0.0852 |

현재까지 가장 높은 성능이었다.

| 모델 | Hits |
|---|---:|
| Item-Based | 253 |
| User-Based | 144 |
| BPR | 209 |
| Content | 112 |
| **Fusion** | **310** |

### Unique Hit 분석

| 모델 | Unique Hits |
|---|---:|
| Item only | **130** |
| User only | 70 |
| BPR only | **124** |
| Content only | 61 |
| 4개 모두 공통 | 8 |

각 모델이 **상당히 다른 정답을 잡고 있다**는 것을 확인했다. Item-Based(253 Hits) 대비 Fusion에서는 새롭게 얻은 정답 115, 기존 Item-Based 정답 중 손실 58로, `253+115-58=310` — **순증가 +57 Hits**였다.

### 코드 오류 수정과 재학습 불필요 확인

Case 3 마무리 시점에 `print("3개 모델 공통:", all_three)`에서 `NameError: all_three is not defined`가 발생했다 — 4모델 구조로 확장하면서 3모델 시절 출력 코드가 남아있던 단순 오류였고, `all_four`로 수정하고 빠져 있던 User-Based Hits/Unique Hits 출력도 추가했다. 이 오류 수정 때문에 모델을 다시 학습해야 하는지 확인했으나 **전혀 필요 없었다** — Item/User/BPR/Content cache, 평가 subset, TF-IDF가 모두 저장되어 있고 `FORCE_REBUILD_*` 플래그가 모두 `False`였기 때문에 재실행이 **약 3초**밖에 걸리지 않았다. 이후 Case 3 출력이 두 번 나온 것을 보고 의아했는데, 코드가 두 번 출력한 게 아니라 실행 명령 자체가 두 번 호출된 것이었고, 둘 다 캐시를 이용해 동일한 결과가 빠르게(3.1초/2.9초) 나온 것임을 확인했다.

---

## 5. 전체 구조 비교와 핵심 발견

| 구조 | P@10 | R@10 | HR | NDCG | Hits |
|---|---:|---:|---:|---:|---:|
| Item Baseline | 0.0633 | 0.0814 | 0.3925 | 0.0913 | 253 |
| Case 1 Item→BPR | 0.0695 | 0.0888 | 0.4525 | 0.0978 | 278 |
| Case 1 Reverse BPR→Item | 0.0695 | 0.0874 | **0.4675** | 0.0977 | 278 |
| Case 2 (Multi-Retriever + BPR Ranking) | 0.0673 | 0.0862 | 0.4425 | 0.0953 | 269 |
| **Case 3 Fusion** | **0.0775** | **0.1001** | **0.4675** | **0.1134** | **310** |

현재 최고 성능은 **Case 3**.

### Case 2 vs Case 3 — "모델만 더 추가하면 같아지는 것 아닌가?"

"Case 2를 개선해서 모델 두 개 더 추가하면 Case 3랑 같아지는 거 아니냐"는 질문을 스스로 던졌고, 답을 정리했다.

- **Case 2**: `여러 Retriever → UNION → BPR 하나가 최종 Ranking` — Multi-Retriever + **Single Ranker**
- **Case 3**: `여러 Retriever → 각 모델의 Rank 신호까지 유지 → Weighted Fusion` — Multi-Retriever + **Multi-Signal Ranking**

Case 2에 모델을 더 추가해도 최종 Ranking이 BPR 하나뿐이라면 Case 3와 같아지지 않는다 — Case 2의 최종 Ranking 단계까지 Item/User/BPR/Content 점수를 모두 결합하도록 바꿔야 비로소 Case 3에 가까워진다.

### Case 2에서 얻은 중요한 교훈

Case 2는 Candidate Recall≈0.2045로 Case 1보다 후보 안에 정답을 더 많이 확보했음에도 최종 Hits는 269로 Case 1(278)보다 낮았다 — **좋은 Candidate Retrieval과 좋은 Ranking은 별개의 문제**이며, Case 2에서는 BPR 단독 Ranking이 병목이었을 가능성을 확인했다.

### XGBoost/LambdaMART로 바로 넘어가지 않기로 한 판단

Case 3가 가장 높게 나오자 "Fusion을 XGBoost ranking으로 바꿔볼까"라는 생각이 들었지만, Learning-to-Rank로 가려면 user-item 후보/각종 rank/score/feature/label 구조를 새로 만들고 Train/Validation/Test도 다시 나눠야 하며 잘못하면 Test label이 Ranker 학습에 쓰이는 데이터 누수 위험도 있다는 것을 확인해, **지금 바로 넘어가지 않고 간단한 Ablation만 먼저 추가**하기로 했다.

---

## 6. 내가 직접 판단한 부분

- **Candidate@30이 무엇인지, BPR이 Test를 보고 판단하는지 직접 확인**하며 개념을 스스로 검증
- **Item-Based와 BPR 역할을 바꾼 Reverse 실험을 추가하자고 제안**
- 실험 코드 구현에서 **직접 잡아야 할 영역과 AI에게 맡겨도 되는 영역을 스스로 구분**
- Reverse 실행이 느린 이유를 "30개만 재정렬하는데 왜 느리지"라고 의문을 갖고 **구조적 원인(scoring 공간과 ranking 대상의 불일치)을 직접 파악**
- **"BPR이 후보를 넓혔다"는 표현이 부정확하다고 스스로 지적**하고 Candidate 생성력 비교표로 재해석 수정
- Case 3에서 **각 모델을 모두 쓰는 게 항상 정답은 아니라고 판단**, Ablation의 필요성을 스스로 제기
- **"Case 2에 모델만 추가하면 Case 3와 같아지지 않냐"는 질문을 직접 던지고**, Single Ranker vs Multi-Signal Ranking의 구조적 차이로 답을 정리
- **XGBoost로 바로 넘어가지 않고 Ablation부터 하자는 우선순위 판단**
- "추천 프로젝트가 원래 이렇게 실험이 많고 복잡한가"라는 솔직한 피로감을 제기하면서도 **무한 확장보다 결론을 만드는 단계로 넘어가는 것이 중요하다는 방향을 스스로 정리**

## 7. GPT(Claude)가 보완한 부분

Case 1/Case 1 Reverse/Case 3 구현 코드 작성, Candidate@30·Test 데이터 흐름 등 개념 설명, Item-Based 4가지 최적화(캐싱, unique source item, norm 사전계산, batch sparse multiplication) 설계 및 구현, Case 3 가중치 설계와 diagnostic 코드, Unique Hit 분석 코드, 코드 오류(`all_three` → `all_four`) 원인 설명, Case 2와 Case 3의 구조적 차이(Single Ranker vs Multi-Signal Ranking) 설명, XGBoost/LambdaMART로 넘어가지 않는 것이 낫다는 의견 제시, Ablation·Weight Grid·Top-N 실험 순서 제안 — 코드 구현과 구조 설명은 GPT 비중이 높았지만, 각 결과를 보고 표현을 수정하거나("BPR이 넓혔다"→정정), 다음 실험을 결정하는 것은 직접 판단했다.

---

## 8. 오늘의 회고

오늘 추천시스템 관점에서 얻은 핵심을 정리하면: ① **Item-Based는 현재 가장 강한 단일 모델**(순수 Hits 253으로 BPR 209, User 144, Content 112보다 높음). ② 하지만 **Item-Based 자체 Ranking도 완벽하지 않다**(Top-30 정답 468개 중 Top-10은 253개). ③ **BPR 역시 Candidate 안에는 정답을 갖고 있지만 자체 Ranking이 약하다**(Top-30 정답 422, Top-10 209). ④ **Item-Based의 후보 생성력이 BPR보다 현재 더 강하다**(Candidate Hits 468 vs 422). ⑤ 서로 다른 모델들이 **상당히 다른 정답을 잡는다**(Unique Hits Item 130, BPR 124, User 70, Content 61). ⑥ 따라서 **여러 신호를 결합할 이유가 실제 데이터에서 존재**하며, Case 3 Fusion이 310 Hits로 가장 높았다. ⑦ **Candidate가 좋아도 Ranking이 나쁘면 최종 성능은 떨어질 수 있다**(Case 2가 대표적 사례). ⑧ 현재 가장 유망한 구조는 **Multi-Retriever + Multi-Signal Rank Fusion**(Case 3)이다.

구현 측면에서도 얻은 것이 컸다. 기존에는 실험할 때마다 similarity·모델 계산·후보 생성을 매번 다시 했지만, 오늘부터는 `Retrieval → Cache → Rank/Fusion만 반복 실험`하는 구조로 바뀌어 Case 3 가중치를 바꿔 재평가하는 데 약 3초밖에 걸리지 않게 됐다 — 이는 앞으로 Weight Grid Search를 할 때 결정적으로 중요하다.

> **오늘은 모델 하나를 구현한 날이라기보다, 추천시스템을 '모델들의 조합과 역할' 관점에서 설계하고 실험한 날이었다.** Item-Based와 BPR의 역할을 서로 바꿔보고, Candidate 생성력과 Ranking 능력을 구분해서 관찰하고, 여러 모델을 하나로 합쳤을 때 실제로 서로 다른 정답을 보완하는지 데이터로 확인하면서, "어떤 모델이 가장 좋은가"에서 "각 단계(Retrieval/Ranking/Fusion)에서 어떤 조합이 필요한가"로 사고가 한 단계 더 나아갔다.

---

## 9. 현재 프로젝트 상태

```
[완료] Case 1 (Item-Based Candidate → BPR Reranking) 구현·평가 (Hits 253→278)
[완료] Candidate@30 개념 정리, Test 누수 없음 확인
[완료] Case 1 Reverse (BPR Candidate → Item-Based Reranking) 구현·평가 (Hits 209→278)
[완료] Item-Based 4가지 최적화 (Top-K 캐싱, unique source item, norm 사전계산, batch sparse multiplication) — 평가 시간 30분48초 → 3분55초로 단축
[완료] Item-Based/BPR의 Candidate 생성력·Ranking 능력 재해석 (Item-Based가 Candidate 생성에서 더 강함)
[완료] Case 3 (Multi-Retriever + Weighted Rank Fusion) 구현·평가 (Hits 310, 현재 최고)
[완료] Unique Hit 분석 (Item 130 / BPR 124 / User 70 / Content 61 / 공통 8)
[완료] Case 1/1Reverse/2/3 전체 구조 비교표 작성
[완료] Case 2 vs Case 3 구조적 차이 정리 (Single Ranker vs Multi-Signal Ranking)
[완료] XGBoost/LambdaMART를 지금 바로 도입하지 않기로 결정
[미완료] Case 3 Ablation (모델 조합별 기여도 확인)
[미완료] Case 1/2/3 최종 구조 비교 및 Architecture 확정
[미완료] Fusion Weight Grid Search
[미완료] Candidate Top-N 민감도 실험 (Top-50/100/200)
[미완료] 최종 평가 및 시스템 정리
```

## 10. 다음 계획

현재까지 Reranking(Case 1) / Multi-Retriever(Case 2) / Rank Fusion(Case 3) 구조를 모두 비교했으므로, 이제는 새 구조를 계속 추가하기보다 **최종 Architecture를 좁히고 튜닝하는 단계**로 넘어간다.

| 단계 | 실험 | 목적 |
|---|---|---|
| 1 | **Case 3 Ablation** | 4개 모델이 모두 필요한지 확인 |
| 2 | **Case 1/2/3 최종 비교** | 최종 Hybrid Architecture 선택 |
| 3 | **Fusion Weight Grid Search** | 선택된 구조의 최적 가중치 탐색 |
| 4 | **Candidate Top-N 비교** | Top-50/100/200 등 후보 수 민감도 확인 |
| 5 | **최종 평가** | 고정된 최종 구조로 성능 및 그룹별 특성 정리 |
| 선택 | XGBoost / LambdaMART | Learned Ranking 확장 실험 (필수 아님) |

### 1. Case 3 Ablation

현재 4모델 조합(Item+User+BPR+Content, 310 Hits)에서 몇 개를 제거해 비교한다.

```text
A. Item + User + BPR + Content   ← 현재
B. Item + BPR + Content
C. Item + User + BPR
D. Item + BPR
```

User-Based/Content-Based를 제거하면 성능이 떨어지는지, Item+BPR만으로도 비슷한 성능이 나오는지, 복잡도를 늘릴 만큼 각 모델이 실제로 기여하는지 확인한다. **성능이 비슷하면 더 단순한 구조를 선택**하는 것도 고려한다.

### 2. Case 1/2/3 최종 비교

Baseline, Case 1(Item→BPR), Case 2(Multi-Retriever→BPR Ranking), Case 3(Multi-Retriever→Weighted Fusion)를 Precision/Recall/HR/NDCG/Hits/Candidate Recall/그룹별 성능/실행시간·복잡도까지 함께 비교해, 단순히 최고 점수가 아니라 **성능 향상 대비 구조 복잡도**도 고려해 선택한다.

### 3. 최종 Architecture 확정 후 Weight Grid Search

Case 3 계열이 최종 구조로 선택되면 `Item 0.4~0.7 / BPR 0.1~0.3 / User 0.0~0.2 / Content 0.0~0.2` (합 1.0) 범위에서 탐색한다. Retrieval cache가 저장되어 있어 가중치 조합 하나 평가에 약 3초면 되지만, Ablation 결과를 반영해 탐색 범위를 줄인다.

### 4. Candidate Top-N 민감도

가중치가 정해진 뒤 Top-50/100/200을 비교해 "후보를 계속 늘리는 것이 실제 최종 성능 향상으로 이어지는가"를 확인한다. Case 2에서 이미 Candidate Recall 증가가 최종 Ranking 개선을 보장하지 않는다는 것을 확인했으므로, 후보 수를 무조건 늘리는 것이 능사가 아니라는 점을 염두에 둔다.

### 5. 최종 평가

Architecture와 Hyperparameter가 정해지면 전체 성능, 단일 Item-Based 대비 개선폭, 사용자 활동량별 성능, 각 모델의 Unique Hit 기여, Candidate Retrieval과 Ranking의 역할, 계산 비용·Cache 활용, 현재 시스템의 한계를 정리한다. 여기까지 하면 프로젝트의 **1차 추천 Architecture 실험을 종료**한다.

### 선택적 후속 실험 — Learned Ranking

추가 확장을 원하면 `Candidate Retrieval → 각 모델 rank/score + user/item feature → XGBoost Ranker/LambdaMART → Top-10` 구조로 발전시킬 수 있지만, 이는 Weighted Fusion의 단순 튜닝이 아니라 새로운 Learning-to-Rank 단계이므로 이번 프로젝트의 필수 실험으로 두지 않는다.

```text
[완료] 단일 모델 실험 → Case 1 Reranking → Case 2 Multi-Retriever → Case 3 Rank Fusion
[다음] Case 3 Ablation → Case 1/2/3 최종 비교 → Architecture 확정
       → Weight Grid Search → Candidate Top-N 실험 → 최종 평가 및 프로젝트 정리
[선택적 확장] XGBoost / LambdaMART Learning-to-Rank
```