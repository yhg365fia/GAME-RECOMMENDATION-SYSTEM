# Day24 — Case 3 Ablation, Cross-Ranker 발견, Hybrid Architecture 반확정

## 1. 오늘 한 일 요약

기존 Item-Based / User-Based / BPR / Content-Based 모델들을 각각 평가하는 단계를 넘어, **실제 하이브리드 추천 시스템에서 각 모델을 어떻게 조합할 것인지 결정하는 실험**을 집중적으로 진행했다.

```text
Case 3 Weighted Rank Fusion Ablation → User-Based 비중 재검토
        ↓
Case 2 Multi-Retriever 구조 실험 → BPR 자기 재랭킹(self-ranking) 문제 발견
        ↓
Cross-Ranker 비교 (Ranker를 Retriever 후보 생성에서 제외)
        ↓
Item Ranker 고정 후 Retriever 비율(BPR/Content/User) Sweep
        ↓
Case 2 vs Case 3 비교 → Retriever→Ranker 구조로 방향 확정
        ↓
Candidate Size Sweep (50/100/150/200) → 100이 최적점 확인
        ↓
Hybrid Architecture 반확정, 프로젝트 완료 계획 수립
```

오늘 결과로 현재 프로젝트의 하이브리드 구조를 거의 결정했다. 이후부터는 새로운 구조를 계속 만드는 것보다 **현재 구조의 미세 조정, Learning-to-Rank, 최종 튜닝** 단계로 넘어가기로 했다.

---

## 2. Case 3 Ablation과 User-Based 비중 재검토

기존 Case 3(Item 0.50 / User 0.15 / BPR 0.20 / Content 0.15 Weighted Rank Fusion)에서 각 모델을 하나씩 제거하고 나머지 Weight를 비율대로 재정규화해 비교했다.

| 실험 | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---:|---:|---:|---:|---:|
| Full | 0.07750 | 0.10006 | 0.4675 | 0.11344 | 310 |
| **- Item** | 0.05975 | 0.07743 | 0.4200 | 0.08810 | 239 |
| **- User** | 0.07925 | 0.10189 | 0.4800 | 0.11254 | 317 |
| **- BPR** | 0.06825 | 0.08741 | 0.4275 | 0.10229 | 273 |
| **- Content** | 0.07525 | 0.09634 | 0.4575 | 0.10868 | 301 |

기여도는 대략 **Item-Based >> BPR > Content-Based >> User-Based** 순으로 나타났고, Item-Based를 빼면 성능이 가장 크게 무너져 Steam 데이터에서 가장 중요한 신호는 **아이템 간 공동 소비 관계**라는 것을 다시 확인했다. 반대로 User-Based를 제거했을 때 오히려 지표가 소폭 상승했는데, 이를 곧바로 "User-Based는 쓸모없다"로 결론 내리지 않고 **"User-Based 자체가 쓸모없는 것인가, 아니면 현재 0.15 Weight가 너무 큰 것인가?"**라는 질문으로 다시 정의했다.

이어서 Item:BPR:Content 상대 비율은 유지한 채 User Weight만 0/0.025/0.05/0.10/0.15로 조절해 sweep했다.

| User Weight | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.07925 | 0.10189 | 0.4800 | 0.11254 | 317 |
| 0.025 | 0.08000 | 0.10285 | 0.4800 | 0.11405 | 320 |
| **0.05** | **0.08000** | 0.10268 | **0.4825** | **0.11475** | **320** |
| 0.10 | 0.07900 | 0.10154 | 0.4775 | 0.11376 | 316 |
| 0.15 | 0.07750 | 0.10006 | 0.4675 | 0.11344 | 310 |

User-Based는 완전히 필요 없는 모델이 아니라 **작은 비중의 보조 신호로 사용할 때 성능이 개선**됐고, 기존 0.15는 과한 Weight였음을 확인했다. 이 과정에서 기존 ablation 구현에서 Weight가 0인 모델이 후보 UNION이나 tie-break에는 여전히 남을 가능성을 발견해, `active_models`만 실제 후보·ranking에 참여하도록 **strict ablation** 구조로 수정했다.

---

## 3. Case 2 실험과 BPR 자기 재랭킹 문제 발견

Fusion이 아니라 Retriever → Ranker 구조를 실험했다. Fusion에서 얻은 모델 중요도를 Candidate 개수에 반영해 후보를 만들었다 — Without User(Item59+BPR24+Content17)와 With User(Item56+BPR22+Content17+User5) 두 구조를 UNION → **BPR Ranker** → Top10으로 평가했다.

| 구조 | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---:|---:|---:|---:|---:|
| Without User | 0.05225 | 0.06870 | 0.3850 | 0.06968 | 209 |
| With User | 0.05225 | 0.06870 | 0.3850 | 0.06968 | 209 |

두 구조의 최종 성능이 **완전히 동일**했다. Candidate Recall은 각각 0.2792 / 0.2720으로 달랐지만, 두 경우 모두 `Final Top10 ↔ Pure BPR Top10` 평균 overlap이 10/10, 완전 동일한 사용자 비율이 100%였다.

여기서 중요한 구조적 문제를 발견했다: BPR Retriever가 이미 **BPR 기준 고득점 Top22~24 후보**를 가져오고, 그 안에 BPR의 global Top10이 자연히 포함된다. 이후 동일한 BPR score로 다시 ranking하면 **BPR이 가져온 Top10이 그대로 다시 Top10이 된다** — 즉 `Multiple Retriever → Same BPR Retriever의 동일 BPR Score Ranker` 구조에서는 Multi-Retriever의 의미가 사실상 사라진다는 것을 확인했다. 이 실험에서 Candidate Recall은 약 0.28까지 올랐지만 최종 Recall은 약 0.069에 그쳤다 — **Candidate Recall이 높다고 최종 추천 성능이 자동으로 높아지는 것은 아니다**라는 것을 다시 확인했다.

---

## 4. Cross-Ranker 실험

BPR self-ranking 문제 때문에 **Ranker 모델을 Retriever 후보 생성에서 제외**하는 Cross-Ranker 구조를 설계했다. 세 구조를 비교했다.

| 실험 | Retriever | Ranker |
|---|---|---|
| 1 | BPR59 + Content41 | Item-Based |
| 2 | Item71 + BPR29 | Content-Based |
| 3 | Item78 + Content22 | BPR |

| 구조 | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---:|---:|---:|---:|---:|
| **BPR59 + Content41 → Item** | **0.08272** | **0.10612** | **0.5225** | **0.11679** | **330** |
| Item71 + BPR29 → Content | 0.06000 | 0.07988 | 0.4275 | 0.08405 | 240 |
| Item78 + Content22 → BPR | 0.07050 | 0.08797 | 0.4750 | 0.09631 | 282 |

Candidate Recall은 오히려 Content Ranker 구조(Item+BPR→Content, 0.29121)가 가장 높았지만, 최종 성능은 Item Ranker 구조가 압도적으로 좋았다 — **후보에 정답이 많은 것과 좋은 Top10을 만드는 것은 별개의 문제**라는 점을 다시 확인했다. 세 Ranker가 완전히 같은 Candidate Pool을 받은 것은 아니므로 이 결과는 "Ranker 알고리즘만의 절대적 순위"가 아니라 **Cross-Ranker Pipeline 조합의 성능 비교**로 해석하기로 했다. Case 2의 Ranker는 **Item-Based로 확정**했다.

---

## 5. Retriever Ratio Sweep과 User-Based 5% 효과, Case 2 vs Case 3

Item Ranker를 고정한 뒤 BPR/Content 비율과 User-Based 5% 추가 효과를 함께 확인하는 7개 실험을 진행했다.

| 후보 구성 | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---:|---:|---:|---:|---:|
| BPR59 + Content41 | 0.08272 | 0.10612 | 0.5225 | 0.11679 | 330 |
| BPR70 + Content30 | 0.08012 | 0.10275 | 0.5025 | 0.11425 | 320 |
| BPR50 + Content50 | 0.08188 | 0.10524 | 0.5175 | 0.11684 | 326 |
| BPR40 + Content60 | 0.08088 | 0.10321 | 0.5050 | 0.11351 | 322 |
| **BPR56 + Content39 + User5** | **0.08722** | **0.11102** | **0.5400** | **0.12155** | **348** |
| BPR67 + Content28 + User5 | 0.08490 | 0.10787 | 0.5275 | 0.11956 | 339 |
| BPR48 + Content47 + User5 | 0.08500 | 0.10941 | 0.5250 | 0.11985 | 339 |

User-Based를 5% 넣었을 때 세 비율 모두 지표가 상승했고(예: 약 59:41 조합에서 Δ Hits +18), User-Based가 BPR/Content에 없는 정답을 새로 가져온 Unique Hits도 44~45개였다. **User-Based는 단독 성능이나 큰 Weight에서는 약하지만, BPR·Content가 놓치는 일부 후보를 보충하는 작은 Retriever 역할에서는 실제 가치가 있다**는 것을 확인했다.

이를 바탕으로 확정한 Case 2 최고 구조(BPR56+Content39+User5 → Item Ranker, Precision@10 0.08722/Hits 348)와 Case 3 최고 구조(Item .5588+BPR .2235+Content .1676+User .05 Fusion, Precision@10 0.08000/Hits 320)를 비교했다.

| Metric | Case 2 (Retriever→Ranker) | Case 3 (Fusion) |
|---|---:|---:|
| Precision@10 | **0.08722** | 0.08000 |
| Recall@10 | **0.11102** | 0.10268 |
| Hit Rate@10 | **0.5400** | 0.4825 |
| NDCG@10 | **0.12155** | 0.11475 |
| Hits | **348** | 320 |

모든 주요 지표에서 Case 2가 앞서, **최종 Hybrid Architecture 방향을 Fusion이 아니라 Multi-Retriever → Ranker 구조로 확정**했다.

---

## 6. Candidate Size Sweep

Retriever 비율(56:39:5)을 고정하고 총 Candidate Size(50/100/150/200)만 바꿔 성능 영향을 확인했다.

| Size | 실제 UNION | Candidate Recall | Scoreable Recall | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 48.36 | 0.17096 | 0.14486 | 0.07804 | 0.09759 | 0.4875 | 0.11039 | 308 |
| **100** | **95.36** | 0.24884 | 0.20550 | **0.08722** | **0.11102** | **0.5400** | **0.12155** | **348** |
| 150 | 141.58 | 0.30045 | 0.24126 | 0.08353 | 0.10778 | 0.5300 | 0.11880 | 334 |
| 200 | 188.05 | **0.33859** | **0.26687** | 0.08400 | 0.10806 | 0.5225 | 0.11967 | 336 |

Candidate Recall은 후보를 늘릴수록 계속 올랐지만(0.171→0.249→0.300→0.339), 최종 Recall은 0.0976→**0.1110**→0.1078→0.1081로 **100에서 정점을 찍고 오히려 소폭 하락**했다. 후보를 100개 이상 주면 정답 자체는 더 많이 후보군에 들어오지만 **현재 Item Ranker가 그 추가 정답을 Top10으로 충분히 끌어올리지 못한다**는 것을 확인했고, 현재 구조에서는 **Candidate Size=100이 최적점**이라고 결론지었다. 이 결과는 이후 Learning-to-Rank 실험의 필요성과도 연결된다 — 150~200 후보에는 더 많은 정답이 이미 있으므로, XGBoost/LambdaMART가 현재 Item Ranker보다 후보를 잘 재정렬한다면 추가 성능 향상 가능성이 있다.

---

## 7. 오늘 이해한 추천시스템 개념

- **Retriever와 Ranker는 역할이 다르다**: Retriever는 "정답이 있을 가능성이 있는 후보를 최대한 확보", Ranker는 "그 후보 안에서 실제 Top10에 들어갈 아이템을 선택"하는 것이 목적이며, 따라서 Candidate Recall과 최종 Recall은 서로 다른 문제다.
- **Candidate Recall이 높다고 최종 성능이 높은 것은 아니다**: Content Ranker 실험과 Candidate Size 150/200에서 후보 안 정답은 늘었지만 최종 성능은 오히려 감소했다 — **Retrieval Bottleneck과 Ranking Bottleneck을 분리해서 봐야 한다.**
- **같은 모델로 Retrieval과 Ranking을 동시에 하면 두 단계가 사실상 중복될 수 있다**: BPR Top-N으로 후보를 만든 뒤 동일 BPR score로 다시 랭킹하면 최종 결과가 Pure BPR Top10과 100% 동일해진다는 것을 직접 확인했다.
- **User-Based의 역할을 재해석**: 큰 Weight(0.15)에서는 오히려 성능을 낮췄지만 약 5%로 낮추자 여러 실험에서 성능이 개선됐다 — 메인 모델이 아니라 **다른 Retriever가 놓치는 일부 후보를 보충하는 weak auxiliary signal**로 이해하는 것이 적절하다.
- **Steam 데이터에서는 Item-Based의 가치가 매우 크다**: Ablation에서도 Item 제거가 가장 치명적이었고, Cross-Ranker에서도 Item Ranker가 가장 높은 최종 성능을 냈다.

---

## 8. 실험 설계에서 내가 직접 판단한 부분

오늘은 코드 구현이나 속도 최적화보다 **실험 결과를 보고 다음 가설을 정하고 구조를 좁혀가는 판단**이 많았던 날이었다. 다만 오늘의 실험 설계는 처음부터 끝까지 혼자 짠 것은 아니고, **큰 방향과 "이걸 확인해야 한다"는 질문은 직접 세우되, 구체적인 비율/후보 수/실험 개수 같은 세부 sweep 설계는 GPT의 도움을 받아 자동화한 절반 협업 구조**였다.

내가 직접 판단한 것:

- User-Based 제거 시 성능이 오히려 올라간 결과를 보고 "User-Based가 쓸모없는 것인가, Weight가 과한 것인가"로 문제를 다시 정의
- Case 2에서 두 구조(With/Without User)의 최종 성능이 완전히 동일하다는 결과를 보고 **동일 모델을 Retriever와 Ranker에 함께 쓰는 것이 의미가 있는지 의문 제기**
- 이 의문에서 **Ranker 자체를 후보군에서 제외하는 Cross-Ranker 구조를 제안**
- 필요 없어진 self-ranker 실험은 정리하고 넘어감
- Case 2와 Case 3를 비교한 뒤 **Fusion이 아니라 Retriever→Ranker 구조로 최종 방향 결정**
- Candidate Size 실험 결과를 보고 **하이브리드 구조를 더 확장하지 않고 여기서 수렴시키기로 결정**
- 프로젝트를 무한히 확장하지 않고 **6단계 완료 계획**(구조 탐색→미세조정→Learning-to-Rank→최종 튜닝→최종 분석→보고서/종료)을 명확히 설정

## 9. GPT(AI)가 담당한 부분 — 실험 자동화와 최적화

오늘은 두 영역을 GPT에게 크게 의지했다.

**① 실험 설계의 절반(구체적 sweep 값·실험 개수 자동화)**: Case 3 Ablation 코드, User Weight Sweep의 구체적인 값(0/0.025/0.05/0.10/0.15) 설계, Case 2 Weighted Multi-Retriever 구현, Cross-Ranker 실험 코드, Retriever Ratio Sweep의 7가지 조합 설계, Candidate Size Sweep의 구체적인 값(50/100/150/200) 설계와 구현, 결과표·지표 비교 자동화 — "무엇을 확인하고 싶다"는 방향은 직접 제시했지만, 몇 개의 조합을 어떤 값으로 돌릴지 같은 세부 실험 설계는 GPT가 제안하고 자동화한 부분이 컸다.

**② 속도 최적화는 전적으로 GPT가 처리했다**: 초기 Cross-Ranker 실험(BPR59+Content41→Item Ranker)이 400명 평가에 약 18.1분 걸렸던 문제의 원인 분석(매 사용자 source item마다 전체 37,567개 Item과 cosine similarity를 반복 계산)부터, Interaction Matrix cache → Normalized Item Matrix 저장 → 필요한 source item의 Top30 neighbor batch 계산 → Item Neighbor Cache 저장 → User Score Cache 저장까지 이어지는 **캐싱 구조 설계와 구현을 전부 GPT가 맡았다.** 그 결과 7개 Ratio Sweep 전체가 1.75분, Candidate Size Sweep 전체가 7.81초로 단축됐다. 이 최적화 작업은 직접 코드를 짜거나 구조를 설계한 것이 아니라 결과만 확인했다.

또한 BPR self-ranking 문제의 원인 분석, Item neighbor/user score cache 최적화, 실험 결과를 Candidate Stage/Ranking Stage로 나눠 해석하는 프레임도 GPT가 제시했다. 어떤 구조를 최종적으로 선택할지, 실험을 어디까지 할지, 어떤 가설을 더 확인할지는 결과를 본 뒤 직접 판단하며 진행했다.

---

## 10. 오늘의 회고

오늘의 가장 큰 성과는 특정 metric 하나를 올린 것보다, **수많은 가능한 Hybrid 조합 중 하나의 구조로 수렴했고, 왜 그 구조를 선택했는지를 실험 결과로 설명할 수 있게 된 것**이었다. Fusion(Case 3)이 아니라 Retriever→Ranker(Case 2) 구조를 선택한 이유, Item-Based를 최종 Ranker로 쓰는 이유, User-Based를 5%만 섞는 이유가 모두 숫자로 뒷받침된다.

오늘 이전까지는 계속 **"어떤 Hybrid Architecture가 좋은가?"**를 찾는 단계였다면, 오늘 이후부터는 질문이 **"현재 선택한 Hybrid Architecture의 성능을 얼마나 더 개선할 수 있는가?"**로 바뀐다. 즉 프로젝트가 **Architecture Exploration 단계에서 Optimization/Finalization 단계로 넘어갔다.**

> **오늘은 여러 후보 구조를 놓고 좁혀가는 하루였다.** Case 3 Ablation에서 User-Based의 역할을 다시 정의하고, Case 2에서 우연히 발견한 BPR self-ranking 문제가 Cross-Ranker라는 더 나은 구조로 이어졌으며, Retriever 비율과 Candidate Size를 sweep하며 "후보가 많다고 늘 좋은 것은 아니다"라는 것을 반복해서 확인했다. 실험의 세부 설계와 속도 최적화는 GPT에 크게 의존했지만, 각 결과를 보고 다음 질문을 던지고 구조를 좁혀가는 판단은 직접 했다.

---

## 11. 현재 반확정 최종 하이브리드 구조

Hybrid Architecture 탐색은 오늘 기준으로 일단 멈춘다.

```text
[Candidate Retrieval]
BPR 56 + Content-Based 39 + User-Based 5   (총 Candidate Budget = 100)
        ↓
[Final Ranker]
Item-Based
        ↓
[Output]
Top-10
```

| Metric | Score |
|---|---:|
| Precision@10 | 0.08722 |
| Recall@10 | 0.11102 |
| Hit Rate@10 | 0.5400 |
| NDCG@10 | 0.12155 |
| Hits | 348 |

이 구조는 완전한 최종 확정이 아니라 **반확정 Final Baseline Architecture**로 사용한다. 앞으로 새로운 Hybrid 구조를 계속 늘리기보다는 이 구조 위에서 성능을 개선한다.

## 12. 현재 프로젝트 상태

```
[완료] Case 3 Ablation (모델별 기여도: Item >> BPR > Content >> User)
[완료] User-Based Weight Sweep → 0.05 최적, strict ablation 구조로 수정
[완료] Case 2 With/Without User 실험 → BPR self-ranking 문제 발견
[완료] Cross-Ranker 실험 → Item-Based를 최종 Ranker로 확정
[완료] Retriever Ratio Sweep (7종) + User 5% 추가 효과 검증
[완료] Case 2 vs Case 3 비교 → Retriever→Ranker 구조로 최종 방향 확정
[완료] Candidate Size Sweep (50/100/150/200) → 100이 최적점
[완료] (AI) Item Ranker 속도 최적화 — 18.1분 → 이후 실험 수초~2분대로 단축
[완료] Hybrid Architecture 반확정 (BPR56+Content39+User5 → Item Ranker, Hits 348)
[완료] 프로젝트 완료 6단계 계획 수립
[미완료] 반확정 구조 미세 조정 (Retriever 비율 소폭 조정, Item Ranker 파라미터 등)
[미완료] Learning-to-Rank (XGBoost Ranker / LambdaMART) 학습 및 적용
[미완료] 최종 성능 튜닝
[미완료] 최종 성능 분석 및 보고서, 프로젝트 종료
```

## 13. 다음 작업 시작 시 바로 할 것

다음 시간에는 **새로운 Hybrid 구조를 만들지 않는다.** 현재 반확정 구조(`BPR56+Content39+User5 → Candidate100 → Item Ranker → Top10`)를 기준으로 시작해 다음 순서로 진행한다.

1. **반확정 구조 미세 조정**: Retriever 비율 소폭 조정, Item Ranker 관련 파라미터, 필요한 최소 Ablation — 대규모 구조 탐색은 다시 하지 않는다.
2. **Learning-to-Rank 원리 학습 및 적용**: XGBoost Ranker / LambdaMART. 핵심 질문은 "현재 Item Ranker가 활용하지 못한 추가 relevant candidate들을 학습 기반 Ranker가 더 잘 Top10으로 올릴 수 있는가?"
3. **최종 성능 튜닝**: 가장 좋은 구조 하나만 남기고 주요 hyperparameter·candidate size·ranking parameter·weight를 마지막으로 조정 (새 모델/구조는 끝없이 추가하지 않음)
4. **최종 성능 분석**: Precision/Recall/HR/NDCG, 사용자 sparsity group별 성능, Candidate/Ranking 성능, Baseline 대비 개선, 모델별 역할 정리
5. **최종 보고서 및 프로젝트 종료**: 문제 정의 → Baseline → 개별 모델 → 실패한 실험 → 원인 분석 → Hybrid 설계 → Ablation → Architecture 선택 → Ranking 개선 → 최종 모델이라는 전체 흐름을 보여주도록 README와 프로젝트 구조를 최종 정리

오늘을 기준으로 **Hybrid Architecture 탐색은 사실상 종료**하고, 프로젝트의 마지막 성능 개선 단계로 진입한다.