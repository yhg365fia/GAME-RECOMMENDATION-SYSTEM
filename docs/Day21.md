# Day21 — BPR 파라미터 최적화 및 최종 성능 분석

## 1. 오늘 한 일 요약

BPR을 Hybrid 단계로 넘기기 전에 **Standalone 모델로서 충분히 실험하고 최종 버전을 결정하는 작업**을 진행했다. 단순히 Grid Search를 돌려 가장 높은 숫자를 고르는 방식이 아니라 **결과 관찰 → 이상한 점 발견 → 원인 추론 → 가설 설정 → 통제된 실험 → 결과 재해석 → 다음 실험 결정**이라는 방식으로 진행했다.

확인한 주요 변수: Explicit False Fine-Tuning의 실제 효과, Iteration 10/15의 차이, Base Regularization, Latent Factors, Factors×Regularization의 상호작용, 최적점 주변 Local Search.

```text
Explicit False 효과를 동일 조건(positive_only=True)에서 공정 재검증
        ↓
Regularization 실험 (0.0001~0.020) → 0.005가 최고
        ↓
Factors 실험 (20/40/80) → NDCG는 80에서, 나머지는 40에서 최고 → metric 해석 재검토
        ↓
Factors×Regularization 상호작용 실험 → 60/0.005가 전 지표 최고
        ↓
Local Search (55/60/65, reg 0.005~0.008) → 최종 60/0.006 확정
        ↓
Standalone BPR 최종 결정, 전체 모델 비교표 갱신
        ↓
Hybrid로 넘어가기 전 개발 철학 정립
```

최종 Standalone BPR: **iterations=15, factors=60, lr=0.05, reg=0.006, Explicit False Fine-Tuning 사용(epoch=1, lr=0.01, reg=0.001), positive_only=True** — Precision@10 0.05225 / Recall@10 0.068695 / HR@10 0.3850 / NDCG@10 0.069681 / Hits 209.

---

## 2. Explicit False 효과 공정 재검증

이전에 Explicit False를 적용했을 때 성능이 크게 상승했지만, 당시 기존 BPR은 `positive_only=False`, Explicit False BPR은 `positive_only=True`로 평가되어 **Test relevant item의 정의 자체가 달랐다.** 그래서 "False를 negative로 넣었더니 엄청나게 올랐다"고 단정할 수 없었고, 오늘은 이 문제를 먼저 해결해 동일하게 `positive_only=True` 조건으로 맞추고 비교했다.

| Iter | Explicit False | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---:|:---:|---:|---:|---:|---:|---:|
| 10 | X | 0.02850 | 0.03731 | 0.2125 | 0.04032 | 114 |
| 10 | O | 0.02925 | 0.03855 | 0.2350 | 0.04282 | 117 |
| 15 | X | 0.02525 | 0.03415 | 0.2150 | 0.03863 | 101 |
| **15** | **O** | **0.03650** | **0.04630** | **0.2825** | **0.04875** | **146** |

10 iteration에서는 Explicit False를 넣어도 개선폭이 작았지만, 15 iteration에서는 Precision 0.02525→0.03650, HR 0.2150→0.2825, Hits 101→146으로 훨씬 큰 차이가 났다. 이를 보고 **"False fine-tuning이 시작될 당시 latent representation의 상태에 따라 효과가 달라질 수 있다"**고 판단했다 — 학습 구조를 바꾸면 기존의 optimal hyperparameter까지 달라질 수 있다는 점을 구체적으로 체감했다. **데이터 정의 + 학습 objective + 학습 단계가 바뀌면 최적 parameter도 함께 바뀔 수 있다**는 것이 오늘의 첫 배움이었고, BPR의 가장 큰 개선은 parameter tuning이 아니라 **"Steam의 False interaction을 모델이 무엇이라고 이해해야 하는가?"**라는 문제 정의에서 시작됐다는 점을 다시 확인했다.

---

## 3. Regularization / Factors / 상호작용 / Local Search 실험

### Regularization 실험 (Factors=40, Iterations=15 고정)

| Reg | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---:|---:|---:|---:|---:|---:|
| 0.0001 | 0.03225 | 0.04107 | 0.2550 | 0.04373 | 129 |
| 0.001 | 0.03650 | 0.04630 | 0.2825 | 0.04875 | 146 |
| **0.005** | **0.04575** | **0.05823** | **0.3625** | **0.06127** | **183** |
| 0.010 | 0.04200 | 0.05252 | 0.3250 | 0.05865 | 168 |
| 0.020 | 0.03150 | 0.04029 | 0.2650 | 0.04180 | 126 |

`reg=0.01`에서 Fine-Tuning Loss가 더 높고 Pair Accuracy도 더 낮았음에도 실제 추천 성능은 `reg=0.001`보다 좋았다. "Loss가 더 높은데 왜 성능이 더 좋지?"라는 의문에서, Train에서 pair 관계를 완벽히 맞힌 모델보다 **적당히 규제된 모델이 새로운 Test interaction에 더 잘 일반화**할 수 있다는 관점으로 해석을 옮겼다. Regularization을 "학습을 방해하는 penalty"가 아니라 **"Train 데이터에 너무 자신만만해지지 않게 제한해 unseen interaction에 더 잘 대응하게 만드는 장치"**로 재정의했고, training objective가 좋아진 것과 실제 추천 품질이 좋아진 것은 다른 문제라는 것을 다시 확인했다.

### Factors 실험 (Reg=0.005 고정)

| Factors | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.04300 | 0.05666 | 0.3375 | 0.06197 | 172 |
| **40** | **0.04575** | 0.05823 | **0.3625** | 0.06127 | **183** |
| 80 | 0.04475 | **0.05924** | 0.3350 | **0.06400** | 179 |

Factors=80에서 NDCG가 가장 높게 나와 처음엔 "표현력이 좋아졌는데 과적합 때문에 Precision/HR에서 상쇄된 것 아닐까" 생각했다. 하지만 NDCG는 맞힌 개수가 아니라 **정답을 얼마나 높은 순위에 배치했는가**까지 반영하는 지표라는 점을 다시 따져보고, "성능은 좋아졌지만 과적합이 상쇄했다"가 아니라 **"이미 맞히고 있던 정답들을 더 좋은 순위에 올리는 데는 도움이 됐지만, 더 많은 사용자에게 정답을 만들어내는 능력 자체가 개선된 것은 아니다"**로 해석을 수정했다. 이를 통해 Precision(추천 중 정답 비율), Recall(정답 중 찾아낸 비율), HR(최소 1회 성공한 유저 비율), NDCG(정답의 순위 품질), Hits(총 정답 수)가 **점수판이 아니라 모델 행동을 관찰하는 서로 다른 센서**라는 것을 체감했다.

### Factors × Regularization 상호작용 실험

Factors와 Regularization을 독립 변수가 아니라 **상호작용하는 변수**로 보기 시작했다 — "Factor가 커지면 과적합을 막을 regularization도 같이 커져야 하는 것 아닌가?"

| Factors | Reg | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.001 | 0.03150 | 0.04089 | 0.2450 | 0.04148 | 126 |
| **60** | **0.005** | **0.05125** | **0.06567** | **0.3750** | **0.06807** | **205** |
| 80 | 0.0075 | 0.04575 | 0.06024 | 0.3475 | 0.06439 | 183 |
| 80 | 0.010 | 0.04250 | 0.05772 | 0.3375 | 0.05868 | 170 |
| 100 | 0.010 | 0.04175 | 0.05277 | 0.3100 | 0.05927 | 167 |

예상과 달리 80을 잘 규제한 조합이 아니라 **60/0.005가 모든 주요 지표에서 동시에 최고**였다. "더 큰 모델을 잘 규제해서 쓰는 게 항상 답이 아니라, 이 데이터에 필요한 표현 용량 자체가 60 근처일 수 있다" — 복잡도는 많을수록 좋은 게 아니라 **데이터에 필요한 만큼이면 된다**는 결론을 얻었다.

### Local Search로 전환

`20→40→80→100`으로 크게 벌려가던 탐색에서 60이 최고를 찍자, 곧바로 120을 시도하는 대신 **"최적점이 이 근처에 있겠다"**고 판단해 탐색 방향을 Local Search로 전환했다.

| Factors | Reg | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---:|---:|---:|---:|---:|---:|---:|
| 55 | 0.005 | 0.04875 | 0.06092 | 0.3500 | 0.06918 | 195 |
| 60 | 0.005 | 0.05125 | 0.06567 | 0.3750 | 0.06807 | 205 |
| **60** | **0.006** | **0.05225** | **0.06870** | **0.3850** | **0.06968** | **209** |
| 65 | 0.005 | 0.04575 | 0.06132 | 0.3650 | 0.06256 | 183 |

같은 Factors=60에서도 `reg 0.005→0.006`이라는 작은 차이가 모든 지표를 개선했고, Factors=65는 60보다 복잡하지만 오히려 대부분 지표가 하락했다 — **모델 capacity와 regularization 사이에는 균형점이 있고, 어느 하나를 계속 늘리는 방식으로는 찾을 수 없다**는 것을 확인했다.

마지막으로 60/0.007, 60/0.008, 63/0.007, 63/0.008을 추가 확인했으나 `0.006→0.007→0.008`로 갈수록 모든 지표가 순차적으로 하락했고 Factors=63도 60을 넘지 못했다. "0.0058, 0.0062처럼 더 세밀하게 찾으면 조금 더 오를 수도 있지만, 지금 목적에서 그 차이는 중요하지 않다"고 판단하며 탐색을 마무리했다. **최적화에서는 어디까지 할지를 판단하는 것도 실력**이며, 프로젝트 목표는 BPR benchmark competition이 아니라 Hybrid Recommendation System 완성이므로 **충분히 설명 가능한 최적점이 확보되면 다음 문제로 넘어가는 것이 더 좋은 개발 판단**이라고 정리했다.

---

## 4. 최종 BPR과 전체 모델 성능 비교

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
random_state = 42
```

| Metric | Score |
|---|---:|
| Precision@10 | 0.05225 |
| Recall@10 | 0.068695 |
| Hit Rate@10 | 0.3850 |
| NDCG@10 | 0.069681 |
| Micro F1@10 | 0.057473 |
| Hits | 209 |

### 전체 모델 성능 비교

> **Evaluation note:** 아래에는 서로 다른 실험 시점의 Legacy 결과가 함께 포함되어 있으며, 일부는 split·ground-truth 정의·평가 코드가 완전히 동일하지 않다. 따라서 절대적인 모델 간 우열을 단정하기보다 당시 모델 탐색 흐름을 기록한 참고값으로 본다.

| 모델 | P@10 | R@10 | HR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Content-Based | 0.0268 | 0.0238 | 0.2250 | 0.0286 |
| User-Based CF | 0.0545 | 0.0427 | 0.3011 | 0.0655 |
| **Item-Based CF** | **0.0783** | **0.0880** | **0.4800** | **0.1078** |
| Funk SVD biased | 0.0003 | 0.0006 | 0.0025 | 0.0004 |
| Funk SVD unbiased | 0.0037 | 0.0030 | 0.0350 | 0.0042 |
| **Final BPR** | **0.05225** | **0.06870** | **0.3850** | **0.06968** |

BPR은 Precision에서 User-Based와 거의 비슷하고 Recall/HR/NDCG는 더 높게 나오며, **Item-Based 다음으로 강한 단일 추천 모델**이 되었다. 초기 Biased Funk-SVD 성능(P@10 0.0003)이 극히 낮았던 것을 생각하면 단순 parameter tuning 이상의 의미가 있다.

### Iteration은 오늘 재최적화하지 않은 이유

Factors/Regularization이 바뀌었으니 iteration도 다시 열어야 하나 고민했지만, 이전에 이미 5/10/15/30 비교와 Explicit False 적용 전후의 optimal iteration 변화까지 확인한 경험이 있어, 지금 다시 grid를 확장하면 BPR 자체의 미세최적화에 과도하게 들어갈 수 있다고 판단했다. 현재 목적은 "Best BPR benchmark"가 아니라 "최종 Hybrid Recommendation System"이므로 iteration은 `15`로 유지하고, **Hybrid 안에서 BPR의 역할이 실제로 중요하다고 확인되면 그때 다시 최적화**하기로 했다 — Standalone 최적 BPR과 Hybrid 최적 BPR은 다를 수 있기 때문이다.

---

## 5. 내가 직접 판단한 부분

- **좋은 결과를 바로 믿지 않고 비교 조건의 공정성부터 확인** — Explicit False 적용 후 성능 상승을 보고도 "평가 조건이 달랐는데 정말 False 때문이라고 할 수 있나?"를 먼저 검증
- **Loss가 나빠졌는데 Test 성능이 좋아진 결과에 의문을 제기**하고, Train objective와 Test recommendation objective가 다른 목적을 가진다는 것을 실험으로 확인
- **Factors=80에서 NDCG만 좋아진 것을 단순히 "좋다"로 넘기지 않고** "왜 NDCG는 오르는데 HR은 떨어지지?"라고 질문해 metric별 의미를 재해석
- **처음 세운 가설("과적합이 상쇄")이 부정확하다는 것을 스스로 인정**하고 "Ranking quality의 일부만 좋아졌다"로 해석을 수정
- **Factors와 Regularization을 독립 변수가 아닌 상호작용 변수로 보자는 발상 전환**
- **60이라는 예상 밖 결과를 보고 탐색 방향을 큰 폭 증가에서 Local Search로 전환**
- **더 세밀하게 찾을 수 있음에도 "지금 목적에 중요한가"를 기준으로 탐색을 멈춤**
- **이번 BPR 작업 전체에서 우선순위(Problem Definition > Data Signal > Hyperparameter)를 스스로 정리**
- **Iteration을 오늘 다시 열지 않기로 결정** — Hybrid에서 BPR 역할이 확인된 뒤로 미룸

## 6. GPT(Claude)가 보완한 부분

Explicit False 재검증을 위한 `positive_only=True` 통일 조건 설계 및 실험 코드 지원, Regularization/Factors/상호작용/Local Search 각 단계의 실험 스크립트 작성, 결과 표 정리와 1차 해석 제안(예: Loss와 Test metric의 관계, NDCG와 HR의 트레이드오프 설명), 최종 파라미터 확정과 전체 모델 비교표 정리 지원 — 코드 작성과 실험 설계 지원은 GPT 비중이 높았지만, **질문을 만들고 실험 방향을 제안하고 결과에서 이상한 부분을 발견하고 다음 실험을 왜 해야 하는지 판단하는 것**은 직접 수행했다.

---

## 7. 오늘의 회고 — 개발 철학 정립

오늘 구체적으로 배운 것들:

- **좋은 실험은 parameter가 아니라 질문에서 시작된다** — "LR 몇으로 하지?"가 아니라 "왜 NDCG와 HR이 반대로 움직였지?"가 먼저였고, 그 질문이 factors와 regularization을 바꾼 이유였다.
- **여러 metric을 보는 이유를 처음 제대로 체감했다** — 지표들은 점수판이 아니라 모델의 서로 다른 행동을 관찰하는 창이며, 숫자가 다르게 움직이는 순간이 오히려 가장 흥미로웠다.
- **Loss가 모델의 목적 전체를 대표하지 않는다** — 만드는 것은 Loss가 낮은 시스템이 아니라 사용자에게 좋은 Top-N을 주는 추천시스템이므로, 앞으로도 "Loss가 내려갔는가"와 "실제 모델 행동은 어떻게 바뀌었는가"를 함께 본다.
- **복잡한 모델이 좋은 모델은 아니다** — Factors=100까지 올려도 60보다 나빴다.
- **모델 개선은 내가 직접 판단할 수 있는 일이다** — 코드 일부를 AI가 작성했어도 질문·실험 방향·이상 발견·다음 실험 판단은 직접 했기에, AI를 많이 썼지만 오히려 이전보다 더 "내가 만든 모델"이라는 느낌이 강했다.

이를 바탕으로 앞으로의 개발 철학을 정리했다: **점수를 올리는 것을 목표로 삼지 말고, 모델을 이해하면서 더 나은 행동을 하게 만드는 것을 목표로 삼는다.** 구체적으로 ① 모델부터 고르지 않고 문제부터 정의한다, ② Baseline을 반드시 남긴다, ③ 한 번의 실험에는 하나의 질문을 둔다(가설→통제 변수→실험→결과→해석), ④ Loss 하나만 보지 않는다, ⑤ 결과가 나오면 코드부터 수정하지 않고 왜 이런 결과가 나왔는지 먼저 생각한다, ⑥ AI에게 "최적화해줘"가 아니라 "내 가설을 실험하게 해줘"라고 요청하고, 문제 정의·가설·결과 해석·의사결정은 직접 맡는다.

BPR 전체에서 얻은 가장 큰 교훈은 **Hyperparameter보다 Data Signal이, Data Signal보다 Problem Definition이 중요하다**는 우선순위였다 — `is_recommended=False`를 모델이 어떤 signal로 이해하는가의 문제가 factors/epoch/lr보다 훨씬 근본적이었다. 아무리 좋은 optimizer와 parameter를 써도 잘못된 문제를 잘 풀고 있다면 좋은 추천시스템이 되지 않는다는 것을 이번 BPR 여정 전체로 확인했다.

> **오늘은 BPR의 점수를 높인 날이 아니라, 서로 다르게 움직이는 숫자들에서 모델의 행동을 읽고, 내가 직접 질문과 가설을 만들어 모델을 개선한 날이었다. AI가 코드를 많이 작성하더라도 문제를 정의하고 실험을 설계하고 결과의 의미를 판단하는 사람이 나라면 이 프로젝트는 여전히 내가 만드는 프로젝트라는 것을 처음 강하게 느꼈다. 앞으로도 단순히 점수를 올리는 것이 아니라 모델을 이해하면서 더 나은 행동을 하게 만드는 개발자가 되고 싶다.**

---

## 8. 현재 프로젝트 상태

```
[완료] Explicit False 효과의 공정한 재검증 (positive_only=True 통일)
[완료] Regularization 실험 (0.0001~0.020) → 0.005 확인
[완료] Factors 실험 (20/40/80) → metric별 트레이드오프 해석
[완료] Factors × Regularization 상호작용 실험 → 60/0.005 확인
[완료] Local Search (55~65 × 0.005~0.008) → 60/0.006 최종 확정
[완료] Standalone BPR 최종 버전 결정 (P@10 0.05225, HR@10 0.3850, Hits 209)
[완료] 전체 모델(Content/User/Item/Funk SVD/BPR) 성능 비교표 갱신 — BPR이 Item-Based 다음으로 강한 단일 모델로 확인
[완료] 추천시스템 개발 철학 정리 (문제 정의 > 데이터 signal > hyperparameter)
[미완료] Hybrid 관련 개념 학습 (『핸즈온 추천 시스템』 Hybrid 파트)
[미완료] 기존 프로젝트 전체를 Hybrid 관점에서 재분석
[미완료] Hybrid Architecture 설계 및 Baseline 구현
```

## 9. 다음 시작점

다음 시간에는 **BPR을 추가로 미세조정하지 않는다.** Standalone BPR은 `iterations=15, factors=60, regularization=0.006, Explicit False=True`로 종료한다.

```text
Hands-On Hybrid 완독
↓
Hybrid 종류 공부
↓
전체 프로젝트 결과 재분석
↓
Hybrid의 문제 정의
↓
Architecture 설계
↓
Baseline 구현
```

순서로 진행하며, Hybrid 결과에서 BPR이 중요한 역할을 한다고 판단되면 그때 `iterations / factors / regularization`을 다시 연다. 프로젝트 최종 로드맵은 Phase 1(Hybrid 공부) → Phase 2(재분석) → Phase 3(Architecture 설계) → Phase 4(Baseline 구현) → Phase 5(기여도 분석) → Phase 6(Hybrid 최적화) → Phase 7(최종 정량·정성평가, 약 2일) → Phase 8(장점·한계 분석) → Phase 9(다음 공부 방향 정리) → Phase 10(프로젝트 최종 종료: 회고·README·GitHub 정리) 순서로 확정했다. 이후에는 잠시 대형 프로젝트를 쉬고 더 어려운 Kaggle 문제 또는 가벼운 ML/DL 학습을 진행할 계획이다.