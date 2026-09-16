# Day19 — Funk SVD 마무리와 BPR 원리·소규모 구현

## 1. 오늘 한 일 요약

기존 Funk SVD의 마지막 실험으로 **bias를 제거한 unbiased Funk SVD**를 평가하고, 그 결과를 바탕으로 모델 기반 추천의 다음 단계인 **BPR(Bayesian Personalized Ranking)**로 넘어갔다. BPR은 코드 구현에 앞서 Funk SVD와의 목적 차이, pointwise/pairwise ranking의 차이, positive/negative item과 `(u,i,j)` triplet 구조, score difference, sigmoid, likelihood, BPR loss, latent vector update, SGD와 negative sampling, epoch와 update 횟수의 관계, sampling bias 가능성, 주요 hyperparameter까지 개념을 먼저 정리했다. 이후 `bpr_experiment.py`로 **유저 1,000명 규모 소규모 BPR 실험**을 직접 수행했고, 처음에는 거의 학습되지 않았던 모델이 parameter를 조정한 뒤 정상적으로 ranking을 학습하는 것까지 확인했다.

```text
Funk SVD bias 제거(unbiased) 평가 → Hit Rate 다소 개선되었지만 여전히 매우 낮음
        ↓
GPT 분석: bias는 일부 원인, 근본 원인은 rating-prediction과 ranking objective의 불일치
        ↓
Funk SVD → BPR로 전환 결정 (모델 기반 추천의 마지막 모델로 확정)
        ↓
BPR 수학 개념 학습 (score diff → sigmoid → likelihood → loss → gradient)
        ↓
소규모 BPR 실험 1차 (거의 학습 안 됨, loss 0.693 근처 고정)
        ↓
parameter 조정 후 2차 실험 (loss 0.693 → 0.382로 정상 감소, 학습 확인)
        ↓
내일부터 기존 프로젝트 구조에 BPR 정식 연결 예정
```

---

## 2. Funk SVD Bias 제거(unbiased) 실험과 결론

기존 Funk SVD에서 추천이 일부 게임에 과도하게 집중되는 문제의 원인 중 하나로 item/user bias를 의심했고, **bias를 제거한 Funk SVD**를 직접 학습·평가했다.

### 최종 평가 결과 (400명, Top-10)

**Macro Average**

| Metric | Result |
|---|---:|
| Precision@10 | 0.0037 |
| Recall@10 | 0.0030 |
| Hit Rate@10 | 0.0350 |
| NDCG@10 | 0.0042 |

**Micro Average**

| Metric | Result |
|---|---:|
| Precision@10 | 0.0037 |
| Recall@10 | 0.0039 |
| F1@10 | 0.0038 |

전체 Hit 15 / 전체 추천 4,000개 / 전체 Test positive 3,850개.

**Review Group별 결과**

| Group | Precision | Recall | HR | NDCG |
|---|---:|---:|---:|---:|
| 10~15개 | 0.001 | 0.0020 | 0.01 | 0.00214 |
| 16~25개 | 0.001 | 0.0020 | 0.01 | 0.00113 |
| 26~45개 | 0.003 | 0.0027 | 0.03 | 0.00259 |
| 46~78개 | 0.010 | 0.0053 | 0.09 | 0.01087 |

interaction이 많은 유저일수록 어느 정도 성능이 올라가는 경향은 여전히 나타났다.

### 판단

bias를 제거하자 기존 Biased Funk SVD보다 Hit Rate가 어느 정도 개선되어, 기존 추천 집중 현상에 bias term의 영향이 실제로 존재했다는 것은 확인했다. 하지만 bias 제거 후에도 Precision/Recall/Hit Rate가 여전히 매우 낮고, **Half-Life 2: Episode One, Half-Life: Opposing Force, Half-Life: Blue Shift, F.E.A.R., Half-Life 2: Lost Coast** 같은 특정 게임이 여러 유저에게 반복 추천되는 현상이 계속 나타나, **단순히 bias 하나만의 문제가 아니라고 판단**했다.

GPT는 이 결과를 두고 "bias는 문제의 일부였지만, 더 근본적인 문제는 Funk SVD의 학습 목적과 현재 프로젝트의 평가 목적이 다르다는 것"으로 정리했다. Funk SVD는 $\hat r_{ui}$를 실제 값과 맞추는 **rating prediction / pointwise optimization**에 가깝지만, 현재 프로젝트는 Precision@10/Recall@10/Hit Rate@10/NDCG@10을 쓰는 **Top-N ranking 문제**다. 즉 원하는 것은 "게임의 정확한 점수를 맞추는 것"이 아니라 "사용자가 좋아할 게임을 다른 게임보다 위에 올리는 것"이라는 차이가 있다. 이 때문에 Funk SVD 파라미터 튜닝을 계속하기보다 **ranking objective 자체를 학습하는 BPR로 넘어가는 것이 적절하다**고 결론지었다.

오늘 모델 기반 추천의 전체 흐름을 다음과 같이 확정했다: `User-Based CF → Item-Based CF → Funk SVD → BPR → 모델 기반 추천 종료`. BPR 이후에는 모델을 계속 추가하지 않고 `모델 비교 → Hybrid → 최종 Ranking` 방향으로 넘어갈 예정이다.

---

## 3. BPR 핵심 개념 이해

### BPR이 왜 필요한가

BPR(Bayesian Personalized Ranking)의 핵심 목적은 **Positive item이 Negative item보다 높은 점수를 가지도록 학습하는 것**이다. Funk SVD는 $(u,i)$ 관계를 보고 예측값을 계산하지만, BPR은 $(u,i,j)$ triplet(u=user, i=positive item, j=negative item)을 사용한다.

### Funk SVD vs BPR — 문제 정의 차이

처음에는 "Funk SVD는 regression이고 BPR은 classification인가?"라는 의문을 가졌으나, 다음과 같이 정리했다.

- **Funk SVD**: 회귀적 관점에 가까움. $\hat r_{ui}=p_u^Tq_i$이고 $(r_{ui}-\hat r_{ui})^2$를 줄인다. "이 아이템의 값을 얼마나 정확하게 예측하는가"가 중요.
- **BPR**: 단순 classification이라기보다 **Pairwise Ranking**. $P(i>j\mid u)$, 즉 "유저에게 item i와 j 중 어느 것이 더 위에 와야 하는가"를 학습한다. sigmoid와 log loss를 쓰기 때문에 classification과 수학적으로 비슷한 부분은 있지만 최종 목적은 분류가 아니라 순위 학습이다.

참고로 ranking을 classification으로 푸는 방식도 있다 — 모델이 $P(y=1\mid u,i)$를 예측해 각 item의 positive 확률을 계산한 뒤 정렬하면 ranking이 만들어진다. 즉 **Pointwise Classification**("이 item을 좋아할까?")과 **Pairwise BPR**("A와 B 중 무엇을 더 좋아할까?")의 차이를 이해했다.

### BPR의 핵심 수학

유저/아이템 latent vector $p_u$, $q_i$에 대해 positive score $x_{ui}=p_u^Tq_i$, negative score $x_{uj}=p_u^Tq_j$를 계산하고, 그 차이 $x_{uij}=x_{ui}-x_{uj}=p_u^T(q_i-q_j)$를 사용한다. $x_{uij}>0$이면 원하는 방향(올바른 ranking), $x_{uij}<0$이면 잘못된 ranking이다.

점수 차이는 실수이므로 0~1 값으로 변환하기 위해 **sigmoid** $\sigma(x)=\frac{1}{1+e^{-x}}$를 사용해 $\sigma(x_{uij})$를 구한다. 예를 들어 $x_{uij}=5$면 $\sigma(5)\approx0.993$으로 positive가 위에 있을 가능성이 매우 높다는 뜻이다.

**Likelihood**는 확률을 더하는 것이 아니라, 여러 ranking 관계가 동시에 관찰될 확률이므로 **곱**한다 ($\prod\sigma(x_{uij})$) — 이는 처음에 헷갈렸던 부분으로, joint probability는 $P(A,B,C)=P(A)P(B)P(C)$처럼 곱해진다는 것을 다시 이해했다. 확률을 계속 곱하면 값이 너무 작아지므로 log를 취해 $\sum\log\sigma(x_{uij})$를 최대화하는 문제로 바꾸고, 머신러닝은 보통 loss를 최소화하므로 부호를 뒤집어 $-\sum\log\sigma(x_{uij})$를 쓴다.

### 최종 BPR Loss

$$ L = -\log\sigma(x_{uij}) + \lambda(\|p_u\|^2+\|q_i\|^2+\|q_j\|^2) $$

- **Ranking Loss** $-\log\sigma(x_{uij})$: positive가 negative보다 위에 오도록 학습
- **Regularization** $\lambda(\|p_u\|^2+\|q_i\|^2+\|q_j\|^2)$: latent vector가 지나치게 커지는 것 방지

한 triplet만 볼 때는 $\sum$을 생략해도 원리를 이해하는 데 문제없고, 전체 학습 데이터에서는 $\sum L_{uij}$가 된다.

### 어떤 latent vector가 업데이트되는가

BPR에서는 $p_u$, $q_i$, $q_j$ 세 벡터를 모두 업데이트한다 — user는 positive item과 가까워지고, positive item은 user와 가까워지며, negative item은 user와 멀어지는 방향으로 학습되어 $p_u^Tq_i - p_u^Tq_j$가 점점 커진다.

### BPR 실제 학습 방식과 연산량

전체 게임 수가 매우 많을 때 "모든 positive-negative 조합을 비교하면 계산량이 폭발하지 않는가"라는 의문(예: positive 25개 × 전체 아이템 124,975개)이 있었지만, 실제 BPR은 모든 조합을 만들지 않는다. 대신 반복적으로 `(u,i,j)` 하나를 샘플링해 ① $p_u$ 가져오기 → ② $q_i$ 가져오기 → ③ $q_j$ 가져오기 → ④ score 계산 → ⑤ loss 계산 → ⑥ 세 벡터 update를 수행하고 다음 triplet을 다시 샘플링하는, **작은 계산을 매우 많이 반복하는 구조**다.

### Epoch의 의미

처음에는 "loss 최적화 반복 수가 epoch인가?"라고 생각했지만, 정확히는 `(u,i,j)` 하나에 대한 loss 계산+파라미터 업데이트가 **1 step/1 update**이고, 이런 update를 학습 데이터 전체에 대해 한 바퀴 수행하는 것이 **1 epoch**라고 정리했다. 즉 epoch는 update 한 번이 아니라 전체 학습 반복 단위다.

### Sampling Bias 문제의식

"random sampling을 계속하면 특정 user나 특정 item만 많이 학습될 가능성이 있지 않은가"라는 의문을 스스로 제기했다. 실제로 가능하며, interaction이 많은 유저는 positive interaction 기반 sampling에서 더 자주 학습될 수 있다. 이 때문에 interaction-based sampling, uniform user sampling, uniform negative, popularity-based negative, hard negative 등 다양한 sampling 방식이 존재하며, **"어떤 데이터를 얼마나 자주 보여주는가"도 모델의 중요한 일부**라는 것을 이해했다.

### BPR 주요 Hyperparameter

- `n_factors`: latent vector 차원
- `learning_rate`: 한 번 update할 때 얼마나 크게 움직일지
- `regularization`: latent vector 과대화 방지
- `epochs`: 전체 학습 반복 횟수
- `negative sampling`: 어떤 negative를 얼마나 뽑을지 — BPR에서는 이 자체가 매우 중요한 학습 전략이라는 점을 확인

---

## 4. 소규모 BPR 실험

이론 이해 후 `bpr_experiment.py`를 만들어 소규모 실험을 했다.

### 실험 데이터

- 전체 Train: 37,113,471건
- Positive 5개 이상 유저: 1,598,086명
- 샘플 유저: 1,000명, 샘플 interaction: 9,927건 (True 8,801 / False 1,126)
- 학습에 사용한 item 수: 3,316, positive interaction 수: 8,801

### 구현한 코드 구조

- `sigmoid()`: BPR probability 계산
- `_sample_negative()`: 해당 유저가 interaction하지 않은 게임에서 random negative 샘플링
- `_update()`: 하나의 triplet에 대해 $x_{ui}$, $x_{uj}$, $x_{uij}$ 계산 후 $p_u, q_i, q_j$ 업데이트
- `fit()`: epoch loop 및 positive interaction 반복
- `predict_all()`: 한 user에 대해 모든 item score 계산
- `recommend()`: seen item 제외 후 Top-K 추출

### 1차 실험 (거의 학습 안 됨)

파라미터: `n_factors=40, epochs=5, learning_rate=0.01, reg=0.01`

```text
Epoch 1 Loss = 0.69320
Epoch 5 Loss = 0.69313
```

거의 변화가 없었다 ($\ln 2 \approx 0.6931$이므로 모델이 초기 random 상태에서 거의 벗어나지 못한 상태). 실제 triplet에서도 `positive_score=0.000202, negative_score=-0.000607, score difference=0.000809, probability=0.5002`로 방향은 맞지만 거의 학습되지 않았다.

GPT는 이 결과를 "구현이 실패한 것이 아니라 update 크기와 반복 횟수가 너무 작아서 실제 학습이 충분히 진행되지 않은 것에 가깝다"고 분석했고, 학습 여부 자체를 확인하기 위해 파라미터를 `n_factors=40, epochs=30, learning_rate=0.05, reg=0.001`로 변경해 재실험을 제안했다.

### 2차 실험 (정상 학습 확인)

```text
Epoch  1 = 0.69315
Epoch  5 = 0.69262
Epoch 10 = 0.68810
Epoch 15 = 0.65781
Epoch 20 = 0.59076
Epoch 25 = 0.48446
Epoch 30 = 0.38198
```

$0.693 \rightarrow 0.382$로 명확하게 감소했다. 최종 예시 triplet에서 `positive_score=1.93685, negative_score=-0.15881`로 $x_{uij}=2.09566$, $\sigma(2.09566)\approx0.89048$ — 모델이 해당 pair에서 "positive item이 negative item보다 위에 있어야 한다"는 관계를 상당히 강하게 학습했다.

### 추천 Score에 대한 이해

초기에는 추천 score가 0.002 수준이었지만 학습 후에는 2.97, 2.70, 2.29, 2.26 등으로 점수 차이가 크게 벌어졌다. BPR에서는 이 score가 확률처럼 0~1이어야 하는 것은 아니며, 중요한 것은 $score_i > score_j$라는 **상대적 순위**다. 즉 절대적인 score 자체보다 순서가 핵심이라는 것을 확인했다.

---

## 5. 내가 직접 판단한 부분

- Funk SVD bias 제거 후 Hit가 개선됨을 직접 확인했지만, ranking 성능은 여전히 매우 낮다고 판단
- Funk SVD와 ranking 목적의 불일치가 핵심 문제라는 GPT의 방향에 동의하고, 추가 Funk SVD 튜닝보다 **BPR로 넘어가기로 결정**
- likelihood, sigmoid, loss 사이의 관계를 직접 질문하며 이해
- BPR의 조합 수와 계산량 문제를 스스로 의심하고 질문
- sampling bias 문제를 스스로 제기
- BPR이 실제로 골고루 학습되는지 의문을 제기
- 소규모 실험에서 1차 결과(loss 정체)를 코드 실패로 단정하지 않고, **parameter를 바꿔 실제로 학습되는지 직접 검증**

## 6. GPT(Claude)가 보완하거나 정정한 부분

`BPR = Classification`이라는 단순 정의를 Pairwise Ranking으로 정정, sigmoid와 likelihood가 같은 개념이 아니라 연결된 개념이라는 점 구분, likelihood는 확률의 합이 아니라 joint probability이므로 곱이라는 점 설명, `σ(5)≈0.993`의 loss가 정확히 0이 아니라 약 0.007임을 설명, epoch와 single update의 차이 설명, 모든 positive-negative 조합을 계산하지 않고 sampling+SGD를 이용한다는 설명, BPR에서도 sampling bias가 존재할 수 있음을 설명, 1차 BPR 결과가 코드 오류보다는 "거의 학습되지 않은 상태"라고 해석, 30 epoch/lr 0.05 실험을 통해 실제 학습 여부를 확인하도록 제안 — 오늘도 개념 설명과 실험 설계 제안은 GPT 비중이 높았지만, 결과를 해석하고 다음 실험 방향을 최종 선택하는 것은 직접 판단했다.

---

## 7. 오늘의 회고

오늘의 시행착오는 세 가지였다. **① BPR을 classification으로 단순화**했던 것 — `Funk SVD=Regression, BPR=Classification`으로 이해하려 했으나 `Funk SVD=Pointwise/Rating Prediction, BPR=Pairwise Ranking`으로 수정했다. **② Likelihood의 의미 혼동** — 처음엔 여러 확률을 더하는 것으로 생각했으나 $P(A,B,C)=P(A)P(B)P(C)$처럼 joint probability에서는 곱한다는 점을 이해했다. **③ 첫 BPR 학습이 거의 진행되지 않음** — loss가 0.693에 머무른 것을 코드 실패로 판단하지 않고 parameter를 바꿔 추가 실험했고, 결과적으로 0.693→0.382까지 정상적으로 감소하는 것을 확인했다.

오늘 가장 중요하게 배운 것은 **"좋은 추천 시스템에서 '점수를 잘 맞추는 것'과 '순위를 잘 만드는 것'은 다른 문제일 수 있다"**는 점이다. Funk SVD를 직접 구현하고 실패 원인을 분석했기 때문에, BPR의 의미가 단순한 새 알고리즘 추가가 아니라 **문제 정의와 objective function이 왜 중요한가**라는 관점으로 연결되었다. 또한 BPR에서도 loss 자체만이 아니라 sampling, update 횟수, negative 구성, latent dimension, learning rate, regularization 등 **학습 데이터가 모델에게 어떻게 전달되는지**가 중요하다는 것도 이해했다.

오늘은 단순 코드 복붙보다 **모델 수학 → 알고리즘 구조 → 실제 코드 동작**을 연결하는 데 집중했다: `Ranking 문제 → Positive>Negative → Score Difference → Sigmoid → Likelihood → Negative Log-Likelihood → BPR Loss → Gradient Update → Latent Vector 학습 → Top-N Ranking`. 실제 코드에서 loss가 0.693 근처에서 시작해 점차 감소하고 $P(i>j)$가 0.5에서 0.89 수준까지 올라가는 결과를 직접 확인하면서 BPR의 학습 원리를 수치로 검증했다.

> **오늘은 Funk SVD의 한계를 bias 제거 실험으로 끝까지 확인한 뒤, BPR의 수학적 원리부터 sampling·SGD 구조까지 이해하고 실제 소규모 구현에서 loss가 0.693 → 0.382로 감소하는 것을 확인하며 pairwise ranking 학습이 실제로 작동한다는 것까지 검증했다.**

---

## 8. 현재 프로젝트 상태

```
[완료] Funk SVD biased=False(unbiased) 학습 및 400명 평가
[완료] Bias 제거 효과 판단 (일부 개선되었지만 근본 해결은 아님 → 단일 원인 아님으로 결론)
[완료] Funk SVD → BPR 전환 결정 및 모델 기반 추천 전체 로드맵 확정
[완료] BPR 이론 학습 (pointwise vs pairwise, score diff, sigmoid, likelihood, loss, gradient, epoch, sampling bias, hyperparameter)
[완료] bpr_experiment.py 소규모 구현 (1,000명 샘플)
[완료] 1차 실험(학습 거의 안 됨) → 2차 실험(파라미터 조정 후 정상 학습, loss 0.693→0.382) 검증
[미완료] 기존 프로젝트 pipeline에 BPR 정식 연결
[미완료] 전체 데이터 BPR 학습 및 400명 동일 조건 평가
[미완료] Funk SVD와 BPR 성능 비교
[미완료] False를 explicit negative로 활용하는 추가 실험
```

## 9. 다음 할 일

내일은 소규모 실험을 다시 할 필요 없이 **"BPR prototype은 정상 작동한다. 이제 기존 프로젝트 구조에 연결한다"**에서 시작한다.

1. **기존 Funk SVD 파이프라인 구조 확인**
   - 데이터 로딩
   - 학습 / 저장 / 추천 / 평가 연결 방식 파악
2. **BPR 연결 구조 설계**
   - 그대로 사용: `mf_train`, `mf_test`, games, 기존 평가 지표
   - 새로 구현: negative sampling, `(u,i,j)` 학습, BPR update
   - 수정: 추천 함수와 모델 save/load
3. **`bpr.py` 정식 구현**
   - 전체 Train item universe 사용
   - `True`만 positive로 사용
   - small sample로 학습 → 저장 → load → 추천 확인
4. 전체 학습 전 속도·메모리 확인 후 BPR 전체 학습
5. **기존과 동일한 400명 평가**
   - P@10 / R@10 / HR@10 / NDCG@10
   - 그룹별 결과
   - 반복 추천 및 인기 편향 분석
6. **Funk SVD와 BPR 비교**: ranking objective가 실제 성능을 개선했는지 분석
7. **추가 실험**
   - 기본 BPR: `True vs unseen`
   - `False`를 explicit negative로 활용한 BPR 비교
8. 결과 정리 후 **모델 기반 추천 파트 종료 → Hybrid로 이동**