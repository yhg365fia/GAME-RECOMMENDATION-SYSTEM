# Day20 — BPR 전체 학습, False signal 실험, 모델 비교와 Hybrid 방향 설정

## 1. 오늘 한 일 요약

기존 BPR 파이프라인을 실제 전체 데이터에서 안정적으로 돌리고 `iterations`에 따른 성능 차이를 확인한 뒤, **Steam의** **`is_recommended=False`** **정보를 BPR이 제대로 활용하지 못하고 있다는 문제를 발견하고 이를 explicit negative로 추가 학습하는 실험까지** 진행했다. 단순히 BPR을 돌린 날이라기보다 **BPR을 실제 프로젝트 데이터에 맞게 다시 정의하고, 모델 성능 차이를 추천시스템 관점에서 해석하기 시작한 날**에 가까웠다.

```text
implicit BPR 전체 데이터 학습
        ↓
5 / 10 / 15 / 30 iterations 비교 → 10 iterations가 기존 BPR 최고 성능
        ↓
기존 모델들과 성능 비교 → Item-Based가 BPR보다 높은 이유 탐색
        ↓
BPR이 False를 positive interaction처럼 취급한다는 문제 발견
        ↓
True-only BPR + True > False 추가 학습 구조 구현
        ↓
Explicit False 학습 후 성능 크게 상승 (P@10 0.0278 → 0.0520)
        ↓
평가 조건(positive_only) 차이라는 통제 변수 문제 발견 → 공정 재평가 필요
        ↓
BPR 개선을 조금 더 진행한 뒤 Hybrid로 넘어가기로 결정

```

---

## 2. implicit BPR 전체 학습과 iteration 실험

### 직접 구현 vs implicit 라이브러리

처음 직접 구현했던 BPR은 Python loop로 interaction 하나하나에 대해 `negative sampling → score 계산 → gradient → factor update`를 수행했는데, 37M 규모에서는 현실적으로 너무 느렸다. 그래서 전체 데이터 학습에서는 `implicit` 라이브러리의 BPR을 사용했다.

오늘 이해한 핵심은 **BPR 알고리즘 자체가 다른 것이 아니라 구현 최적화 수준이 다르다**는 것이었다.

| 구분 | 직접 구현 | implicit |
|---|---|---|
| 처리 단위                    | single interaction, Python loop | CSR Sparse Matrix 기반 배치 |
| 자료구조                     | dict / set / 함수 호출              | 연속된 factor array        |
| 실행 환경                    | 순수 Python                       | Cython/C 수준 최적화, 멀티코어   |
| negative sampling/update | 직접 구현                           | 최적화된 구현                 |

따라서 기존 직접 구현은 **BPR 원리를 이해하기 위한 구현**, `implicit`은 **대규모 데이터를 실제로 학습시키기 위한 구현**으로 역할을 구분할 수 있었다.

### implicit으로 전환했을 때 학습 속도가 크게 빨라진 이유

직접 구현한 BPR에서는 Python의 `for` loop 안에서 interaction마다 `negative sampling → dot product → gradient 계산 → user/item factor update`를 하나씩 수행했다. 이 구조는 알고리즘 자체는 이해하기 쉽지만, 3,700만 건 규모에서는 매 update마다 Python 레벨의 반복문·함수 호출·`dict`/`set` 접근 비용이 누적되어 전체 학습 시간이 매우 커졌다.

반면 `implicit`으로 전환한 뒤에는 동일한 BPR 원리를 사용하면서도 학습이 수십 초~1분대에 끝날 정도로 크게 빨라졌다. 이유는 **알고리즘이 단순해진 것이 아니라, 같은 연산을 훨씬 효율적인 실행 구조로 처리하기 때문**이었다.

```text
직접 구현 BPR
Python loop
→ interaction 하나씩 처리
→ negative sampling
→ score / gradient 계산
→ factor update
→ Python 호출 비용이 매번 발생

implicit BPR
CSR Sparse Matrix
→ 필요한 interaction만 압축 저장
→ Cython/C 수준에서 핵심 학습 loop 실행
→ user/item factor를 연속된 array로 처리
→ 멀티코어로 병렬 연산
→ Python 레벨 반복 비용을 크게 줄임
```

특히 `CSR(Compressed Sparse Row)` 구조는 대부분이 0인 User×Item 행렬 전체를 dense하게 저장하지 않고 **실제로 interaction이 존재하는 위치만 저장**하므로, Steam처럼 매우 큰 sparse recommendation 데이터에서 메모리와 접근 비용을 크게 줄일 수 있다. 또한 `num_threads=0` 설정을 통해 사용 가능한 CPU 코어를 활용하고, negative sampling과 latent factor update 역시 Python 함수 호출을 반복하는 대신 최적화된 내부 loop에서 처리된다.

즉 오늘의 속도 개선은 **"더 단순한 BPR을 사용해서"가 아니라 "같은 BPR을 대규모 sparse 데이터에 맞는 자료구조와 컴파일된 연산으로 실행했기 때문"**이라고 이해했다. 이를 통해 추천시스템에서는 모델 수식 자체뿐 아니라 **자료구조, 구현 언어, 벡터화·병렬화 같은 시스템 수준의 최적화도 실제 학습 가능성을 결정하는 중요한 요소**라는 점을 배웠다.

### Iteration 실험 결과 (평가 유저 400명 동일)

| Iteration | P@10 | R@10 | HR@10 | NDCG@10 | Micro F1 | Hits |
|---:|---:|---:|---:|---:|---:|---:|
| 5                                                    | 0.0315     | 0.0349     | 0.2475     | 0.0398     | 0.0321     | 126     |
| **10**                                               | **0.0335** | **0.0380** | **0.2600** | **0.0452** | **0.0341** | **134** |
| 15                                                   | 0.0278     | 0.0319     | 0.2225     | 0.0369     | 0.0283     | 111     |
| 30                                                   | 0.0295     | 0.0368     | 0.2175     | 0.0424     | 0.0301     | 118     |

기존 BPR에서는 `10 iterations`가 가장 좋았다. 처음에는 "iteration 증가 → 학습 증가 → 성능 증가"라고 생각할 수 있었지만 실제 결과는 그렇지 않았고, 특히 15 iteration에서는 **Train AUC는 증가했는데 Test 추천 성능은 감소**했다. 이를 통해 **BPR training objective가 더 잘 최적화되는 것과 실제 Top-N 추천 지표가 좋아지는 것은 같은 일이 아니다**라는 점을 직접 확인했다.

### 실험 설정 구조 개선

iteration 실험마다 코드 아래쪽까지 내려가 `iterations=...`, `MODEL_PATH=...`를 찾는 것이 불편해, 실험 파라미터를 `main.py` 상단 Config 영역으로 모았다.

```python
BPR_ITERATIONS = 15
BPR_FACTORS = 40
BPR_LEARNING_RATE = 0.05
BPR_REGULARIZATION = 0.001

```

모델 파일 이름도 `f"implicit_bpr_{BPR_ITERATIONS}epoch_model.npz"`처럼 자동 생성되도록 구성해, 이제 `BPR_ITERATIONS = 10` 한 줄만 바꾸면 학습 iteration과 저장 모델 이름이 동시에 바뀐다. 이는 모델 자체보다 **반복 실험을 쉽게 만드는 실험 환경 개선**이었다.

---

## 3. 핵심 발견 — False signal과 Explicit Negative 학습

### 전체 모델 성능 비교에서 생긴 의문

> **Evaluation note:** 아래 단일 모델 수치는 실험 시점별 split, ground-truth 정의(`positive_only` 포함), 평가 코드가 완전히 동일하지 않은 Legacy 결과가 섞여 있어 절대적인 모델 간 우열 비교에는 주의가 필요하다. 당시 실험 흐름을 이해하기 위한 참고값으로 본다.

| 모델 | P@10 | R@10 | HR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Content-Based                   | 0.0268     | -          | -          | -          |
| User-Based CF                   | 0.0545     | 0.0427     | 0.3011     | 0.0655     |
| **Item-Based CF**               | **0.0783** | **0.0880** | **0.4800** | **0.1078** |
| Funk SVD biased                 | 0.0003     | 0.0006     | 0.0025     | 0.0004     |
| Funk SVD unbiased               | 0.0037     | 0.0030     | 0.0350     | 0.0042     |
| BPR 10 iter                     | 0.0335     | 0.0380     | 0.2600     | 0.0452     |

여기서 **"왜 훨씬 복잡한 머신러닝 모델인 BPR보다 단순한 Item-Based CF가 훨씬 잘 맞지?"**라는 질문이 생겼고, 이것이 오늘의 중요한 사고 확장점이 되었다.

### 원인 발견 — BPR은 False를 positive처럼 취급하고 있었다

Item-Based 코드를 다시 확인하면서 중요한 차이를 발견했다. 기존 Item-Based interaction matrix는 `True → +1, False → -1, 없음 → 0`으로 구성되어 있어, Item-Item cosine similarity 계산 시 **전체 사용자들의 좋아요/싫어요 패턴이 모두 similarity에 반영**된다.

반면 `implicit` BPR은 sparse matrix 값이 `+1`인지 `-1`인지가 아니라 **non-zero인가 아닌가**를 중심으로 interaction을 해석하기 때문에, 사실상 다음처럼 학습되고 있었다.

```text
[Steam 데이터의 실제 의미]        [기존 BPR이 학습한 것]
True   = 좋아함                   True   = positive
False  = 싫어함          →        False  = positive
Unseen = 모름                     Unseen = negative candidate

```

이건 단순 hyperparameter 문제가 아니라 **추천 문제 정의 자체의 문제**였다.

### 2단계 학습 구조 구현

`implicit` 자체에서는 False를 바로 explicit negative로 쓸 수 없어 학습을 두 단계로 구성했다.

```text
1단계: True interaction만 사용 → implicit BPR → True > Unseen 학습
2단계: 각 User의 True item > False item → Explicit Negative Fine-Tuning

```

```python
BPR_ITERATIONS = 15              # True > Unseen 15 iterations
BPR_EXPLICIT_FALSE_EPOCHS = 1    # True > False 1 epoch

```

False fine-tuning에서는 약 **3,928,771 pairs**가 학습되었고 `pair_accuracy ≈ 0.9079`를 기록했다. 즉 실제 False interaction이 명확한 negative signal로 모델에 들어가기 시작했다.

### Explicit False 적용 결과

| Metric | 15 iter (기존) | 15 iter + False |
|---|---:|---:|
| Precision\@10                        | 0.0278 | **0.0520** |
| Recall\@10                           | 0.0319 | **0.0670** |
| Hit Rate\@10                         | 0.2225 | **0.3950** |
| NDCG\@10                             | 0.0369 | **0.0718** |
| Micro F1                             | 0.0283 | **0.0572** |
| Hits                                 | 111    | **208**    |

즉 factor 수, learning rate, iteration을 조정한 것보다 **False를 무엇으로 정의하는가가 훨씬 큰 성능 변화를 만들어냈다.** 오늘 가장 중요한 실험 결과였다.

### 해석 수정 — 평가 조건이라는 통제 변수 문제

처음에는 "False 학습을 넣었더니 성능이 엄청 올랐다"고 바로 생각했지만, GPT가 **기존 BPR과 Explicit-False BPR의 평가 조건이 같지 않다**는 점을 지적했다.

```text
기존 BPR         : positive_only=False, Test relevant = 3,850
False-negative BPR: positive_only=True,  Test relevant = 3,273

```

평가에서도 False를 정답에서 제외했기 때문에 **성능 상승 전체를 False 학습 효과라고 단정하면 안 된다.** 정확한 검증은 `기존 BPR(positive_only=True)` vs `Explicit False BPR(positive_only=True)`로 비교해야 한다.

### 10 iter vs 15 iter 역전 현상

기존 BPR에서는 `10 iter > 15 iter`였는데, False fine-tuning 후에는 `15+False (P@10 0.0520) > 10+False (P@10 0.0348)`로 역전되었다.

처음에는 "15 iteration에는 False 정보가 없는 거 아니야?"라는 의문이 생겼는데, 맞다 — 15 iteration 자체는 `True > Unseen`만 학습하고 그 뒤에 동일한 1 epoch의 `True > False` fine-tuning이 들어간다. 즉 **False 학습량은 같지만 False 학습이 시작되는 latent representation의 상태가 다르다.** 따라서 문제 정의가 바뀌면 optimal iteration도 바뀔 수 있다는 새로운 가설이 생겼다.

---

## 4. 내가 직접 판단한 부분

- 전체 BPR 학습 실행 및 5/10/15/30 iteration 반복 실험, 각 성능 결과 비교
- 모델 path와 실험 구조 문제를 발견하고 **실험 parameter를 한곳에 모을 필요성 제기**
- 전체 모델 성능표를 본 뒤, **"왜 수식도 복잡하고 실제 학습까지 하는 BPR보다 단순한 Item-Based/User-Based가 더 잘 맞는가?"**라는 의문을 직접 제기했다.
- 여기서 단순히 "BPR 튜닝이 부족해서 그런가?"라고만 보지 않고, **추천시스템은 implicit 데이터 구조를 가지므로 일반적인 ML 모델 복잡도보다 현재 데이터 구조와 추천 목적에 맞는 collaborative signal이 더 강하게 작동한 것 아닐까**라는 가설을 세웠다.
- 동시에 **BPR도 parameter tuning이나 implicit 문제 정의를 개선하면 더 좋아질 수 있고, Random Forest·Boosting·Ensemble·NN 같은 다른 모델을 쓰면 추가 발전 가능성이 있지 않을까**라고 추측했다.
- 더 나아가 Content-Based / User-Based / Item-Based / Model-Based가 **각자 학습하는 정보, 잘 맞추는 부분, 놓치는 부분이 다르기 때문에 성능 차이가 발생하는 것 아닐까**라고 생각했고, 이것이 **각 모델의 강점을 합치는 Hybrid가 필요하지 않을까**라는 질문으로 이어졌다.
- Hybrid 역시 단순히 추천 결과를 몇 개씩 섞는 것인지, 아니면 각 모델의 score와 feature를 모아 다시 학습하는 **앙상블/Ranking 구조**로 가야 하는지 고민했다.
- Item-Based에서 `False=-1`을 실제로 쓰고 있는지 직접 코드를 다시 확인했고, 그 과정에서 **BPR에서도 False를 명확한 negative signal로 학습시키자**는 실험 방향을 결정했다.
- 37M interaction 전체 학습과 Explicit False 실험을 직접 실행하고 결과를 비교했다.
- **BPR을 어디까지 튜닝할지 프로젝트 전체 관점에서 판단**했다. Grid Search식 전수 탐색보다는, 문제 정의와 핵심 parameter를 짧게 검증한 뒤 전체 모델 비교와 Hybrid 설계로 넘어가는 것이 현재 프로젝트 목적에 더 맞다고 정리했다.

## 5. GPT(Claude)가 주로 도운 부분

`implicit` BPR의 구조 설명, Python custom BPR과 optimized library의 속도 차이 설명, iteration별 평가 결과 비교, 실험 parameter config 구조 정리, 기존 README 모델 성능 비교, Item-Based의 False 사용 방식 분석, **`implicit`****이 negative 값을 explicit negative로 처리하지 않는 문제 발견**, True-only + explicit False fine-tuning 구조 설계, **평가 조건 차이(****`positive_only`****) 지적**, Hybrid/Ensemble/Ranking 구조 정리, BPR 튜닝 범위를 제한하고 Hybrid로 넘어가는 프로젝트 방향 제안.

---

## 6. 개념적 성장 — 모델 복잡도 vs signal, 그리고 Hybrid

### 전체 모델 표를 보고 세운 가설과 수정된 이해

전체 모델 성능을 비교했을 때 가장 먼저 든 생각은 **"복잡한 ML 모델인 BPR이 왜 Item-Based보다 약하지? 내가 loss 최소화나 parameter tuning을 충분히 안 해서 그런가?"**였다. 하지만 여기서 한 단계 더 나아가, 추천시스템에서는 모델 복잡도보다 **데이터가 어떤 feedback 구조를 가지고 있고, 각 모델이 어떤 signal과 objective를 사용하느냐가 더 중요할 수 있다**고 추측했다.

이 추측에서 맞았던 부분은 컸다. 실제로 각 모델은 같은 데이터를 보더라도 전혀 다른 관계를 사용한다.

```text
Content-Based → 게임 자체의 장르/태그/텍스트가 무엇과 비슷한가
User-Based    → 나와 비슷한 유저가 무엇을 소비했는가
Item-Based    → 이 게임과 어떤 게임이 함께 소비되는가
BPR           → 이 유저에게 positive item이 negative item보다 위에 오는가
```

특히 현재 Steam 데이터는 37M interaction / 37K items 규모라 **item 간 co-consumption 패턴을 직접 사용하는 Item-Based가 매우 강한 signal을 얻을 수 있다.** 반대로 BPR은 수많은 interaction을 제한된 차원의 latent vector로 압축해 일반화하는 대신, 원본 관계의 일부를 잃을 수 있다. 따라서 **"수식이 더 복잡하고 학습을 한다 = 반드시 더 좋은 추천 성능"은 아니다**라는 점을 배웠다.

또 하나 중요한 수정은 **튜닝보다 먼저 문제 정의를 봐야 한다**는 것이었다. 기존 `implicit` BPR은 True와 False를 모두 non-zero interaction으로 받아 사실상 positive처럼 다루고 있었지만, Item-Based는 전체 matrix에서 `True=+1`, `False=-1` 정보를 similarity에 반영하고 있었다. 즉 BPR의 낮은 성능에는 단순히 `factors`, `lr`, `regularization`을 덜 튜닝한 것뿐 아니라 **True / False / Unseen을 어떻게 정의했는가**가 크게 영향을 주고 있었다.

여기서 **loss를 더 낮추는 것 자체가 목표가 아니라는 점**도 다시 확인했다. Train AUC나 BPR loss가 좋아져도 P@10, Recall@10, HR@10, NDCG@10이 같이 좋아진다는 보장은 없다. 추천시스템에서는 모델 내부 목적함수보다 **실제 Top-N ranking 목적과 evaluation metric이 맞는지**가 더 중요하다.

### 다른 ML 모델을 쓴다면 어디에 들어가는가

Random Forest, Boosting, Ensemble, NN을 쓰면 무조건 Item-Based를 이긴다는 뜻은 아니지만, **각 모델이 잡은 signal을 다시 조합하거나 ranking하는 단계**에서는 충분히 발전 가능성이 있다는 것을 배웠다.

예를 들어 LightGBM/XGBoost는 단순히 `user_id, app_id → 추천 여부`를 바로 예측하는 것보다, 여러 추천기가 뽑은 후보에 다음과 같은 feature를 붙여 최종 순위를 학습하는 방식이 더 자연스럽다.

```text
ItemCF score
BPR score
Content similarity
Popularity
Genre affinity
True ratio
Interaction count
        ↓
LightGBM / XGBoost Ranker
        ↓
Final Ranking
```

Two-Tower 같은 NN 구조 역시 `user embedding`과 `item embedding`을 따로 학습해 대규모 후보를 빠르게 찾는 **retrieval 모델**로 이해할 수 있었고, 단순히 "NN이라 더 강하다"가 아니라 **어떤 단계와 목적에 쓰는 모델인지가 중요하다**는 관점을 갖게 되었다.

### Hybrid에 대한 이해 확장

처음에는 Hybrid가 각 모델에서 추천 몇 개씩 뽑아 합치는 방식인지, 아니면 머신러닝의 Ensemble처럼 score를 합쳐야 하는지 궁금했다. 정리해보니 둘 다 Hybrid가 될 수 있지만 구조가 다르다.

```text
[단순 Weighted Hybrid]
final_score
= w1 * ItemCF
+ w2 * BPR
+ w3 * Content

[Multi-stage Hybrid]
Content-Based ─┐
Item-Based CF ─┤
BPR ───────────┤
Popularity ────┤
               ▼
         Candidate Pool
               ↓
   각 모델 score + user/item feature
               ↓
        LightGBM / XGBoost
               ↓
            Final Rank
               ↓
             Top-10
```

머신러닝 관점에서는 blending/stacking/ensemble과 비슷하지만, 추천시스템에서는 `Candidate Generation → Ranking → Re-ranking`이라는 **multi-stage recommender 구조**로 이해하는 것이 더 정확하다. 특히 지금 프로젝트에서는 Content-Based, Item-Based, BPR이 서로 다른 signal을 잡는다는 사실을 이미 실험으로 확인했기 때문에, **최고 점수 모델 하나만 남기는 것보다 각 모델의 장점을 Hybrid에서 재사용하는 방향**이 더 자연스럽다고 판단했다.

### 모델 개선 범위 결정

"BPR을 더 튜닝할까 vs 이제 Hybrid로 넘어갈까"를 고민한 끝에 **BPR을 아주 짧게만 더 개선·검증한 뒤 Hybrid로 넘어간다**로 결정했다. `lr 10개 × reg 10개 × factors 10개 × iterations 10개` 같은 Grid Search는 프로젝트 목표와 맞지 않는다. 현재 목적은 "모델 하나 최고점 만들기"가 아니라 **"여러 추천 방법 경험 → 구조/한계 비교 → 적합한 모델 선택 → Hybrid 설계"**이기 때문이다.

## 7. 오늘의 회고

오늘 가장 중요한 배움은 **"추천 모델의 성능은 모델의 복잡도만으로 결정되지 않는다. 데이터에서 무엇을 positive/negative로 정의하고, 어떤 signal을 학습하며, 어떤 objective를 최적화하는지가 훨씬 중요할 수 있다"**는 것이었다.

Funk SVD에서는 `rating prediction objective ≠ Top-N ranking objective` 문제를 봤고, BPR에서는 `False를 positive처럼 취급 ≠ Steam 사용자의 실제 선호 구조` 문제를 직접 발견했다. 결국 오늘은 **모델 코드를 더 복잡하게 만드는 것보다 문제 정의를 제대로 하는 것이 성능에 더 큰 영향을 줄 수 있다는 걸 실험으로 확인한 날**이었다.

또한 프로젝트를 바라보는 관점이 **"어떤 ML 모델이 가장 좋냐"에서 "각 추천 방식이 어떤 signal을 잡느냐"로** 바뀌기 시작했다는 점에서도 의미가 컸다. 다만 Explicit False 실험의 성능 상승폭을 그대로 신뢰하지 않고 평가 조건(`positive_only`)이라는 통제 변수 문제를 인지한 것도 중요했다 — 다음 세션의 첫 작업은 공정한 baseline 재평가다.

> **오늘은 BPR을 단순히 구현하고 튜닝한 것이 아니라, Steam의 True/False/Unseen 구조를 추천 문제에 맞게 다시 정의하면서 "좋은 모델보다 좋은 문제 정의가 먼저다"라는 걸 실제 성능 변화로 확인했고, 이제 각 모델의 서로 다른 signal을 결합하는 Hybrid 단계로 넘어갈 준비를 시작했다.**

---

## 8. 현재 프로젝트 상태

```
[완료] implicit 라이브러리 기반 BPR 전체 데이터(37M) 학습
[완료] 5/10/15/30 iteration 비교 실험 (기존 BPR은 10 iter 최고)
[완료] 실험 parameter Config 구조 개선 (main.py 상단 집중 + 모델명 자동 생성)
[완료] 기존 전체 모델(Content/User/Item/Funk SVD/BPR) 성능 비교표 정리
[발견] implicit BPR이 False를 positive처럼 취급하고 있던 문제 (문제 정의 자체의 오류)
[완료] True-only BPR + True>False Explicit Negative Fine-Tuning 구조 구현 (3,928,771 pairs, pair_accuracy 0.9079)
[완료] Explicit False 적용 결과 확인 (15 iter 기준 P@10 0.0278 → 0.0520)
[발견] 기존 BPR과 Explicit-False BPR의 평가 조건(positive_only) 불일치 → 성능 상승폭을 그대로 단정할 수 없음
[발견] False fine-tuning 후 optimal iteration이 10 → 15로 역전되는 현상
[미완료] positive_only=True 조건으로 공정한 baseline 재평가
[미완료] regularization / factors 소규모 실험 및 BPR 최종 버전 결정
[미완료] 전체 모델 비교표 최종 정리 및 BPR 단계 종료
[미완료] Hybrid Recommendation 설계

```

## 9. 다음에 바로 해야 할 일

다음 세션에서는 먼저 **BPR 마지막 소규모 개선 단계**를 진행한다. 특히 regularization 실험 전에 **공정한 baseline 평가부터 하는 것**이 맞다.

1. 기존 BPR을 `positive_only=True`로 재평가 → False 학습 효과를 공정하게 비교
2. Explicit-False BPR 기준으로 iteration 결과 다시 정리 (10 / 15 중심)
3. regularization 소규모 실험 (0.0001 / 0.001 / 0.01)
4. 필요하면 factors 40 vs 80 한 번 비교
5. BPR 최종 버전 결정
6. Content / User / Item / Funk SVD / BPR 전체 모델 비교표 정리
7. BPR 단계 종료
8. Hybrid Recommendation 설계 시작