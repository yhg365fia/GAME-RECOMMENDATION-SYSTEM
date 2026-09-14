# Day18 — Funk-SVD 평가 결과 분석 및 실패 원인 탐색

## 1. 오늘 한 일 요약

저번에 구현·평가까지 완료했던 **Funk-SVD(Matrix Factorization)**의 성능이 지나치게 낮게 나온 원인을 분석했다.

| Metric | Result |
|---|---:|
| Precision@10 | 0.0003 |
| Recall@10 | 0.0006 |
| Hit Rate@10 | 0.0025 |
| NDCG@10 | 0.0004 |
| Micro Precision@10 | 0.0003 |
| Micro Recall@10 | 0.0003 |
| Micro F1@10 | 0.0003 |
| 전체 Hits | 1 |
| 전체 추천 | 4,000 |
| Test 정답 | 3,273 |

400명의 사용자에게 10개씩 총 4,000개를 추천했지만 실제 Test 데이터와 겹친 추천은 단 1개뿐이었다. 오늘의 핵심 문제는 단순한 하이퍼파라미터 튜닝이 아니라 **"Funk-SVD가 왜 거의 아무것도 맞히지 못했는가?"**를 분석하는 것이었다.

```text
가설1: positive_only=True로 인한 정답 축소가 원인 아닐까?
        ↓
positive_only=False로 재평가 → 정답 수는 늘었지만 Hits는 그대로 → 가설 기각
        ↓
랜덤 사용자 추천 결과 확인 → 생소하고 마이너한 게임이 다수 등장
        ↓
추천 게임들의 Train 통계(interaction 수, True 비율) 분석
        ↓
400명 전체의 반복 추천 게임 집계
        ↓
"높은 True 비율 게임이 랭킹 상단을 반복 지배"하는 패턴 발견
        ↓
GPT와 함께 item bias(b_i) 및 rating-prediction vs ranking objective mismatch로 해석
        ↓
Bias 제거(biased=False) Ablation 실험 계획으로 마무리
```

---

## 2. 첫 번째 가설 검증 — Test에서 False를 제외한 것이 문제인가?

### 가설

평가 코드에는 `positive_only=True` 옵션이 있어, Test 데이터 중 `is_recommended == True`인 게임만 정답으로 사용하고 있었다. 반면 이전 User-based/Item-based CF 평가에서는 True/False 여부와 관계없이 사용자가 실제로 interaction한 Test 게임 자체를 정답으로 사용했다. 이 차이 때문에 "Funk-SVD만 정답 수가 지나치게 줄어들어 성능이 낮게 나온 것 아닐까?"라는 가설을 세웠다.

### 코드 확인 — 학습 자체는 문제 없음

`fit()`에서는 Train 데이터를 다음과 같이 처리하고 있었다.

```python
ratings = train_df[["user_id", "app_id", "is_recommended"]].copy()
ratings["is_recommended"] = ratings["is_recommended"].astype(int)  # True→1, False→0
```

즉 Train 단계에서는 True/False 둘 다 정상적으로 학습에 쓰이고 있었고, `main.py`에서도 `recommender.fit(train_df)`로 필터링 없이 그대로 넘기고 있었다. 추천 후보 생성 시에도 Train에서 interaction한 게임은 True/False 관계없이 모두 제외되고 있었다. 즉 **"Train에서 False를 삭제해서 잘못 학습된 것"은 아니었다** — 문제 후보는 Test 평가 기준 쪽으로 좁혀졌다.

### 재평가 (positive_only=False)

`positive_only=True`를 `False`로 바꿔 Test의 모든 interaction을 정답으로 사용하도록 재평가했다.

| Metric | 기존(True만) | False 포함 |
|---|---:|---:|
| Precision@10 | 0.0003 | 0.0003 |
| Recall@10 | 0.0006 | 0.0006 수준 |
| Hit Rate@10 | 0.0025 | 0.0025 |
| 전체 Hits | 1 | 1 |
| 전체 추천 | 4,000 | 4,000 |
| Test 정답 | 3,273 | 3,850 |

Test 정답 수는 3,273 → 3,850으로 약 577개 늘었지만 **Hits는 여전히 1개**였다.

### 결론

"False를 정답에서 제외했기 때문에 성능이 낮다"는 가설은 **사실상 기각**되었다. 문제는 평가 정답 수가 아니라 **Funk-SVD가 만드는 Top-10 랭킹 자체가 실제 Test interaction과 거의 맞지 않는 것**이라고 판단을 전환했다.

---

## 3. 원인 재탐색 — Rating Prediction vs Ranking, 그리고 Item Bias

### Funk-SVD의 학습 목적 재확인

$$ \hat r_{ui} = \mu+b_u+b_i+p_u^Tq_i $$

모델은 $(r_{ui}-\hat r_{ui})^2$를 줄이는 방향으로 학습한다. 즉 Funk-SVD가 직접 배우는 문제는 **"사용자가 이 아이템에 어떤 rating을 줄 것인가?"**에 가깝다. 하지만 현재 프로젝트가 원하는 것은 **"수많은 게임 중 사용자가 실제로 interaction할 게임을 Top-10 안에 넣을 수 있는가?"**다. 실제 정답 게임의 예측 점수가 0.84로 나쁘지 않아도, 다른 게임들이 0.91/0.90/0.89를 받으면 정답 게임은 Top-10 밖으로 밀려 Hit=0이 된다. 즉 **rating prediction loss를 잘 줄이는 것과 Top-N ranking을 잘하는 것은 동일하지 않다**는 점을 이해했다.

### 랜덤 사용자 결과에서 이상 현상 발견

실제 Test(Barotrauma, Sid Meier's Civilization VI, Half-Life 2, Blackwake, Crusader Kings III)와 Funk-SVD 추천 결과(From Frontier, REVOLVER360 RE:ACTOR, DUSK '82, Tales From Off-Peak City Vol. 1, Deadhunt 등)를 비교하니 생소하고 마이너한 게임이 다수 등장했다. 여기서 "Funk-SVD가 사용자 개인 취향보다 어떤 특정 게임의 높은 rating 자체에 끌리고 있는 것 아닐까?"라는 새 의문이 생겼다.

### 확인 방향을 직접 설정

단순히 "성능이 낮다"에서 멈추지 않고, 왜 특정 게임들이 추천되는지 직접 확인해야 한다고 판단해 두 가지를 확인하기로 했다: ① 추천된 게임들의 **Train 평가 통계**(interaction 수, True 비율), ② **여러 사용자에게 동일 게임이 반복 추천되는가**.

### 추천 게임의 Train 통계

| 게임 | interaction | True | True Ratio |
|---|---:|---:|---:|
| Kick Ass Commandos | 213 | 201 | 0.9437 |
| Wally and the FANTASTIC PREDATORS | 143 | 142 | 0.9930 |
| Hook 2 | 198 | 198 | **1.0000** |
| Mission in Snowdriftland | 115 | 111 | 0.9652 |
| Duck Souls | 130 | 127 | 0.9769 |
| The Citadel | 240 | 230 | 0.9583 |
| Swordlord | 16 | 16 | **1.0000** |
| Mute Crimson+ | 65 | 64 | 0.9846 |
| 100 hidden dogs | 130 | 125 | 0.9615 |
| Knight Swap | 39 | 37 | 0.9487 |

추천 게임 대부분의 True 비율이 **94~100%**에 몰려 있었다.

### 400명의 반복 추천 게임 분석

400명 × Top-10 결과를 모아 동일 게임의 반복 빈도를 확인했다.

| 게임 | 추천 횟수 | interaction | True Ratio |
|---|---:|---:|---:|
| DUSK '82: ULTIMATE EDITION | 26 | 108 | 0.9815 |
| OXXO | 22 | 95 | 0.9684 |
| Paperball | 22 | 92 | 0.9783 |
| Everyday Genius: SquareLogic | 18 | 152 | 0.9737 |
| Blockwick 2 | 18 | 89 | 0.9888 |
| 100 hidden turtles | 17 | 73 | **1.0000** |
| Wally and the FANTASTIC PREDATORS | 17 | 143 | 0.9930 |
| Zombie Estate 2 | 17 | 229 | 0.9782 |
| Master of Magic Classic | 16 | 81 | **1.0000** |
| Hook 2 | 15 | 198 | **1.0000** |
| Tales From Off-Peak City Vol. 1 | 15 | 186 | 0.9946 |

여기서도 반복 추천되는 게임 대부분이 True Ratio ≈ 0.97~1.00에 가까웠다. 400명 중 같은 게임이 20여 명에게 추천되었다고 완전한 recommendation collapse는 아니지만, **높은 True 비율을 가진 일부 게임이 서로 다른 사용자들의 Top-N에 체계적으로 자주 등장하는 경향**은 명확히 확인되었다.

### GPT의 해석 — Item Bias 가설

$\hat r_{ui}=\mu+b_u+b_i+p_u^Tq_i$에서 **item bias $b_i$**가 영향을 미치고 있을 가능성을 제시받았다. `Hook 2`(198 interaction, 198 True)처럼 "누가 평가하더라도 거의 항상 높은 rating을 받는다"는 정보를 모델이 학습하면, 사용자 개인 선호($p_u^Tq_i$)가 아주 강하지 않아도 높은 $b_i$만으로 최종 예측 rating이 높게 형성될 수 있다.

```text
높은 True ratio → 높은 예상 rating → Top-N에 자주 등장
→ 하지만 실제 사용자가 다음에 interaction할 게임은 아님 → Hit = 0
```

---

## 4. 내가 직접 판단한 부분

- **가설 1을 직접 세우고 실험으로 검증**: "False를 정답에서 제외해서 점수가 낮다"는 가설을 세우고 `positive_only=False`로 직접 조건을 바꿔 재평가 → Test 정답이 3,273→3,850으로 늘었는데도 Hits가 그대로 1인 것을 확인하고 **가설을 스스로 기각**
- **결과가 바뀌지 않은 것을 근거로 문제 범위를 재설정**: 정답 수 문제가 아니라 "Top-10 랭킹 자체가 실제 interaction과 맞지 않는다"는 방향으로 전환
- **랜덤 사용자 추천 결과를 보고 패턴을 직접 포착**: 마이너 게임이 많이 나온다는 것을 보고 "특정 게임의 높은 rating 자체에 끌리는 것 아닐까"라는 새 가설 제시
- **추천 게임의 Train 통계(interaction 수·True 비율)를 확인하자고 판단**하고, **400명 전체의 반복 추천 빈도까지 조사하자고 스스로 범위를 넓힘**
- 결과를 보고 "True rating이 높은 게임들이 자주 추천되는 것 같다"는 해석을 직접 제시
- **지나친 추가 분석보다 다른 모델로 넘어가는 것이 낫지 않을지 판단**: $b_i$가 Top-N을 얼마나 수치적으로 지배하는지 완전히 분해·증명하는 것은 현재 프로젝트 목적상 필수적이지 않다고 보고, 대신 **bias를 제거한 간단한 ablation 실험**을 다음 단계로 선택

## 5. GPT(Claude)가 지원한 부분

main/evaluation/Funk-SVD 코드에서 False 처리 위치 확인 및 Train에서는 True/False가 모두 학습되고 있음을 코드로 확인, `positive_only=False` 진단 실험 제안, rating prediction과 ranking objective의 차이 설명, 추천 게임별 interaction count/True ratio 진단 코드 및 400명 전체 추천 빈도 분석 코드 제공(`groupby("app_id")`, `Counter` 활용), 높은 True ratio와 item bias 사이의 관계 설명, `biased=False` ablation 실험 제안, 이후 BPR 같은 ranking 기반 MF로 넘어가는 방향 제시 — 4,000개의 추천을 수작업으로 확인하는 대신 `groupby`/`Counter` 기반 집계 코드를 받아 실행함으로써, 직접 손으로 훑는 대신 통계로 패턴을 빠르게 발견할 수 있었다. GPT가 분석용 코드를 제공했고, 실제 실행 결과를 보고 모델의 특징과 문제를 해석하는 데는 직접 집중했다.

---

## 6. 오늘의 회고

오늘 가장 큰 시행착오는 **낮은 점수의 원인을 Test의 False 제거 문제라고 처음 추정했던 것**이다. 하지만 이 가설을 바로 폐기하지 않고 실제 코드를 변경해 재평가했고(Test 정답 3,273→3,850, Hits 1→1), 데이터로 가설을 기각할 수 있었다. 이후 랜덤 사용자의 추천 목록에서 이상한 게임들이 다수 등장한다는 점에 주목했고, 추천 아이템 통계와 전체 사용자 반복 추천 통계를 직접 확인함으로써 새로운 원인 가설(item bias)을 만들었다.

오늘의 핵심은 단순히 "결과가 안 좋다"에서 끝난 것이 아니라 `결과 확인 → 가설 설정 → 코드 변경 → 재실험 → 가설 기각 → 추천 결과 관찰 → 새 가설 설정 → 통계 확인 → 원인 범위 축소`라는 실험적인 문제 해결 과정을 밟았다는 점이다. 또한 이 과정에서 **Rating Prediction과 Ranking은 다른 문제**라는 것, **미관측 아이템은 negative가 아니라는 것**(관측된 True=1/False=0만 loss에 포함되므로 모델이 "좋아한 게임을 미관측 게임보다 위에 두라"는 ranking objective를 직접 배우지 않음), **Top-N은 절대 점수보다 상대적 순위가 중요하다는 것**, 그리고 **"좋아할 확률" $P(\text{Like}\mid\text{played})$과 "실제로 플레이할 확률" $P(\text{future interaction})$은 다르다**는 것을 개념적으로 정리했다. 어떤 마이너 게임을 플레이한 사람의 99%가 좋아한다고 해도, 특정 사용자가 그 게임을 다음에 플레이할 가능성은 매우 낮을 수 있다는 점이 이번 분석에서 가장 중요한 통찰이었다.

> **오늘은 Funk-SVD의 낮은 성능을 단순 실패로 넘기지 않고, 평가 기준 문제라는 첫 가설을 직접 실험으로 기각한 뒤 추천 결과와 데이터 통계를 통해 높은 긍정률 아이템이 랭킹 상단에 반복적으로 등장하는 현상을 발견했고, rating prediction과 Top-N ranking의 목적 차이라는 더 근본적인 문제까지 연결해 이해했다.**

---

## 7. 현재 프로젝트 상태

```
[완료] "False 정답 제외가 원인"이라는 가설 검증 및 기각 (positive_only=False 재평가)
[완료] 랜덤 사용자 추천 결과 정성 확인 및 이상 패턴 포착
[완료] 추천 게임의 Train interaction/True ratio 통계 분석
[완료] 400명 전체 반복 추천 게임 집계 및 패턴 확인
[완료] Item bias(b_i)와 rating-vs-ranking objective mismatch로 원인 가설 정리
[미완료] b_i가 Top-N을 실제로 얼마나 지배하는지 수치적 분해 검증 (현재 프로젝트 범위상 생략 판단)
[미완료] biased=False Ablation 실험 실행 및 비교
[미완료] Funk-SVD 단계 최종 종료 판단
[미완료] BPR 등 ranking 기반 MF로의 전환
```

## 8. 다음 시간 시작 지점

1. Funk-SVD 생성 시 `biased=False` 옵션 추가 (latent interaction $p_u^Tq_i$만으로 추천 생성)
2. 기존 모델과 구분하기 위해 `model_path="models/funk_svd_unbiased_model.pkl"`로 신규 모델 저장
3. 동일 조건(400명, 그룹당 100명, Top-10, 동일 random_state, 동일 Train/Test split)으로 재평가
4. 기존 Biased Funk-SVD(P@10=0.0003, HR@10=0.0025)와 비교해 세 가지 시나리오로 해석
   - 성능이 크게 상승 → item bias가 Top-N 랭킹을 방해한 것이 주요 원인
   - 여전히 매우 낮음 → rating prediction objective와 ranking 문제의 mismatch가 더 근본적인 원인
   - 성능이 더 감소 → bias도 어느 정도 유용했지만 Funk-SVD 구조 자체가 현재 문제와 맞지 않는다는 의미
5. 결과 해석 후 Funk-SVD 단계 종료, 이후 Top-N ranking을 직접 학습하는 **BPR(Bayesian Personalized Ranking)** 등 ranking 기반 MF로 이동 — Funk-SVD는 $r_{ui}\approx\hat r_{ui}$를 학습하지만 BPR은 $score(u,i^+) > score(u,j^-)$를 직접 학습해 Precision@10/Recall@10/HR@10/NDCG@10 평가와 더 직접적으로 연결된다는 점을 다음 단계 전환 근거로 정리했다.

---

## 9. 다음 시간

다음 시간에는 **BPR(Bayesian Personalized Ranking)**로 넘어간다.

- Funk-SVD와 BPR의 차이 이해
- Positive / Negative item 정의
- 기존 MF Train/Test split 그대로 사용
- 기본 BPR 구현
- Top-10 추천 연결
- 기존 평가 지표(P@10, R@10, HR@10, NDCG@10)로 평가
- Funk-SVD와 성능 비교

핵심 목표는 rating 예측이 아니라 **Top-N ranking을 직접 학습하는 모델이 실제로 더 잘 맞는지 확인하는 것**이다.