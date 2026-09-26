# Day16 — Matrix Factorization 이해와 Funk SVD 시스템 연결

## 1. 오늘 한 일 요약

Model-based CF 단계로 넘어가기 위해 **Matrix Factorization(MF) 개념을 정리하고, Funk SVD를 실제 프로젝트 파이프라인에 연결**하는 작업을 진행했다. 초점은 수식을 처음부터 직접 구현하는 것이 아니라, **MF/Funk SVD의 작동 구조를 이해한 뒤 GPT가 제공한 구현을 프로젝트 데이터 파이프라인과 연결해 실제 학습이 돌아가는 상태까지 만드는 것**이었다.

완료한 것:

- Matrix Factorization의 추천 원리, latent vector, SGD 개념 이해
- Funk SVD와 MF의 관계 정리, Ranking 문제와의 연결 확인
- Memory-based CF와 Model-based CF의 평가 구조 차이 파악
- 대규모 MF용 Train/Test split 구조 재설계 및 저장 (4천만 건 이상)
- Surprise `SVD` 모델 구현 및 학습 파이프라인 연결
- Funk SVD 학습 정상 작동 확인 (10만 건 샘플 + 전체 Train 3,700만 건)

---

## 2. 핵심 개념 정리

### Memory-based CF → Model-based CF (MF)

| 구분 | User/Item-based CF | Matrix Factorization |
|---|---|---|
| 사용 데이터 | 기존 interaction 벡터 그대로 | interaction으로부터 새로운 벡터를 학습 |
| 적합도 계산 | 존재하는 벡터 간 cosine similarity | 학습된 latent vector의 dot product |
| 학습 단위 | 사용자/아이템 개별 비교 | 전체 데이터를 이용한 global 학습 |

`R ≈ PQᵀ, r̂_ui = p_uᵀq_i`

**Latent vector**($p_u, q_i$)는 RPG 선호도처럼 사람이 정의한 feature가 아니라, interaction을 잘 설명하도록 모델이 스스로 생성한 representation이다. 이 때문에 MF는 Item-based CF보다 추천 이유를 설명하기 어렵다.

### Funk SVD와 SGD

Funk SVD는 MF라는 큰 개념을 실제로 학습($P, Q$ 추정)하는 대표적 방법이다. 예측 오차 $e_{ui}=r_{ui}-p_u^Tq_i$를 **SGD(Stochastic Gradient Descent)**로 줄여나가며 $P, Q$를 반복 업데이트한다. 즉 SGD 자체가 추천 알고리즘이 아니라 **Funk SVD의 파라미터를 찾아가는 최적화 방법**이라는 점을 정리했다. 현재 프로젝트 수준에서는 "MF 적용 ≈ Funk SVD 방식의 MF 적용"으로 이해해도 무리가 없으며, 이후 BPR·ALS 같은 다른 MF 학습 방식과는 구분된다.

### 추천 모델을 보는 4가지 질문 (일반화된 프레임)

1. User를 어떻게 표현하는가?
2. Item을 어떻게 표현하는가?
3. User-Item 적합도를 어떻게 계산하는가?
4. 어떤 loss로 학습하는가?

MF는 `학습된 latent vector → dot product → prediction error + SGD`, Two-Tower는 `Neural Network embedding → similarity → 별도 loss` 구조로 대응된다. 이후 Two-Tower나 Deep Recommendation으로 넘어가도 **User Representation → Item Representation → Matching Score → Ranking**이라는 동일한 관점으로 볼 수 있다는 연결을 이해했다.

### Ranking 문제에서의 Funk SVD 활용

현재 프로젝트는 rating prediction이 아니라 **Top-K Ranking 문제**다. Funk SVD의 예측 점수 $\hat r_{ui}$를 정확한 평점이 아닌 **ranking score**로 사용하기로 했다.

```
Funk SVD 학습 → prediction score → 이미 interaction한 게임 제외
→ 내림차순 정렬 → Top-K → Precision/Recall/HR/NDCG
```

다만 Funk SVD의 loss는 "좋아하는 게임 score > 싫어하는 게임 score"라는 순위 목적을 직접 최적화하지 않는다는 한계도 확인했다 (추후 BPR 탐색 여지로 남김).

---

## 3. 문제 해결 과정 — Train/Test 구조 재설계와 구현

### 구조 재검토

기존 evaluation은 별도 Train/Test 파일 없이 **평가 시점에 사용자별로 70:30 split**하는 구조였다. 이는 Memory-based CF에서는 문제 없었지만, 모든 유저의 interaction을 동시에 사용해 학습하는 **global model인 Funk SVD에는 맞지 않는 구조**임을 확인했다. 사용자 한 명 평가할 때마다 SVD를 재학습하는 것은 잘못된 구조라는 점을 직접 지적하고, "전체 Train으로 한 번 학습 → 각 유저 추천/평가" 구조로 방향을 정리했다.

### 최종 Train/Test 설계

처음에는 기존과 동일하게 약 400명의 sampled user만 70:30으로 나누려 했으나, Train이 8,403건밖에 나오지 않는 것을 보고 "학습 프로젝트라면 전체 데이터를 활용하는 게 낫지 않나"라는 문제를 제기해 구조를 변경했다.

- 평가 대상 사용자(interaction 10~78개, **666,781명**, interaction 12,564,053건): 70% Train / 30% Test
- 그 외 사용자: 100% Train
- 결과: 전체 41,154,794건 = Train 37,113,471 + Test 4,041,323 (원본과 정확히 일치, 누락 없음 확인)

재계산 비용을 없애기 위해 `data/split/mf_train.parquet`, `mf_test.parquet`로 저장해 이후 파라미터 튜닝, BPR, Two-Tower 등에서 재사용 가능하도록 했다.

### 속도 최적화

666,781명에 대해 `train_test_split()`을 매번 호출하면 66만 번 실행되는 문제가 있었다. `random_state=42` 고정 시 interaction 개수가 같은 유저는 동일한 **position pattern**으로 나뉜다는 점을 이용해, interaction 개수별로 1회씩(최대 69회)만 계산하고 재사용하도록 최적화했다. 병목이 70:30 계산 자체가 아니라 Python에서 함수를 66만 번 호출하는 반복 구조라는 점을 이해했고, 최적화 후 전체 split이 약 5초로 단축됐다.

> 대용량 데이터는 데이터 행 수보다 Python loop를 얼마나 줄이고 Pandas/NumPy 일괄 연산을 활용하는지가 성능에 큰 영향을 준다는 점을 실제 프로젝트에서 경험했다.

### 코드 구조

```text
preprocessing.py   → 원본 데이터 로딩 / 기존 전처리
data_split.py       → MF Train/Test 생성 / 저장 / 로드
models/funk_svd.py  → Funk SVD 학습 / prediction / recommendation
evaluation.py        → Precision@K / Recall@K / Hit Rate@K / NDCG@K
```

### Surprise Funk SVD 구현

```python
from surprise import Dataset, Reader, SVD

train_df["is_recommended"] = train_df["is_recommended"].astype(int)  # False→0, True→1
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(train_df[["user_id", "app_id", "is_recommended"]], reader)
trainset = data.build_full_trainset()

model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
model.fit(trainset)
```

| 파라미터 | 의미 |
|---|---|
| `n_factors` | latent vector 차원 수 |
| `n_epochs` | 전체 데이터 반복 학습 횟수 |
| `lr_all` | SGD learning rate |
| `reg_all` | latent factor 과대화를 막는 regularization |
| `model.predict(user_id, app_id).est` | 예측 점수, 이후 Top-K ranking에 사용 |

### 실행 문제와 검증

`models/hybrid.py`를 직접 실행하며 `ModuleNotFoundError: No module named 'data_split'` 오류 발생 — `models/` 내부 파일을 직접 실행할 때 import 기준 경로가 달라진 것이 원인이었다. `python -m models.hybrid`처럼 project root 기준으로 module을 실행하는 방식으로 해결 방향을 확인했다.

최종적으로 Train(37,113,471행)/Test(4,041,323행) 로드 → Surprise Dataset 생성 → `fit()` 완료까지 확인했고, 10만 건 샘플(유저 98,022 / 아이템 9,360, `is_recommended` True 85,868 / False 14,132)로 동작을 검증했다. 즉 **Data → Split → 저장 → 로드 → Surprise Dataset → Funk SVD → fit** 전체 연결이 성공했다.

---

## 4. 내가 직접 판단한 부분

- **MF vs CF의 근본 차이**: 존재하는 벡터의 유사도 계산 vs 벡터 자체의 학습이라는 차이를 스스로 정리
- **Funk SVD ≠ 별개 모델**: MF라는 상위 개념과 학습 방법(Funk SVD)의 관계를 구분
- **직접 구현 여부**: 개념 이해가 끝난 상태에서, 프로젝트 목적상 밑바닥 구현보다 `Surprise.SVD` 활용을 선택
- **"한 명씩 학습하는 게 이상하지 않나?"**: 기존 사용자별 70:30 구조를 MF에 그대로 적용하면 안 된다는 것을 지적 → global model이라는 점을 코드 구조 문제로 연결
- **"400명 샘플만으로는 부족하지 않나?"**: Train이 8천 건 수준밖에 안 나오는 것을 보고 전체 사용자 데이터를 활용하는 방향으로 구조 자체를 변경
- **Split을 저장해 재사용하자는 제안**과 **66만 번 split이 정말 필요한가라는 속도 문제 제기** → position 재사용 최적화로 이어짐
- **GPT 답변의 객관성 요청**: GPT가 몇 차례 기존 구조(`train.csv/test.csv` 존재 가정, 400명 샘플 유지 등)를 잘못 전제하고 흔들리는 것을 확인하고, "내 말에 맞춰 답을 바꾸지 말고 프로젝트 구조·설계 근거로 답해달라"고 명시적으로 요청

## 5. GPT(Claude)가 주도한 부분

MF/Funk SVD·latent vector·SGD 개념 설명, Ranking 문제와 Funk SVD 연결, Two-Tower 관점으로의 확장 설명, MF용 Train/Test 구조 설계안과 `data_split.py` 작성, Parquet 저장/로드 구조, 대규모 split 최적화 아이디어, Surprise `SVD` 모델 코드 및 `train_funk_svd()`/`recommend_funk_svd()` 구조, import 오류 원인 설명 등 — 오늘은 코드를 백지에서 직접 짜기보다 **GPT 코드를 적극적으로 가져와 프로젝트에 연결하는 방식**으로 진행했음을 그대로 남긴다.

---

## 6. 오늘의 회고

오늘은 "이해 100% → 직접 구현" 방식을 택하지 않고, **큰 원리 이해 → 완성 코드 확보 → 프로젝트 연결 → 실행 → 오류 수정 → 필요한 부분만 질문**하는 방식으로 진행했다. 7월부터 이어온 동일 데이터셋에 대한 피로, 프로젝트 재진입 비용, User/Item CF → MF로 넘어가며 추상도가 올라간 점, 구현보다 설계·평가 구조 자체를 손봐야 하는 중반부 작업의 특성이 겹치며 하기 싫다는 느낌이 강했던 날이었다. 그럼에도 완벽하게 이해하고 넘어가기보다 **시스템을 한 단계 앞으로 이동시키는 것**을 우선순위로 두고 마무리했다.

기존에는 "비슷한 사용자/게임을 찾는다"는 관점이 중심이었다면, 오늘부터는 **User/Item을 어떤 representation으로 표현하고, matching score를 어떻게 학습하며, 이를 어떻게 ranking으로 연결할 것인가**라는 Model-based Recommendation의 관점으로 넘어오기 시작했다. 동시에 모델의 학습 방식이 바뀌면 데이터 파이프라인과 평가 구조도 함께 바뀌어야 한다는 것을 실제 프로젝트에서 체감했다.

> **오늘은 Funk SVD를 완전히 직접 구현하거나 깊게 파고든 날이라기보다, MF가 기존 CF와 어떻게 다른지 이해한 뒤 GPT의 구현 도움을 적극 활용하여 4천만 건 규모의 데이터 split부터 Funk SVD 학습까지, Model-based 단계가 돌아갈 수 있는 기반을 만든 날이었다.**

---

## 7. 현재 프로젝트 상태

```
[완료] Matrix Factorization 기본 개념 / Latent Vector / SGD
[완료] Funk SVD와 MF 관계 정리, Ranking 문제와 Funk SVD 연결
[완료] MF Train/Test 설계 및 전체 interaction split
[완료] Parquet 저장 / 로드
[완료] Surprise SVD 구현 및 샘플 데이터 fit
[미완료] 학습 데이터 규모 결정
[미완료] 사용자별 Candidate Generation / Top-K Recommendation
[미완료] Precision@10 / Recall@10 / Hit Rate@10 / NDCG@10
[미완료] 정성평가 및 기존 모델과 비교
```

## 8. 다음 시간 시작점

1. 10만 건 샘플 결과가 MF 학습 데이터로 적절한지 확인 → 학습 데이터 규모/샘플링 방식 결정
2. Funk SVD 실제 학습
3. 유저별 Candidate Item 생성 → Train에서 본 게임 제외 → `model.predict()`로 score 계산
4. Top-10 Ranking 생성 → Test 정답과 비교
5. 기존 Precision/Recall/HR/NDCG 평가 파이프라인 연결

(전체 3,700만 건 학습은 파이프라인 완성 후 진행 여부 결정)