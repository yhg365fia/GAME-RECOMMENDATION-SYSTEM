# Day14 회고 — Item-Based CF 구현 완료와 정량평가

## 1. 오늘 한 일 요약

- 전날 이해 단계까지 진행했던 **Item-Based Collaborative Filtering의 `recommend()` 로직을 완성**함
- `sims`의 행/열 구조와 sparse matrix의 `row.data`, `row.indices`, `top_k_pos`를 다시 확인하며 **행별 Top-K 로직을 직접 이해**함
- 각 Train source item별 Top-K candidate를 추출하고, 동일 candidate로 들어오는 similarity를 합산하는 **candidate aggregation** 구현
- Train item exclusion, positive similarity filtering, Top-N ranking까지 적용해 Item-Based 추천 파이프라인 완성
- User-Based와 동일한 `recommend(app_id_list, top_n=10, exclude_user_idx=None)` 인터페이스를 유지하여 기존 evaluation pipeline을 그대로 재사용
- 그룹별 10명씩 총 40명으로 정상 실행 여부를 확인한 뒤, 그룹별 100명씩 총 400명 정량평가 수행
- User-Based와 Item-Based의 Macro/Micro Precision·Recall·HitRate·NDCG 비교
- User-Based에서 발견한 `n_recommended < 10` 문제가 Item-Based에서 실제로 완화되는지 확인
- `n_games ↔ precision` 상관관계와 그룹별 성능 차이 분석
- 단건 추천에서 AppID뿐 아니라 게임 이름까지 확인할 수 있도록 정성평가용 출력 준비
- PUBG, Terraria 입력으로 실제 추천 결과 출력까지 확인했으나, **본격적인 정성평가와 Content-Based 쏠림 비교는 다음 세션으로 이월**

## 2. 구현 및 개념 이해 과정 (시간순 — 가설과 깨달음)

**1) Item-Based의 similarity matrix 구조부터 다시 고정**

전날 만들어둔 구조는 다음과 같음.

```python
source_items = self.item_matrix[col_idx]

sims = cosine_similarity(
    source_items,
    self.item_matrix,
    dense_output=False
)
```

```text
self.item_matrix = 전체 Item × 전체 User
source_items     = Train Item × 전체 User
sims             = Train Item × 전체 Item
```

즉 `sims`에서 **행은 사용자의 Train source item**, **열은 전체 candidate item**임.

User-Based에서는 target user 하나에 대해 Top-K user를 한 번 뽑았지만, Item-Based에서는 Train item이 여러 개이므로 **각 source item 행마다 Top-K item을 따로 추출해야 함**을 재확인함.

**2) `shape`, `row.data`, `row.indices`, `top_k_pos`의 역할 정리**

오늘 가장 많이 혼동한 부분은 각 index가 어느 공간을 가리키는지였음.

```text
sims.shape[0] = Train source item 개수
sims.shape[1] = 전체 item 개수
```

Sparse matrix에서:

```python
row = sims.getrow(row_idx)
```

를 수행했을 때,

```text
row.data
= 해당 행에 저장된 non-zero similarity 값

row.indices
= 각 similarity가 가리키는 실제 전체 item index

top_k_pos
= row.data 내부에서 Top-K similarity가 위치한 자리
```

임을 정리함.

따라서 `top_k_pos` 자체는 실제 game index가 아니며,

```python
row.indices[top_k_pos]
```

를 해야 실제 candidate item index를 얻을 수 있음.

**3) Self-similarity 제거**

각 Train item은 자기 자신과 similarity 1을 가지므로 다음과 같이 제거함.

```python
for row_idx, item_idx in enumerate(col_idx):
    sims[row_idx, item_idx] = 0

sims.eliminate_zeros()
```

여기서:

```text
row_idx  = source_items 안에서 몇 번째 Train item인가
item_idx = 전체 item matrix에서 해당 게임의 실제 index
```

임을 다시 확인함.

또한 **Self-similarity 제거와 Train item exclusion은 서로 다른 과정**이라는 점도 유지됨. 자기 자신과의 similarity를 0으로 만들어도, 다른 Train item이 candidate로 다시 등장할 수 있으므로 최종 단계의 `predicted_scores[col_idx] = -np.inf`가 별도로 필요함.

**4) Positive Top-K 추출 구현**

최종 Top-K 구조:

```python
for row_idx in range(sims.shape[0]):
    row = sims.getrow(row_idx)

    positive_mask = row.data > 0
    positive_data = row.data[positive_mask]
    positive_indices = row.indices[positive_mask]

    if len(positive_data) == 0:
        continue

    k = min(self.k, len(positive_data))

    top_k_pos = np.argpartition(
        positive_data,
        -k
    )[-k:]

    topk_sims_list.append(positive_data[top_k_pos])
    topk_indices_list.append(positive_indices[top_k_pos])
```

User-Based와 비교 조건을 맞추기 위해 **positive similarity만 후보로 사용**함.

**5) Candidate aggregation 구현**

User-Based에서 사용하던 weighted prediction 공식을 그대로 가져오지 않고, Item-Based에서는 source item과 candidate item 사이의 similarity 자체를 추천 근거로 사용함.

```python
predicted_scores = np.zeros(self.item_matrix.shape[0])

for topk_indices, topk_sims in zip(
    topk_indices_list,
    topk_sims_list
):
    predicted_scores[topk_indices] += topk_sims
```

예를 들어:

```text
A → X : 0.8
B → X : 0.6
C → X : 0.3
```

이면:

```text
X score = 1.7
```

이 됨.

즉 **여러 source item에서 동시에 유사하다고 판단된 candidate일수록 높은 최종 score를 받는 구조**임.

**6) Train item exclusion 및 Top-N ranking까지 연결**

이미 플레이한 Train item은:

```python
predicted_scores[col_idx] = -np.inf
```

로 제외하고,

```python
valid_idx = np.where(predicted_scores > 0)[0]
```

로 최종적으로 positive score candidate만 남긴 뒤 Top-N을 추출함.

최종 출력은 evaluation pipeline과 동일하게:

```python
pd.DataFrame({"app_id": recommended_app_ids})
```

형식을 유지함.

**7) `exclude_user_idx`는 Item-Based에서 사용하지 않지만 인터페이스는 유지**

Item-Based에서는 User self-exclusion이 필요하지 않지만, 기존 `run_evaluation()`이 동일한 인자를 넘기도록 설계되어 있으므로:

```python
def recommend(self, app_id_list, top_n=10, exclude_user_idx=None):
```

형식을 유지함.

Item-Based 내부에서는 해당 값을 사용하지 않지만, **모델별 evaluation 코드를 따로 수정하지 않기 위한 공통 인터페이스 유지**라는 의미가 있음.

## 3. Item-Based CF 핵심 구조 정리

### 3-1. 최종 추천 흐름

```text
사용자 Train AppID
        ↓
전체 Item index로 변환
        ↓
Train Item의 Item-User 벡터 추출
        ↓
Train Item × 전체 Item cosine similarity
        ↓
각 source item self-similarity 제거
        ↓
Positive similarity만 유지
        ↓
각 source item별 Top-K candidate 추출
        ↓
동일 candidate의 similarity 합산
        ↓
이미 플레이한 Train item 제외
        ↓
Positive score 후보만 유지
        ↓
Top-N ranking
        ↓
DataFrame["app_id"]
```

### 3-2. User-Based vs Item-Based — 오늘 실제 구현에서 확인한 차이

| 구분 | User-Based CF | Item-Based CF |
|---|---|---|
| Matrix 기준 | User × Item | Item × User |
| Source | Target User 1명 | Target User의 Train Item 여러 개 |
| Similarity | User ↔ User | Item ↔ Item |
| Top-K | Target User 기준 한 번 | Source Item마다 각각 |
| Candidate 생성 | Neighbor User의 interaction에서 생성 | 각 source item의 similar item에서 직접 생성 |
| Candidate score | User similarity × interaction의 weighted prediction | 동일 candidate로 들어오는 item similarity 합산 |
| 주요 구조적 문제 | User overlap 부족 시 neighbor/candidate 부족 | 여러 source에서 candidate가 생성되어 coverage가 넓어짐 |

## 4. 정량평가 결과 및 핵심 분석

### 4-1. 40명 소표본 실행 확인

그룹별 10명씩 총 40명을 먼저 평가함.

```text
Precision@10: 0.0800
Recall@10:    0.0938
HitRate@10:   0.5000
NDCG@10:      0.1110

Micro Precision@10: 0.0800
Micro Recall@10:    0.0814
Micro F1@10:        0.0807
```

40명 모두 추천 10개를 전부 채움.

이를 통해 코드가 정상 실행되고, User-Based보다 높은 성능이 나타날 가능성을 확인함.

### 4-2. 400명 본 평가

그룹별 100명씩 총 400명을 평가함.

```text
평가 완료: 400명
스킵: 0명
실행 시간: 1975.3초
```

**Macro Average**

```text
Precision@10: 0.0783
Recall@10:    0.0880
HitRate@10:   0.4800
NDCG@10:      0.1078
```

**Micro Average**

```text
Precision@10: 0.0783
Recall@10:    0.0813
F1@10:        0.0797
```

**전체 평가량**

```text
전체 Hits:       313
전체 추천 개수:  4,000
전체 Test 개수:  3,850
```

### 4-3. User-Based vs Item-Based

| Model | Aggregate | Precision@10 | Recall@10 | F1@10 | HitRate@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| User-Based | Macro | 0.0545 | 0.0427 | - | 0.3011 | 0.0655 |
| User-Based | Micro | 0.0626 | 0.0454 | 0.0526 | - | - |
| **Item-Based** | **Macro** | **0.0783** | **0.0880** | - | **0.4800** | **0.1078** |
| **Item-Based** | **Micro** | **0.0783** | **0.0813** | **0.0797** | - | - |

모든 주요 지표가 User-Based보다 상승함.

특히:

```text
HitRate@10
0.3011 → 0.4800
(+0.1789)
```

으로 상승폭이 가장 크게 나타남.

Macro Recall도:

```text
0.0427 → 0.0880
```

으로 약 두 배 수준으로 증가함.

### 4-4. Review Group별 결과

| Review Group | Precision | Recall | HitRate | NDCG | n_users |
|---|---:|---:|---:|---:|---:|
| 10–15개 | 0.046 | 0.1147 | 0.30 | 0.0999 | 100 |
| 16–25개 | 0.048 | 0.0761 | 0.33 | 0.0771 | 100 |
| 26–45개 | 0.089 | 0.0886 | 0.58 | 0.1025 | 100 |
| 46–78개 | **0.130** | 0.0726 | **0.71** | **0.1516** | 100 |

활동량이 많은 그룹으로 갈수록 **Precision과 HitRate는 뚜렷하게 증가**함.

특히 46–78개 그룹은 HitRate@10이 0.71로, 100명 중 71명에게 Top-10 안에서 최소 하나 이상의 test item을 추천함.

반면 Recall은 활동량에 따라 단조 증가하지 않음.

### 4-5. `n_recommended` 부족 문제 확인

Item-Based에서는 전 그룹이:

```text
mean = 10.0
min  = 10
max  = 10
```

을 기록함.

전체:

```text
n_recommended 평균: 10.00 / 10
10개 전부 추천된 유저 비율: 100%
```

따라서 User-Based에서 발견했던 **candidate 부족 문제는 현재 평가 범위에서 완전히 사라짐**.

`n_games ↔ n_recommended` Pearson correlation이 `nan`으로 나온 것도 모든 유저의 `n_recommended`가 10으로 동일해 분산이 0이기 때문임.

## 5. 내가 직접 내린 분석·판단

- Item-Based가 User-Based보다 Precision·Recall·HitRate·NDCG가 모두 높았으며, **Steam 데이터에서 User-User overlap보다 Item-Item collaborative signal이 더 안정적으로 작동할 가능성**이 높다고 판단함.
- HitRate가 약 0.18 상승한 것을 통해 Item-Based가 **Top-10 안에서 적어도 하나의 relevant item을 찾는 능력**에서 특히 큰 개선을 보였다고 해석함.
- 처음에는 HitRate 상승에 비해 Precision·Recall 상승폭이 상대적으로 작아 아쉬움을 느꼈지만, Precision/Recall은 test relevant item 수와 metric 구조의 영향을 받기 때문에 절대값만으로 과소평가해서는 안 된다고 판단함.
- 모든 평가 유저에게 추천 10개를 생성했다는 점은 단순 정확도 상승보다 중요한 결과라고 판단함. **Item-Based가 User-Based의 candidate coverage 문제를 구조적으로 완화했다는 증거**로 해석함.
- Item-Based에서는 모든 유저의 추천 개수가 10으로 같아지면서 Macro Precision과 Micro Precision이 동일해졌고, User-Based에서 존재했던 추천 개수 편차의 영향이 사라졌다고 판단함.
- `n_games ↔ precision` 상관계수가 `r=0.3387 (p<0.001)`로 상승한 것을 보고, 유저의 Train item 수가 많을수록 더 많은 source item에서 Item-Item similarity 정보를 얻을 수 있어 활동량이 Item-Based 성능에 더 직접적인 영향을 줄 수 있다고 판단함.
- 그룹별 결과에서 활동량이 높은 유저일수록 Precision과 HitRate가 크게 증가한 것을 Item-Based의 정보량 증가와 연결해 해석함.
- 현재 Item-Based의 가장 눈에 띄는 장점은 단순 accuracy보다 **candidate generation과 coverage의 안정성**일 가능성이 있다고 판단함.

## 6. GPT 피드백으로 수정한 부분

- **`n_games ↔ precision`을 Item-Based 정보량만으로 설명한 해석**: Train item이 많아지면 source 정보량이 증가하는 것은 맞지만, 동시에 70:30 split에서 test relevant item 수도 증가해 Precision@10의 가능한 ceiling 자체가 높아짐. 따라서 현재 correlation은 **Train source 정보량 증가 + test size 증가에 따른 metric ceiling 변화**가 함께 작용했을 가능성이 있다고 수정함.
- **"Precision@10 최대값이 0.3"이라는 해석**: Precision@10 자체의 이론적 최대는 1.0이며, interaction 수가 적은 유저에서는 test relevant item이 3개 정도밖에 없어 실제 달성 가능한 최대 Precision이 약 0.3으로 제한될 수 있다는 의미로 정정함.
- **활동량이 증가하면 모든 지표가 좋아진다는 해석**: Precision과 HitRate는 증가하지만 Recall은 `0.115 → 0.076 → 0.089 → 0.073`으로 동일한 증가 패턴을 보이지 않음. 따라서 활동량이 높은 유저는 **Top-10에서 정답을 하나 이상 찾을 가능성과 정답 개수는 증가하지만, 전체 relevant item 중 회수 비율은 증가하지 않는다**고 수정함.
- **HitRate 0.71의 의미**: "정확도가 71%"가 아니라, 해당 그룹 100명 중 71명이 Top-10 안에서 최소 하나의 test item을 추천받았다는 의미로 정정함.
- **Macro와 Micro 중 무엇을 봐야 하는지에 대한 고민**: 사용자 한 명 한 명의 평균 경험을 보는 Macro를 주 지표로, 전체 hit/추천/정답 수를 합산하는 Micro를 보조 지표로 함께 유지하기로 정리함. 모든 유저의 추천 개수가 동일하므로 Precision에서는 Macro와 Micro가 동일하게 나타남.
- **`exclude_user_idx`를 Item-Based에서 제거하려 했던 방향**: Item-Based 내부에서는 사용하지 않지만 기존 evaluation pipeline이 동일 인터페이스를 기대하므로, 인자는 남겨두고 사용하지 않는 방식이 모델 교체 구조상 더 적절하다고 수정함.

## 7. 오늘 확정한 구현·실험 방향

- Item-Based baseline은 현재 구현으로 고정
- `k=30` 유지
- 각 source item마다 행별 Top-K 추출
- Positive similarity만 candidate로 사용
- Candidate aggregation은 **similarity sum** 유지
- Train에서 플레이한 모든 게임을 source item으로 사용
- 현재 단계에서는 `is_recommended=False`를 별도의 negative weight로 반영하지 않음
- Train item은 최종 recommendation에서 제외
- 기존 `recommend()` 출력 형식과 evaluation pipeline 유지
- Macro/Micro/HitRate/NDCG를 모두 기록
- 모델별 baseline을 먼저 완료한 뒤 파라미터·조건·시스템 개선 수행

향후 전체 실험 순서는 다음과 같이 확정함.

```text
각 모델 baseline 구현·평가
        ↓
Steam 데이터에 적합 / 부적합한 모델 비교
        ↓
각 모델이 잘 잡는 핵심 signal 분석
        ↓
각 모델의 구조적 단점 분석
        ↓
Parameter / 조건 / System 조정
        ↓
문제점 보완 실험
        ↓
적합한 모델들의 장점을 결합한 Hybrid
```

따라서 현재 단계에서는 다음을 적용하지 않음.

- `k=30 → 50/100` 조정
- sum → mean 변경
- positive-only source
- negative weighting
- normalization 변경
- 추가 filtering
- Hybrid

이들은 **Model-Based까지 baseline 비교를 끝낸 이후의 개선 단계**에서 진행하기로 함.

## 8. 오늘 보류한 문제

**정성평가 관련**

- PUBG, Terraria 입력으로 실제 추천 결과가 출력되는 것까지는 확인함
- 추천 결과가 실제 취향 관점에서 얼마나 합리적인지 본격적으로 평가하지 않음
- 한 개 게임 입력뿐 아니라 여러 취향 조합을 넣었을 때 어떤 후보가 올라오는지 확인 필요
- 장르가 다른 게임이 추천될 경우 단순 오류인지, 실제 플레이층 overlap을 잡은 collaborative signal인지 분석 필요

**추천 쏠림 관련**

- Content-Based에서 발견한 장르·태그 기반 쏠림이 Item-Based에서도 재현되는지 확인 필요
- 동일 인기 게임이 서로 다른 입력에서도 반복 추천되는지 확인 필요
- Item-Based에서는 단순 feature similarity가 아니라 **공동 interaction이 많은 인기/hub item이 여러 source에서 반복적으로 높은 점수를 받을 가능성**이 있음
- CBF와 Item-Based에서 모두 쏠림이 발견된다면, 현상 자체보다 **쏠림의 원인이 어떻게 다른지** 비교할 필요가 있음

**추후 개선 관련 — Model-Based 이후로 보류**

- `k` 증가/감소 실험
- sum과 mean aggregation 비교
- positive source만 사용하는 조건
- positive/negative preference 가중치
- score normalization
- candidate/ranking system 조정
- Hybrid 구성

## 9. 성찰 및 느낀 점

오늘은 전날 구조적으로 이해해둔 Item-Based CF를 실제 추천 로직과 정량평가까지 연결하면서, **모델의 구조적 차이가 실제 지표와 candidate coverage 차이로 어떻게 드러나는지 확인한 날**이었다.

특히 구현 과정에서 `shape[0]`, `shape[1]`, `row.data`, `row.indices`, `top_k_pos`처럼 비슷해 보이는 index들을 계속 혼동했지만, 이를 단순 Python 문법으로 외우기보다 **"현재 이 값은 어느 행렬 공간에서 무엇을 가리키는가"**를 추적하는 방식으로 정리하면서 코드의 기능적 의미를 이해할 수 있었다.

또한 User-Based에서 발견했던 추천 개수 부족 문제가 Item-Based에서 400명 전원 Top-10 충족으로 사라졌고, 동시에 Precision·Recall·HitRate·NDCG가 모두 상승하면서, "어떤 모델이 더 좋다"는 단순 결론보다 **Steam의 희소한 interaction 구조에서 어떤 종류의 collaborative signal이 더 안정적으로 작동하는가**를 생각하게 됐다.

한편 높은 상관계수나 그룹별 지표를 바로 모델의 장점으로 해석하려 했던 부분에서 test size와 metric ceiling 같은 평가 구조 자체의 영향도 함께 봐야 한다는 점을 확인했다. 모델을 이해하는 것뿐 아니라 **평가 지표가 왜 그렇게 움직였는지를 분리해서 설명하는 것**도 추천시스템 분석의 일부라는 걸 배웠다.

## 10. 다음에 해야 할 것 (To-Do)

**바로 다음 시작 지점**: Item-Based의 정량평가는 완료했으므로, 다음에는 코드 수정이나 파라미터 개선이 아니라 **정성평가와 Content-Based 쏠림 비교**부터 진행한다.

- [ ] 다양한 게임 및 취향 조합으로 Item-Based 정성평가
- [ ] 입력 게임과 추천 게임의 실제 관계 분석
- [ ] 단순 장르 유사성 외에 플레이층 기반 관계를 포착하는지 확인
- [ ] 서로 다른 입력에서도 동일 유명 게임이 반복 추천되는지 확인
- [ ] Item-Based의 popularity / hub-item 쏠림 여부 확인
- [ ] Content-Based에서 발견한 쏠림 현상과 직접 비교
- [ ] CBF와 Item-Based에서 쏠림이 발생한다면 원인이 어떻게 다른지 분석
- [ ] 정량평가 + 정성평가를 종합해 Item-Based의 최종 장점·단점 정리
- [ ] Item-Based baseline 종료 후 다음 baseline 모델 단계로 이동

**다음 세션 시작 문장** (생각 복구용):

> "Item-Based의 정량평가는 완료했고, 400명 모두 Top-10을 채우며 User-Based보다 Precision·Recall·HitRate·NDCG가 상승했다. 이제 실제 게임 조합을 넣어 추천 결과를 정성적으로 평가하고, Content-Based에서 발견한 추천 쏠림이 Item-Based에서도 나타나는지와 그 원인이 무엇인지 비교한다."

## 11. 오늘의 한 문장 회고

오늘은 전날 구조적으로 이해한 Item-Based CF를 실제 추천 로직으로 완성하고, 400명 평가에서 User-Based보다 높은 성능과 100% Top-10 추천 충족률을 확인함으로써 **Steam 데이터에서 Item-Item collaborative signal이 User-User signal보다 더 안정적으로 candidate를 생성할 가능성을 정량적으로 확인한 날**이었다.
