# Day13 회고 — Item-Based CF 구조 이해와 추천시스템에 대한 확장된 문제의식

## 1. 오늘 한 일 요약

- Item-Based Collaborative Filtering 구현을 시작했으나, 오늘의 핵심은 **코드량보다 User-Based CF와 Item-Based CF가 구조적으로 왜 다른지를 이해한 것**이었음
- 기존 `userbase.py`를 복사해 `itembase.py`의 뼈대로 삼고, 외부 인터페이스(`recommend(app_id_list, top_n=10, exclude_user_idx=None)`)는 유지한 채 내부 로직만 Item-Based 관점으로 재해석
- `query_vec`, `cosine_similarity()`, `.getrow(0)`이 각각 정확히 무엇을 계산/의미하는지 재점검
- `interaction_matrix`를 transpose하여 `Item × User` 구조(`item_matrix`)로 사용하는 이유를 정리
- Self-similarity 제거와 Train item(이미 플레이한 게임) 제거가 서로 다른 과정이라는 걸 Item-Based 관점에서도 재확인
- 여러 source item의 similarity를 하나의 candidate score로 합치는 aggregation 문제를 고민하다가, representation·embedding·LLM 역할 분리까지 사고를 확장
- 행별 Top-K 코드를 이해하는 단계까지 진행하고 오늘 구현을 마무리 (aggregation 구현은 다음으로 이월)
- 오늘의 문제의식이 확장되어 별도의 사업 기획안 문서를 작성함 (본 회고에서는 기술적 맥락만 기록)

## 2. 구현 및 개념 이해 과정 (시간순 — 가설과 깨달음)

**1) 기존 User-Based 코드를 기반으로 Item-Based 작성 시작**
`evaluation.py`와 main pipeline을 그대로 재사용하기 위해, 외부 인터페이스는 유지하고 내부 추천 로직만 User-Based → Item-Based로 바꾸는 방향을 설정함.

**2) User-Based의 `query_vec`을 다시 이해**
`col_idx`(Train 게임들의 column index)로 만든 `query_vec`을 처음엔 "게임을 나타내는 벡터"로 혼동했으나, 다시 보니 `[1, 1, 0, 0]`은 "이 User가 어떤 게임들과 interaction했는가"를 나타내는 **User vector**라는 걸 확인함.

**3) `cosine_similarity()`가 실제로 비교하는 대상 재확인**
`query_vec(1×n_items)`와 `interaction_matrix(n_users×n_items)`를 비교하면 User-User similarity가 나온다는 걸 재확인. 핵심 원리: **어떤 similarity가 계산되는지는 `.getrow()`가 아니라 matrix의 각 행이 무엇을 의미하는지가 결정한다.**

**4) `.getrow(0)`에 대한 오해 수정**
`.getrow(0)`/`.getrow(1)`이 User-Based/Item-Based를 구분하는 스위치처럼 느껴졌으나, 이는 단순히 "유일한 similarity 결과 행을 꺼내는 것"일 뿐이고, 알고리즘 종류는 vector orientation과 matrix의 행 의미로 결정된다는 걸 정리함.

**5) User×Item → Item×User 전환**
Item-Based에서는 게임끼리 비교해야 하므로 `interaction_matrix.T`로 전환. 이제 한 행이 "이 게임에 각 User가 어떻게 interaction했는가"를 나타내는 Item vector가 됨.

**6) Item-Based에서 별도 query vector가 불필요한 이유**
User-Based는 `app_id_list`로 query User를 새로 재구성해야 했지만, Item-Based에서는 `app_id_list`의 게임들이 이미 `item_matrix`의 실제 행으로 존재하므로 `source_items = self.item_matrix[col_idx]`로 바로 가져오면 됨. "기존 코드와 일관성을 위해 Item-Based도 query를 만들어야 한다"는 처음 생각을, "외부 인터페이스만 일관되면 되고 내부 구조까지 억지로 맞출 필요는 없다"로 수정함.

**7) Self-similarity 제거 ≠ Seen-item exclusion**
Item similarity matrix에서 자기 자신(`PUBG↔PUBG`)을 0으로 만드는 것과, 이미 플레이한 게임(`PUBG`, `CS2`) 전체를 최종 후보에서 제거하는 것은 별개의 과정이라는 걸 확인함. `PUBG↔CS2`처럼 서로 다른 게임 간 유사도가 높다면 CS2는 여전히 PUBG의 추천 후보가 될 수 있으므로, 두 과정 모두 필요함 (`predicted_scores[col_idx] = -np.inf`는 Item-Based에서도 그대로 필요).

**8) 기존 User-Based 코드 재검토 중 발견한 사실**
Query User 벡터는 `np.ones(len(col_idx))`로 만들어, target user의 interaction 여부(`+1`)만 표현하고, neighbor의 interaction은 원본 `+1/-1`을 그대로 사용한다는 점을 재확인. 즉 target 쪽은 "interaction 여부", neighbor 쪽은 "추천/비추천 정보"를 사용하는 비대칭 구조라는 걸 정리 (추후 재검토 대상으로 기록).

**9) 행별 Top-K 필요성 이해**
User-Based는 similarity 결과가 한 행이라 전체에서 Top-K를 뽑으면 됐지만, Item-Based는 source item마다 한 행씩(`PUBG→전체`, `CS2→전체`, `L4D2→전체`) 결과가 나오므로, 전체 `sims.data`에서 Top-30을 뽑으면 특정 source item이 결과를 독점할 수 있음. 따라서 source item별로 각각 Top-30을 뽑는 **행별 Top-K** 구조가 필요하다는 걸 확인하고, 이를 위한 반복문 코드를 오늘 복사해둔 상태로 마무리.

**10) Aggregation에 대한 고민 → 사고 확장**
"서로 다른 source item의 similarity를 왜 단순히 더해서 하나의 점수로 만드는가"라는 의문에서 출발해, representation과 similarity의 관계, embedding·딥러닝의 역할, LLM과 추천시스템의 역할 분리, multi-interest·exploration/exploitation 문제까지 사고를 확장함 (자세한 내용은 4번 참고).

## 3. Item-Based CF 핵심 구조 정리

### 3-1. User-Based vs Item-Based — 구조적 차이

| 구분 | User-Based CF | Item-Based CF |
|---|---|---|
| Matrix 방향 | User × Item | Item × User (transpose) |
| Query 필요 여부 | 필요 (User를 새로 재구성) | 불필요 (Item이 이미 행으로 존재) |
| Similarity 대상 | User ↔ User | Item ↔ Item |
| Similarity 결과 shape | 1 × n_users (한 행) | Train item 개수 × n_items (여러 행) |
| Top-K 방식 | 전체에서 한 번 | Source item마다 행별로 |
| Similarity → Output 거리 | User 유사도를 구한 뒤, 그 이웃의 interaction까지 다시 연결해야 Item(output)이 나옴 | Similarity 자체가 이미 Item(output) 공간에 존재 |

### 3-2. Self-similarity 제거와 Seen-item exclusion

| 과정 | 대상 | 목적 |
|---|---|---|
| Self-similarity 제거 | `PUBG↔PUBG` 같은 자기 자신 | 자명한 값(trivial)이 Top-K를 차지하는 것 방지 |
| Seen-item exclusion | Train에서 이미 interaction한 전체 아이템 | 이미 한 게임을 다시 추천하는 것 방지 |

두 과정은 서로 대체할 수 없고, 둘 다 필요함.

## 4. 오늘 정리한 핵심 개념

**① 왜 여러 similarity를 합산(sum)하는가**
추천시스템은 최종적으로 1차원 ranking(1위, 2위, 3위 …)을 만들어야 하므로, 여러 source item과 후보 item 간의 관계를 결국 하나의 candidate score로 축약해야 함. `sum`은 수학적으로 유일한 정답이 아니라 **여러 관계를 하나의 ranking score로 만들기 위해 선택한 가장 기본적인 aggregation 규칙**일 뿐이며, 평균·최대값·Top-N 평균·가중치 부여·ML 기반 scoring 등 다른 방식도 가능함. 중요한 질문은 "합해야 하는가"가 아니라 "어떤 aggregation이 목적에 가장 적합한가"임.

**② Representation과 Similarity는 별개의 문제**
Cosine similarity 결과가 scalar 하나(예: 0.73)라고 해서 비교 자체가 "2차원적"인 게 아니라, User가 100만 명이면 Item 벡터는 100만 차원이고 similarity는 그 고차원 공간 전체에서 계산되는 것. Representation(객체를 어떤 벡터로 나타낼 것인가)과 Similarity(그 벡터들을 어떤 방식으로 하나의 수치로 비교할 것인가)는 분리해서 생각해야 하며, 추천 성능에는 similarity 공식만큼 representation 자체도 중요함.

**③ Embedding과 딥러닝의 역할**
PCA는 데이터의 분산 구조를 보존하며 차원을 축소하는 반면, 추천모델의 embedding은 **추천/예측을 잘할 수 있는 representation 자체를 데이터로부터 학습**한다는 차이가 있음. 즉 딥러닝은 similarity 수학을 없애는 게 아니라, similarity를 계산하기 좋은 공간 자체를 학습하는 역할을 함. 이는 이미지 embedding, LLM/텍스트 embedding에도 동일하게 적용되는 원리.

**④ 두 번의 정보 압축**
- 1차 압축: Item A와 Item B 사이의 수많은 User interaction 관계 → cosine similarity 하나의 값(예: 0.73)
- 2차 압축: 여러 source item 관점의 similarity(`PUBG→Apex=0.8`, `CS2→Apex=0.7`, `L4D2→Apex=0.3`) → aggregation → 하나의 candidate score

두 과정 모두 정보 손실이 발생할 수 있으며, 중요한 질문은 "압축을 피할 수 있는가"가 아니라 "최종 목적에 필요한 정보를 최대한 유지하며 어떻게 압축할 것인가"임.

**⑤ Multi-interest와 Exploration/Exploitation**
한 사용자가 서로 다른 관심 영역(경쟁 FPS, 협동/공포, 캐주얼/힐링)을 동시에 가질 수 있는데, 단순 추천모델은 이를 하나의 preference score로 뭉갤 수 있음. 여기서 Relevance/Diversity/Novelty/Serendipity/Exploration/Exploitation 같은 개념들이 왜 필요한지 연결됨 — "가장 좋아할 것 같은 것만 계속 보여주는" 방식은 장기적으로 추천이 특정 영역에 고립될 위험이 있음.

**⑥ 행동 예측과 의도 이해의 차이**
관측된 행동(예: 특정 카테고리 반복 클릭)이 곧 장기적 핵심 취향이라고 단정할 수 없음 — 장기 취향, 현재 세션의 목적, 일시적 관심, 단순 호기심 등 여러 원인이 있을 수 있음. 즉 `관측된 행동 ≠ 현재 의도 ≠ 장기적 취향`이며, 행동 pattern을 정확히 예측하는 것과 그 행동이 왜 발생했는지 해석하는 것은 다른 문제라는 걸 정리함.

**⑦ LLM과 추천시스템의 역할 분리**
Cosine similarity, sparse matrix 연산, embedding retrieval 같은 대규모 계산은 기존 수학/ML/DL 시스템이 더 효율적일 수 있음. LLM은 이런 계산을 대체하기보다, 계산 결과를 해석하거나(사용자의 현재 상태, 새로운 관심 영역 탐색 여부, 기존 취향 유지 정도) 추천 전략을 조율하는 **상위 역할**에 배치하는 것이 더 합리적일 수 있다는 아이디어로 정리함. 구체적 시스템 설계까지는 발전시키지 않고 아이디어 수준으로 남김.

## 5. 내가 직접 내린 분석·판단

- User-Based와 Item-Based의 차이는 "similarity 대상 하나를 바꾸는 것" 이상이며, User-Based는 User similarity를 구한 뒤 이웃의 interaction까지 다시 연결해야 Item(output)이 나오는 반면, Item-Based는 similarity 결과 자체가 이미 output 공간(Item)에 존재한다는 구조적 차이를 스스로 정리함.
- 추천시스템의 input/output 관점(User→Item)이 알고리즘 이해에 핵심적이라는 걸 확인함.
- 여러 similarity를 단순 합산하는 것이 처음엔 직관적으로 불편했으나, 최종적으로 ranking이 필요하므로 정보 축약 자체는 불가피하며 "합을 써야 하는가"가 아니라 "어떤 aggregation이 목적에 맞는가"가 진짜 질문이라고 판단함.
- 추천시스템이 하나의 취향만 반복적으로 강화해서는 안 될 수 있고, relevance와 exploration 사이의 균형이 필요하다고 판단함.
- 행동 예측의 정확도와 의도 이해는 서로 다른 문제이며, 이는 향후 고도화된 개인화 시스템에서 중요해질 수 있다고 판단함.

## 6. GPT 피드백으로 수정한 부분

- **`.getrow(0)`이 User-Based/Item-Based를 결정한다는 오해**: 알고리즘 종류는 row 선택이 아니라 matrix의 각 행이 User인지 Item인지로 결정된다는 걸 정정함.
- **User-Based의 query vector가 "게임 벡터"라는 오해**: 값의 위치가 게임이라 game vector처럼 보였으나, 실제로는 "게임을 feature로 사용해 한 User를 표현한 벡터"라는 걸 정정함.
- **Item-Based에서도 query vector를 만드는 게 일관성 있다는 생각**: Item은 이미 `item_matrix`의 실제 행으로 존재하므로 새 query를 만드는 건 중복이며, "인터페이스는 동일하게 유지하되 내부 구조는 각 모델에 맞게 설계해야 한다"로 수정함.
- **Self-similarity 제거로 Train item exclusion까지 해결된다는 생각**: `A↔A` 제거와 "A를 source로 했을 때 이미 플레이한 B가 후보가 되는 것"을 막는 건 별개의 문제이며, 두 exclusion이 모두 필요하다고 정정함.
- **Cosine similarity가 "2차원적"이라는 인식**: 결과가 scalar 하나라는 이유로 단순한 비교처럼 느껴졌으나, 실제 벡터가 수십만~수백만 차원이면 similarity 역시 그 고차원 공간 전체에서 계산된다는 걸 정정함.
- **LLM이 추천시스템의 모든 계산을 담당해야 한다는 방향성**: 대규모 similarity·sparse matrix·embedding retrieval은 기존 ML/DL 시스템이 더 효율적일 수 있으며, LLM은 계산 결과를 해석하거나 전략을 조율하는 상위 역할에 배치하는 게 더 합리적일 수 있다는 방향으로 수정함.

## 7. 오늘 확정한 구현 방향

- `evaluation.py`와 main pipeline은 최대한 유지 — User-Based/Item-Based 비교 조건을 동일하게 유지하기 위함
- `itembase.py` 내부만 우선 수정, pipeline 전체는 현재 단계에서 변경하지 않음
- 기존 `build_interaction_matrix()`는 그대로 유지하고, `interaction_matrix.T`로 Item 관점을 만들어 사용
- 전체 Item×Item dense similarity matrix는 생성하지 않음 (메모리 문제) — Train source item과 전체 item 간 sparse similarity만 계산하고 각 source에서 Top-K만 사용
- K는 User-Based와 동일하게 30으로 우선 설정 (비교 조건 통일)
- Aggregation은 우선 가장 기본적인 similarity 합산(sum) baseline으로 구현 후, 실제 결과를 보고 개선 여부 판단

## 8. 오늘 보류한 문제

**Item-Based scoring 관련**
- 각 source item별 Top-30 similarity를 candidate별로 어떻게 효율적으로 합칠 것인가
- 단순 합(sum)과 평균(mean)의 차이가 실제로 어떤 영향을 주는가
- 음수 similarity를 가진 item을 후보에 포함할 것인가
- source item의 +1/-1 preference를 어떻게 반영할 것인가
- Train item 개수가 많은 유저에게 sum 방식이 과도하게 높은 score를 만들지는 않는가

**추천 품질 관련**
- Item-Based에서 `n_recommended < 10` 현상이 실제로 감소하는가
- User-Based보다 candidate coverage가 증가하는가
- accuracy가 오르더라도 diversity가 줄어들 가능성은 없는가
- 여러 관심사가 하나의 score로 뭉개지는 문제를 어떻게 평가할 수 있는가

이 문제들은 기본 Item-Based 구현을 마친 뒤 실제 데이터로 확인하고 판단하기로 함.

## 9. 성찰 및 느낀 점

오늘 가장 큰 변화는 코드를 "User-Based를 Item-Based 문법으로 바꾼다"는 관점이 아니라, **코드 한 줄 한 줄이 어떤 데이터 구조와 추천 논리를 표현하는지 설명하면서 바꾼다**는 방식으로 접근했다는 점이다. User×Item matrix의 행·열 의미 → transpose → vector representation → cosine similarity → Top-K → aggregation → ranking → information compression → multi-interest → exploration/exploitation까지, 처음엔 독립적으로 보이던 개념들이 하나의 추천 파이프라인 안에서 서로 연결되기 시작했다. 오늘의 학습은 구현량보다 **모델 내부에서 데이터가 어떻게 이동하는지 머릿속으로 추적할 수 있게 된 것** 자체에 더 큰 가치가 있었다.

## 10. 다음에 해야 할 것 (To-Do)

**바로 다음 시작 지점**: 새 코드를 추가하기보다, 오늘 복사해둔 행별 Top-K 코드(`for row_idx in range(sims.shape[0])`, `sims.getrow(row_idx)`, `k = min(self.k, len(row.data))`, `argpartition`, `row.data[top_k_pos]`, `row.indices[top_k_pos]`)를 한 줄씩 직접 설명할 수 있을 정도로 이해하는 것부터 시작한다.

- [ ] 행별 Top-K 코드 한 줄씩 이해
- [ ] Candidate aggregation 구현 — 동일 candidate로 들어오는 여러 source item의 similarity를 합쳐 최종 score 생성
- [ ] Train item exclusion 적용 (`predicted_scores[col_idx] = -np.inf`)
- [ ] Positive score filtering 적용 (`score > 0`만 후보로, User-Based와 비교 조건 통일)
- [ ] 기존 `recommend()`와 동일한 출력 형식(`pd.DataFrame({"app_id": recommended_app_ids})`) 유지
- [ ] 기존 evaluation pipeline(Macro/Micro/Hit Rate/NDCG) 그대로 재사용해 평가
- [ ] User-Based에서 발견된 `n_recommended` 부족 현상이 Item-Based에서 실제로 완화되는지 확인

**다음 세션 시작 문장** (생각 복구용):
> "현재 `sims`는 유저 Train item 각각과 전체 Item 사이의 similarity 행렬이다. 이제 각 행에서 Top-30을 꺼내는 코드를 이해하고, 그 Top-30들이 가리키는 동일 candidate들의 similarity를 합쳐 한 User의 추천 score를 만드는 단계로 넘어간다."

## 11. 오늘의 한 문장 회고

오늘은 Item-Based CF 코드를 많이 완성한 날이 아니라, User와 Item이 각각 어떤 공간에서 벡터가 되고 similarity가 어떻게 최종 추천 점수로 변환되는지를 이해하면서, 단순한 '유사도 합' 뒤에 존재하는 정보 압축·다중 취향·탐색과 개인화의 구조적 문제까지 처음으로 연결해 본 날이었다.