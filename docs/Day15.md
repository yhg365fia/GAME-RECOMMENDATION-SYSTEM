# Day15 회고 — Item-Based CF 정성평가

## 1. 오늘 한 일 요약

- Day14에서 완료한 Item-Based CF 정량평가(User-Based 대비 전 지표 상승)에 이어, **실제 추천 결과가 사용자 관점에서도 납득 가능한지 확인하는 정성평가**를 진행함
- Content-Based 정성평가(Day08)에서 발견했던 **다중 취향 쏠림 현상이 Item-Based에서도 나타나는지** 재현 실험을 설계함
- 총 6개 실험(A~F)을 설계하고 실행함: 카드류 단일 취향, 슈팅류 단일 취향, 카드+슈팅 혼합, PUBG 단일 게임, RDR2 단일 게임, Stardew Valley 단일 게임
- Party Animals 입력 시 빈 DataFrame이 반환되는 것을 확인하고 해당 실험을 Stardew Valley로 대체함
- 각 실험 결과를 Day08 Content-Based 결과와 직접 비교하여, 두 모델이 각각 어떤 신호를 포착하는지 정리함
- 특히 카드+슈팅 혼합 실험(C)에서 Content-Based와 Item-Based가 **정반대 방향으로 쏠리는 현상**을 확인함
- 정성평가 결과를 바탕으로 Item-Based CF의 최종 장단점을 정리하고, Hybrid 설계 필요성에 대한 문제의식을 남김
- Name = NaN metadata mismatch 문제를 발견했으나 수정하지 않고 To-Do로만 기록함
- Item-Based baseline 단계를 마무리하고, 다음 baseline 모델(Model-Based CF)로 이동하기로 확정함

## 2. 실험 설계 및 진행 과정

**1) 정성평가 목적 정의**

정량평가에서 이미 확인한 Item-Based의 성능 우위가 실제 추천에서도 체감되는지, 그리고 Content-Based에서 발견했던 다중 취향 쏠림이 collaborative signal 기반 모델에서도 재현되는지를 확인하는 것을 목적으로 정의함.

**2) 실험 설계**

| 실험 | 입력 | 핵심 목적 |
|---|---|---|
| A | Slay the Spire + Balatro | 카드/덱빌딩 취향 포착 여부 |
| B | Counter-Strike 2 + Left 4 Dead 2 | 슈팅 취향 포착 여부 |
| C | Slay the Spire + Balatro + CS2 + L4D2 | 서로 다른 두 취향을 섞었을 때 쏠림 여부 |
| D | PUBG: BATTLEGROUNDS | 단일 게임에서 여러 특징을 얼마나 잘 확장하는지 |
| E | Red Dead Redemption 2 | 콘텐츠 유사성과 실제 소비 취향의 차이 확인 |
| F | Stardew Valley | 생활/농장이라는 세부 특성과 broader taste의 차이 확인 |

C 실험은 Day08 Content-Based에서 수행했던 핵심 장르 혼합 실험과 **동일 입력**으로 설계하여, 두 모델의 결과를 직접 비교할 수 있도록 함.

**3) 실행 중 발견한 문제**

Party Animals를 입력했을 때 빈 DataFrame이 반환됨을 확인함. 원인 분석은 하지 않고 해당 게임을 정성평가 대상에서 제외한 뒤 Stardew Valley로 대체함.

## 3. Item-Based CF 정성평가 핵심 구조 정리

### 3-1. 비교 축

- Content-Based: "이 게임은 무엇인가?" — 장르·태그·테마 등 콘텐츠 자체의 semantic 유사성
- Item-Based CF: "이 게임을 플레이한 사람들은 다른 어떤 게임을 함께 소비하는가?" — 공통 user interaction 기반 collaborative signal

### 3-2. 실험별 결과 요약

| 실험 | 주요 추천 | 관찰된 패턴 |
|---|---|---|
| A (카드류) | Monster Train, Inscryption, Griftlands, Hades, Binding of Isaac, Dead Cells 등 | 직접적인 카드/덱빌딩 + 로그라이크·빌드 구성이라는 상위 구조 공유 게임 혼합 |
| B (슈팅류) | Half-Life 2, TF2, Left 4 Dead, Garry's Mod, CS:Source, PAYDAY 2 등 | FPS 계열 다수, 세부 특징(좀비/협동)은 약하게 반영 |
| C (혼합) | Half-Life 2, TF2, Left 4 Dead, Garry's Mod, Portal 2, Monster Train, CS:Source, PAYDAY 2, Hades, POSTAL 2 | 슈팅 계열로 강하게 쏠림, B의 Top-5와 완전히 동일 |
| D (PUBG) | GTA V, Rainbow Six Siege, CS2, DayZ, Rust, ARK, Z1 BR, Call of Duty, Dead by Daylight | 경쟁 FPS·생존 PvP·배틀로얄·온라인 액션으로 폭넓게 확장 |
| E (RDR2) | Witcher 3, God of War, Cyberpunk 2077, Mafia, Spider-Man, Days Gone 등 | 대형 싱글플레이·스토리·액션·오픈월드로 확장, GTA V는 미포함 |
| F (Stardew) | Terraria, Don't Starve Together, Slime Rancher, Starbound, Hades, Portal 2, Undertale 등 | 콘텐츠 근접 게임 + 샌드박스·제작·인디·스토리로 확장 |

## 4. 정성평가 결과 및 핵심 분석

### 4-1. 실험 A — Slay the Spire + Balatro

Monster Train, Inscryption, Griftlands는 카드/덱빌딩이라는 입력 취향과 직접적으로 연결됨. Hades, Binding of Isaac, Dead Cells는 카드게임은 아니지만 로그라이크·반복 플레이·빌드 구성이라는 상위 플레이 구조를 공유함.

```text
카드/덱빌딩 → 로그라이크 → 빌드 구성 → 전략적 반복 플레이
```

정성적으로는 직접적인 카드게임과 행동적으로 가까운 인접 장르가 혼합된 추천으로 정리함.

### 4-2. 실험 B — Counter-Strike 2 + Left 4 Dead 2

Half-Life 2, TF2, Left 4 Dead, Garry's Mod, Portal 2, CS:Source, PAYDAY 2, POSTAL 2, The Forest, Counter-Strike 등 FPS 계열이 약 7개 포함됨. 다만 L4D2가 가진 좀비/공포/협동/생존이라는 구체적 특징은 추천 결과에서 강하게 유지되지 않고, 전반적인 슈팅 취향만 두드러짐.

Valve 계열 게임이 다수 포함된 것은 "Valve cluster" 같은 정식 용어가 아니라, 동일한 게임 생태계의 아이템들이 공통 user interaction에 의해 높은 collaborative signal을 형성했을 가능성으로 해석함.

### 4-3. 실험 C — 카드 2개 + 슈팅 2개 혼합 ⭐

**Top-10**: Half-Life 2, TF2, Left 4 Dead, Garry's Mod, Portal 2, Monster Train, CS:Source, PAYDAY 2, Hades, POSTAL 2

직접적인 카드/덱빌딩 게임은 Monster Train 정도만 남았고, 슈팅 계열이 Top-10 대부분을 차지함.

```text
B의 Top-5: Half-Life 2 / TF2 / Left 4 Dead / Garry's Mod / Portal 2
C의 Top-5: Half-Life 2 / TF2 / Left 4 Dead / Garry's Mod / Portal 2
```

Slay the Spire와 Balatro를 추가했음에도 B의 Top-5가 그대로 유지됨. 이는 슈팅 계열의 interaction density가 높아 collaborative signal이 더 안정적으로 형성됐고, similarity-sum aggregation 과정에서 해당 후보들의 점수가 강하게 누적됐을 가능성으로 정리함(원인 검증은 보류).

### 4-4. C 실험과 Day08 Content-Based의 비교 ⭐⭐⭐

| 모델 | 동일한 4개 입력 결과 |
|---|---|
| Content-Based (Day08) | 카드/덱빌딩 쪽으로 강하게 쏠림 |
| Item-Based CF (Day15) | 슈팅/CS2·L4D2 쪽으로 강하게 쏠림 |

두 모델 모두 서로 다른 두 취향을 균형 있게 보존하지 못했지만, **쏠리는 방향이 정반대**라는 점이 이번 정성평가에서 가장 중요한 발견임. 원인 후보도 서로 다름 — CBF는 TF-IDF feature/IDF·평균 유사도 구조, Item-CF는 interaction density·user overlap·similarity-sum aggregation 구조. 현재 aggregation이 "모든 취향을 일정 수준 만족하는 게임"을 찾는 구조가 아니라 "각 source의 similarity를 합산해 총점이 높은 게임"을 뽑는 구조이기 때문일 가능성으로 정리함.

### 4-5. 실험 D — PUBG: BATTLEGROUNDS

GTA V, Rainbow Six Siege, CS2, DayZ, Rust, ARK, Z1 Battle Royale, Call of Duty, Dead by Daylight 등이 추천됨.

```text
경쟁 FPS   → Rainbow Six Siege / CS2 / Call of Duty
생존·PvP   → DayZ / Rust / ARK
배틀로얄   → Z1 Battle Royale
온라인 액션 → GTA V
```

PUBG 하나에서 여러 인접 취향으로 자연스럽게 확장됨. Day08 CBF에서도 PUBG는 배틀로얄/생존 계열로 잘 수렴했으므로, CBF는 장르/태그 자체를 따라가는 방식으로, Item-CF는 실제 PUBG 플레이어가 함께 소비할 법한 인접 취향까지 확장하는 방식으로 각각 좋은 결과를 냈다고 정리함.

### 4-6. 실험 E — Red Dead Redemption 2

Witcher 3, God of War, Cyberpunk 2077, Mafia, Spider-Man Remastered, Jedi: Fallen Order, Far Cry 5, Fallout 4, Days Gone, Detroit: Become Human이 추천됨. 오픈월드·스토리텔링·전투 중심의 대형 싱글플레이 게임으로 확장됨.

GTA V가 추천 목록에 없다는 점은 아쉬운 부분으로 확인함. RDR2와 GTA V는 총기·오픈월드·범죄·스토리·Rockstar라는 직접적 공통점이 많기 때문임.

Day08 Content-Based RDR2 결과(Red Dead Online, GTA V, Metal Gear Solid V 등)와 비교하면 차이가 선명함 — CBF는 RDR2와 콘텐츠적으로 가까운 게임을, Item-CF는 RDR2를 좋아하는 사람이 함께 좋아할 법한 넓은 대형 스토리/액션 게임을 찾음.

### 4-7. 실험 F — Stardew Valley

Terraria, Don't Starve Together, Hades, Portal 2, Slime Rancher, Bloons TD 6, Undertale, Witcher 3, To the Moon, Starbound가 추천됨.

```text
콘텐츠 자체도 가까운 게임: Terraria / Don't Starve Together / Slime Rancher / Starbound
유저 취향상 연결되는 게임: Hades / Portal 2 / Bloons TD 6 / Undertale / Witcher 3 / To the Moon
```

Slime Rancher는 수집 → 관리/육성 → 자원 획득 → 확장이라는 생활형 플레이 루프 측면에서 특히 좋은 추천으로 평가함. Item-CF가 농장 게임만 좁게 찾기보다 제작·샌드박스·인디·스토리 등 broader taste로 확장하는 모습을 다시 확인함.

## 5. 내가 직접 내린 분석·판단

- 카드/덱빌딩 게임을 넣었을 때 실제로 카드/덱빌딩 게임이 여러 개 추천되었으므로 기본적인 취향은 제대로 잡았다고 판단함 (실험 A)
- CS2 + L4D2 입력에 대해 슈팅게임이 많이 나온 것으로 보아 해당 취향은 잘 잡아낸 것으로 판단했으나, 좀비 같은 키워드나 세부적인 특성까지 잘 잡은 추천은 아니라는 한계도 함께 확인함 (실험 B)
- 카드+슈팅 혼합 입력(C)에서 슈팅 게임 쪽이 더 인기 있고 interaction도 많아 Item-User matrix가 상대적으로 dense할 수 있고, 그로 인해 Item-Based가 유사 게임과 가중치를 더 안정적으로 잡아내면서 슈팅 쪽이 우세해진 것 아닐까 하는 가설을 제시함
- PUBG 결과에 대해 "저번보다 훨씬 잘 잡은 것 같다"고 평가하며, PUBG의 다양한 특성과 연관되는 게임이 실제로 많이 나타난 점을 긍정적으로 판단함 (실험 D)
- RDR2 결과에서 오픈월드·스토리텔링·전투·무법자 느낌과 PlayStation에서 유명한 게임들이 많이 나타난 점을 긍정적으로 평가하며, "스토리텔링이고 무법자 느낌에 싸우는 오픈월드 게임을 잘 찾아냈다는 점에서 추천 성능은 인정한다"고 판단함. 다만 GTA V가 없다는 점은 아쉬움으로 남김 (실험 E)
- Content-Based와 Hybrid하면 훨씬 잘 잡을 것 같다는 문제의식을 제기함 — CBF의 세부 콘텐츠 포착 능력과 Item-CF의 인접 취향 확장 능력이 상호보완될 가능성이 있다고 판단함

## 6. GPT 피드백으로 수정한 부분

- **Valve 계열 쏠림을 "Valve cluster"로 표현한 부분**: 추천시스템 정식 용어가 아니므로, 동일한 게임 생태계의 아이템들이 공통 user interaction에 의해 높은 collaborative signal을 형성했을 가능성이라는 표현으로 수정함
- **C 실험의 슈팅 쏠림 가설을 원인으로 확정하려던 부분**: "슈팅 게임이 더 인기 있어서"라는 가설은 설득력이 있지만 아직 확정된 원인은 아니며, interaction 수·shared user 수·sparsity·cosine similarity 분포·candidate 중복 등을 함께 봐야 한다고 수정함. 현재는 CS2/L4D2 계열의 interaction과 user overlap이 상대적으로 풍부하여 안정적인 collaborative signal을 형성했고, similarity-sum aggregation 과정에서 점수가 강하게 누적됐을 가능성 정도로만 가설을 유지하고, 검증은 이후 모델 분석 단계로 보류함
- **RDR2 결과에서 PlayStation 게임이 많이 나온 것을 PlayStation bias로 해석하려던 부분**: 플랫폼 영향으로 확정하기에는 근거가 부족하며, 결과 전체를 보면 플랫폼보다 대형 싱글플레이 + 강한 스토리 + 액션 + 오픈월드/탐험이라는 취향이 더 강하게 나타난 것으로 해석하는 것이 안전하다고 수정함
- **RDR2 추천 전체를 "무법자" 테마로 묶으려던 부분**: Mafia와 Cyberpunk 등은 범죄/무법자 측면에서 연결되지만 God of War나 Witcher 3는 그렇지 않으므로, 모든 게임을 하나의 테마로 묶기는 어렵다고 수정함
- **Hybrid가 다중 취향 쏠림을 자동으로 해결해줄 것이라는 기대**: CBF는 카드 쪽, Item-CF는 슈팅 쪽으로 각각 쏠렸기 때문에 단순히 두 score를 더하면 또 다른 쏠림이 생길 수 있으며, 현재는 "Hybrid를 하면 무조건 해결된다"가 아니라 "Hybrid 설계가 필요한 이유와 고려해야 할 문제를 발견했다" 정도로 정리하는 것이 정확하다고 수정함

## 7. 오늘 확정한 실험 방향

- Item-Based CF의 baseline 정성평가는 오늘 6개 실험(A~F)으로 완료하고 추가 실험은 진행하지 않음
- Content-Based와 Item-Based는 "어느 쪽이 더 낫다"가 아니라 **서로 다른 signal을 포착한다**는 관점으로 최종 결론을 정리함
- Day08 CBF와 Day15 Item-CF가 동일 혼합 입력(C 실험)에서 정반대로 쏠린 결과를 이후 모델 비교·Hybrid 설계의 핵심 근거로 삼기로 함
- Hybrid는 지금 설계하지 않고, "필요성과 고려해야 할 문제를 발견한 단계"로만 남겨두고 Model-Based CF 이후로 보류함
- Item-Based 단계에서는 k, sum/mean, normalization 등 하이퍼파라미터를 건드리지 않는다는 Day14의 원칙을 그대로 유지함 — 모든 baseline을 기본형으로 완성한 뒤 비교하는 것이 목표이기 때문

## 8. 오늘 보류한 문제

**정성평가 관련**

- C 실험에서 제시한 "슈팅 계열 interaction density가 높아 쏠렸다"는 가설의 실제 검증(interaction 수, shared user 수, sparsity, cosine similarity 분포 등) — 모델 분석 단계로 보류
- Party Animals 입력 시 빈 DataFrame이 반환되는 원인 분석 — 진행하지 않음

**metadata 관련**

- Name = NaN metadata mismatch 문제 — 원인 파악 및 수정은 하지 않고 To-Do로만 기록

**Hybrid 관련**

- Content-Based + Item-Based Hybrid의 구체적 설계(가중치 결합 방식, 쏠림 재발 방지 방법) — Model-Based baseline 완료 이후로 보류

## 9. 성찰 및 느낀 점

오늘은 Day14에서 완성한 Item-Based CF의 정량적 우위가 실제 추천 결과에서도 체감되는지를 확인하면서, **모델의 구조적 특성이 정성적 결과로 어떻게 드러나는지**를 훨씬 구체적으로 확인한 날이었다.

특히 C 실험에서 Slay the Spire와 Balatro를 추가로 넣었음에도 B 실험의 Top-5가 한 글자도 다르지 않게 그대로 유지된 부분은, 단순히 "쏠림이 있다"는 것을 넘어서 **현재 aggregation 구조가 서로 다른 취향을 균형 있게 담아내는 설계가 아니라는 점**을 직접 눈으로 확인한 경험이었다. 그리고 이 결과를 Day08 Content-Based와 나란히 놓았을 때, 같은 입력에서 정반대 방향으로 쏠린다는 사실은 어느 한쪽 모델의 결함이라기보다 **두 모델이 완전히 다른 종류의 정보를 사용하고 있다는 증거**로 읽는 것이 더 정확하다는 것을 배웠다.

또한 RDR2 실험에서 "PlayStation 게임이 많이 나왔다"는 관찰을 그대로 원인으로 단정하지 않고, 더 넓은 패턴(대형 싱글플레이 + 스토리 + 액션 + 오픈월드)으로 다시 짚어본 과정은, 정성평가에서도 **눈에 보이는 표면적 공통점과 실제 구조적 원인을 구분해서 봐야 한다**는 점을 다시 확인시켜줬다.

## 10. 다음에 해야 할 것 (To-Do)

**바로 다음 시작 지점**: Item-Based CF의 정량평가와 정성평가가 모두 끝났으므로, 다음은 Item-Based를 더 개선하는 것이 아니라 **Model-Based CF baseline 선정 및 구현**으로 넘어간다.

- [ ] Name = NaN metadata mismatch 문제 기록만 유지, 수정은 이후 진행
- [ ] Steam implicit interaction 구조(binary +1/-1/0)에 적합한 Model-Based CF baseline(SVD/Matrix Factorization 계열 포함) 후보 조사
- [ ] Model-Based CF 기본형 구현
- [ ] 기존 evaluation pipeline(Precision/Recall/HitRate/NDCG, Macro/Micro)으로 동일하게 정량평가
- [ ] 필요 시 Model-Based에 대한 간단한 정성평가 진행
- [ ] Content-Based / User-Based / Item-Based / Model-Based baseline 비교 정리
- [ ] 각 모델이 잘 잡는 signal과 구조적 장단점 종합 정리
- [ ] 이후 Hybrid 설계 시 C 실험(카드+슈팅 혼합)의 쏠림 비교 결과를 근거로 활용

**다음 세션 시작 문장** (생각 복구용):

> "Item-Based CF의 정량평가와 정성평가를 모두 마쳤다. Content-Based는 카드 쪽, Item-Based는 슈팅 쪽으로 동일 입력에서 정반대로 쏠리는 것을 확인했고, 두 모델이 서로 다른 signal(콘텐츠 semantic vs collaborative interaction)을 포착한다는 결론을 내렸다. 이제 Model-Based CF baseline을 선정하고 구현하는 단계로 넘어간다."

## 11. 오늘의 한 문장 회고

오늘은 Day14 정량평가에서 확인한 Item-Based CF의 우위를 6개 정성 실험으로 추가 점검하면서, Content-Based와 Item-Based가 **동일한 입력에서도 정반대 방향으로 쏠린다는 사실**을 발견함으로써 두 모델이 "누가 더 낫다"가 아니라 **서로 다른 종류의 신호를 포착하고 있다는 것**을 데이터로 확인한 날이었다.