# Day25 — Hybrid 미세조정 완료, Learning-to-Rank 설계 및 최종 시스템 구체화

## 1. 오늘 한 일 요약

오늘은 새로운 Hybrid 구조를 계속 추가하는 날이 아니라, 지금까지 만든 추천시스템을 **최종 구조에 가깝게 다듬고 Learning-to-Rank(LTR) 단계로 넘어간 날**이었다. 현재 기준 Hybrid 구조는 다음과 같다.

```text
BPR Top-56 + Content-Based Top-39 + User-Based Top-5
        ↓ Seen 제거 + Candidate UNION
Candidate 약 100개
        ↓
Item-Based Ranker
        ↓
Top-10 Recommendation
```

```text
Retriever 비율 Fine-Tuning (BPR/Content 비율 미세조정)
        ↓
BPR 27개 Grid Search → Final BPR 재확정
        ↓
Retrieval ≠ Ranking 재확인
        ↓
Learning-to-Rank(XGBoost Ranker) 도입 결정
        ↓
1차 LTR Feature(Score+Rank 8개) 설계, Exp1/2/3 단계적 계획 수립
        ↓
LTR 학습용 사용자와 최종 평가 사용자 분리 설계
        ↓
XGBoost 코드 구성 및 환경 문제 해결
        ↓
실제 사용자 입력 기반 최종 추천 시스템 구조 논의
        ↓
신규 사용자 BPR 처리 방식 설계 (Item Embedding 평균 + Fine-Tuning)
        ↓
MAP@10 / Popularity / Novelty 최종 평가 계획 추가
        ↓
README 가독성 개선 및 구조 재편 (약 1,669줄 → 약 943줄)
```

---

## 2. Retriever 비율 Fine-Tuning

Candidate Size 100, User-Based 비중 5를 고정하고 BPR/Content 비율만 미세 조정했다.

| BPR/Content/User | UNION Recall | Scoreable Recall | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50/45/5 | 0.2453 | 0.2038 | 0.08656 | **0.11111** | 0.5300 | **0.12157** | 345 |
| 53/42/5 | 0.2470 | 0.2050 | 0.08672 | 0.11072 | 0.5325 | 0.12120 | 346 |
| **56/39/5** | 0.2488 | 0.2055 | **0.08722** | 0.11102 | **0.5400** | 0.12155 | **348** |
| 59/36/5 | 0.2538 | 0.2087 | 0.08666 | 0.10997 | 0.5350 | 0.12139 | 346 |
| 62/33/5 | **0.2540** | **0.2097** | 0.08612 | 0.10957 | 0.5275 | 0.12087 | 344 |

BPR 비중을 높일수록 Candidate Recall과 Scoreable Recall은 계속 좋아졌지만, 최종 Top-10 성능은 같은 방향으로 움직이지 않았다 — 특히 `62/33/5`는 Candidate Recall이 가장 높았음에도 Hits는 344로 오히려 낮았다. `50/45/5`의 NDCG가 미세하게 높았지만 차이가 사실상 크지 않아, 전체 지표를 고려해 **기존 56/39/5를 그대로 유지**하기로 했다.

---

## 3. BPR 27개 Grid Search와 Final BPR 재확정

BPR의 핵심 hyperparameter(factors/regularization/iterations)에 대해 총 **27개 조합**을 비교했다. 이때 Hybrid 조건(Candidate Size 100, BPR 56/Content 39/User 5)은 고정하고 BPR 파라미터만 바꿔 통제했다.

**최종 1위**: factors=60, regularization=0.006, iterations=15 — BPR Candidate Recall 0.183040, UNION Candidate Recall 0.248277, Item Scoreable Recall 0.205562, Precision@10 0.086599, Recall@10 0.109385, HR@10 0.522500, NDCG@10 0.120657, Hits 345.

> **재현성 메모:** 위 Retriever Ratio 표의 `56/39/5` 결과(P@10 0.08722, Hits 348)와 이 Grid Search의 동일 nominal BPR 설정 결과(P@10 0.086599, Hits 345)는 **서로 다른 실험 실행에서 기록된 값**이다. 같은 하이퍼파라미터명만으로 재학습 상태·캐시·실행 조건까지 완전히 동일했다고 확인할 근거는 없으므로 하나의 수치로 강제 통일하지 않고 각 실험 결과를 그대로 보존한다. 프로젝트의 최종 공식 성능은 Day27의 Final400 결과를 기준으로 한다.

```python
factors = 60
regularization = 0.006
iterations = 15
learning_rate = 0.05
explicit_false = True
explicit_false_epochs = 1
explicit_false_learning_rate = 0.01
explicit_false_regularization = 0.001
```

2위 조합(factors=40, reg=0.01, iterations=20)은 UNION Candidate Recall이 0.252776으로 1위보다 높았지만 NDCG@10은 0.117489로 더 낮았다 — 여기서도 **Candidate Recall↑ ≠ Final Ranking Performance↑**가 다시 확인되었다.

---

## 4. Retrieval ≠ Ranking 재확인, Learning-to-Rank 진입

Candidate Size Sweep(Day24)에서 이미 "Candidate Recall은 후보를 늘릴수록 계속 증가하지만 Final Top-10은 100에서 최고"였던 현상을, 오늘 Retriever Ratio 실험과 BPR Grid Search 모두에서 **똑같이** 재확인했다. 즉 **좋은 후보를 많이 가져오는 능력과 그 후보를 정확한 순서로 배치하는 능력은 별개**라는 것이 오늘 세 번째로 관찰됐고, 이것이 Learning-to-Rank로 넘어가는 가장 중요한 근거가 되었다.

BPR 튜닝 자체는 Candidate Retriever 품질 향상에는 의미가 있었지만, Candidate Recall이 높아지는 것만으로 최종 Top-10이 좋아지지는 않았다. 따라서 지금부터의 핵심 문제를 **"Retriever를 더 복잡하게 만드는 것보다, 이미 확보한 Candidate 100개를 더 잘 Ranking하는 것"**으로 재정의했고, Hybrid 구조 자체에 대한 탐색은 사실상 여기서 마무리하고 Ranking 단계 개선(LTR)으로 넘어갔다.

---

## 5. XGBoost Learning-to-Rank 설계

### 실험 통제 조건

```text
Candidate Retrieval: BPR 56 + Content 39 + User 5 (약 100개)  ← 고정
변경되는 것: Item-Based Ranker → XGBoost Ranker
```

이렇게 통제하면, 성능이 좋아졌을 때 "Candidate Generation이 달라져서"가 아니라 **"Ranking 모델이 후보 순서를 더 잘 결정했다"**고 해석할 수 있다.

### XGBoost Ranker의 역할

기존 Weighted Rank Fusion에서는 `Item 0.50 + BPR 0.20 + Content 0.15 + User 0.15`처럼 사람이 직접 가중치를 정했다. XGBoost Ranker는 각 모델의 score를 Feature로 제공하면 Tree Split을 통해 **어떤 score가 중요한지, 어떤 score 조합일 때 정답인지, 어떤 rank 조건일 때 더 위로 올려야 하는지**를 스스로 학습한다 — 선형 결합(`0.5×Item+0.3×BPR+0.2×Content`)보다 "Item score는 낮지만 BPR+Content가 동시에 높으면 그 조합을 별도로 높게 평가"하는 것 같은 비선형 조건도 학습할 수 있다는 것이 핵심 차이다.

### 1차 Feature Set — Score + Rank 8개

처음엔 모델 Score 4개만 쓰려 했지만, **각 모델의 Score와 Rank를 모두 사용**하는 방향으로 수정했다. Score는 "얼마나 강하게 추천하는가", Rank는 "그 모델 내부에서 다른 Candidate 대비 몇 번째인가"를 표현하므로 두 정보를 함께 주는 것이 더 유용하다고 판단했다.

```text
Score: item_score_norm, bpr_score_norm, content_score_norm, user_score_norm
Rank:  item_rank, bpr_rank, content_rank, user_rank
→ 총 8개 Feature (Score는 Candidate Set 내부에서 사용자별 Min-Max Normalization)
```

### 단계적 Feature 추가 계획 (Exp1 → Exp2 → Exp3)

처음부터 15~20개 Feature를 다 넣지 않고 **Ablation처럼 단계적으로 추가**하기로 했다.

- **Exp1 — Model Signal Only (8개)**: 위 Score+Rank 8개. 목적은 "기존 4개 추천 모델의 출력만으로 XGBoost가 Item-Based Ranker보다 나은 순위를 학습할 수 있는가?"
- **Exp2 — Candidate/User Context 추가 (10개)**: `source_count`(해당 Candidate를 몇 개 Retriever가 동시에 추천했는지), `user_interaction_count`(해당 사용자의 Train interaction 수) 추가.
- **Exp3 — Item Statistics 추가 (12개)**: `game_popularity`(Train interaction 수), `game_positive_ratio`(Train True 비율) 추가. 이 값들은 반드시 **Train 데이터만으로 계산**해 Test Leakage를 막아야 한다.
- **선택적 확장**: Exp3까지 의미가 있을 경우에만 genre/tag/developer/publisher similarity 등을 추가하며, 프로젝트 막바지이므로 무작정 늘리지 않기로 했다.

### LTR 학습 데이터 구조

한 행은 `(user_id, candidate app_id)` 쌍이며, 각 Candidate마다 Feature를 붙여 하나의 Ranking DataFrame을 만든다. `user_id`/`app_id`는 identifier로만 쓰고 numerical Feature에는 포함하지 않는다.

### 왜 LTR 학습 사용자를 별도로 둬야 하는가

기존 400명은 지금까지 모든 Hybrid 실험에서 써온 **최종 평가 사용자**다. LTR은 학습하는 ML 모델이므로 이 400명의 Test 정답을 학습에 쓰면 "시험 문제를 먼저 보고 학습 → 같은 시험으로 평가"가 되어 오염된다. 그래서 **LTR 학습용 사용자 약 1,000명(Train 800 / Validation 200)**을 별도로 두고, 기존 400명은 계속 Final Evaluation 전용으로 유지하기로 했다. 사용자 한 명은 하나의 Ranking Problem(Candidate 약 95개)을 만들며, 1,000명이면 약 1,000개의 Ranking 사례가 된다.

---

## 6. XGBoost 코드 구성과 환경 문제

1차 코드는 `Candidate Retrieval(BPR56+Content39+User5) → Feature(Score+Rank 8개) → objective=rank:ndcg → 평가=ndcg@10` 조건으로 설계했다. 코드 작업 중 **기존 Hybrid Cache가 `user_id/rank/app_id`만 저장하고 실제 모델 score를 버리고 있다는 점**을 발견해, LTR에는 각 모델의 **실제 raw score를 보존하는 Feature 생성 과정**이 별도로 필요하다는 것을 확인했다. BPR(`model.recommend()`의 item_indices/scores), Content(TF-IDF cosine similarity), User-Based(이웃 가중합), Item-Based(candidate로 들어오는 item-item similarity 합) 네 모델 모두 실제 numerical score를 갖고 있어 Feature로 활용 가능함을 확인했다.

실행 시 `ModuleNotFoundError: No module named 'xgboost'`가 발생했는데, 코드 문제가 아니라 `.venv`에 XGBoost가 설치되어 있지 않았던 것이 원인이었다. `pip install xgboost`로 설치하고 버전 확인 후 해결했다.

**예상 실행시간**(Feature Engineering이 XGBoost 학습 자체보다 병목일 것으로 판단): Train/Interaction Matrix 준비 2~8분, LTR 1,000명 Feature 생성 30~90분, XGBRanker 학습 수십 초~2분, 기존 400명 Feature 생성+평가 10~35분, 전체 최초 실행 약 45분~2시간. 한번 Feature Cache를 만들면 이후 Feature 실험은 Cache→Column 선택→재학습→평가만 하면 되므로 상대적으로 저렴할 것으로 예상했다.

---

## 7. 실제 추천 시스템 구조와 신규 사용자 BPR 문제

프로젝트를 평가 코드로만 끝내지 않고, 사람이 직접 게임을 입력해 추천을 받는 시스템까지 만들기로 했다 — `사용자가 플레이한 게임 입력 → 추천 엔진 실행 → Top-10 출력` 구조이며, `main.py`(실제 사용)와 `evaluation.py`(정량 평가)가 **동일한 최종 recommender class를 공유**하도록 설계했다.

### 신규 사용자 문제

Content-Based/Item-Based/User-Based는 플레이한 게임 목록만 있어도 어느 정도 추천이 가능하지만, BPR은 기존 학습 사용자에 대한 `user latent factor`가 있어야 하는데 새 사용자는 이 매핑이 없다는 문제를 발견했다.

### 두 가지 해결 방법 비교

- **방법 A — Item Embedding 평균**: 플레이한 게임들의 BPR Item Factor를 평균해 pseudo user vector $p_u^{(0)}=\frac{1}{|I_u|}\sum_{i\in I_u}q_i$를 만든다. 빠르지만 BPR이 실제로 이렇게 학습되도록 설계된 것은 아니라는 heuristic 한계가 있다.
- **방법 B — User Factor만 Fine-Tuning**: 기존 Item Factor는 freeze하고, `Played Item=Positive, Unseen Item=Negative`로 두어 $p_u^Tq_i > p_u^Tq_j$가 되도록 **새 User Vector 하나만** BPR objective로 학습한다. 전체 3,700만 interaction을 재학습하는 게 아니라 60차원 벡터 하나만 업데이트하면 된다.

### 최종 채택 — 두 방법의 결합

어느 하나를 버리지 않고 **Item Embedding 평균으로 좋은 초기 위치를 만든 뒤, BPR Ranking Objective로 Fine-Tuning**하는 방식을 채택했다.

```text
게임 입력 → BPR Item Embedding 조회 → 평균 → 새 User Vector 초기값
→ Item Factor Freeze → 새 User Vector만 BPR Fine-Tuning → 최종 User Factor
→ BPR Candidate Score (이후 기존 Hybrid 구조 그대로 사용)
```

예상 학습 비용은 `Played Games 20 × Negative/Positive 5 × Epoch 30 ≈ 3,000 updates` 수준으로 전체 BPR 재학습 대비 매우 작으며, 목표 실행시간은 sub-second~수초 수준으로 오히려 Item-Based/User-Based 계산이 더 큰 병목이 될 가능성이 있다고 판단했다.

---

## 8. 평가 지표 확장과 README 재편

### MAP@10 추가

기존 Precision/Recall/HR/NDCG@10에 **MAP@10**(Mean Average Precision)을 추가하기로 했다 — 관련 아이템을 맞힌 위치에서 Precision을 계산해 "관련 아이템을 얼마나 상위에 안정적으로 배치했는가"를 보는 보조 Ranking Metric이다. `MAPE(Mean Absolute Percentage Error, 회귀 지표)`와 이름이 비슷하지만 완전히 다른 지표라는 것을 명확히 구분했다.

### Popularity Bias / Novelty 추가

정확도만으로는 좋은 추천시스템인지 알 수 없어, 추천이 인기 게임에 과도하게 쏠리는지도 확인하기로 했다. 기본 Popularity(Train interaction count) 대신 초인기 게임의 왜곡을 줄이기 위해 $\log(1+\text{interaction count})$ 기반 **Mean Log Popularity@10**을 쓰기로 했고, 반대 관점의 **Novelty@10**($-\log_2(\text{interaction count}_i/\text{total interactions})$, 인기 낮은 게임일수록 값이 큼)도 함께 추가했다. 이를 통해 "정확도↑ + Popularity↑↑ + Novelty↓"가 나타나면 "정확도는 좋아졌지만 인기 아이템 편향도 함께 증가했다"고 해석할 수 있게 된다.

### 최종 평가 체계

```text
Accuracy/Ranking: Precision@10, Recall@10, HR@10, NDCG@10, MAP@10
Recommendation Characteristic: Mean Log Popularity@10, Novelty@10
정성평가: 실제 게임 입력 → Top-10 생성 → 추천 게임 정보·Popularity 함께 확인
```

### README 재편

기존 README는 정보는 충분했지만 Project Goals/Development Roadmap/Current Progress/Project Status가 상당 부분 반복되고 성능표·현재 상태가 여러 구역에서 재등장했다. 약 1,669줄을 정보는 최대한 유지하면서 약 943줄로 재구성하고, `At a Glance → Current Architecture → Current Best Result → Key Conclusions → Model Performance → Hybrid Experiment History → Model Details → Debugging → Evaluation → Performance Optimization → Project Structure → Roadmap → Current Status` 순서로 바꿨다. Funk SVD/BPR/Debugging처럼 기록상 중요하지만 첫 화면에 다 펼칠 필요는 없는 내용은 `<details>`로 접어, **"위쪽은 Portfolio README, 아래쪽은 Technical Report"** 형태로 개선했다.

---

## 9. 내가 직접 판단하고 결정한 부분

- **56/39/5 Retriever 비율 유지 결정**: NDCG가 미세하게 높은 50/45/5 대신 전체 지표를 고려해 기존 비율 유지
- **BPR Grid Search 결과에서 factors=60/reg=0.006/iter=15를 Final BPR로 재확정**: UNION Candidate Recall이 더 높은 2위 조합 대신 NDCG가 더 좋은 1위를 선택
- **더 이상 새로운 Hybrid 구조를 추가하지 않고 Ranking 개선(LTR)으로 문제를 재정의**
- **1차 LTR Feature를 Score 4개가 아니라 Score+Rank 8개로 설정**
- **Feature를 한 번에 다 넣지 않고 Exp1→Exp2→Exp3 단계적 Ablation으로 설계**
- **기존 400명은 계속 Final Evaluation 전용으로, LTR 학습에는 별도 1,000명을 쓰기로 분리 결정**
- **실제 사용자가 게임을 입력하는 최종 시스템까지 구현하기로 함**
- **신규 사용자 BPR 문제를 포기하지 않고, Item Embedding 평균 + User-only Fine-Tuning을 함께 쓰는 결합 방식 채택**
- **MAP@10, Popularity, Novelty를 최종 평가에 추가**
- **프로젝트 막바지에는 새 구조 탐색보다 통합과 평가에 집중하기로 방향 전환**

## 10. GPT(AI)가 담당한 부분

Retriever 실험 결과 해석, BPR Grid 결과 비교, LTR/LambdaMART 개념 설명, Feature 후보 구조화, Score+Rank 8 Feature 설계 지원, Exp1/2/3 실험 설계, Feature DataFrame 구조 설계, LTR 코드 구조 초안, 실행시간 병목 추정, XGBoost 설치 오류 원인 파악, `main.py` 최종 구조 제안, 신규 사용자 BPR 대안(방법 A/B) 비교 설명, MAP/Popularity/Novelty 지표 설명, README 구조 재편 — 개념 설명과 설계 초안 제시가 GPT 비중이 컸고, 그 중 어떤 안을 실제로 채택할지(비율 유지, Final BPR 선택, Feature 단계, 신규 사용자 결합 방식 등)는 직접 판단했다.

---

## 11. 오늘의 회고

오늘의 핵심을 압축하면: **Retriever는 좋은 Candidate를 Top-100 안에 넣는 역할, Ranker는 그 100개의 최종 순서를 정하는 역할**이며, Candidate Recall↑이 Final Top-10 Performance↑을 보장하지 않는다는 것을 오늘만 세 번(Retriever Ratio, BPR Grid, 그리고 Day24의 Candidate Size Sweep) 확인했다. **XGBoost LTR은 기존 추천 모델의 Score/Rank와 User/Item Context를 이용해 Ranking 규칙 자체를 ML로 학습**하는 시도이고, **신규 사용자 BPR은 전체 재학습 없이 기존 Item Latent Space를 유지한 채 새 User Vector 하나만 적응**시키면 된다는 것도 이해했다. 마지막으로 **최종 평가는 Accuracy만이 아니라 Popularity Bias/Novelty도 함께 봐야 한다**는 것을 정리했다.

> **오늘은 기존 Hybrid의 Retriever와 BPR 튜닝을 사실상 마무리하고, "후보를 더 찾는 문제"에서 "이미 찾은 후보를 더 잘 정렬하는 문제"로 초점을 옮겨 XGBoost Learning-to-Rank의 Feature·학습·평가·실제 서비스 구조까지 설계한 날이었다.**

---

## 12. Day25 종료 시점 프로젝트 상태

```
[완료] Hybrid Architecture 탐색
[완료] Candidate Size 탐색
[완료] Retriever Ratio Fine-Tuning (56/39/5 유지)
[완료] BPR Hyperparameter Search (27개 조합) → Final BPR 재확정 (factors=60/reg=0.006/iter=15)
[완료] Retrieval ≠ Ranking 재확인 및 LTR 도입 결정
[완료] LTR 원리 학습, Feature 구조 설계 (Score+Rank 8개)
[완료] Feature 추가 Exp1/2/3 계획 (8→10→12개)
[완료] LTR DataFrame 구조, 학습/평가 사용자 분리 설계 (1,000명 학습 vs 기존 400명 평가)
[완료] XGBoost 코드 구성 및 환경 문제 해결
[완료] 신규 사용자 처리 설계 (Item Embedding 평균 + BPR Fine-Tuning 결합)
[완료] 최종 recommender 구조 및 main.py 통합 방향 설계
[미완료] LTR 실제 실행 및 결과 (Exp1/2/3)
[미완료] MAP@10 / Popularity / Novelty 구현
[미완료] 최종 시스템 통합 (main.py, 신규 사용자 파이프라인)
[미완료] README 최종 최신화, 프로젝트 최종 회고
```

## 13. 다음 계획

1. XGBoost 환경 완료 확인
2. **LTR Experiment 1**: 4개 model score(정규화) + 4개 model rank = 8 Features로 학습, 기존 Item-Based Ranker와 비교
3. Exp1이 가능성을 보이면 **Experiment 2**: `source_count`, `user_interaction_count` 추가 (10 Features)
4. **Experiment 3**: `game_popularity`, `game_positive_ratio` 추가 (12 Features, Train 데이터만으로 계산)
5. 필요할 경우에만 genre/tag 등 Content 관련 Feature 추가
6. 최종 LTR Feature Set 결정
7. **Item-Based Ranker vs Final XGBoost Ranker** 비교 후 최종 Ranker 확정 및 모델 저장
8. 평가 시스템 확장 (MAP@10, Mean Log Popularity@10, Novelty@10)
9. 최종 recommender class 구현
10. `main.py` 연결: 사용자 게임 입력 → 신규 User Factor 초기화 → User-only BPR Fine-Tuning → BPR/Content/User Candidate → Final Ranker → Top-10 출력
11. End-to-End 실제 사용 테스트
12. 최종 정량/정성 평가
13. README / Docs / 최종 회고 정리