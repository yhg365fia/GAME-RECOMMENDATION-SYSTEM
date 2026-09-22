# 🎮 Steam Game Recommendation System

Steam 게임 데이터를 활용하여 **Content-Based / Collaborative Filtering / Matrix Factorization / Pairwise Ranking / Hybrid Recommendation**을 단계적으로 구현하고, 동일한 평가 환경에서 추천 구조와 성능을 비교하는 프로젝트입니다.

현재까지 다음 작업을 완료했습니다.

- Steam 메타데이터 전처리 및 Parquet 캐싱
- User-based / Global Train-Test Split 설계
- TF-IDF + Cosine Similarity 기반 Content-Based Recommendation
- User-Based / Item-Based Collaborative Filtering 구현 및 정량·정성평가
- Precision@K / Recall@K / Hit Rate@K / NDCG@K + Macro/Micro 평가 시스템 구축
- User-Based CF 데이터 누수 버그 발견·수정 및 Sparsity 한계 정량 검증
- Item-Based CF의 강한 co-consumption signal 확인 및 Standalone 최고 성능 확보
- Funk SVD 구현·평가·Ablation 후 `rating prediction objective ≠ Top-N ranking objective`로 단계 종료
- BPR 직접 구현 → `implicit` 기반 37M 전체 학습 → Explicit False Fine-Tuning → 최종 튜닝 완료 (**Final BPR**: P@10 0.05225 / HR@10 0.3850 / NDCG@10 0.06968)
- Hybrid 전용 동일 평가환경 구축 — Global MF Split + 동일 평가 사용자 400명 + `positive_only=True`
- Hybrid Item-Based Baseline 구축 — Hits 253
- **Case 1 (Item→BPR Reranking) / Case 1 Reverse (BPR→Item Reranking) 완료** — 각각 Hits 278
- **Case 2 (Multi-Retriever→BPR Ranking) 1차 실험 완료 → BPR 자기 재랭킹(self-ranking) 문제 발견**
- **Case 3 (4-Model Weighted Rank Fusion) 완료 및 Ablation 수행** — Item-Based가 압도적 기여, User-Based는 5% 비중일 때 최적
- **Cross-Ranker 실험으로 Ranker를 Retriever 후보 생성에서 제외** → Item-Based가 최종 Ranker로 확정
- **Retriever 비율 Sweep + User-Based 5% 추가 검증** → BPR56+Content39+User5 → Item Ranker 조합이 현재까지 최고 성능
- **Case 2 vs Case 3 최종 비교** → Retriever→Ranker 구조(Case 2 개선판)가 모든 지표에서 Fusion(Case 3)을 상회 → **Hybrid Architecture 반확정**
- **Candidate Size Sweep (50/100/150/200)** → Candidate Size 100이 최적점
- Retrieval/Neighbor 결과를 캐싱해 반복 실험 속도를 대폭 단축 (18분대 실험 → 수초~2분대)

현재는 새로운 Hybrid 구조를 계속 만드는 단계가 아니라, **반확정된 `BPR56+Content39+User5 → Item Ranker` 구조를 기준으로 미세 조정하고, Learning-to-Rank(XGBoost/LambdaMART) 적용 가능성을 검토한 뒤 최종 성능을 확정하는 단계**입니다.

향후에는 아래 **Hybrid Architecture 확정 → Learning-to-Rank 검토 → 최종 튜닝 → 최종 성능 분석 → 보고서/README 정리 → 프로젝트 종료 → FastAPI/Streamlit 배포**까지 확장하는 것을 목표로 합니다.

---

# 📊 전체 모델 성능 비교 (Top-10, 평가 유저 400명)

Standalone 모델, Hybrid Architecture 탐색 전체 과정, 그리고 현재 반확정된 최종 구조를 순서대로 비교합니다.

## Standalone Model Comparison

| 모델 | P@10 | R@10 | HR@10 | NDCG@10 | 비고 |
|---|---:|---:|---:|---:|---|
| Content-Based (TF-IDF) | 0.0268 | 0.0238 | 0.2250 | 0.0286 | 콘텐츠 semantic 유사성 |
| User-Based CF | 0.0545 | 0.0427 | 0.3011 | 0.0655 | User neighborhood, Sparsity에 취약 |
| **Item-Based CF** | **0.0783** | **0.0880** | **0.4800** | **0.1078** | **Standalone 최고 성능**, 강한 co-consumption signal |
| Funk SVD (biased) | 0.0003 | 0.0006 | 0.0025 | 0.0004 | item bias가 랭킹 지배 |
| Funk SVD (unbiased) | 0.0037 | 0.0030 | 0.0350 | 0.0042 | bias 제거로 개선되나 절대 성능은 낮음 |
| **Final BPR** | **0.05225** | **0.06870** | **0.3850** | **0.06968** | Explicit False 활용 + 최종 튜닝 |

**핵심 관찰**: 추천 성능은 모델 복잡도보다 **데이터 구조 + 추천 목적 + 모델이 사용하는 signal**에 크게 좌우되었습니다. Hybrid 단계에서는 Standalone 순위보다 **각 모델이 다른 모델이 놓친 정답을 실제로 보완하는가**를 더 중요하게 봤습니다.

### Hybrid 전용 Item-Based Baseline

| Metric | Hybrid Baseline |
|---|---:|
| Precision@10 | 0.06325 |
| Recall@10 | 0.08144 |
| Hit Rate@10 | 0.3925 |
| NDCG@10 | 0.09129 |
| Hits | 253 |

Hybrid 실험은 모두 **같은 Global Split / 같은 400명 / 같은 Positive 정의 / 같은 Top-K / 같은 Evaluator**를 사용하며, 이 baseline(Hits 253)을 기준으로 모든 Architecture를 비교합니다.

## Hybrid Architecture 전체 실험 비교

| 단계 | 구조 | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Item-Based 단독 | 0.0633 | 0.0814 | 0.3925 | 0.0913 | 253 |
| Case 1 | Item Top-30 → BPR Rerank | 0.0695 | 0.0888 | 0.4525 | 0.0978 | 278 |
| Case 1 Reverse | BPR Top-30 → Item Rerank | 0.0695 | 0.0874 | 0.4675 | 0.0977 | 278 |
| Case 2 (1차) | Multi-Retriever → BPR Ranking | 0.0673 | 0.0862 | 0.4425 | 0.0953 | 269 |
| Case 3 (초기 가중치) | 4-Model Weighted Rank Fusion | 0.0775 | 0.1001 | 0.4675 | 0.1134 | 310 |
| Case 3 (User 5% 튜닝) | Item .5588+BPR .2235+Content .1676+User .05 Fusion | 0.0800 | 0.1027 | 0.4825 | 0.1148 | 320 |
| Cross-Ranker 최적 | BPR59+Content41 Candidate → **Item Ranker** | 0.0827 | 0.1061 | 0.5225 | 0.1168 | 330 |
| **Case 2 최종 (반확정)** | **BPR56+Content39+User5 → Item Ranker** | **0.0872** | **0.1110** | **0.5400** | **0.1216** | **348** |

> Case 2는 1차 실험에서 BPR을 Retriever와 Ranker에 동시에 써서(자기 재랭킹 문제) 성능이 저평가되었고, Cross-Ranker 실험으로 Ranker를 Item-Based로 바꾸면서 크게 개선되었습니다. 자세한 경위는 아래 「Hybrid Recommendation 실험 결과 및 설계 방향」 섹션을 참고하세요.

## 최종 최적 성능표 (현재 반확정 구조)

```text
[Candidate Retrieval]  BPR 56 + Content-Based 39 + User-Based 5   (Candidate Budget = 100)
        ↓
[Final Ranker]          Item-Based
        ↓
[Output]                 Top-10
```

| Metric | Item-Based Baseline | **현재 반확정 최종 구조** | 개선폭 |
|---|---:|---:|---:|
| Precision@10 | 0.06325 | **0.08722** | +37.9% |
| Recall@10 | 0.08144 | **0.11102** | +36.3% |
| Hit Rate@10 | 0.3925 | **0.5400** | +37.6% |
| NDCG@10 | 0.09129 | **0.12155** | +33.2% |
| Hits | 253 | **348** | +95 |

> 이 구조는 완전한 최종 확정이 아니라 **반확정(semi-confirmed) Final Baseline Architecture**입니다. 이후 미세 조정과 Learning-to-Rank 검토를 거쳐 최종 확정할 예정입니다.

---

# 📌 Project Goals

## ✅ Current

- Steam 메타데이터 전처리 / Global Train-Test Split / Parquet 캐싱 / 프로젝트 모듈화
- Content-Based / User-Based / Item-Based CF 구현 및 정량·정성평가 완료
- Funk SVD 구현·평가·원인 진단·Ablation 완료 및 단계 종료
- BPR 전체 학습 / Explicit False Fine-Tuning / 공정 재평가 / 최종 튜닝 완료 (**Final BPR 확정**)
- Item-Based / User-Based / BPR의 Collaborative 관계 및 latent preference 차이 재정리
- Hybrid 평가 사용자 400명 고정 및 Item-Based Baseline 구축
- **Case 1 / Case 1 Reverse / Case 2 / Case 3 구현·평가 완료**
- **Case 3 Ablation 및 User-Based Weight Sweep 완료** (User 5%가 최적)
- **Case 2의 BPR 자기 재랭킹 문제 발견 및 Cross-Ranker 실험으로 해결**
- **Retriever 비율 Sweep + Candidate Size Sweep 완료**
- **Hybrid Architecture 반확정** (BPR56+Content39+User5 → Item Ranker, Hits 348)
- Hybrid Retrieval/Neighbor Cache 구조 구축으로 반복 실험 속도 대폭 단축

---

## 🚧 In Progress

- 반확정 구조의 **미세 조정** (Retriever 비율 소폭 조정, Item Ranker 파라미터, 필요 최소 Ablation)
- **Learning-to-Rank(XGBoost Ranker / LambdaMART) 원리 학습 및 적용 가능성 검토**

---

## 🚀 Future

- Learning-to-Rank 적용 시 기존 Item Ranker 대비 개선 여부 확인 (Candidate Size 150/200에서 확보된 여유 정답을 학습 기반 Ranker가 더 잘 끌어올릴 수 있는지 검증)
- 최종 성능 튜닝 (구조 확정 후 hyperparameter/candidate size/ranking parameter 최종 조정, 새 모델·구조 추가는 하지 않음)
- 최종 정량·정성평가 (Precision/Recall/HR/NDCG, 사용자 sparsity group별 성능, Candidate/Ranking 역할, Baseline 대비 개선, 모델별 기여)
- MAP@K, Popularity Baseline / Popularity Bias / Coverage 분석
- 프로젝트 최종 회고, README/GitHub 최종 정리, 프로젝트 공식 종료
- 가족 대상 발표 준비·실시
- FastAPI & Streamlit Deployment (선택적, 구조 확정 이후)

---

# 🛠 Tech Stack

## Language
- Python

## Data Processing
- Pandas / NumPy / PyArrow (Parquet 캐싱) / SciPy (Sparse Matrix, CSR/LIL)

## Machine Learning
- Scikit-learn
- **Surprise** (Funk SVD 구현)
- **implicit** (대규모 BPR 학습 — Cython/C 최적화, CSR 기반, 멀티코어)

### Algorithms
- TF-IDF Vectorization / Cosine Similarity
- User-Based CF (Neighborhood-based, Top-K)
- Item-Based CF (행별 Top-K, Similarity Sum Aggregation)
- **Matrix Factorization / Funk SVD** (Surprise `SVD`, biased/unbiased 실험 완료 — 단계 종료)
- **BPR (Bayesian Personalized Ranking)** — 직접 구현(원리 이해용) + `implicit`(전체 학습용), Explicit Negative Fine-Tuning(`True > False`) 자체 구현 — 단계 완전 종료
- **Hybrid Recommendation** — Reranking(Case 1) / Multi-Retriever(Case 2) / Weighted Rank Fusion(Case 3) / Cross-Ranker 구조 모두 구현·비교 완료
- **Learning-to-Rank (검토 중)** — XGBoost Ranker / LambdaMART

## Evaluation
- Precision@K / Recall@K / Hit Rate@K / NDCG@K (Macro) + Micro Precision/Recall/F1
- Stratified Sampling (리뷰 수 구간 기반), Qualitative Experiment
- Leave-N-Out 유저별 Split (Memory-based CF) / **Global Train/Test Split (Model-Based CF, Hybrid 공통)**
- Pearson Correlation 기반 가설 검증
- 추천 게임 단위 Train 통계 분석(`groupby("app_id")`), `Counter` 기반 반복 추천 빈도 분석
- **Candidate Recall / Scoreable Recall / Hit Overlap / Unique Hit / Recovered Hit / Lost Hit 분석**

## Visualization / Deployment
- Matplotlib / FastAPI / Streamlit

---

# 📂 Project Structure

```text
Game-Recommendation-System/
│
├── data/                                      # Git 제외
│   ├── raw/
│   ├── cache/                                 # 전처리/로딩 결과 캐싱 (Parquet)
│   └── split/                                 # Global MF Train/Test
│       ├── mf_train.parquet
│       └── mf_test.parquet
│
├── docs/
│   ├── Day01.md
│   ├── ...
│   ├── Day23.md                               # Case 1/1Rev/2/3 구현 및 초기 비교
│   └── Day24.md                               # Ablation, Cross-Ranker, 구조 반확정
│
├── hybrid_arctech_experiment/
│   ├── baseline_item.py                       # Hybrid Item-Based 기준선
│   ├── exp_a_reranking.py                     # Case 1 / Case 1 Reverse
│   ├── exp_b_multi_retriever.py               # Case 2 (Cross-Ranker 포함)
│   └── exp_c_fusion.py                        # Case 3 및 Ablation / Weight Sweep
│
├── models/
│   ├── content_base.py
│   ├── userbase.py
│   ├── itembase.py
│   ├── Funk_SVD.py
│   ├── bpr.py
│   ├── hybrid.py                              # 최종 Hybrid 구현용
│   │
│   ├── run_model/
│   │   ├── runsvd.py
│   │   └── run_bpr.py
│   │
│   └── saved_model/                           # Git 제외
│       ├── bpr_grid/                          # BPR 최종/실험 모델
│       ├── hybrid_cache/                      # Item neighbor / BPR candidate cache
│       ├── case3_cache/                       # Item/User/BPR/Content retrieval cache
│       └── results/
│           ├── BPR / Hybrid 실험 결과
│           ├── hybrid_sampled_users.csv
│           ├── item_top200_candidates.csv
│           └── Architecture별 평가 CSV
│
├── notebook/
├── preprocessing.py
├── data_split.py
├── evaluation.py
├── main.py
├── Readme.md
├── requirements.txt
└── .gitignore
```

`models/saved_model/` 내부의 대형 모델과 cache는 Git에서 제외하며, 반복 실험 시 로컬에서 재사용합니다. `models/hybrid.py`는 최종 Architecture가 확정된 뒤 선택된 구조(현재 반확정: BPR+Content+User Retriever → Item Ranker)를 정식 구현하기 위한 파일입니다.

---

# ⚙️ Current Recommendation Pipeline

## Content-Based Pipeline

```text
Raw Steam Dataset → Data Loading (Parquet Cache) → Data Validation
      ↓
User-based Train/Test Split (런타임 유저 단위 적용)
      ↓
Data Preprocessing (Name Dedup, AppID 기준 정리) → Combined Features
      ↓
TF-IDF Vectorization → Name → AppID → Index 변환
      ↓
Cosine Similarity → Top-N Recommendation (AppID 기준 중복 제거)
      ↓
Recommendation Evaluation (정량 지표 + 정성적 실험)
```

## User-Based CF Pipeline

```text
User History (user_id, app_id, is_recommended)
      ↓
build_interaction_matrix() → Sparse User x Item Matrix (+1 / -1 인코딩)
      ↓
유저별 Train/Test Split (Leave-N-Out, 70/30)
      ↓
Query Vector 생성 (Train App ID만 사용, Test 누수 방지)
      ↓
Cosine Similarity → 자기 자신 제외(exclude_user_idx) → Top-K Neighbor (k=30)
      ↓
이웃 가중합 예측 점수 (Σ 유사도×상호작용 / Σ|유사도|)
      ↓
이미 플레이한 게임 제외 + 양수 점수만 채택 → Top-N 추천
      ↓
Recommendation Evaluation (구간별 Breakdown 포함)
```

## Item-Based CF Pipeline

```text
build_interaction_matrix() (User-Based와 동일 함수 재사용)
      ↓
interaction_matrix.T → item_matrix (Item x User)
      ↓
유저의 Train App ID → Source Items 행 조회 (Query Vector 불필요)
      ↓
Cosine Similarity (Source Items vs 전체 item_matrix)
      ↓
Self-Similarity 0 처리 → Source Item별 행 단위 Positive Top-K (k=30)
      ↓
동일 Candidate Similarity Sum Aggregation
      ↓
Train Item 제외 (-np.inf) → Positive Score만 채택 → Top-N 반환
      ↓
Recommendation Evaluation (기존 evaluation.py 재사용)
```

> **Hybrid에서 Item-Based는 Candidate Retrieval이 아니라 Final Ranker로 확정되었습니다.** Cross-Ranker 실험 결과, Candidate Recall이 가장 높은 조합(Content Ranker)보다 Item-Based를 Ranker로 쓴 조합의 최종 성능이 압도적으로 높았기 때문입니다 (자세한 내용은 아래 섹션 참고).

## Model-Based CF (Funk SVD) Pipeline — 완료·단계 종료

```text
Global Train/Test Split → mf_train.parquet (37,113,471건) / mf_test.parquet (4,041,323건)
      ↓
Surprise Dataset 변환 → SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02).fit()
      ↓
recommend() — bias 반영 score, seen-item 제거, 벡터화 Top-N
      ↓
400명 평가: Precision@10 0.0003, Hits 1/4000
      ↓
[가설1] positive_only=False 재평가 → Hits 그대로 1 → 기각
      ↓
[가설2] 추천 게임 Train 통계 + 반복 추천 분석 → item bias(b_i) 지배 패턴 발견
      ↓
[Ablation] biased=False → Hit Rate 0.0025→0.0350 개선, 그러나 여전히 낮음
      ↓
rating-vs-ranking objective mismatch가 근본 원인 → Funk SVD 단계 종료
```

## Model-Based CF (BPR) Pipeline — 완료·단계 종료

```text
[1단계] True interaction만 사용 → implicit BPR 학습 (True > Unseen)
  iterations=15 / factors=60 / learning_rate=0.05 / regularization=0.006
      ↓
[2단계] Explicit Negative Fine-Tuning (True item > False item, epochs=1 / lr=0.01 / reg=0.001)
      ↓
positive_only=True 조건으로 공정 재평가 → Regularization/Factors/상호작용/Local Search
      ↓
Final BPR 확정 → predict_all(user) → seen item 제외 → Top-K 추출
```

> **Final BPR**: P@10 0.05225 / R@10 0.06870 / HR@10 0.3850 / NDCG@10 0.06968 / Hits 209. Standalone BPR 단계는 완전히 종료했으며, Hybrid에서는 BPR을 **Candidate Retriever**(56% 비중)로 사용합니다. BPR을 Retriever와 Ranker에 동시에 쓰면 두 단계가 사실상 중복된다는 것을 Case 2 실험에서 확인해, 현재 구조에서는 BPR을 Ranker로 쓰지 않습니다.

---

# 🧩 Recommendation Identifier Flow

```text
Game Name (사용자 입력) → [동명이인 시 후보 선택] → AppID (고유 식별자)
      ↓ game_to_idx
Index (tfidf_matrix 상의 위치) → TF-IDF Vector → Cosine Similarity
      ↓
추천 Index → AppID → Game Name
```

> Game Name은 중복될 수 있지만 AppID는 고유하므로 내부 로직은 전부 **AppID 기준**으로 동작합니다. 모든 모델과 Hybrid가 동일한 원칙을 유지합니다.

---

# 🔍 Qualitative Experiment Findings (Content-Based)

### 발견 1 — 텍스트에 없는 특성은 포착 불가
`Party Animals` 입력 시 장르/인원수는 유사했지만 "동물 캐릭터" 테마는 전혀 반영되지 않음. TF-IDF는 텍스트 메타데이터에 명시된 정보만 학습하므로 비주얼/테마적 특성은 원천적으로 포착 불가.

### 발견 2 — 장르 혼합 시 쏠림 현상
카드/덱빌딩 + 슈팅을 함께 입력했을 때 추천이 카드/덱빌딩 계열로 완전히 쏠리고 슈팅은 하나도 포함되지 않음.

---

# 🐛 User-Based CF 데이터 누수 진단 및 수정

첫 평가에서 Precision@10이 **0.6254**로 비정상적으로 높게 나와 원인을 진단했습니다.

**원인**: 평가용 `interaction_matrix`가 Train/Test 미분리 원본으로 생성되어, 평가 대상 유저 본인의 전체 데이터가 이웃 후보 풀에 남아 Test 정답이 예측 점수에 직접 새어 들어감.

**수정 과정**: `exclude_user_idx` 파라미터 추가 → 1차 수정 후에도 지표 동일 → 호출 체인 추적 → `run_evaluation()` 호출 시 `user_to_idx` 인자 전달 누락으로 제외 로직이 무력화된 것 확인 → 수정 후 Precision@10 0.0545로 정상화.

**교훈**: 여러 파일/함수로 나뉜 파이프라인에서는 내부 로직보다 **호출 체인(인자 전달 경로) 검증을 우선시**하는 디버깅 원칙(Outside-In)을 세우는 계기가 되었습니다.

---

# 🐛 NDCG 계산 구현 오류 수정

추천 결과를 `set(result["app_id"])`로 바로 변환해 **추천 순위 정보가 소실**되고 있는 것을 발견. `recommended_list`(순서 보존, NDCG용)와 `recommended_ids`(집합, Precision/Recall용)로 역할을 분리해 수정했고, 수정 후 NDCG@10은 **0.0655**로 확인했습니다.

---

# 📐 Macro vs Micro 평가 및 Sparsity 정량 검증

`n_recommended`가 유저마다 크게 달라(평균 7.28/10) Macro 평균에 편향이 생길 수 있어 **Micro Precision/Recall/F1**을 추가 도입했습니다.

| 비교 변수 | Pearson r | p-value |
|---|---|---|
| n_games ↔ n_recommended | +0.3096 | < 0.0001 |
| n_games ↔ precision | +0.2504 | < 0.0001 |

review_group별 추이와 함께 **"interaction이 적을수록 안정적인 이웃 형성이 어렵다"는 가설과 일관된 방향**으로 해석했습니다.

---

# 🔗 Self-Similarity 처리 원칙 (User-Based / Item-Based 공통)

- **User-Based CF**: 본인의 interaction(Test 포함)이 neighbor로 사용되는 **데이터 누수** 문제
- **Item-Based CF**: `A↔A=1.0`이 항상 최고 유사도로 Top-K를 차지하는 **trivial self-similarity** 문제

```python
similarities = cosine_similarity(...)
similarities[self_idx] = 0
# 이후 Top-K 선정
```

---

# 🧠 Item-Based CF 구조 이해 및 구현

| 구분 | User-Based CF | Item-Based CF |
|---|---|---|
| Matrix 방향 | User × Item | Item × User (`interaction_matrix.T`) |
| Query 필요 여부 | 필요 | 불필요 (Item이 이미 행으로 존재) |
| Similarity 대상 | User ↔ User | Item ↔ Item |
| Candidate Score | User similarity × interaction의 weighted prediction | 동일 candidate로 들어오는 item similarity 합산 |

**구현 완료 사항**: Self-similarity 0 처리 → Source Item별 행 단위 Positive Top-K(k=30) → Candidate Aggregation → Train Item Exclusion 및 Positive Score Filtering → Top-N 반환.

---

# 🧮 Model-Based CF: Funk SVD → BPR (완료·종료)

### Memory-based CF와의 핵심 차이

| 구분 | Memory-based CF | Model-Based CF |
|---|---|---|
| User/Item 표현 | 이미 존재하는 interaction 벡터 | 데이터로부터 학습된 latent vector |
| 학습 단위 | 유저 평가 시점마다 개별 계산 | 전체 interaction을 한 번에 학습하는 Global Model |

### Global Train/Test 설계

- 평가 대상 유저(interaction 10~78개, **666,781명**): 70% Train / 30% Test, 그 외 유저: 100% Train
- 전체 41,154,794건 = Train 37,113,471건 + Test 4,041,323건
- `train_test_split()` 호출을 666,781회 → 최대 69회로 최적화 (interaction 개수별 position 재사용, 약 5초)

### Funk SVD — 원인 진단과 최종 결론

**원인 진단**: 가설 1(`positive_only` 정답 축소)은 재평가로 기각, 가설 2(item bias 지배)는 추천 게임 대부분이 True Ratio 0.94~1.00인 패턴으로 확보.

**Unbiased Ablation 결과** (400명, Top-10): Precision@10 0.0037 / Recall@10 0.0030 / Hit Rate@10 0.0350 / NDCG@10 0.0042. Hit Rate가 약 14배 개선되어 bias의 영향은 확인했지만 절대 수치는 여전히 매우 낮아, **rating-prediction objective와 Top-N ranking objective의 근본적 불일치**를 최종 원인으로 결론짓고 단계를 종료했습니다.

### BPR — 이론, 전체 학습, 파라미터 최적화

**이론**: `(u,i,j)` triplet으로 positive가 negative보다 높은 점수를 갖도록 직접 학습합니다.

$$ x_{uij} = p_u^T(q_i-q_j), \quad L = -\log\sigma(x_{uij}) + \lambda(\|p_u\|^2+\|q_i\|^2+\|q_j\|^2) $$

**직접 구현 vs implicit**: 직접 구현은 원리 이해용, `implicit`(Cython/C 최적화, CSR 기반, 멀티코어)은 대규모 데이터 학습용으로 역할을 구분했습니다.

**False signal 문제**: `implicit`은 non-zero 여부만 보기 때문에 `is_recommended=False`를 positive처럼 취급하고 있었습니다. `1단계: True > Unseen` → `2단계: True > False (Explicit Negative Fine-Tuning)` 구조로 해결했습니다.

**파라미터 최적화 히스토리** (모두 `positive_only=True`로 공정 비교):

| 실험 | 핵심 결과 |
|---|---|
| Explicit False 효과 (15 iter) | P@10 0.02525→0.03650, Hits 101→146 |
| Regularization (factors=40 고정) | 0.0001~0.020 중 **0.005**에서 전 지표 최고 |
| Factors (reg=0.005 고정) | 20/40/80 중 metric별 트레이드오프 확인 (NDCG는 80, 나머지는 40 유리) |
| Factors×Reg 상호작용 | **60/0.005**가 20~100 범위에서 전 지표 동시 최고 |
| Local Search (55~65 × 0.005~0.008) | **60/0.006**이 최종 최고 (P@10 0.05225, Hits 209) |

**Final BPR**:

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
```

BPR 단계에서 얻은 핵심 결론: **Problem Definition > Data Signal > Hyperparameter**. `is_recommended=False`를 모델이 어떤 의미로 받아들이는지 바로잡는 것이 factors/regularization/iteration 미세 조정보다 먼저였습니다.

---

# 🧩 Hybrid Recommendation 실험 결과 및 설계 방향

Hybrid 단계에서는 Standalone 점수가 높은 모델을 단순히 섞는 것이 아니라, **같은 interaction을 각 모델이 어떤 관계로 표현하고, Retrieval/Ranking/Fusion 각 단계에서 다른 모델의 실패를 실제로 얼마나 보완하는가**를 중심으로 설계했습니다.

### Collaborative 모델 관계 재해석

| 모델 | 직접적으로 보는 관계 | 핵심 질문 |
|---|---|---|
| **Item-Based** | Item ↔ Item | 이 게임과 같이 소비되는 게임은? |
| **User-Based** | User ↔ User | 이 사용자와 비슷한 사용자는? |
| **BPR** | User ↔ Item latent preference | 이 사용자에게 어떤 item을 더 높은 순위에 둬야 하는가? |
| **Content-Based** | Item metadata ↔ User profile | 행동 데이터와 다른 semantic signal을 제공하는가? |

### Hybrid 공통 평가환경

Global MF Train/Test Split 재사용, 동일 평가 사용자 400명(리뷰 수 구간별 100명), `positive_only=True`, 최종 추천 Top-10, 동일 `evaluation.py`. Hybrid Baseline은 Item-Based Top-10의 **Hits 253**입니다.

### Case 1 — Item-Based Candidate → BPR Reranking (및 Reverse)

```text
Item-Based Top-30 → BPR Reranking → Top-10     (Hits 253→278)
BPR Top-30 → Item-Based Reranking → Top-10     (Hits 209→278)
```

두 방향 모두 자체 Top-10이 놓친 정답을 후보(Top-30) 안에서 재발견했습니다. Candidate Generation 능력만 비교하면 **Item-Based(Recall@30 0.1491, Hits 468)가 BPR(Recall@30 0.1350, Hits 422)보다 강했습니다.**

Reverse 실험에서 Item-Based가 BPR 후보 30개만 재정렬함에도 느렸던 이유는, scoring 정의상 각 source item의 Top-K neighbor를 **전체 Item 공간**에서 찾아야 했기 때문이었습니다. Item→Item Top-K 캐싱, unique source item(3,321개)만 계산, norm 사전계산, batch sparse multiplication으로 최적화해 평가 시간을 1847.7초→234.7초(캐시 구축 포함), cached reranking 자체는 0.3초까지 단축했습니다.

### Case 2 — Multi-Retriever → Ranking, 그리고 self-ranking 문제

```text
Item-Based + User-Based + Content Candidate → UNION → BPR Ranking → Top-10
```

1차 실험 결과 Candidate Recall은 Case 1보다 높아졌지만(0.2045) 최종 Hits는 269로 오히려 낮아졌습니다 — **좋은 Candidate Retrieval과 좋은 Final Ranking은 별개의 문제**임을 확인했습니다.

이후 BPR을 Retriever로도 사용한 변형에서 더 심각한 구조적 문제를 발견했습니다: **BPR Retriever가 이미 BPR 기준 고득점 후보를 가져오고, 그 안에 BPR의 global Top10이 포함된 상태에서 동일 BPR score로 다시 랭킹하면 결과가 Pure BPR Top10과 100% 동일**해집니다. 즉 같은 모델을 Retriever와 Ranker에 동시에 쓰면 두 단계가 사실상 중복될 수 있습니다.

### Cross-Ranker 실험 — Ranker를 후보 생성에서 제외

이 문제를 해결하기 위해 **Ranker 모델은 Retriever 후보 생성에 참여하지 않는** Cross-Ranker 구조로 전환했습니다.

| Retriever | Ranker | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---|---:|---:|---:|---:|---:|
| **BPR59 + Content41** | **Item-Based** | **0.0827** | **0.1061** | **0.5225** | **0.1168** | **330** |
| Item71 + BPR29 | Content-Based | 0.0600 | 0.0799 | 0.4275 | 0.0841 | 240 |
| Item78 + Content22 | BPR | 0.0705 | 0.0880 | 0.4750 | 0.0963 | 282 |

Candidate Recall은 Item+BPR→Content 조합(0.2912)이 가장 높았지만, 최종 성능은 **BPR+Content→Item Ranker**가 압도적으로 좋았습니다 — 후보에 정답이 많은 것과 좋은 Top10을 만드는 것은 별개라는 것을 다시 확인했고, **Item-Based를 최종 Ranker로 확정**했습니다.

### Retriever 비율 Sweep + User-Based 5% 효과

Item Ranker를 고정하고 BPR/Content 비율 및 User-Based 5% 추가 효과를 sweep했습니다.

| 후보 구성 | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---:|---:|---:|---:|---:|
| BPR59 + Content41 | 0.0827 | 0.1061 | 0.5225 | 0.1168 | 330 |
| BPR70 + Content30 | 0.0801 | 0.1028 | 0.5025 | 0.1143 | 320 |
| BPR50 + Content50 | 0.0819 | 0.1052 | 0.5175 | 0.1168 | 326 |
| BPR40 + Content60 | 0.0809 | 0.1032 | 0.5050 | 0.1135 | 322 |
| **BPR56 + Content39 + User5** | **0.0872** | **0.1110** | **0.5400** | **0.1216** | **348** |
| BPR67 + Content28 + User5 | 0.0849 | 0.1079 | 0.5275 | 0.1196 | 339 |
| BPR48 + Content47 + User5 | 0.0850 | 0.1094 | 0.5250 | 0.1199 | 339 |

User-Based를 5% 비중으로 넣자 모든 조합에서 지표가 상승했고(Unique Hits 44~45개), **User-Based는 메인 모델이 아니라 BPR/Content가 놓치는 후보를 보충하는 weak auxiliary retriever**로 역할이 재정의되었습니다.

### Case 3 (Weighted Rank Fusion) Ablation

Case 3 기존 가중치(Item 0.50/User 0.15/BPR 0.20/Content 0.15)에서 각 모델을 하나씩 제거한 결과:

| 실험 | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---|---:|---:|---:|---:|---:|
| Full | 0.0775 | 0.1001 | 0.4675 | 0.1134 | 310 |
| - Item | 0.0598 | 0.0774 | 0.4200 | 0.0881 | 239 |
| - User | 0.0793 | 0.1019 | 0.4800 | 0.1125 | 317 |
| - BPR | 0.0683 | 0.0874 | 0.4275 | 0.1023 | 273 |
| - Content | 0.0753 | 0.0963 | 0.4575 | 0.1087 | 301 |

기여도는 **Item-Based >> BPR > Content-Based >> User-Based** 순으로 나타났고, User Weight를 0.05로 낮춘 뒤 재평가하니 Fusion 성능도 개선(Hits 310→320)되었습니다. 다만 동일 조건에서 Retriever→Ranker 구조(Case 2 최종, Hits 348)가 Fusion(Case 3, Hits 320)보다 모든 지표에서 높아, **최종 방향을 Fusion이 아니라 Multi-Retriever→Ranker 구조로 확정**했습니다.

### Case 2 vs Case 3 구조적 차이

```text
Case 2: 여러 Retriever → Candidate UNION → 하나의 Ranker(Item-Based)가 최종 Ranking
        (Multi-Retriever + Single Ranker)

Case 3: 여러 Retriever → 각 모델의 Rank 신호까지 유지 → Weighted Fusion
        (Multi-Retriever + Multi-Signal Ranking)
```

Case 2에 모델을 더 추가해도 최종 Ranking이 하나의 모델뿐이라면 Case 3와 같아지지 않습니다. 핵심 차이는 **최종 Ranking을 하나의 모델이 독점하는지, 여러 모델의 판단을 함께 쓰는지**입니다.

### Candidate Size Sweep

Retriever 비율(56:39:5)을 고정하고 총 Candidate Size(50/100/150/200)를 비교했습니다.

| Size | Candidate Recall | Scoreable Recall | P@10 | R@10 | HR@10 | NDCG@10 | Hits |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.1710 | 0.1449 | 0.0780 | 0.0976 | 0.4875 | 0.1104 | 308 |
| **100** | 0.2488 | 0.2055 | **0.0872** | **0.1110** | **0.5400** | **0.1216** | **348** |
| 150 | 0.3005 | 0.2413 | 0.0835 | 0.1078 | 0.5300 | 0.1188 | 334 |
| 200 | **0.3386** | **0.2669** | 0.0840 | 0.1081 | 0.5225 | 0.1197 | 336 |

Candidate Recall은 후보를 늘릴수록 계속 올랐지만 최종 Recall은 100에서 정점을 찍고 소폭 하락했습니다 — **현재 Item Ranker가 100개를 초과하는 추가 후보를 Top10으로 충분히 끌어올리지 못한다**는 뜻이며, 이것이 Learning-to-Rank 검토의 근거가 되었습니다.

### Unique Hit / Complementarity 분석 (Case 3 기준)

| 모델 | Standalone Hits | Unique Hits |
|---|---:|---:|
| Item-Based | 253 | 130 |
| User-Based | 144 | 70 |
| BPR | 209 | 124 |
| Content-Based | 112 | 61 |
| 4개 모델 모두 공통 | - | 8 |

각 모델이 상당히 다른 정답을 잡고 있다는 것을 확인했고, 이는 여러 신호를 결합할 근거가 되었습니다. Fusion(Case 3)에서는 Item-Based Hits(253) 대비 새로 얻은 정답 115, 손실 58로 순증가 +57이었습니다.

### 현재 반확정 구조와 다음 단계

```text
[Candidate Retrieval] BPR 56 + Content-Based 39 + User-Based 5  (Budget=100)
        ↓
[Final Ranker] Item-Based
        ↓
[Output] Top-10
```

| Metric | Score |
|---|---:|
| Precision@10 | 0.08722 |
| Recall@10 | 0.11102 |
| Hit Rate@10 | 0.5400 |
| NDCG@10 | 0.12155 |
| Hits | 348 |

이 구조는 **반확정(semi-confirmed) Final Baseline**이며, 다음 단계는 ① 이 구조 내에서의 미세 조정, ② Learning-to-Rank(XGBoost/LambdaMART) 적용 검토, ③ 최종 튜닝, ④ 최종 성능 분석입니다. 새로운 Hybrid 구조를 계속 늘리는 탐색은 여기서 종료합니다.

---

# 🐛 알려진 이슈 (To-Do)

- **Name = NaN metadata mismatch**: Item-Based CF 정성평가 중 일부 게임 이름이 NaN으로 조회. 원인 분석 보류
- **Party Animals 입력 시 빈 DataFrame 반환**: 정성평가 대상에서 제외. 원인 분석 보류
- **`ModuleNotFoundError: No module named 'data_split'`**: 하위 폴더 직접 실행 시 import 경로 문제. Project Root import 구조로 해결 방향 정리
- **User-Based Candidate Coverage**: Hybrid 실험에서 400명 중 최대 362명만 후보 생성, 평균 후보 수가 적어(약 14.6개) Sparsity 영향이 여전히 큼
- **동일 모델을 Retriever와 Ranker에 함께 쓰면 두 단계가 사실상 중복될 수 있음**: BPR Retriever+BPR Ranker 조합에서 확인, Cross-Ranker 구조로 해결
- **Candidate Size를 무작정 늘리는 것이 능사가 아님**: 100 초과 시 Candidate Recall은 오르지만 최종 성능은 하락 — 현재 Item Ranker의 한계로 추정, Learning-to-Rank로 개선 여부 검토 예정
- **Hybrid Architecture는 아직 반확정 단계**: 미세 조정과 Learning-to-Rank 검토 이후 최종 확정 예정
- **실험 저장 경로 통일 필요**: `models/saved_model/`, `results/`, `case3_cache/`, `hybrid_cache/`의 Config Source 단일화 필요 (최종 리팩토링 예정)
- **Dependency 동기화 필요**: `implicit`, Surprise, PyArrow 등 실행 환경과 `requirements.txt` 최종 동기화 필요

---

# ✅ Implemented Features

## Content-Based
Steam Metadata Loading (Parquet Caching), Data Validation, User-based Train/Test Split, Combined Features, TF-IDF Vectorization, Cosine Similarity Recommendation, Multi-Game Recommendation, AppID 기반 식별 및 동명이인 처리, 정성적 실험

## User-Based CF
Sparse Interaction Matrix 구축, Cosine Similarity 기반 Top-K 이웃 탐색, 자기 자신 제외 로직, Query Vector 생성(Test 누수 방지), 이웃 가중합 예측 점수 계산, Sparsity 정량 검증

## Item-Based CF
User×Item → Item×User 구조 변환, Self-Similarity 0 처리, Source Item별 행 단위 Positive Top-K, Candidate Aggregation, Train Item Exclusion, 정량·정성평가, **Hybrid 최종 Ranker로 채택**

## Model-Based CF — Funk SVD (완료·종료)
Global Train/Test Split 설계 및 최적화, Surprise 기반 학습, `recommend()` Top-N 구현, MF 전용 Evaluation, 400명 정량평가(Biased/Unbiased), `positive_only` 가설 기각, item bias 가설 확보, `biased=False` Ablation 완료, objective mismatch 분석

## Model-Based CF — BPR (완료·종료)
BPR 이론 학습, `implicit` 기반 전체 데이터(37M) 학습, False signal 문제 발견 및 Explicit Negative Fine-Tuning, `positive_only=True` 공정 재평가, Regularization/Factors/상호작용/Local Search, **Final BPR 확정**, **Hybrid Candidate Retriever(56%)로 재사용**

## Hybrid Recommendation — Architecture 탐색 완료, 반확정 구조 미세 조정 중
- Item-Based Hybrid Baseline, Item-Based Top-200 Candidate Cache
- **Case 1 / Case 1 Reverse (Reranking)** 완료
- **Case 2 (Multi-Retriever)** 1차 실험 → self-ranking 문제 발견 → **Cross-Ranker**로 해결
- **Case 3 (Weighted Rank Fusion)** 완료 및 **Ablation / User Weight Sweep** 완료
- **Retriever 비율 Sweep(7종) + Candidate Size Sweep(4종)** 완료
- Common/Unique/Recovered/Lost Hit 분석
- **Hybrid Architecture 반확정**: BPR56+Content39+User5 → Item Ranker (Hits 348)
- Item→Item Neighbor Cache / BPR Candidate Cache / Case3 Retrieval Cache로 반복 실험 속도 최적화 (일부 실험 18분→수초)
- [진행 중] 반확정 구조 미세 조정, Learning-to-Rank(XGBoost/LambdaMART) 검토

## Evaluation (공통)
리뷰 수 구간 기반 층화 표집, Precision/Recall/Hit Rate/NDCG@K, Macro/Micro Precision·Recall·F1, Pearson Correlation, 구간별 Breakdown, 정성평가, Hybrid Candidate Recall/Scoreable Recall/Hit Overlap/Unique Hit/Recovered Hit/Lost Hit

---

# 🚀 Development Roadmap

## ✅ V1. Content-Based Recommendation
전체 완료

## ✅ V2. Collaborative Filtering

### User-Based CF / Item-Based CF
전체 완료

### Model-Based CF

#### 1. Funk SVD — 완료·종료
- [x] Memory-based vs Model-Based(Global Model) 학습 구조 차이 파악
- [x] Global Train/Test Split 설계 및 대규모 split 성능 최적화
- [x] Surprise 기반 구현, Top-N 추천 함수, evaluation 연결, 400명 정량평가
- [x] 성능 저조 원인 진단 및 `biased=False` Ablation
- [x] objective mismatch로 단계 종료

#### 2. BPR — 완료·종료
- [x] BPR pairwise ranking 구조 이해 및 소규모 구현
- [x] `implicit` 기반 전체 데이터(37M) 학습, iteration 비교
- [x] False explicit negative 문제 발견 및 Fine-Tuning 구조 구현
- [x] 동일 평가조건 공정 재평가, Regularization/Factors/상호작용/Local Search
- [x] **Final BPR 결정 및 단계 종료**

### Model Comparison & Selection
- [x] Content/User/Item/Funk SVD/BPR 성능 비교 및 signal·관계 구조 재해석
- [ ] Coverage / Personalization / Explainability / 실행시간 추가 비교

---

## 🚧 V3. Hybrid Recommendation — Architecture 탐색 완료

- [x] Hands-On Recommendation Hybrid 파트 학습, Item/User/BPR 관계 재해석
- [x] Hybrid 평가 사용자 400명 고정, Item-Based Hybrid Baseline
- [x] Case 1 (Item→BPR Reranking) / Case 1 Reverse 완료
- [x] Case 2 (Multi-Retriever→BPR Ranking) 1차 실험
- [x] Case 3 (4-Model Weighted Rank Fusion) 완료
- [x] Common/Unique/Recovered/Lost Hit 분석
- [x] **Case 3 Ablation + User-Based Weight Sweep** (User 5% 최적)
- [x] **BPR self-ranking 문제 발견 → Cross-Ranker 구조로 전환**
- [x] **Item-Based를 최종 Ranker로 확정**
- [x] **Retriever 비율 Sweep(7종) + User 5% 추가 검증**
- [x] **Case 2 vs Case 3 최종 비교 → Retriever→Ranker 구조 확정**
- [x] **Candidate Size Sweep(50/100/150/200) → 100 최적**
- [x] **Hybrid Architecture 반확정** (BPR56+Content39+User5 → Item Ranker)

### V3.5 — 반확정 구조 미세 조정 & Learning-to-Rank (진행 중)
- [ ] Retriever 비율 소폭 조정, Item Ranker 파라미터, 필요 최소 Ablation
- [ ] Learning-to-Rank 원리 학습 (XGBoost Ranker / LambdaMART)
- [ ] Candidate 150/200에서 확보된 여유 정답을 학습 기반 Ranker가 더 잘 활용하는지 검증
- [ ] 최종 Architecture 확정

### 추가 평가 (예정)
- [ ] MAP@K
- [ ] Popularity Baseline / Popularity Bias / Coverage
- [ ] Sparsity / Personalization / Explainability / 실행시간 비교

---

## 🚧 V4. 최종 평가 및 마무리

- [ ] 최종 성능 튜닝 (구조 고정 후 hyperparameter만 조정, 새 모델·구조 추가 없음)
- [ ] 최종 정량·정성평가 (단일 모델/Case별 비교, 그룹별 성능, Unique Hit 기여, 실행 비용)
- [ ] 최종 시스템 분석 (잘된 점/한계/기술적 한계 vs 데이터적 한계)
- [ ] 향후 추천시스템/ML 공부 방향 결정
- [ ] 프로젝트 최종 회고, README 최종 완성, 프로젝트 공식 종료

## 🚧 V5. 발표 및 다음 단계
- [ ] 가족 대상 발표 준비·실시
- [ ] 발표 후 휴식 또는 짧은 ML/DL 실험으로 전환

## (참고) Deployment
- [ ] FastAPI / Streamlit (구조 확정 이후 선택적 진행)

---

# 📊 Evaluation

Memory-based CF는 유저 단위 런타임 Split을, Model-Based CF와 Hybrid는 저장된 **Global MF Train/Test Split**을 재사용합니다. Hybrid 단계에서는 `hybrid_sampled_users.csv`에 저장한 **동일 평가 사용자 400명**을 모든 Architecture에서 사용해 실험 조건을 통제합니다.

### Recommendation Quality
Precision@K / Recall@K / Hit Rate@K / NDCG@K / Micro Precision·Recall·F1 / Candidate Recall / Scoreable Recall / MAP(예정)

### Sampling Strategy

| Review Group | Users |
|---|---:|
| 10–15 | 100 |
| 16–25 | 100 |
| 26–45 | 100 |
| 46–78 | 100 |
| **Total** | **400** |

### Hybrid Evaluation Summary

| Experiment | Hits | 핵심 목적/발견 |
|---|---:|---|
| Item-Based Baseline | 253 | Hybrid 기준선 |
| Case 1 Item→BPR | 278 | Item 후보의 ranking 보완 |
| Case 1 Reverse BPR→Item | 278 | BPR 후보를 Item signal로 재정렬 |
| Case 2 (1차, BPR Ranker) | 269 | Candidate Recall↑에도 self-ranking으로 성능 정체 |
| Case 3 (초기 가중치) | 310 | 4모델 Fusion 첫 결과 |
| Case 3 (Ablation+User5%) | 320 | Item>>BPR>Content>>User 기여도, User 5%가 최적 |
| Cross-Ranker (BPR+Content→Item) | 330 | Item-Based를 Ranker로 확정 |
| **Case 2 최종 (BPR56+Content39+User5→Item)** | **348** | **현재 반확정 최종 구조** |

### Performance
- 대규모 Global Split 연산 최적화 (66만 회 → 최대 69회)
- BPR 전체 학습은 `implicit`의 Cython/C 최적화 및 CSR 구조 활용
- 기존 Item-Based 400명 평가: 약 1811~1848초 → Reverse cache 구축 포함 234.7초, cached reranking 자체 0.3초
- Case 3 Retrieval Cache 재사용 시 전체 Fusion 재평가: 약 3초
- Item Ranker 최적화(캐싱): 초기 Cross-Ranker 실험 약 18.1분 → 이후 Ratio Sweep 7종 전체 1.75분, Candidate Size Sweep 전체 7.81초

---

# 📈 Current Progress

| Module | Status |
|:---|:---:|
| Data Loading / Caching / Validation / Preprocessing | ✅ |
| Content-Based Recommendation 전체 | ✅ |
| User-Based Collaborative Filtering 전체 | ✅ |
| Item-Based Collaborative Filtering 전체 | ✅ |
| Model-Based CF Global Train/Test Split | ✅ |
| Funk SVD 학습·추천·평가·Ablation 전체 | ✅ |
| BPR 전체 학습 / False Fine-Tuning / 최종 튜닝 | ✅ |
| Final BPR 결정 | ✅ |
| Hybrid 평가 사용자 400명 고정 / Baseline 구축 | ✅ |
| Case 1 / Case 1 Reverse | ✅ |
| Case 2 1차 실험 및 self-ranking 문제 발견 | ✅ |
| Case 3 Fusion 및 Ablation | ✅ |
| Cross-Ranker 실험 | ✅ |
| Retriever 비율 Sweep / Candidate Size Sweep | ✅ |
| Hybrid Architecture 반확정 | ✅ |
| 반확정 구조 미세 조정 | 🚧 |
| Learning-to-Rank(XGBoost/LambdaMART) 검토 | 🚧 |
| 최종 성능 튜닝 및 분석 | ⏳ |
| MAP / Popularity / Coverage 추가 분석 | ⏳ |
| 프로젝트 최종 회고 / README 정리 / 종료 | ⏳ |
| Deployment | ⏳ |

---

# 📌 Project Status

**Current Version:** `V3.4 - Hybrid Architecture 반확정 (BPR56+Content39+User5 → Item Ranker, Hits 348), 미세 조정 및 Learning-to-Rank 검토 단계`

### Completed

- Content-Based / User-Based / Item-Based CF 전체 파이프라인 구현 및 평가
- Global MF Train/Test Split 설계 및 저장 (41,154,794 interactions)
- Funk SVD 구현·평가·Ablation 완료 및 단계 종료
- **Final BPR 확정** — P@10 0.05225 / HR@10 0.3850 / NDCG@10 0.06968
- Hybrid 동일 평가환경 구축, Item-Based Hybrid Baseline(Hits 253)
- **Case 1 / Case 1 Reverse (Reranking) — Hits 278**
- **Case 2 1차 실험 → BPR self-ranking 문제 발견**
- **Case 3 (Weighted Rank Fusion) 및 Ablation — Item 기여 최대, User 5%가 최적**
- **Cross-Ranker 실험 → Item-Based를 최종 Ranker로 확정**
- **Retriever 비율 Sweep + User 5% 검증 → BPR56+Content39+User5 확정**
- **Case 2 vs Case 3 최종 비교 → Retriever→Ranker 구조 채택**
- **Candidate Size Sweep → 100이 최적점**
- **Hybrid Architecture 반확정** (Precision@10 0.08722, HR@10 0.5400, Hits 348 — Baseline 대비 +37~38%대 전지표 개선)
- Retrieval/Neighbor Cache 구조로 반복 실험 속도 대폭 단축

### In Progress

- **반확정 구조 미세 조정** (Retriever 비율, Item Ranker 파라미터)
- **Learning-to-Rank(XGBoost/LambdaMART) 원리 학습 및 적용 가능성 검토**

### Next Milestone

➡️ 반확정 구조(BPR56+Content39+User5→Item Ranker) 미세 조정
➡️ Learning-to-Rank(XGBoost Ranker/LambdaMART) 학습 및 적용 — Candidate 150/200의 여유 정답을 더 잘 끌어올리는지 검증
➡️ 최종 Architecture 확정 및 최종 성능 튜닝
➡️ 최종 정량·정성평가, 시스템 한계 분석, 향후 공부 방향 결정
➡️ MAP@K, Popularity Baseline 및 Popularity Bias/Coverage 분석
➡️ 프로젝트 최종 회고 → README 최종 완성 → 공식 종료
➡️ 가족 발표 준비·실시 → 휴식 및 다음 학습 탐색
➡️ (선택) FastAPI / Streamlit Deployment