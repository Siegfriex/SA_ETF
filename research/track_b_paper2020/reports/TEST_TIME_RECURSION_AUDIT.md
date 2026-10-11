# TEST_TIME_RECURSION_AUDIT — 테스트 구간 조건부분산의 의미

대상: Lee, Kim & Lee (2020) 영문 원문 (물리 PDF 34–37쪽 = 기사 5–8쪽). 인용은 `pdftotext -layout` 로 추출한 원문 문장이다.
결론 먼저: **원문은 테스트 구간에서 Eq.(9) 의 분산 lag 를 무엇으로 채우는지 명시하지 않는다. 이 항목은 `UNRESOLVED` 다.**

## 1. 원문 근거 (그대로 인용)

| # | 위치 | 원문 | 이 문제에 주는 정보 |
|---|---|---|---|
| E1 | p.5 Eq.(9) | `x_s = (y̆_s^T, σ̆_s^T)^T, y̆_s = (y²_{s−p},…,y²_{s−1}), σ̆_s = (σ²_{s−q},…,σ²_{s−1})` | 입력은 *참* 조건부분산 σ² 의 lag. 참값은 미지 |
| E2 | p.5 Step 2 | "the conditional variance σ²_t of (9) is unknown. As a remedy, replace σ²_t with the initial estimates σ̃²_t … based on the training time series using a moving average method" | 학습 단계에서 σ² 대신 proxy σ̃² 를 쓴다 |
| E3 | p.5 Step 3 | "we estimate g in (1) with ĝ using the SVR with σ²_t replaced by σ̃²_t. Then, the estimate σ̂²_t of σ²_t is obtained as: **σ̂²_t = ĝ(y²_{t−1},…,y²_{t−p}, σ̃²_{t−1},…,σ̃²_{t−q})**" | σ̂² 를 만드는 *유일한 명시 식*. 입력은 관측 proxy σ̃² |
| E4 | p.5 Step 4 | "using the same proxy formula as in Step 2 for the validation time series, MAE = (1/n′)Σ\|σ̂²_t − σ̃²_t\|" | validation 에서도 proxy 를 관측값으로 계산 → σ̂² 는 E3 식 그대로(관측 proxy 입력) |
| E5 | p.6 Step 5 | "perform Steps 2 and 3 using the training and validation time series together to determine the final model, **which is used in obtaining the residuals**" | 잔차는 최종 모형으로 얻는다. 테스트 입력 구성은 말하지 않음 |
| E6 | p.6 Remark 2 | "One may consider **iteratively updating σ̂²_t until a specific convergent criterion is satisfied**, as observed in Chen, Härdle and Jeong [13] and Lee, Lee and Moon [38]. However, **the ĝ obtained from this iterative approach** makes a function either 'flatten out' … or drastically follow the peak outliers … disregarded in our study." | 기각 대상은 *ĝ 를 다시 추정하는* 반복 절차 |
| E7 | p.6 §3 | (parametric GARCH 의 경우) "the residuals ê_t are **recursively** obtained via ê_t = y_t/σ̂_t, σ̂²_t = ω̂ + α̂y²_{t−1} + β̂σ²_{t−1} … with the initials y_0 = 0, σ²_0 = y_1" | QMLE 모수모형의 재귀. σ²_{t−1} 에 hat 이 없고 초기값 σ²_0 = y_1 은 차원이 맞지 않음 |
| E8 | p.7 §3 끝 | "This CUSUM procedure can be applied to the residuals obtained through the SVR-GARCH models" | SVR 잔차의 구성 방식은 추가 설명 없음 |
| E9 | p.7 §4 Step 2–3 | "Follow the estimation scheme described in Section 3 … Conduct the CUSUM of squares test … We utilize the remaining n number of time series as a testing set." | 'Section 3 의 추정 방식' 이라 함 — §2 의 SVR 절차를 가리키는지 §3 의 재귀를 가리키는지 모호 |

한국어 연구판(물리 1–29쪽)은 보조자료이며, 테스트 구간 lag 구성에 대해 원문 이상의 정보를 주지 않는다 (§3 재귀식과 Remark 2 를 번역·해설할 뿐).

## 2. 세 방식의 정의와 비교

ĝ 는 Step 5 에서 학습+검증 결합 구간으로 적합한 최종 SVR. 테스트 시점 t = n+1,…,2n (코드에서는 테스트 0-기반 j).

| 항목 | **A. Observed-Proxy Lag** | **B. Predicted-Variance Lag** | **C. Iterative Re-estimation** |
|---|---|---|---|
| 수학적 정의 | σ̂²_t = ĝ(y²_{t−1}, σ̃²_{t−1}), σ̃²_{t−1} = (1/5)Σ_{j=1..5} y²_{t−j} | σ̂²_t = ĝ(y²_{t−1}, σ̂²_{t−1}), σ̂²_n := σ̃²_n | 학습 구간에서 σ̃² → ĝ⁽¹⁾ → σ̂²⁽¹⁾ 를 입력으로 ĝ⁽²⁾ 재적합 … 수렴까지 |
| 최초 입력 시점 | t=n+1: (y²_n, σ̃²_n) | t=n+1: (y²_n, σ̃²_n) — A 와 동일 | 학습 시작점 |
| 분산 lag 생성 주체 | 관측 데이터 (이동평균) | 모형 자신 (직전 예측) | 모형 (반복마다 ĝ 갱신) |
| 필요 관측 정보 | y_{t−5..t−1} | y_{t−1} 과 직전 예측 | 학습 구간 전체 |
| 초기 분산 | 불필요 (관측 proxy, 이력 carry) | σ̃²_n (결합 구간 마지막 proxy) — **ASSUMPTION** | 원문 없음 |
| SVR 재학습 | 없음 | 없음 (ĝ 고정) | **있음** (반복마다) |
| 미래정보 사용 | 없음 (t−1 까지) — 테스트로 검증 | 없음 — 테스트로 검증 | 해당 없음 (학습 단계) |
| 원문 직접 명시 | E3 식 자체 (학습·검증에 대해), E4 | 없음. E7 은 모수모형에 대한 유사 구조 | E6 에서 *기각* |
| 원문에 없는 가정 | 테스트 구간에도 E3 식을 그대로 적용; 경계에서 proxy 이력 carry | 초기값, 재귀 사용 자체 | — |
| 계산 경로 (코드) | `forecast_test_variance(…, OBSERVED_PROXY)` | `forecast_test_variance(…, RECURSIVE_PREDICTION)` | 미구현 (원문이 기각) |

**B 와 C 는 다른 알고리즘이다.** C 는 ĝ 를 반복 재적합하는 *학습* 절차이고 Remark 2 는 그 결과 ĝ 가 평평해지거나 이상치를 따라간다고 기각한다. B 는 ĝ 를 한 번 적합한 뒤 고정하고 *적용* 할 때만 입력을 재귀로 공급한다. Remark 2 는 B 를 금지하지도 지지하지도 않는다.

**B 의 성질 (판정 아님).** ĝ 는 σ̃² (5일 평균, 잡음이 큼) 를 입력으로 학습됐는데 B 는 더 매끈한 σ̂² 를 입력한다 — 학습·적용 입력 분포가 다르다 (train/serve mismatch). B 는 parametric GARCH 필터(E7)와 구조가 같다.

**A 가 B 보다 덜 기각하고 size 가 과소한 이유 (측정).** A 대 원문의 power 차이는 양방향이라 (원문보다 높은 셀 34개) 이 메커니즘만으로 설명되지 않는다. target σ̃²_t 의 5 항 중 4 항(y²_{t−4..t−1})은 t−1 에 이미 관측된 값이고, 입력 σ̃²_{t−1} 은 y²_{t−5..t−1} 이다. 따라서 ĝ 는 단기 실현분산에 가까운 함수를 학습하고, 테스트에서 σ̂² 가 새 변동성 수준을 며칠 안에 따라간다. 이것은 누출이 아니라 정당한 1-step 예측이지만, 그 결과 ê² 의 수준 변화가 줄고 ê² 에 음의 자기상관이 생긴다. ablation 기준 300회(n=500, 5개 조건, 동일 DGP 경로) 측정치는 `reports/ABLATION_RESULTS.csv` 와 METHOD_FIDELITY_AUDIT §3 에 있다:
- corr²(σ̂²_t, mean(y²_{t−4..t−1})): A > B
- LRV(ê²)/τ̂² (Bartlett, 대역폭 ⌊n^{1/3}⌋): A < 1 (CUSUM 분모 τ̂² 가 실제 장기분산을 과대평가 → T̂ 가 작아짐 → size 과소), B > 1

## 3. 판정

| 정책 | 분류 | 근거 |
|---|---|---|
| A (관측 proxy lag) | `ASSUMPTION` — 원문 E3 식을 테스트 구간에 문자 그대로 연장 | E3–E5. 가장 문자적인 독해지만 테스트 구간 적용은 명시 안 됨 |
| B (예측 분산 lag) | `SENSITIVITY` | E7 의 모수모형 재귀와 같은 구조. SVR 에 대한 직접 근거 없음 |
| C (반복 재추정) | 원문이 기각 — 구현 안 함 | E6 |
| 테스트 구간 정책 전체 | **`UNRESOLVED`** | 어느 쪽도 원문으로 확정 불가 |

원문 수치에 더 가깝다는 이유로 B 를 primary 로 올리지 않는다. 직전 세션 채팅에서 "B 를 primary 로 올려 1000회 재실행" 을 제안했으나, 이번 감사 기준(원문 충실성 우선)에 따라 **그 제안은 철회**한다.
