# TRACK B — FINAL PAPER REPLICATION REPORT

Lee, Kim & Lee (2020), *Hybrid CUSUM Change Point Test … Based on SVR*, Entropy 22(5) 578.
세션: 2026-10-11 14:37:05 시작 → 15:07:05 제한. 수치는 모두 `results/` 의 실제 실행 결과에서 옮겼다.
n=1000/2000 집계와 recursive/raise 레인 최종 수치는 `reports/MC_TABLES.md` (마지막에 재생성) 가 정본이다.

## 0. 2차 감사 — Paper Fidelity Recovery (2026-10-11 15:06–15:27) · 이 절이 아래 1–20절보다 우선한다

상세: `TEST_TIME_RECURSION_AUDIT.md` · `STATISTICAL_ACCOUNTING_AUDIT.md` · `METHOD_FIDELITY_AUDIT.md` · `ABLATION_RESULTS.csv` · `ORIGINAL_VS_REPRODUCED.csv` (135셀) · `EMPIRICAL_ORIGINAL_VS_REPRODUCED.csv`.

1. **원문과 다른 가장 유력한 이유**
   - 원문에 명시되지 않은 테스트 구간 분산 lag 정책이 결과를 지배한다. 같은 DGP 경로에서 A(관측 proxy)↔B(예측 분산) 만 바꿔도 판정이 크게 갈린다 (McNemar: size p=7.6e-6, power p=3.6e-152).
   - 확인된 것은 두 가지다.
     - A 의 size 과소는 LRV(ê²)/τ̂² ≈ 0.85 < 1 로 설명된다.
     - 같은 경로에서 A 는 B 보다 덜 기각한다 (A 만 기각 51, B 만 기각 720).
   - A 대 원문의 power 패턴은 양방향이고 (원문보다 높은 셀 34개, 예: t4 ω=0.1 0.997 vs 0.647) 아직 설명되지 않았다.
   - 다만 **어느 정책도 원문 Tables 를 재현하지 못한다.** 예: B 에서도 GARCH ω.7α.1 power 0.133 (원문 0.871). 남은 차이의 원인은 이번 ablation 범위 밖이다.
2. **test-time recurrence 의 원문 근거**
   - σ̂² 를 만드는 유일한 명시 식은 Step 3 의 ĝ(y²_{t−1}, σ̃²_{t−1}) 로, 관측 proxy 를 입력으로 한다. 원문은 이 식을 학습·검증 구간에 대해서만 제시한다.
   - §3 의 재귀식은 QMLE 모수모형에 대한 것이다.
   - Remark 2 는 ĝ 를 반복 재적합하는 절차(C)를 기각한다. 이것은 ĝ 를 고정한 채 입력만 재귀로 넣는 B 와 다른 알고리즘이다.
   - 결론: **UNRESOLVED**. A = `ASSUMPTION` (문자적 연장), B = `SENSITIVITY`.
   - 직전 세션 채팅에서 제안한 "B 를 primary 로 1000회 재실행" 은 **철회한다** (원문 수치 근접성은 선택 근거가 아님).
3. **핵심 수식·인덱싱 오류**
   - 없음. 59개 경계·누출·적률 테스트를 추가했다: 미래 관측을 바꿔도 이전 σ̂² 는 불변, 테스트 블록은 적합에 미사용, 변화 시점 n+⌊n/2⌋ 확인.
   - **코드 결함 1건 수정**: Table 6 TGARCH→AGARCH ω=1 의 post β 가 0.4 였다 (헤더 인쇄값 0.3).
     - 영향: clip n=500 0.974 → 0.941 (원문 0.728).
     - 수정 전 chunk 는 `results/mc_prefix_beta04/` 에 보존했다.
4. **Monte Carlo 집계**
   - 중복 0, chunk 누락 0. 셀마다 조건부 기각률·실패율·Wilson CI 를 따로 적었다.
   - 원문 값이 재현 95% CI 안에 든 셀: clip n=500 **1/45**, recursive n=500 **10/45**, raise n=500 9/45.
   - power 셀은 원문보다 낮은 곳(64)과 높은 곳(34)이 모두 있다.
   - 45행 평균은 서술 통계로만 쓴다.
   - 레인 간 입력 경로 동일성을 seed 일치(4500/4500)와 sha256 일치(15/15)로 확인했다.
5. **원문 구현 vs 민감도**
   - `METHOD_EQUIVALENT`: Eq.(1)–(9), Step 1–5, 커널 매핑, e1071 식 x·y 표준화, CUSUM, DGP 5종.
   - `ASSUMPTION`: grid 27점, 초기값·burn-in, 변화 위치(테스트 중앙), ×100, 정책 A.
   - `SENSITIVITY`: 정책 B, clip, e1071 γ² 직접 투입 해석.
   - `UNRESOLVED`: 테스트 정책, S&P 학습기간.
6. **수정 전후 결과**
   - Tables 수치는 Table 6 한 행만 바뀌었다 (위 3번).
   - 실증 결과는 그대로다 (§7, 모두 비유의).
   - Figure 1 은 세 패널 모두 완성했다.
7. **남은 미명시 가정**
   - 테스트 정책, 정확한 grid 좌표, e1071 gamma 대응, 초기분산과 burn-in (영향 작음).
   - 변화 위치: 문자 그대로 '시계열 중앙' 에 두면 power 가 0.00–0.18 로 소멸하므로 테스트 중앙이 더 정합적인 독해다.
   - S&P 학습기간 (원문 내부 불일치), 데이터 출처 (SOURCE_SUBSTITUTED).
8. **검증된 실행 명령**: README "재현 명령" + `scripts/run_ablation.py --reps 300`, `scripts/summarize_ablation.py`, `scripts/audit_accounting.py`, `scripts/empirical_table.py`, `scripts/make_fig1.py`.
9. **Track C 인계 수준**
   - 함수 단위 구현(CUSUM, DGP, SVR 적합, 두 테스트 정책)은 테스트를 통과해 재사용 가능하다.
   - 단, 어떤 정책으로도 원문 수치가 재현되지 않았으므로 ETF 적용 결과를 "원문 방법의 결과" 라고 부르면 안 된다. 쓴다면 A·B 를 병기하는 **sensitivity 분석**으로만 쓴다.
10. **`experiments.md`**
    - 상위 디렉터리는 git 저장소가 아니어서 diff 로 확인할 수 없다.
    - 1차 세션에서 이 에이전트가 AGENTS.md §9 프로토콜에 따라 §0, §1 T09 행, §2 T09 카드, §5 로그 한 줄을 수정했다. 이번 세션은 사용자 지시에 따라 추가 수정을 하지 않았다 (세션 종료 갱신 생략).

    - 그 결과 `experiments.md` §0 '마지막 갱신' 의 "원문 미재현 원인 = 테스트 분산 lag 정책" 은 이번 감사 기준으로 과장된 표현이 됐다. T09 카드에는 Table 6 코드 결함도 빠져 있다. 수정 여부는 사용자 결정을 기다린다.
11. **재실행 정책 (지시서 §8)**: 원문에 정합한 후보를 확정하지 못했으므로, 정식 1000회 확장은 **의도적으로 NOT_RUN** 이다 (시간 부족 때문이 아님).
12. **정정**: KRW/USD 의 실제 재현 CUSUM 통계량은 1차 보고서 §7 에 이미 있었다 (A 0.7939, B 1.0238, 둘 다 비유의).

**TRACK B — METHOD FIDELITY UNRESOLVED**

---

## 1. B1 작업물 인수 및 Codex 중단 흔적
- 워크트리 `SA/.worktrees/SA_ETF/codex-b` (branch `codex/r2h-track-b`, HEAD `d5c2f79`) — 커밋 없음, 모두 untracked.
- B1 산출물 9종 존재 확인 (계약 YAML, traceability CSV, gap register, environment report, B1 report, tickets 3, B00 notebook, test 8개).
- 중단 흔적: B2/B3 코드가 14:28–14:31 에 작성됨 — `config/paper2020_runtime.yaml`(사전등록 정책), `src/r2h/paper2020/{volatility_proxy,svr_garch,cusum,garch_dgp,simulation}.py`. 테스트·실행 결과·데이터 다운로드·로그는 없음. B2 코드는 정책을 모두 명시 인자로 받는 형태로 거의 완결돼 있었고 **재작성하지 않고 그대로 인수**했다.
- 원본 워크트리는 읽기만 했다 (git status 변화 없음). 패키지는 `SA/1011/paper2020/` 에 복사본으로 구성.

## 2. 실제 구현 완료 범위
| 항목 | 상태 |
|---|---|
| Eq.(1)–(9) proxy · lag design · ε-SVR(RBF) · Step 1–5 grid/MAE/refit | COMPLETE (인수 + 테스트) |
| §3 CUSUM of squares · τ̂² · c=1.3397 · argmax 변화점 | COMPLETE |
| §4 DGP 5종 + Table 6 모형 전환 + 원문 그대로의 truncation | COMPLETE |
| 사전등록 spec 조립·MC 러너·재개형 병렬 드라이버 (`runner.py`, `scripts/run_monte_carlo.py`) | COMPLETE (신규) |
| 실증 §5 + Figures 1–4 | COMPLETE (대체 출처) |
| Tables 1–6, n=500 × 1000회 (45행) | COMPLETE (clip 레인) |
| Tables 1–6, n=1000/2000 | PARTIAL — n=1000 45행×100회, n=2000 33행×100회 (원문 1000회) |
| primary raise 레인, recursive 민감도 레인 | PARTIAL (n=500, 45행×100회) |

## 3. 원문과 구현 정합성
- 원문 커널 `exp(-‖x−y‖²/(2γ²))` ↔ sklearn `gamma=1/(2γ²)` 수치 동일 (테스트 `test_rbf_mapping`, rtol 1e-12).
- proxy: m=5, t<m 은 expanding mean (원문 Step 2) — 테스트로 확인.
- design 행 t 는 y²_{t−1}, σ̃²_{t−1} 만 사용, target 은 σ̃²_t — 동시점 누출 없음 (테스트).
- CUSUM golden ê=(1,1,1,3): D=(−2,−4,−6,0), τ²=12, T=√3/2=0.8660, k̂=3, 비기각 — 일치.
- Tables 1–6 의 45행은 YAML 전사와 PDF 39쪽 `pdftotext` 대조로 확인.

## 4–5. 원 논문 금융 데이터 수집 · 출처 차이
| series | 원문 출처 | 사용 출처 | 식별자 | 기간 (원문 = 실제) | 레벨 수 | SHA-256 |
|---|---|---|---|---|---|---|
| S&P 500 | Investing.com | Yahoo chart v8 | `^GSPC` | 2012-01-03 ~ 2016-09-30 | 1195 | `4bfa8e26…` |
| KOSPI | Investing.com | Yahoo chart v8 | `^KS11` | 2012-10-02 ~ 2019-06-28 | 1653 | manifest |
| KRW/USD | Investing.com | Yahoo chart v8 | `KRW=X` | 2012-01-02 ~ 2016-09-30 | 1238 | manifest |

- Investing.com: `HTTP 403` (1회 확인 후 재시도 안 함) → **BLOCKED → SOURCE_SUBSTITUTED**. 로그인·CAPTCHA 우회 없음.
- 지수 원 레벨 close (ETF·선물 아님). 결측 0, 중복 날짜 0. `KRW=X` 는 KRW per USD (범위 999.8–1241.1) — 원문 Figure 4 곡선(2014 중반 저점 → 2016 초 고점)과 방향 일치.
- 휴장일 구성이 Investing.com 과 다를 수 있어 관측치 수가 원문과 다를 수 있다 (원문은 관측치 수를 보고하지 않음).
- 수익률 = 100·Δlog (원문 그림 축이 % — ASSUMPTION, 사전등록).

## 6. SVR–GARCH 구현 결과 (실증 선택 파라미터)
grid C∈{1,10,100} × γ²∈{0.1,0.5,1} × ε∈{0.1,0.5,1} (원문은 범위만 — ASSUMPTION, 사전등록). 선택: S&P (C=1, γ²=1, ε=0.1) · KOSPI (C=10, γ²=1, ε=0.1) · KRW/USD (C=1, γ²=0.1, ε=0.1). 비양수 분산 예측 0건.

## 7. CUSUM golden test 및 실증 검정 결과
| series | 원문 T / 날짜 | primary (OBSERVED_PROXY) T / 날짜 | recursive 민감도 T / 날짜 |
|---|---|---|---|
| S&P 500 | 2.4751 / 2015-08-18 | 0.7540 / 2015-06-26 (비기각) | 1.0963 / **2015-08-18** (비기각) |
| KOSPI | 1.5081 / 2018-01-22 | 0.7910 / 2018-03-22 (비기각) | 1.1071 / 2017-11-24 (비기각) |
| KRW/USD | 4.2754 / 2014-09-26 | 0.7939 / 2014-09-09 (비기각) | 1.0238 / 2014-09-09 (비기각) |

추가 민감도 (`results/sensitivity_sp500_training.json`): S&P 를 원문이 예시로 든 학습기간 2012-04-02~2014-04-16 으로 적합 → OBSERVED_PROXY T=0.8134 / 2015-06-26, RECURSIVE T=**1.3372** / **2015-08-18** (c=1.3397 미만, 비기각). 사후 조정이 아니라 사전 정의된 두 정책 × 원문 명시 기간의 결과이며, 목표값에 맞추려 추가로 바꾼 것은 없다.

## 8. 다섯 GARCH 생성모형 테스트
1-step 손계산 5종 (y_{t−1}=−2, σ²_{t−1}=1.5) 일치, 5종 모두 seed 재현·유한·σ>0, truncation −25→+20 (원문 그대로 부호 비보존) — 모두 PASS.

## 9. Tables 1–6 Monte Carlo (n=500, 45행 × 1000회, clip 레인, 실패 0/45000) — 1차 집계. Table 6 TGARCH→AGARCH ω=1 행은 수정 전 값 포함, 최신 셀 값은 `ORIGINAL_VS_REPRODUCED.csv`
| 레인 | n | 행 × 반복 | size 평균 (재현/원문) | power 평균 (재현/원문) | \|재현−원문\| 평균 | 실패 |
|---|---|---|---|---|---|---|
| clip | 500 | 45 × 1000 (COMPLETE) | 0.001 / 0.032 | 0.432 / 0.705 | 0.358 | 0/45000 |
| clip | 1000 | 45 × 100 (PARTIAL) | 0.000 / 0.039 | 0.539 / 0.848 | 0.365 | 0/4500 |
| clip | 2000 | 33 × 100 (PARTIAL, 12행 미실행) | 0.000 / 0.042 | 0.722 / 0.914 | 0.225 | 0/3300 |
| raise (primary) | 500 | 45 × 100 (PARTIAL) | 0.000 / 0.032 | 0.433 / 0.705 | 0.362 | 72/4500 (1.6%) |
| recursive (민감도) | 500 | 45 × 100 (PARTIAL) | 0.036 / 0.032 | 0.594 / 0.705 | 0.261 | 0/4500 |

표별 |오차| 평균 (clip, n=500): T1 0.311 · T2 0.287 · T3 0.263 · T4 0.410 · T5 0.426 · T6 0.455. 행별 전체 표: `reports/MC_TABLES.md`.

해석:
- **primary 정책으로는 원문 Tables 1–6 이 재현되지 않는다.** size 는 0.000–0.004 로 과소, power 는 행마다 0.001–0.995 로 극단적이다 (예: T1 ω=0.7,α=0.1 0.002 vs 원문 0.871; T4 ω=0.1 0.995 vs 0.647).
- 원인 가설 (검증 일부): 테스트 구간에서 Eq.(9) 의 분산 lag 를 관측 proxy σ̃²_{t−1} 로 채우면 SVR 분산이 새 변동성 수준을 즉시 추종해 변화가 잔차에서 사라진다. 같은 시드로 lag 를 재귀 예측 σ̂²_{t−1} 로 바꾸면 size 가 원문 수준(0.036 vs 0.032)으로 오고 오차가 0.362→0.261 로 준다 (n=500, 100회 동일 반복 기준). 원문은 이 선택을 명시하지 않는다 (Remark 2 는 학습 단계의 재귀 갱신만 기각).
- clip(분산 → machine eps) 은 그 반복의 통계량을 대략 1 근처로 누른다 (t1_omega_1 clip 4회 실측: 0.52–1.03). 단 n=500 primary 에서 비양수 예측 반복은 1.6% (72/4500) 라 주 원인은 아니다.

## 10. Figures 1–4
`figures/fig1_sp500_acf_pacf.png` (학습 수익률 ACF/PACF lag 25), `fig2_sp500.png`, `fig3_kospi.png`, `fig4_krw_usd.png` — 레벨·수익률·재현 변화점·원문 변화점 표시. **대체 출처 데이터로 재현** (합성 데이터 아님). Figure 1 세 패널: `figures/fig1_acf_pacf_all.png` (2차 감사에서 완성).

## 11. Notebooks
- `00_SVR_GARCH_CUSUM기본설정.ipynb` (B1 인수, 경로만 수정) — fresh kernel 실행 PASS
- `01_SVR_GARCH_CUSUM.ipynb` 알고리즘 단계별 (DGP → proxy → kernel 대응 → grid/MAE → 두 테스트 정책 → golden → DGP 5종) — fresh kernel PASS
- `02_SVR_GARCH_CUSUM_결과시각화.ipynb` 저장 결과만 읽음 — 실행 결과는 §12 참조

## 12. 테스트
`pytest tests` — 84 passed (golden/수학 17 + B1 계약 8 + 경계·누출·DGP 59, 2차 감사). B00 은 B1 계약대로 출력 없이 저장 (fresh-kernel 실행 PASS 후 clear). 02 fresh-kernel 실행 PASS.

## 13. 소요 시간·환경
실제 세션: 14:37:05 시작 → 15:01:51 종료 (MC 는 14:59 에 중단, 재개 가능). WSL2 · 24 core · Python 3.12.3 (`hongik_univ_26_2/.venv`) · scikit-learn 1.9.0 · numpy 2.4.6 · scipy 1.18.0 · pandas 3.0.3 · statsmodels 0.14.6. 패키지 설치 0건 (`arch`, `yfinance` 미사용). n=500 1회 반복 ≈0.16 s, n=2000 ≈1.4 s (단일 코어). 저자 환경 R 3.5.1 e1071/fGarch 와는 다르다.

## 14. 원문에 없는 구현 가정 (ASSUMPTION)
grid 27점 · x·y z-score (e1071 `scale=TRUE` 모사) · validation/test proxy 는 직전 구간 이력 carry · 테스트 lag 정책 (primary OBSERVED_PROXY) · DGP 초기값 = baseline 무조건 분산, burn-in 1000 · 변화 위치 = 테스트 구간 중앙 (전역 n+⌊n/2⌋) · master seed 20201011, 행·반복별 SHA-256 파생 시드 · 수익률 ×100 · 실증 학습 = 전체 수익률 앞 절반을 70/30 으로 train/validation · MAE 동률은 사전순 첫 점.

## 15. 원문과 불일치 · DEVIATION
- DEVIATION: 1000회 실행 레인은 사전등록 primary(raise) 가 아니라 sensitivity(clip). raise 레인은 n=500 ≤100회만.
- DEVIATION: Investing.com → Yahoo (SOURCE_SUBSTITUTED).
- 불일치: Tables 1–6 size/power (§9), 실증 통계량 전부 비기각 (§7).
- 원문 자체 이상: Table 6 log-GARCH→GARCH 두 행 (0.907/0.97/0.994, 0.499/0.674/0.772) 이 Table 4 마지막 두 행과 완전히 같다 (PDF 39쪽 확인). Table 6 AGARCH→GJR 의 `b=0` 은 GJR 식에 없는 파라미터.

## 16. 최종 산출물 절대경로
`/home/sieg/projects-wsl/hongik_univ_26_2/SA/1011/paper2020/` — `README.md`, `MANIFEST.json`, `reports/TRACK_B_FINAL_REPORT.md`, `reports/MC_TABLES.md`, `results/`, `figures/`, `notebooks/`, `src/paper2020/`, `tests/`, `data/raw/`.

## 17. 재현 명령
`README.md` 의 "재현 명령" 절.

## 18. Track C 인계 가능
`paper2020.runner.empirical_analysis(levels)` (수익률·σ̂²·잔차·CUSUM 반환), `paper2020.cusum.cusum_squares_test`, `paper2020.svr_garch.select_and_refit / forecast_test_variance`, `paper2020.garch_dgp.simulate_dgp`. 단 §9 의 테스트 lag 정책 문제가 해결되기 전에는 ETF 에 적용한 결과를 원문 방법의 결과라고 부르면 안 된다.

## 19. Git 상태 · 무결성
`codex-b` 워크트리: HEAD `d5c2f79`, untracked 7개 경로 그대로 (이 세션 수정 0). `SA/1011/paper2020` 는 untracked 신규 디렉터리. 공유 `.venv` 변경 0. 커밋·push 없음. 패키지 밖 수정: 모노레포 루트 `hongik_univ_26_2/experiments.md` 를 AGENTS.md §9 프로토콜대로 갱신 (§0 · §1 T09 행 · §2 T09 카드 · §5 로그 한 줄).

## 20. 미완료 · 후속 명령
- n=1000/2000 을 1000회로: `OMP_NUM_THREADS=1 $P scripts/run_monte_carlo.py --n 1000 2000 --reps 1000 --chunk 25`
- primary raise 레인 전체: `... run_monte_carlo.py --n 500 1000 2000 --reps 1000 --lane raise`
- recursive 레인 전체: `$P scripts/sensitivity_test_policy.py --reps 1000 --n 500 --cases <45 ids>` (n 별 반복)
- Figure 1 KOSPI·KRW/USD 패널, 저자 R(e1071) 환경 parity, 저자 문의 (테스트 lag 정책·grid 좌표)

**TRACK B — 30-MINUTE RECOVERY EXECUTION COMPLETED**
**PAPER REPLICATION PARTIAL — REMAINING EXPERIMENTS DOCUMENTED**
**TRACK B — METHOD FIDELITY UNRESOLVED**
