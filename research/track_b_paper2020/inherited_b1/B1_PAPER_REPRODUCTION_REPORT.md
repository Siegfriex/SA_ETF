# B1 PAPER REPRODUCTION REPORT

## 1. 원문 서지·SHA·페이지 범위

- Sangyeol Lee, Chang Kyeom Kim, Sangjo Lee (2020), “Hybrid CUSUM Change Point Test for Time Series with Time-Varying Volatilities Based on Support Vector Regression,” *Entropy* 22(5), 578, DOI `10.3390/e22050578`.
- Source of truth: `/home/sieg/projects-wsl/hongik_univ_26_2/SA/1011/Hybrid_SVR_GARCH_CUSUM_.pdf`.
- SHA-256: `97c95e308af8dd95e8d283f01674932f52349ab8b4d0ea2a93d4dbe45fcf3cd6`.
- 결합 PDF는 45쪽이며 영어 원문은 물리 30–45쪽(논문 1–16쪽)이다. 수식과 표는 layout text와 렌더링을 교차 대조했다.

## 2. Eq. (1)–(9) 완성 매핑

- Eq. (1): 관측식과 미지 조건부분산 함수.
- Eq. (2)–(3): epsilon-insensitive loss와 primal.
- Eq. (4)–(7): Lagrangian, KKT, dual.
- Eq. (8): support-vector kernel 표현.
- Eq. (9): 과거 제곱 관측치와 과거 조건부분산의 lag feature.
- 각 식의 원문 페이지, 정의, B2 함수/노트북/테스트 목표가 `paper_traceability.csv`와 `paper2020_contract.yaml`에 연결됐다.
- RBF 대응은 `gamma_sklearn = 1/(2*paper_gamma_squared)`로 확정했다. 정확한 finite grid는 원문 미명시다.

## 3. §3 CUSUM 정의·검정 계약

- 입력은 SVR–GARCH의 표준화 잔차 `e_hat_t=y_t/sigma_hat_t`의 제곱이다.
- 중심화 경로, 분모 `n`인 `tau_hat_n^2`, `sqrt(n)*tau_hat_n` 정규화, 절댓값 supremum을 고정했다.
- 귀무분포 기준은 standard Brownian bridge의 supremum이며 5% 기각은 `T_hat_n >= 1.3397`이다.
- 기각 여부와 변화점 `argmax`는 서로 다른 출력이다.
- residual `[1,1,1,3]`에 대한 fixture는 `D=[-2,-4,-6,0]`, `tau^2=12`, `T=0.8660254037844386`, 1-based argmax `3`으로 검산됐다.
- 원문의 `sigma_0^2=y_1`, 이전 분산 hat 누락, non-positive prediction, `tau=0`, argmax tie는 보정하지 않고 gap으로 남겼다.

## 4. Tables 1–6 조건표

- 공통: `n={500,1000,2000}`, 총 `2n`, 처음 `0.7n` training, 다음 `0.3n` validation, 나머지 `n` testing, `m=5`, 1,000회, 유의수준 0.05, iid standard normal.
- Table 1 GARCH, Table 2 AGARCH, Table 3 GJR-GARCH, Table 4 TGARCH, Table 5 log-linear GARCH, Table 6 DGP change를 각각 원문 식으로 고정했다.
- 6개 표의 45개 target row를 machine-readable YAML로 전사했다. 별도 감사표는 `paper2020/tickets/B1_simulation_source_audit.md`에 있다.
- `N=20` 식은 음의 극단값도 `+20`으로 보내는 원문 literal rule이며 Table 5와 Table 6 Cases 1–2에만 적용한다.
- H1의 “middle”은 정확한 index가 없고, Table 6의 GJR 목적지 `b=0`, AGARCH baseline 차이는 원문 불일치로 보존했다.

## 5. Figures 1–4 데이터 출처·목표값

| Series | Source period | Target change point | Target T |
|---|---|---:|---:|
| S&P 500 | 2012-01-03–2016-09-30 | 2015-08-18 | 2.4751 |
| KOSPI | 2012-10-02–2019-06-28 | 2018-01-22 | 1.5081 |
| KRW/USD | 2012-01-02–2016-09-30 | 2014-09-26 | 4.2754 |

- 원문 출처는 Investing.com이다. Figure 1은 앞쪽 절반 training 로그수익률의 ACF/PACF를 lag 25까지, Figures 2–4는 원 수준·로그수익률·변화점 선을 표시한다.
- 원 snapshot은 로컬에 없으며 외부 다운로드는 하지 않았다. Track A ETF는 대체 자료가 아니다.
- S&P 전체 시작일과 training 예시 시작일 `2012-04-02`는 충돌한다. 목표값은 regression target일 뿐 post-hoc tuning 기준이 아니다.

## 6. 원문에 누락된 재현 정보

정확한 finite hyperparameter grid, scaling, solver 설정, proxy/lag 경계, test-time recursion, positivity, H1 index, DGP 초기값/burn-in/RNG, log-zero 처리, 원 데이터 snapshot/field/log-return scale, 정확한 empirical midpoint, Monte Carlo 허용오차가 없다. 우선순위와 해결 조건은 `reproducibility_gap_register.md`에 28개 gap으로 정리했다.

## 7. 선택된 환경·라이브러리 정책

- Primary: WSL2/Python 3.12.3/Jupyter의 method-equivalent lane. 명시적 식과 파라미터만 사용한다.
- Secondary: Windows/R 3.5.1/e1071/fGarch environment-parity lane. 현재 R이 없으므로 사용자 승인 후 별도 격리 환경에서만 시도한다.
- 공유 venv는 변경하지 않았다. 현재 `arch`도 없지만, 원문 DGP를 library class로 대체하지 않으므로 B1에는 필요하지 않다.
- 프로젝트-level lockfile은 없으므로 현재 package version은 관측 snapshot으로만 기록했다.

## 8. B00 노트북 학습 구조 및 검증 결과

- 15 cells: scope/source integrity, Eq. mapping/RBF check, five-step procedure, CUSUM fixture, simulations, empirical targets, traceability, environment, phase gate.
- 소스 노트북은 의도적으로 unexecuted 상태이며 외부 I/O, SVR fit, simulation이 없다.
- 정적 B1 tests: **8 passed**.
- `hongik_26_2` fresh kernel: **6/6 code cells executed, 0 errors**. 실행 사본은 repository를 오염시키지 않도록 `/tmp/B00_fresh_kernel_executed.ipynb`에 생성했다.
- 실행 사본 SHA-256: `d1c35e8fac5c31f17f17ae689bfd65240d2221613682dfc3a696aa738c5ed3bb`.

## 9. Phase B2 구현 계획

사용자 승인 후에만 다음 순서로 진행한다.

1. Grid/boundary/test-recursion/positivity 정책을 사용자 결정과 함께 preregister한다.
2. proxy, lag matrix, RBF mapping을 작은 fixture로 구현·검증한다.
3. SVR chronological selection과 combined refit을 구현한다.
4. CUSUM statistic/rejection/location을 golden tests로 구현한다.
5. 다섯 DGP를 표시식 그대로 구현하고 smoke test까지만 안정화한다.
6. B3 승인 후 1,000회 full protocol과 empirical snapshot을 별도 수행한다.

## 10. Git status와 변경 파일

- Worktree: `/home/sieg/projects-wsl/hongik_univ_26_2/SA/.worktrees/SA_ETF/codex-b`
- Branch: `codex/r2h-track-b`
- Baseline HEAD: `d5c2f792022d3c8efdbc52bdecd246b04bf32c45`
- 원본 ETF/Track A 파일 수정: **없음**.
- B2 source/experiment 구현: **없음**.
- B1 변경은 허용된 `config/`, `notebooks/paper2020/`, `reports/r2h/paper2020/`, `paper2020/tickets/`, `tests/paper2020/` 아래의 신규 파일로 한정했다.
- Commit/push/merge/cherry-pick: 수행하지 않음.

**TRACK B / PHASE B1 COMPLETE — AWAITING REVIEW**
