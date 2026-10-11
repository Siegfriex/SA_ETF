# paper2020 — Lee, Kim & Lee (2020) SVR–GARCH CUSUM 재현 패키지

Lee, S., Kim, C. K., & Lee, S. (2020). *Hybrid CUSUM Change Point Test for Time Series with Time-Varying
Volatilities Based on Support Vector Regression.* Entropy 22(5), 578. doi:10.3390/e22050578
원문 PDF: `../Hybrid_SVR_GARCH_CUSUM_.pdf` (영문 원문 = 물리 30–45쪽, Source of Truth)

이 패키지는 Codex B 의 B1 계약(`config/paper2020_contract.yaml`)과 B2 코어 모듈을 인수해 독립 실행형으로 옮긴 것이다.
ETF 데이터와의 결합은 범위 밖이다. 결과와 원문 대비 비교는 `reports/TRACK_B_FINAL_REPORT.md` 에 있다.

## 구조

| 경로 | 내용 |
|---|---|
| `src/paper2020/volatility_proxy.py` | Eq.(9), Step 2 proxy (m=5, expanding start), lag design |
| `src/paper2020/svr_garch.py` | Eq.(2)–(8) ε-SVR(RBF, `gamma=1/(2γ²)`), Step 1–5 grid/MAE/refit, test variance |
| `src/paper2020/cusum.py` | §3 CUSUM of squares, τ̂², c=1.3397, argmax 변화점 |
| `src/paper2020/garch_dgp.py` | §4 생성모형 5종 (GARCH, AGARCH, GJR, TGARCH, log-linear) + 원문 그대로의 truncation |
| `src/paper2020/simulation.py` | Tables 1–6 의 45개 행, 시드 파생, 반복 1회 실행 |
| `src/paper2020/runner.py` | 사전등록 spec(`config/paper2020_runtime.yaml`) 조립, MC 행·실증 분석 |
| `scripts/` | `fetch_data.py` · `run_monte_carlo.py` · `sensitivity_test_policy.py` · `run_empirical.py` · `summarize_mc.py` · `build_notebooks.py` |
| `notebooks/` | 00 기본설정(B1) · 01 알고리즘 단계별 · 02 결과 (저장 결과만 읽음) |
| `data/raw/` | Yahoo Finance 대체 시계열 3종 + `data_manifest.json` (SHA-256) |
| `results/` | `mc/<lane>/*.jsonl` (반복별 원시 결과) · `mc_summary_<lane>.csv` · `empirical_*.{json,csv}` |
| `figures/` | Figure 1–4 재현 그림 |
| `inherited_b1/` | Codex B 의 B1 산출물 (traceability, gap register, tickets) |
| `tests/` | golden/수학 테스트 17개 + B1 계약 테스트 8개 |

## 재현 명령

```bash
cd /home/sieg/projects-wsl/hongik_univ_26_2/SA/1011/paper2020
P=/home/sieg/projects-wsl/hongik_univ_26_2/.venv/bin/python
$P -m pytest -q tests                                        # 25 tests
$P scripts/fetch_data.py                                     # Yahoo 대체 데이터 (Investing.com 403)
$P scripts/run_empirical.py                                  # Figures 1–4, empirical_summary.json
OMP_NUM_THREADS=1 $P scripts/run_monte_carlo.py --n 500 --reps 1000 --chunk 50          # clip 레인
OMP_NUM_THREADS=1 $P scripts/run_monte_carlo.py --n 1000 2000 --reps 1000 --chunk 25    # 남은 반복 (재개 가능)
OMP_NUM_THREADS=1 $P scripts/run_monte_carlo.py --n 500 --reps 1000 --lane raise        # primary 레인 전체
$P scripts/sensitivity_test_policy.py --reps 1000 --cases $(PYTHONPATH=src $P -c "from paper2020.simulation import TABLE_CASES as T; print(' '.join(c.case_id for c in T))")
$P scripts/summarize_mc.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=hongik_26_2 notebooks/*.ipynb
```

chunk 파일이 이미 있으면 건너뛰므로 같은 `--chunk` 로 재실행하면 이어서 돈다. n=1000/2000 은 `--chunk 25` 로 시작했으므로 재개할 때도 25 를 쓴다.
