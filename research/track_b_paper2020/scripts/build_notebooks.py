"""Generate notebooks/01 (algorithm walkthrough) and 02 (results) with nbformat."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = """import sys
from pathlib import Path
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'config' / 'paper2020_runtime.yaml').exists())
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)"""

nb1 = nbf.v4.new_notebook()
nb1.cells = [
    md("# 01 · SVR–GARCH + CUSUM of squares — 알고리즘 단계별 재현\n"
       "Lee, Kim & Lee (2020), *Entropy* 22(5) 578. 원문 Eq.(1)–(9), §2 Step 1–5, §3 CUSUM 을 `src/paper2020` 모듈로 한 단계씩 실행한다.\n\n"
       "- 표기: **ASSUMPTION** = 원문에 없는 구현 선택, **DEVIATION** = 원문과 다른 선택 (`config/paper2020_runtime.yaml`)."),
    code(SETUP),
    md("## 1. 데이터 생성 — GARCH(1,1) baseline ω=α=β=0.3, 길이 2n (n=500), 테스트 중앙에서 ω→1.0 변화\n"
       "초기값·burn-in(1000)은 원문에 없음 → ASSUMPTION."),
    code("""from paper2020.garch_dgp import simulate_dgp, ChangeSpec, DGPModel
from paper2020.runner import dgp_assumptions, MASTER_SEED
n = 500
base = {'omega': 0.3, 'alpha': 0.3, 'beta': 0.3}
sim = simulate_dgp(DGPModel.GARCH, base, length=2*n, assumptions=dgp_assumptions(DGPModel.GARCH), seed=MASTER_SEED,
                   change=ChangeSpec(n + n//2, DGPModel.GARCH, {**base, 'omega': 1.0}))
y = sim.observations
train, val, test = y[:int(0.7*n)], y[int(0.7*n):n], y[n:]
print(len(train), len(val), len(test))
fig, ax = plt.subplots(figsize=(10, 2.5)); ax.plot(y, lw=.6); ax.axvline(n, c='k', lw=.6); ax.axvline(n+n//2, c='r', ls='--'); ax.set_title('y_t (training | validation | test, red = true change)'); plt.show()"""),
    md("## 2. Step 2 — 변동성 proxy σ̃²_t = (1/m)Σ y²_{t-j+1}, m=5 (t<m 은 expanding mean)"),
    code("""from paper2020.volatility_proxy import moving_square_proxy, build_lagged_design, BoundaryPolicy
proxy = moving_square_proxy(train, m=5, boundary_policy=BoundaryPolicy.RESET)
assert np.isclose(proxy[0], train[0]**2) and np.isclose(proxy[10], np.mean(train[6:11]**2))
design = build_lagged_design(train, proxy, p=1, q=1, boundary_policy=BoundaryPolicy.RESET)
print('Eq.(9) x_t = (y_{t-1}^2, sigma~_{t-1}^2):', design.X[:3], 'target sigma~_t^2:', design.target[:3])
# 누출 검사: 행 t 의 feature 는 t-1 까지만 사용
assert np.allclose(design.X[:, 0], train[design.target_indices - 1]**2)"""),
    md("## 3. Eq.(2)–(8) — ε-insensitive SVR, RBF 커널 exp(−‖x−y‖²/(2γ²)) ↔ sklearn `gamma = 1/(2γ²)`"),
    code("""from sklearn.metrics.pairwise import rbf_kernel
from paper2020.svr_garch import paper_gamma_to_sklearn
X = design.X[:4]; g2 = 0.5
paper_K = np.exp(-((X[:, None] - X[None]) ** 2).sum(-1) / (2 * g2))
assert np.allclose(paper_K, rbf_kernel(X, gamma=paper_gamma_to_sklearn(g2)))
print('kernel mapping OK')"""),
    md("## 4. Step 1·4·5 — grid search (C∈{1,10,100}, γ²∈{0.1,0.5,1}, ε∈{0.1,0.5,1}; 원문은 범위만 제시 → ASSUMPTION), validation MAE, 결합 재적합"),
    code("""from paper2020.runner import make_spec
from paper2020.svr_garch import select_and_refit, forecast_test_variance, VariancePredictionPolicy
spec = make_spec('full', clip=True)
fit = select_and_refit(train, val, parameter_grid=spec.parameter_grid, fit_policy=spec.fit_policy, m=5, p=1, q=1,
    validation_proxy_boundary=spec.validation_proxy_boundary, validation_lag_boundary=spec.validation_lag_boundary,
    validation_variance_policy=spec.validation_variance_policy, variance_floor=spec.variance_floor, tie_policy=spec.selection_tie_policy)
scores = pd.DataFrame([{**vars(s.parameters), 'MAE': s.mae} for s in fit.scores]).sort_values('MAE')
print('selected:', fit.best_parameters); scores.head()"""),
    md("## 5. 테스트 구간 조건부 분산과 표준화 잔차 ê_t = y_t/σ̂_t\n"
       "테스트 구간에서 Eq.(9) 의 분산 lag 를 무엇으로 채울지는 원문에 명시가 없다. primary = 관측 proxy(OBSERVED_PROXY), 민감도 = 재귀 예측(RECURSIVE_PREDICTION)."),
    code("""from paper2020.svr_garch import TestVarianceInputPolicy
from paper2020.cusum import standardized_residuals, NonPositiveVariancePolicy, cusum_squares_test, ArgmaxTiePolicy, DegenerateTauPolicy
out = {}
for pol in (TestVarianceInputPolicy.OBSERVED_PROXY, TestVarianceInputPolicy.RECURSIVE_PREDICTION):
    fc = forecast_test_variance(fit, test, m=5, p=1, q=1, input_policy=pol, proxy_boundary=BoundaryPolicy.CARRY,
                                lag_boundary=BoundaryPolicy.CARRY, prediction_policy=VariancePredictionPolicy.RAW, variance_floor=None)
    e = standardized_residuals(test, fc.variance, nonpositive_policy=NonPositiveVariancePolicy.CLIP, variance_floor=spec.variance_floor)
    out[pol.value] = (fc.variance, e, cusum_squares_test(e, tie_policy=ArgmaxTiePolicy.FIRST, degenerate_tau_policy=DegenerateTauPolicy.RAISE))
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(sim.variance[n:], c='k', lw=.8, label='true sigma^2')
for k, (v, e, cu) in out.items():
    ax[0].plot(v, lw=.7, label=f'SVR {k}'); ax[1].plot(cu.centered_path / (np.sqrt(len(e)) * np.sqrt(cu.tau_squared)), label=f'{k}: T={cu.statistic:.3f}, k={cu.change_index_one_based}')
ax[1].axhline(1.3397, c='r', ls=':'); ax[1].axhline(-1.3397, c='r', ls=':'); ax[1].axvline(n//2, c='r', ls='--')
ax[0].legend(); ax[1].legend(); plt.show()
{k: (round(cu.statistic, 4), cu.reject, cu.change_index_one_based) for k, (_, _, cu) in out.items()}"""),
    md("## 6. §3 CUSUM golden test — ê=(1,1,1,3) → D=(−2,−4,−6,0), τ²=12, T=√3/2, k̂=3, 비기각 (c=1.3397)"),
    code("""cu = cusum_squares_test([1., 1., 1., 3.], tie_policy=ArgmaxTiePolicy.FIRST, degenerate_tau_policy=DegenerateTauPolicy.RAISE)
assert np.allclose(cu.centered_path, [-2, -4, -6, 0]) and np.isclose(cu.tau_squared, 12) and np.isclose(cu.statistic, np.sqrt(3)/2)
assert cu.change_index_one_based == 3 and not cu.reject
print('golden OK', cu.statistic)"""),
    md("## 7. 다섯 GARCH 생성모형 (원문 p.8) baseline 표본 경로"),
    code("""from paper2020.simulation import TABLE_CASES
fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
for ax, case in zip(axes, [c for c in TABLE_CASES if c.is_size]):
    r = simulate_dgp(case.baseline_model, case.baseline_parameters, length=1000, assumptions=dgp_assumptions(case.baseline_model), seed=1)
    ax.plot(r.observations, lw=.5); ax.set_ylabel(case.baseline_model.value, fontsize=8)
plt.tight_layout(); plt.show()"""),
    md("## 8. (감사 추가) 테스트 구간 분산 정책 A / B / C — `reports/TEST_TIME_RECURSION_AUDIT.md`\n"
       "- **A** σ̂²_t = ĝ(y²_{t−1}, σ̃²_{t−1}) — 원문 Step 3 식을 테스트에 문자 그대로 연장 (`ASSUMPTION`)\n"
       "- **B** σ̂²_t = ĝ(y²_{t−1}, σ̂²_{t−1}) — §3 모수 GARCH 재귀와 같은 구조 (`SENSITIVITY`)\n"
       "- **C** ĝ 자체를 반복 재적합 — Remark 2 가 기각, 미구현. B ≠ C\n"
       "원문은 테스트 구간 정책을 명시하지 않는다 → `UNRESOLVED`. 아래는 두 정책이 미래 관측을 쓰지 않음을 같은 코드 경로로 보이는 경계 검사다."),
    code("""fcs = {}
te2 = test.copy(); k = 100; te2[k:] = te2[k:] * 5 + 2
for pol in (TestVarianceInputPolicy.OBSERVED_PROXY, TestVarianceInputPolicy.RECURSIVE_PREDICTION):
    v0 = forecast_test_variance(fit, test, m=5, p=1, q=1, input_policy=pol, proxy_boundary=BoundaryPolicy.CARRY, lag_boundary=BoundaryPolicy.CARRY, prediction_policy=VariancePredictionPolicy.RAW, variance_floor=None).variance
    v1 = forecast_test_variance(fit, te2, m=5, p=1, q=1, input_policy=pol, proxy_boundary=BoundaryPolicy.CARRY, lag_boundary=BoundaryPolicy.CARRY, prediction_policy=VariancePredictionPolicy.RAW, variance_floor=None).variance
    assert np.array_equal(v0[:k+1], v1[:k+1]) and not np.array_equal(v0[k+1:], v1[k+1:])
    print(pol.value, ': y_j (j>=100) 변경 -> sigma_hat^2_0..100 불변 OK')"""),
    md("### A 가 변화를 흡수하는 메커니즘 — ê² 의 long-run variance / τ̂²\nA 에서 1 보다 작으면 CUSUM 분모가 실제 장기분산을 과대평가 → T̂ 축소 → size 과소."),
    code("""from paper2020.ablation import _lrv_ratio
for k_, (v, e, cu) in out.items():
    print(f'{k_:22s} LRV(e^2)/tau^2 = {_lrv_ratio(e):.3f}   T = {cu.statistic:.3f}')"""),
]


nb2 = nbf.v4.new_notebook()
nb2.cells = [
    md("# 02 · 결과 — Tables 1–6 (Monte Carlo) 와 Figures 1–4 (실증) 원문 대비\n"
       "이 노트북은 **재계산하지 않는다**. `results/` 의 저장된 실행 결과만 읽는다 (재실행은 README 의 명령)."),
    code(SETUP + "\nimport json\nfrom IPython.display import Image, display"),
    md("## 1. Monte Carlo — lane별 원문 대비 rejection rate\n"
       "- `clip` = 실행 레인 (OBSERVED_PROXY, 비양수 분산 → machine-eps clip; 사전등록상 sensitivity)\n"
       "- `raise` = 사전등록 primary (비양수 분산 시 해당 반복 실패; 실패는 비기각으로 계산한 `reject_rate_all` 병기)\n"
       "- `recursive` = 테스트 분산 lag 를 재귀 예측으로 채우는 민감도 레인\n"
       "- `status`: COMPLETE_1000 / PARTIAL_k (k 회만 실행)"),
    code("""import subprocess
subprocess.run([sys.executable, str(ROOT / 'scripts' / 'summarize_mc.py')], check=True, capture_output=True)
lanes = {l: pd.read_csv(ROOT / 'results' / f'mc_summary_{l}.csv') for l in ('clip', 'raise', 'recursive') if (ROOT / 'results' / f'mc_summary_{l}.csv').exists()}
for l, s in lanes.items():
    print(l, len(s), 'rows;', s.status.value_counts().to_dict())
lanes['clip'].pivot_table(index=['table', 'label'], columns='n', values=['reject_rate_ok', 'paper'], sort=False)"""),
    code("""fig, axes = plt.subplots(1, len(lanes), figsize=(5 * len(lanes), 4.5), sharey=True)
for ax, (l, s) in zip(np.atleast_1d(axes), lanes.items()):
    s5 = s[s.n == 500]
    size = s5.label == 'size'
    ax.scatter(s5.paper[~size], s5.reject_rate_ok[~size], s=14, label='power rows')
    ax.scatter(s5.paper[size], s5.reject_rate_ok[size], s=30, marker='x', c='r', label='size rows')
    ax.plot([0, 1], [0, 1], c='k', lw=.6); ax.set_xlabel('paper (n=500)'); ax.set_title(f'lane {l}')
np.atleast_1d(axes)[0].set_ylabel('replication rejection rate'); np.atleast_1d(axes)[0].legend(); plt.tight_layout(); plt.show()"""),
    code("""rows = []
for l, s in lanes.items():
    for n, g in s.groupby('n'):
        sz, pw = g[g.label == 'size'], g[g.label != 'size']
        rows.append({'lane': l, 'n': n, 'rows': len(g), 'reps/row (min)': g.reps_attempted.min(),
                     'mean size (paper)': round(sz.paper.mean(), 3), 'mean size (repl)': round(sz.reject_rate_ok.mean(), 3),
                     'mean power (paper)': round(pw.paper.mean(), 3), 'mean power (repl)': round(pw.reject_rate_ok.mean(), 3),
                     'MAE vs paper': round((g.reject_rate_ok - g.paper).abs().mean(), 3)})
pd.DataFrame(rows)"""),
    md("## 2. 실증 (§5) — Yahoo Finance 대체 출처 (SOURCE_SUBSTITUTED; Investing.com HTTP 403)"),
    code("""emp = json.loads((ROOT / 'results' / 'empirical_summary.json').read_text())
tab = []
for name, d in emp.items():
    for lane in ('raise', 'clip', 'recursive_clip'):
        v = d.get(lane, {})
        tab.append({'series': name, 'lane': lane, 'paper_T': d['paper']['stat'], 'repl_T': v.get('statistic'),
                    'paper_date': d['paper']['date'], 'repl_date': v.get('change_date'), 'reject': v.get('reject_5pct'),
                    'svr': v.get('best_svr'), 'nonpos_var': v.get('n_nonpositive_variance')})
pd.DataFrame(tab)"""),
    code("""for f in sorted((ROOT / 'figures').glob('fig*.png')):
    print(f.name); display(Image(filename=str(f)))"""),
    code("""manifest = json.loads((ROOT / 'data' / 'raw' / 'data_manifest.json').read_text())
pd.DataFrame(manifest['series']).T[['symbol', 'label', 'paper_period', 'first_date', 'last_date', 'n_levels', 'sha256']]"""),
    md("## 3. (감사 추가) 집계 정합성 — `reports/STATISTICAL_ACCOUNTING_AUDIT.md`\n셀 = 원문 135칸. 조건부 기각률 + 실패율 + Wilson 95% CI. 45행 평균은 서술 통계일 뿐."),
    code("""audit = json.loads((ROOT / 'results' / 'accounting_audit.json').read_text())
ovr = pd.read_csv(ROOT / 'reports' / 'ORIGINAL_VS_REPRODUCED.csv')
print({l: audit['lanes'][l] for l in audit['lanes']}); print(audit['paired_A_vs_B_n500'])
ovr[['table', 'label', 'n', 'paper', 'clip_status', 'clip_rate_cond_ok', 'clip_ci_lo', 'clip_ci_hi', 'raise_rate_cond_ok', 'raise_fail_rate', 'recursive_rate_cond_ok']].head(20)"""),
    md("## 4. (감사 추가) 통제 ablation — 같은 DGP 경로(sha256) 위 한 요인씩, 기준점 A·B 각각 (n=500, 300회)"),
    code("""abl = pd.read_csv(ROOT / 'reports' / 'ABLATION_RESULTS.csv')
abl.pivot_table(index=['test_policy', 'variant'], columns='case_id', values='reject_rate', sort=False).round(3)"""),
    code("""abl.groupby(['test_policy', 'variant'], sort=False)[['mean_lrv_ratio', 'mean_r2_sigma_vs_recent4', 'max_abs_dstat_vs_baseline', 'sec_per_rep']].mean().round(4)"""),
    md("## 5. (감사 추가) 실증 원문 대비 — SOURCE_SUBSTITUTED, 임계값 1.3397 미만은 비유의"),
    code("""pd.read_csv(ROOT / 'reports' / 'EMPIRICAL_ORIGINAL_VS_REPRODUCED.csv')[['series', 'training', 'test_policy', 'nonpos_policy', 'paper_T', 'repl_T', 'repl_reject', 'paper_date', 'repl_date']]"""),
]


for nb, name in ((nb1, "01_SVR_GARCH_CUSUM.ipynb"), (nb2, "02_SVR_GARCH_CUSUM_결과시각화.ipynb")):
    nb.metadata["kernelspec"] = {"name": "hongik_26_2", "display_name": "Python (hongik 26-2 .venv CUDA)", "language": "python"}
    nbf.write(nb, ROOT / "notebooks" / name)
print("ok")
