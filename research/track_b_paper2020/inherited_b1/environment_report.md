# Track B Phase B1 environment report

Snapshot time: **2026-10-11 14:08 KST**. All checks were read-only. No package was installed or changed.

## Execution context

| Item | Observed value | Replication implication |
|---|---|---|
| Worktree | `/home/sieg/projects-wsl/hongik_univ_26_2/SA/.worktrees/SA_ETF/codex-b` | Isolated Track B worktree |
| Branch / baseline | `codex/r2h-track-b` / `d5c2f792022d3c8efdbc52bdecd246b04bf32c45` | Correct starting point verified |
| OS layer | WSL2; Linux `6.18.33.2-microsoft-standard-WSL2`; x86_64 | Different from paper's Windows 10 |
| Locale / timezone | `C.UTF-8` / Asia-Seoul (`KST`, UTC+09:00) | Date and CSV parsing must be explicit |
| Python | 3.12.3 | Primary method-equivalent lane |
| Interpreter | `/home/sieg/projects-wsl/hongik_univ_26_2/.venv/bin/python` | Shared venv; not modified |
| Environment manager | `uv` available; no conda or Poetry observed | No paper2020-specific lock yet |
| Jupyter kernel | `hongik_26_2` | Points to the observed shared interpreter |
| GPU | Not required or assumed | No GPU check was needed |

## Observed Python/Jupyter package snapshot

| Package | Version/status |
|---|---:|
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| SciPy | 1.18.0 |
| matplotlib | 3.11.0 |
| seaborn | 0.13.2 |
| plotly | 6.8.0 |
| scikit-learn | 1.9.0 |
| statsmodels | 0.14.6 |
| PyYAML | 6.0.3 |
| JupyterLab / notebook | 4.6.0 / 7.6.0 |
| ipykernel / nbconvert / nbformat | 7.3.0 / 7.17.1 / 5.10.4 |
| FinanceDataReader | 0.9.202 |
| `arch` | **absent** |
| `yfinance` | **absent** |
| R / Rscript | **absent** |

The worktree has no `pyproject.toml`, requirements file, or lockfile. The observed versions are an environment snapshot, not a reproducible dependency lock. Phase B1 uses only already-installed packages and does not create or alter an environment.

## Original-author environment

The English original states only the following on physical PDF page 36 (article page 7):

- Windows 10
- R 3.5.1
- packages `e1071` and `fGarch`

Package versions, BLAS/compiler, R RNG kind/seed, locale, and function arguments are not reported. This is insufficient to reconstruct exact environment parity by itself.

## Environment policy selected for later phases

1. **Primary lane — method-equivalent Python/Jupyter.** Use explicit parameters and custom equations rather than relying on model-name equivalence. `sklearn.svm.SVR` may be used only with the paper-to-library RBF mapping `gamma_sklearn = 1/(2*gamma_paper_squared)` and an explicit preprocessing policy.
2. **Secondary lane — R parity.** Attempt only after user approval for a separate, pinned environment. It must not modify the shared venv. R is currently absent, so this lane is blocked.
3. **No hidden defaults.** Scaling, tolerance, shrinking, cache, seed/RNG, initialization, burn-in, positivity handling, and tie-breaking must be serialized in run metadata.
4. **No `arch` substitution.** Its absence does not block equation-faithful DGP implementation; library classes must not replace the five paper equations by name.
5. **No target-driven tuning.** The three empirical target statistics/dates and Tables 1–6 values are comparison targets, not tuning objectives.

## Python–R parity risks

- `e1071` and scikit-learn both use libsvm-style RBF coefficients, but the paper writes a width-squared parameter. Passing the paper number directly as sklearn `gamma` is wrong.
- Feature scaling is unreported. `e1071` commonly scales inputs by default, while sklearn does not; exact behavior must be established rather than inferred.
- Solver tolerances and stopping/default behavior are not reported and differ by package/version.
- R and NumPy random-number generators do not yield identical streams from a numerically equal seed.
- Return scale (ratio or percent) changes the meaning of `C`, `epsilon`, the variance proxy, and numerical conditioning.
- Current locale and WSL filesystem behavior are not evidence of the authors' Windows data parsing behavior.

## Reproducibility status

| Lane | Status | Blocking fact |
|---|---|---|
| B00 contract notebook | READY | Existing kernel and B1 packages are sufficient |
| B2 Python method-equivalent | PARTIAL | Protocol decisions in the gap register remain unresolved |
| R 3.5.1 environment parity | BLOCKED | R absent; exact package versions and author call arguments unknown |
| B3 exact empirical figures | BLOCKED | Original Investing.com snapshots absent |
| B3 full 1000-repetition tables | PARTIAL | Compute budget and unresolved simulation protocol decisions |

Detailed source evidence is retained in `paper2020/tickets/B1_empirical_environment_audit.md`.
