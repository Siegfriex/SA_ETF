# B1 simulation source audit

## Scope and source precedence

- Source of truth: Lee, Kim, and Lee (2020), *Entropy* 22(5), 578, English original.
- Local source: `/home/sieg/projects-wsl/hongik_univ_26_2/SA/1011/Hybrid_SVR_GARCH_CUSUM_.pdf`.
- Relevant physical PDF pages: 36--39, corresponding to printed article pages 7--10 of 16.
- Verification method: layout-preserving text extraction was checked against rendered PDF pages. The rendered formulae and table entries below take precedence over extracted text.
- This ticket records the paper's experiment contract; it does not implement or run the experiment.

## Global Section 4 protocol

Source: physical PDF page 36 / printed page 7.

1. The paper evaluates empirical size and power at nominal level `0.05` for `n in {500, 1000, 2000}`.
2. For every replication it generates a time series of total length `2n` from a prescribed DGP.
3. The first `0.7n` observations form the training set.
4. The following `0.3n` observations form the validation set.
5. The remaining `n` observations form the CUSUM testing set.
6. Section 3's estimation scheme is used with moving-average proxy length `m = 5`.
7. Steps 1--3 are repeated `1000` times; rejection proportions are reported as empirical size or power.
8. For power, the paper says only that the change point is "in the middle of the time series."
9. Innovations are `e_t`, `t=1,...,2n`, i.i.d. normal with mean zero and variance one.
10. The authors report Windows 10, R 3.5.1, and packages `e1071` and `fGarch`.
11. Section 3 immediately before Section 4 says only `p=q=1` is considered.

### Split indexing contract and unresolved point

For the paper's listed `n` values, `0.7n` and `0.3n` are integers, so the nominal split counts are unambiguous: `(350,150,500)`, `(700,300,1000)`, and `(1400,600,2000)` for `(train, validation, test)`.

The location of the H1 transition is **not operationally indexed**. A literal midpoint of the generated `2n` series is the boundary between validation and test, which would leave the whole test segment post-change and would not define an internal retrospective change point. It is therefore plausible that "middle" means the midpoint of the `n`-observation testing segment (global index about `n + n/2`), but that is an inference, not a paper-stated fact. Phase B2/B3 must preserve this as an unresolved protocol question and must not silently choose an index convention.

## DGP equations

Source: physical PDF page 37 / printed page 8. In all five models, `y_t = sigma_t e_t` and `e_t ~ iid N(0,1)`.

### GARCH(1,1)

```text
sigma_t^2 = omega + alpha y_{t-1}^2 + beta sigma_{t-1}^2,
alpha >= 0, beta >= 0.
```

Null/baseline parameters: `omega=0.3, alpha=0.3, beta=0.3`.

### AGARCH(1,1)

```text
sigma_t^2 = omega + alpha (y_{t-1} - b)^2 + beta sigma_{t-1}^2,
alpha >= 0, beta >= 0.
```

Null/baseline parameters: `omega=0.3, alpha=0.3, beta=0.4, b=1`.

The shift `b` is inside the squared observation term. It is not an intercept, sign indicator, or threshold on the innovation.

### GJR-GARCH(1,1)

```text
sigma_t^2 = omega + alpha_1 (y_{t-1}^+)^2
                    + alpha_2 (y_{t-1}^-)^2
                    + beta sigma_{t-1}^2,
y_t^+ = max(y_t, 0),
y_t^- = -min(y_t, 0),
alpha_1 >= 0, alpha_2 >= 0, beta >= 0.
```

Null/baseline parameters: `omega=0.3, alpha_1=0.3, alpha_2=0.4, beta=0.3`.

Both positive and negative parts are nonnegative before squaring. This is not safely replaceable by a library's default asymmetric-GARCH parameterization without an explicit equivalence proof.

### TGARCH(1,1)

```text
sigma_t = omega + alpha |y_{t-1}| + beta sigma_{t-1},
alpha >= 0, beta >= 0.
```

Null/baseline parameters: `omega=0.3, alpha=0.3, beta=0.3`.

The recursion is on conditional standard deviation `sigma_t`, not conditional variance `sigma_t^2`.

### Log-linear GARCH(1,1)

The paper calls this a specific variation of EGARCH `(p,q)`:

```text
log(sigma_t^2) = omega + alpha log(y_{t-1}^2)
                       + beta log(sigma_{t-1}^2).
```

Null/baseline parameters: `omega=0.3, alpha=0.3, beta=0.3`.

The paper does not state how `log(0)` or values numerically close to zero are handled.

## Truncated residual contract

Source: physical PDF page 38 / printed page 9.

For Table 5, the paper constructs the CUSUM statistic using

```text
e_t_tilde := e_t_hat I(|e_t_hat| < N) + N I(|e_t_hat| >= N),
N = 20.
```

This is the exact rendered formula. In particular, the second term is `+N`, not `N sign(e_t_hat)` and not `-N` for a negative outlier. Therefore, taken literally, every residual with magnitude at least 20 is mapped to positive 20. A sign-preserving clipping rule would be a different method and must be labeled as a sensitivity implementation if later studied.

For Table 6, the paper says the same truncated residual `e_t_tilde` is used in Cases 1 and 2, the two cases that involve log-linear GARCH. It does not say to use truncation in Cases 3 and 4.

## Tables 1--5: exact targets

Columns in every table are `n=500`, `n=1000`, `n=2000`. A right-arrow row gives the post-change parameters; parameters not shown on that row appear to remain at baseline, but the paper does not separately state this convention in prose.

### Table 1 — GARCH(1,1)

Source: physical PDF page 38 / printed page 9. Baseline `(omega,alpha,beta)=(0.3,0.3,0.3)`.

| Case | Post-change specification | 500 | 1000 | 2000 |
|---|---|---:|---:|---:|
| size | H0 baseline | 0.023 | 0.038 | 0.055 |
| change of omega | `omega=1` | 0.761 | 0.826 | 0.956 |
| change of omega | `omega=0.1` | 0.612 | 0.792 | 0.970 |
| change of alpha,beta | `alpha=0.1, beta=0.2` | 0.355 | 0.651 | 0.949 |
| change of alpha,beta | `alpha=0.4, beta=0.5` | 0.649 | 0.802 | 0.952 |
| mixed parameters | `omega=0.7, alpha=0.1` | 0.871 | 0.969 | 0.981 |
| mixed parameters | `omega=0.1, beta=0.1` | 0.848 | 0.952 | 0.964 |

### Table 2 — AGARCH(1,1)

Source: physical PDF page 38 / printed page 9. Baseline `(omega,alpha,beta,b)=(0.3,0.3,0.4,1)`.

| Case | Post-change specification | 500 | 1000 | 2000 |
|---|---|---:|---:|---:|
| size | H0 baseline | 0.031 | 0.040 | 0.030 |
| change of omega | `omega=1` | 0.881 | 0.951 | 0.975 |
| change of omega | `omega=0.1` | 0.260 | 0.613 | 0.904 |
| change of alpha,beta | `alpha=0.1, beta=0.1` | 0.783 | 0.939 | 0.975 |
| change of alpha,beta | `alpha=0.6` | 0.762 | 0.863 | 0.941 |
| change of b | `b=0` | 0.591 | 0.897 | 0.976 |
| change of b | `b=3` | 0.898 | 0.936 | 0.957 |
| mixed parameters | `omega=0.1, alpha=0.5, beta=0.4, b=0` | 0.565 | 0.726 | 0.846 |
| mixed parameters | `omega=0.1, beta=0.6, b=2` | 0.879 | 0.926 | 0.966 |

### Table 3 — GJR-GARCH(1,1)

Source: physical PDF page 38 / printed page 9. Baseline `(omega,alpha_1,alpha_2,beta)=(0.3,0.3,0.4,0.3)`.

| Case | Post-change specification | 500 | 1000 | 2000 |
|---|---|---:|---:|---:|
| size | H0 baseline | 0.021 | 0.029 | 0.028 |
| change of omega | `omega=1` | 0.851 | 0.912 | 0.947 |
| change of omega | `omega=0.1` | 0.644 | 0.787 | 0.866 |
| change of alpha_i,beta | `alpha_1=0.5, alpha_2=0.8` | 0.418 | 0.695 | 0.879 |
| change of alpha_i,beta | `alpha_2=0.1, beta=0.1` | 0.421 | 0.751 | 0.888 |
| mixed parameters | `omega=0.5, alpha_2=0.6` | 0.657 | 0.863 | 0.928 |
| mixed parameters | `omega=0.1, beta=0.7` | 0.717 | 0.857 | 0.925 |

### Table 4 — TGARCH(1,1)

Source: physical PDF page 39 / printed page 10. Baseline `(omega,alpha,beta)=(0.3,0.3,0.3)`.

| Case | Post-change specification | 500 | 1000 | 2000 |
|---|---|---:|---:|---:|
| size | H0 baseline | 0.037 | 0.049 | 0.059 |
| change of omega | `omega=1` | 0.718 | 0.795 | 0.879 |
| change of omega | `omega=0.1` | 0.647 | 0.840 | 0.902 |
| change of alpha,beta | `alpha=0.1, beta=0.2` | 0.805 | 0.886 | 0.913 |
| change of alpha,beta | `alpha=0.4, beta=0.5` | 0.735 | 0.838 | 0.897 |
| mixed parameters | `omega=0.7, alpha=0.1` | 0.907 | 0.970 | 0.994 |
| mixed parameters | `omega=0.1, beta=0.1` | 0.499 | 0.674 | 0.772 |

### Table 5 — log-linear GARCH(1,1)

Source: physical PDF page 39 / printed page 10. Baseline `(omega,alpha,beta)=(0.3,0.3,0.3)`. The `N=20` truncated residual contract applies.

| Case | Post-change specification | 500 | 1000 | 2000 |
|---|---|---:|---:|---:|
| size | H0 baseline | 0.047 | 0.039 | 0.037 |
| change of omega | `omega=1` | 0.906 | 0.984 | 0.997 |
| change of omega | `omega=0.1` | 0.228 | 0.382 | 0.507 |
| change of alpha,beta | `alpha=-0.1, beta=-0.2` | 0.868 | 0.946 | 0.976 |
| change of alpha,beta | `alpha=-0.4, beta=0.5` | 0.917 | 0.985 | 1.000 |
| mixed parameters | `omega=0.7, beta=0.5` | 0.820 | 0.973 | 0.998 |
| mixed parameters | `omega=0.1, alpha=-0.1` | 0.862 | 0.944 | 0.971 |

## Table 6: DGP changes and exact targets

Source: physical PDF page 39 / printed page 10. Table 6 reports powers only; it has no size row.

### Case 1 — GARCH to log-GARCH

Source baseline shown above the block: GARCH `(omega,alpha,beta)=(0.3,0.3,0.3)`. Truncated residuals apply.

| Destination log-GARCH parameters shown | 500 | 1000 | 2000 |
|---|---:|---:|---:|
| `omega=1` | 0.879 | 0.946 | 0.956 |
| `alpha=0.1, beta=0.1` | 0.668 | 0.918 | 0.969 |

### Case 2 — log-GARCH to GARCH

Source baseline shown above the block: log-GARCH `(omega,alpha,beta)=(0.3,0.3,0.3)`. Truncated residuals apply.

| Destination GARCH parameters shown | 500 | 1000 | 2000 |
|---|---:|---:|---:|
| `omega=1` | 0.907 | 0.970 | 0.994 |
| `alpha=0.1, beta=0.1` | 0.499 | 0.674 | 0.772 |

### Case 3 — TGARCH to AGARCH

Source baseline shown above the block: TGARCH `(omega,alpha,beta)=(0.3,0.3,0.3)`.

| Destination AGARCH parameters shown | 500 | 1000 | 2000 |
|---|---:|---:|---:|
| `omega=1, b=1` | 0.728 | 0.795 | 0.879 |
| `alpha=0.1, beta=0.1, b=1` | 0.851 | 0.891 | 0.919 |

### Case 4 — AGARCH to GJR-GARCH

The second baseline block is AGARCH `(omega,alpha,beta,b)=(0.3,0.3,0.3,1)`. Note that this baseline has `beta=0.3`, unlike Table 2's AGARCH H0 baseline `beta=0.4`.

| Destination GJR parameters as printed | 500 | 1000 | 2000 |
|---|---:|---:|---:|
| `omega=0.7, alpha_1=0.3, alpha_2=0.7, b=0` | 0.549 | 0.861 | 0.958 |
| `alpha_1=0.1, alpha_2=0.3, beta=0.1, b=0` | 0.802 | 0.931 | 0.976 |

`b=0` is printed in both destination rows even though `b` does not occur in the paper's GJR-GARCH equation. This is internally inconsistent and must be retained as a source ambiguity/possible table typo, not interpreted as a GJR parameter.

## Reproduction gaps that the English original does not close

The following details are absent or underspecified on printed pages 7--10 and cannot be filled in as paper facts:

1. Exact H1 transition index and whether the transition is applied only inside the test segment.
2. Initial values for `y_0`, `sigma_0`, or `sigma_0^2` for every recursion.
3. Any burn-in length or disposal rule.
4. Random seed, R RNG kind/version behavior, or whether common random numbers are used across cases.
5. Exact finite grids/step sizes for `C`, `gamma^2`, and epsilon; only continuous ranges are specified elsewhere in the paper.
6. Tie-breaking when multiple hyperparameter combinations have equal validation MAE.
7. Exact SVM solver tolerances, cache behavior, scaling/standardization, and `e1071` defaults explicitly relied upon.
8. Initialization of the moving-average variance proxy at early indices and exact boundary indexing between train/validation/test.
9. Positivity or numerical fallback rules when SVR predicts a non-positive conditional variance.
10. Numerical handling of `log(y_{t-1}^2)` when `y_{t-1}` is zero or tiny.
11. Whether post-change rows leave every unlisted parameter at its baseline value (strongly implied by table layout, but not explicitly defined).
12. Whether `N=20`'s non-sign-preserving formula is intentional or a typographical error.
13. Meaning of the extraneous `b=0` in Table 6's GJR destination rows.
14. Monte Carlo uncertainty measures: no standard errors or confidence intervals accompany the rejection proportions.
15. Exact denominator/convention for empirical rejection fractions beyond `1000` replications.

## Phase B2/B3 guardrails implied by this audit

- Encode all five recursions directly from the displayed equations before comparing to any library implementation.
- Do not substitute an `arch` package class solely by model name.
- Keep TGARCH's standard-deviation recursion distinct from variance recursions.
- Preserve the paper-literal truncated-residual rule as the primary replication; if a sign-preserving alternative is tested, label it separately.
- Record an explicit decision ticket for the H1 change index, initial conditions/burn-in, log-zero handling, unlisted table parameters, Table 6 `b=0`, and RNG policy before claiming exact-protocol replication.
- A smoke run can validate stability, but only 1000 replications for every reported condition can be called the paper protocol.
