# Track B reproducibility gap register

This register separates paper-stated facts from choices that the English original does not operationally specify. No item below may be silently filled from convention or tuned against the published targets.

Severity: **P0** blocks a defensible exact-protocol claim; **P1** materially affects numerical results; **P2** affects parity, reporting, or edge cases. `B2 gate` means a decision is required before implementation; `B3 gate` means method work can proceed but exact numerical reproduction cannot be claimed.

| ID | Severity | Source/evidence | Gap or conflict | Required resolution | Gate |
|---|---|---|---|---|---|
| GRID-01 | P0 | PDF 34 p.5 | Only ranges for `C`, paper `gamma^2`, and `epsilon`; no finite grid coordinates or step sizes | Obtain author code/supplement or preregister a clearly labeled method-equivalent grid | B2 |
| ENV-01 | P0 | PDF 36 p.7 | Windows 10/R 3.5.1/e1071/fGarch stated, but package versions, calls, RNG, locale absent; R unavailable locally | Keep Python and R-parity lanes separate; seek approval for isolated R environment | B3 |
| ENV-02 | P1 | PDF 34 p.5; local sklearn 1.9.0 | Scaling and solver defaults/tolerances are unreported | Explicitly lock scaling and solver metadata; compare with documented e1071 behavior before parity claim | B2 |
| MATH-01 | P2 | PDF 32 p.3 | Eq. (3) displays slack `>=0` while prose says `>0` | Follow displayed equation/standard SVR and record discrepancy | No |
| MATH-02 | P2 | PDF 32 p.3 | Prose calls scalar intercept `b` a vector and says `w,b` match `x` dimension | Treat `b` as scalar per displayed model; retain source error | No |
| MATH-04 | P0 | PDF 34–35 pp.5–6 | First usable lag and proxy behavior at train/validation/test boundaries are unspecified | Define reset/carry candidates and decide before B2; sensitivity path separate | B2 |
| MATH-05 | P1 | PDF 34 p.5 | SVR target is implied to be variance proxy but not independently written as a target vector | Mark proxy-target choice as inference and test construction explicitly | B2 |
| MATH-06 | P0 | PDF 35–36 pp.6–7 | Final model's test-time variance recursion/feature update is unspecified | Do not invent recursive rule; resolve author-code evidence or preregister candidates | B2 |
| MATH-07 | P2 | PDF 35 p.6 Remark 3 | MAPE sentence calls `sigma_tilde^2` unobservable, which conflicts with its proxy construction | Preserve wording; use MAE only in exact lane | No |
| MATH-08 | P1 | PDF 35 p.6 | Parametric residual recursion prints `beta_hat*sigma_{t-1}^2` without a hat on prior variance | Separate literal transcription from likely recursive-estimate interpretation | B2 |
| MATH-09 | P1 | PDF 35 p.6 | Printed initial condition `sigma_0^2=y_1` can be negative and is dimensionally inconsistent | Do not silently replace by `y_1^2`; require explicit policy branches | B2 |
| MATH-10 | P0 | PDF 34–36 pp.5–7 | No policy for non-positive SVR variance prediction or square root | Select fail-closed/literal and separately labeled stabilization policy before B2 | B2 |
| MATH-11 | P2 | PDF 36 p.7 | `tau_hat=0`, floating negative variance, and argmax ties are unspecified | Define deterministic error and tie policies; golden-test them | B2 |
| MATH-12 | P2 | PDF 36 p.7 | SVR-residual Brownian-bridge justification is asserted in prose, not proved there | Do not overstate guarantees beyond paper claim | No |
| SIM-01 | P0 | PDF 36 p.7 | H1 change is only “in the middle of the time series”; midpoint of generated `2n` is the test boundary | Establish whether change is test midpoint (`n+n/2`) or another index; preserve alternatives | B3 |
| SIM-02 | P0 | PDF 36–39 pp.7–10 | DGP initial values, burn-in, seed/RNG, and common-random-number policy absent | Preregister explicit method-equivalent policy; exact parity remains limited without author code | B3 |
| SIM-03 | P1 | PDF 36 p.7 | Rounding convention for generic `0.7n/0.3n` absent | Listed n values are exact integers; constrain exact reproduction to those n values | No |
| SIM-04 | P1 | PDF 38 p.9 | `N=20` truncation maps both positive and negative extremes to `+20` literally | Paper-literal rule is primary; sign-preserving clipping only as separate sensitivity | B2 |
| SIM-05 | P1 | PDF 37–38 pp.8–9 | `log(y^2)` at zero/tiny values has no numerical policy | Define literal failure and separately labeled floor sensitivity | B2 |
| SIM-06 | P1 | PDF 39 p.10 Table 6 | GJR destination rows print extraneous `b=0` although GJR equation has no `b` | Preserve as table inconsistency; do not add `b` to GJR implementation | No |
| SIM-07 | P1 | PDF 38–39 pp.9–10 | Table 6 AGARCH source baseline uses `beta=0.3`; Table 2 baseline uses `beta=0.4` | Use each printed table's own baseline; do not normalize them | No |
| SIM-08 | P2 | Tables 1–6 | Rows omit unchanged post-change parameters and do not state the carry-forward convention | Treat carry-forward as inferred; serialize the full parameter vector in each B3 run | B3 |
| SIM-09 | P1 | Tables 1–6 | No Monte Carlo SE/CI or numerical acceptance tolerance | Report binomial uncertainty and preregister comparison tolerances before B3 | B3 |
| DATA-01 | P0 | PDF 40–42 pp.11–13; local inventory | Original Investing.com snapshots, URLs, symbols, fields, and hashes are absent locally | Acquire/licence and hash original-equivalent snapshots after explicit user approval | B3 |
| DATA-02 | P0 | PDF 40 p.11 | S&P full start `2012-01-03` conflicts with training example start `2012-04-02`; other exact midpoints absent | Preserve both S&P interpretations; determine row-count midpoint after data recovery | B3 |
| DATA-03 | P0 | PDF 40–41 pp.11–12 | Log-return formula/base/percent scale, price field, missing-day and first-row rules absent | Preregister transformation candidates without looking at target fit; report sensitivity separately | B3 |
| DATA-04 | P1 | PDF 40 p.11 Figure 1 | ACF confidence bands and PACF estimator method are unreported | Compare R 3.5.1 defaults in parity lane; label Python alternative | B3 |
| DATA-05 | P1 | Investing.com source | Current downloadable history may differ from the 2020-era snapshot | Store acquisition time, URL/licence, raw checksum, and immutable copy | B3 |

## Current blockers by fidelity level

- **Level 1 mathematical reconstruction:** source formulas are mapped; B2 implementation is gated by explicit edge-case policies, not by missing formula transcription.
- **Level 2 algorithmic reproduction:** blocked from an *exact* claim by GRID-01, MATH-04, MATH-06, and MATH-10. A labeled method-equivalent implementation is possible after user decisions.
- **Level 3 simulation replication:** blocked from exact-protocol claim by SIM-01, SIM-02, GRID-01, and ENV-01. Smoke execution must never be reported as the 1000-repetition protocol.
- **Level 4 empirical reproduction:** blocked by DATA-01–03. The target dates/statistics alone are not sufficient data evidence.

## Non-negotiable separation rules

1. Paper-literal, inferred method-equivalent, typo-corrected, and sensitivity results must have distinct labels and output paths.
2. Rejection (`T >= 1.3397`) and change-location argmax are separate outputs.
3. Retrospective CUSUM must not be reframed as an online alarm.
4. ETF or Track A data must not substitute for the paper's three original series.
5. No result may be tuned post hoc to published table cells, statistics, or dates.
