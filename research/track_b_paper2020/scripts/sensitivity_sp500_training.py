"""S&P 500 sensitivity: paper's quoted example training period 2012-04-02..2014-04-16
(70/30 train/validation inside it), test = every return after 2014-04-16. Both test policies."""
import dataclasses, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from paper2020.runner import make_spec, VARIANCE_FLOOR
from paper2020.svr_garch import select_and_refit, forecast_test_variance, TestVarianceInputPolicy, VariancePredictionPolicy
from paper2020.cusum import standardized_residuals, NonPositiveVariancePolicy, cusum_squares_test, ArgmaxTiePolicy, DegenerateTauPolicy
df = pd.read_csv(ROOT / "data/raw/sp500.csv", parse_dates=["date"])
r = pd.Series(100 * np.diff(np.log(df.close.to_numpy())), index=df.date.iloc[1:])
fitb = r["2012-04-02":"2014-04-16"].to_numpy(); test = r[r.index > "2014-04-16"]
out = {}
for pol in TestVarianceInputPolicy:
    spec = dataclasses.replace(make_spec("full", clip=True), test_variance_input_policy=pol)
    ntr = int(round(0.7 * fitb.size))
    fit = select_and_refit(fitb[:ntr], fitb[ntr:], parameter_grid=spec.parameter_grid, fit_policy=spec.fit_policy, m=5, p=1, q=1,
        validation_proxy_boundary=spec.validation_proxy_boundary, validation_lag_boundary=spec.validation_lag_boundary,
        validation_variance_policy=spec.validation_variance_policy, variance_floor=spec.variance_floor, tie_policy=spec.selection_tie_policy)
    # NOTE: returns between 2012-01-04..2012-03-30 are not used; history carried = fit block
    fc = forecast_test_variance(fit, test.to_numpy(), m=5, p=1, q=1, input_policy=pol, proxy_boundary=spec.test_proxy_boundary,
        lag_boundary=spec.test_lag_boundary, prediction_policy=VariancePredictionPolicy.RAW, variance_floor=None)
    e = standardized_residuals(test.to_numpy(), fc.variance, nonpositive_policy=NonPositiveVariancePolicy.CLIP, variance_floor=VARIANCE_FLOOR)
    cu = cusum_squares_test(e, tie_policy=ArgmaxTiePolicy.FIRST, degenerate_tau_policy=DegenerateTauPolicy.RAISE)
    out[pol.value] = {"statistic": round(cu.statistic, 4), "reject": cu.reject, "change_date": test.index[cu.change_index_one_based - 1].strftime("%Y-%m-%d"),
                      "n_fit": int(fitb.size), "n_test": int(test.size), "best": vars(fit.best_parameters), "nonpos": int((fc.variance <= 0).sum())}
print(json.dumps(out)); (ROOT / "results/sensitivity_sp500_training.json").write_text(json.dumps(out, indent=2))
