"""P0 leakage / boundary / DGP-fidelity tests on the real code paths."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper2020.garch_dgp import ChangeSpec, DGPAssumptions, DGPModel, simulate_dgp  # noqa: E402
from paper2020.runner import dgp_assumptions, make_spec  # noqa: E402
from paper2020.simulation import CASE_BY_ID, SimulationProtocol, TABLE_CASES  # noqa: E402
from paper2020.svr_garch import (  # noqa: E402
    TestVarianceInputPolicy,
    VariancePredictionPolicy,
    forecast_test_variance,
    select_and_refit,
)
from paper2020.volatility_proxy import BoundaryPolicy, build_lagged_design, moving_square_proxy  # noqa: E402

N = 200


@pytest.fixture(scope="module")
def series():
    case = CASE_BY_ID["t1_size"]
    y = simulate_dgp(case.baseline_model, case.baseline_parameters, length=2 * N,
                     assumptions=dgp_assumptions(case.baseline_model), seed=123).observations
    return y[: 7 * N // 10], y[7 * N // 10 : N], y[N:]


def _fit(tr, va):
    s = make_spec("smoke", clip=True)
    return select_and_refit(tr, va, parameter_grid=s.parameter_grid, fit_policy=s.fit_policy, m=5, p=1, q=1,
                            validation_proxy_boundary=s.validation_proxy_boundary,
                            validation_lag_boundary=s.validation_lag_boundary,
                            validation_variance_policy=s.validation_variance_policy,
                            variance_floor=s.variance_floor, tie_policy=s.selection_tie_policy)


def _forecast(fit, te, policy):
    return forecast_test_variance(fit, te, m=5, p=1, q=1, input_policy=policy, proxy_boundary=BoundaryPolicy.CARRY,
                                  lag_boundary=BoundaryPolicy.CARRY, prediction_policy=VariancePredictionPolicy.RAW,
                                  variance_floor=None).variance


@pytest.mark.parametrize("policy", list(TestVarianceInputPolicy))
@pytest.mark.parametrize("k", [0, 1, 57, N - 1])
def test_future_test_values_do_not_change_earlier_variance(series, policy, k):
    """sigma_hat^2_t uses y_{t-1} at most: perturbing y_j (j>=k) leaves sigma_hat^2_0..sigma_hat^2_k unchanged."""
    tr, va, te = series
    fit = _fit(tr, va)
    v0 = _forecast(fit, te, policy)
    te2 = te.copy()
    te2[k:] = te2[k:] * 7.0 + 3.0
    v1 = _forecast(fit, te2, policy)
    np.testing.assert_array_equal(v0[: k + 1], v1[: k + 1])
    if k + 1 < N:
        assert not np.array_equal(v0[k + 1 :], v1[k + 1 :])


def test_test_block_never_enters_fitting(series):
    """select_and_refit has no access to the test block; scalers/SVR/selection are fit on train(+val) only."""
    tr, va, _ = series
    f1, f2 = _fit(tr, va), _fit(tr.copy(), va.copy())
    assert f1.best_parameters == f2.best_parameters
    assert [s.mae for s in f1.scores] == [s.mae for s in f2.scores]
    np.testing.assert_array_equal(f1.combined_observations, np.concatenate((tr, va)))


def test_validation_proxy_carries_training_history(series):
    tr, va, _ = series
    v = moving_square_proxy(va, m=5, boundary_policy=BoundaryPolicy.CARRY, history=tr)
    np.testing.assert_allclose(v[0], np.mean(np.r_[tr[-4:], va[:1]] ** 2))
    np.testing.assert_allclose(v[4], np.mean(va[:5] ** 2))


def test_proxy_contains_current_square_but_feature_does_not():
    """sigma_tilde_t^2 includes y_t^2 (Step 2), so it is only ever a TARGET at t and a FEATURE at t+1."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    proxy = moving_square_proxy(y, m=5, boundary_policy=BoundaryPolicy.RESET)
    d = build_lagged_design(y, proxy, p=1, q=1, boundary_policy=BoundaryPolicy.RESET)
    for row, t in zip(d.X, d.target_indices):
        assert row[0] == y[t - 1] ** 2 and row[1] == proxy[t - 1]


def test_recursive_initial_variance_is_last_combined_proxy(series):
    tr, va, te = series
    fit = _fit(tr, va)
    vb = _forecast(fit, te, TestVarianceInputPolicy.RECURSIVE_PREDICTION)
    va_ = _forecast(fit, te, TestVarianceInputPolicy.OBSERVED_PROXY)
    # first test prediction uses (y_{n-1}^2, sigma_tilde_{n-1}^2) in both policies
    assert vb[0] == va_[0]
    x0 = np.array([[va[-1] ** 2, fit.combined_proxy[-1]]])
    assert vb[0] == pytest.approx(float(fit.model.predict_raw(x0)[0]))


def test_scale_invariance_x_and_y_scaling(series):
    """With x and y z-scored (e1071 scale=TRUE analogue) a x100 rescale changes nothing beyond solver tolerance."""
    tr, va, te = series
    f1, f2 = _fit(tr, va), _fit(100 * tr, 100 * va)
    e1 = te / np.sqrt(_forecast(f1, te, TestVarianceInputPolicy.OBSERVED_PROXY))
    e2 = 100 * te / np.sqrt(_forecast(f2, 100 * te, TestVarianceInputPolicy.OBSERVED_PROXY))
    np.testing.assert_allclose(e1, e2, rtol=2e-2)


@pytest.mark.parametrize("case", [c for c in TABLE_CASES if not c.is_size], ids=lambda c: c.case_id)
def test_change_point_is_test_midpoint(case):
    p = SimulationProtocol(n=500, change_index_in_test=250)
    r = simulate_dgp(case.baseline_model, case.baseline_parameters, length=1000,
                     assumptions=DGPAssumptions(initial_sigma2=1.0, burn_in=50), seed=1,
                     change=ChangeSpec(p.global_change_index, case.post_model, case.post_parameters))
    models = r.model_per_observation
    assert p.global_change_index == 750
    assert all(m is case.baseline_model for m in models[:750]) and all(m is case.post_model for m in models[750:])


def test_table6_unlisted_post_parameters_carry_header_values():
    assert dict(CASE_BY_ID["t6_tgarch_to_agarch_omega_1"].post_parameters) == {"omega": 1.0, "alpha": 0.3, "beta": 0.3, "b": 1.0}
    assert dict(CASE_BY_ID["t6_agarch_to_gjr_mixed_2"].post_parameters)["omega"] == 0.3


MOMENTS = [
    (DGPModel.GARCH, {"omega": 0.3, "alpha": 0.3, "beta": 0.3}, "Ey2", 0.3 / 0.4),
    (DGPModel.AGARCH, {"omega": 0.3, "alpha": 0.3, "beta": 0.4, "b": 1.0}, "Ey2", (0.3 + 0.3) / 0.3),
    (DGPModel.GJR_GARCH, {"omega": 0.3, "alpha_1": 0.3, "alpha_2": 0.4, "beta": 0.3}, "Ey2", 0.3 / (1 - 0.35 - 0.3)),
    (DGPModel.TGARCH, {"omega": 0.3, "alpha": 0.3, "beta": 0.3}, "Esigma", 0.3 / (1 - 0.3 * math.sqrt(2 / math.pi) - 0.3)),
    (DGPModel.LOG_LINEAR_GARCH, {"omega": 0.3, "alpha": 0.3, "beta": 0.3}, "Elogsig2",
     (0.3 + 0.3 * (-1.2703628454614782)) / 0.4),
]


@pytest.mark.parametrize("model,params,kind,expected", MOMENTS, ids=[m[0].value for m in MOMENTS])
def test_dgp_stationary_moments(model, params, kind, expected):
    r = simulate_dgp(model, params, length=60_000, assumptions=DGPAssumptions(initial_sigma2=1.0, burn_in=1000), seed=2020)
    got = {"Ey2": np.mean(r.observations**2), "Esigma": np.mean(r.sigma), "Elogsig2": np.mean(np.log(r.sigma**2))}[kind]
    tol = 0.08 * abs(expected) if kind != "Elogsig2" else 0.03
    assert got == pytest.approx(expected, abs=tol)
