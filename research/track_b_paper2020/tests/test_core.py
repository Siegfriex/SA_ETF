"""Golden / mathematical tests for the paper2020 core (Lee, Kim & Lee 2020)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics.pairwise import rbf_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper2020.cusum import (  # noqa: E402
    ArgmaxTiePolicy,
    DegenerateTauPolicy,
    cusum_squares_test,
)
from paper2020.garch_dgp import (  # noqa: E402
    DGPAssumptions,
    DGPModel,
    conditional_sigma,
    literal_truncate_residuals,
    simulate_dgp,
)
from paper2020.simulation import TABLE_CASES, paper_cusum_squares  # noqa: E402
from paper2020.svr_garch import paper_gamma_to_sklearn  # noqa: E402
from paper2020.volatility_proxy import (  # noqa: E402
    BoundaryPolicy,
    build_lagged_design,
    moving_square_proxy,
)


def test_cusum_golden():
    res = cusum_squares_test([1.0, 1.0, 1.0, 3.0], tie_policy=ArgmaxTiePolicy.FIRST,
                             degenerate_tau_policy=DegenerateTauPolicy.RAISE)
    np.testing.assert_allclose(res.centered_path, [-2.0, -4.0, -6.0, 0.0])
    assert res.tau_squared == pytest.approx(12.0)
    assert res.statistic == pytest.approx(0.8660254037844386)
    assert res.change_index_one_based == 3
    assert res.reject is False
    # the simulation module's isolated evaluator must agree
    alt = paper_cusum_squares([1.0, 1.0, 1.0, 3.0])
    assert alt.statistic == pytest.approx(res.statistic) and alt.change_index_one_based == 3


def test_cusum_detects_variance_jump_and_null_size():
    rng = np.random.default_rng(0)
    e = np.r_[rng.standard_normal(500), 2.0 * rng.standard_normal(500)]
    res = cusum_squares_test(e, tie_policy=ArgmaxTiePolicy.FIRST, degenerate_tau_policy=DegenerateTauPolicy.RAISE)
    assert res.reject and abs(res.change_index_one_based - 500) < 40
    # iid N(0,1): empirical 5% size of sup|Brownian bridge| with c=1.3397
    rej = np.mean([paper_cusum_squares(rng.standard_normal(1000)).reject for _ in range(400)])
    assert 0.01 <= rej <= 0.10


def test_rbf_mapping():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(5, 2)), rng.normal(size=(4, 2))
    g2 = 0.5
    paper = np.exp(-((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1) / (2 * g2))
    np.testing.assert_allclose(rbf_kernel(X, Y, gamma=paper_gamma_to_sklearn(g2)), paper, rtol=1e-12)


def test_proxy_m5_expanding_then_rolling():
    y = np.arange(1.0, 8.0)
    proxy = moving_square_proxy(y, m=5, boundary_policy=BoundaryPolicy.RESET)
    sq = y**2
    np.testing.assert_allclose(proxy[:5], np.cumsum(sq[:5]) / np.arange(1, 6))
    np.testing.assert_allclose(proxy[5], sq[1:6].mean())
    np.testing.assert_allclose(proxy[6], sq[2:7].mean())


def test_design_has_no_contemporaneous_leakage():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    proxy = np.array([10.0, 20.0, 30.0, 40.0])
    d = build_lagged_design(y, proxy, p=1, q=1, boundary_policy=BoundaryPolicy.RESET)
    # row for t uses y_{t-1}^2 and proxy_{t-1}; target is proxy_t
    np.testing.assert_allclose(d.X, [[1.0, 10.0], [4.0, 20.0], [9.0, 30.0]])
    np.testing.assert_allclose(d.target, [20.0, 30.0, 40.0])


@pytest.mark.parametrize(
    "model,params,expected_sigma",
    [
        (DGPModel.GARCH, {"omega": 0.3, "alpha": 0.3, "beta": 0.3}, math.sqrt(0.3 + 0.3 * 4 + 0.3 * 1.5)),
        (DGPModel.AGARCH, {"omega": 0.3, "alpha": 0.3, "beta": 0.4, "b": 1.0}, math.sqrt(0.3 + 0.3 * 9 + 0.4 * 1.5)),
        (DGPModel.GJR_GARCH, {"omega": 0.3, "alpha_1": 0.3, "alpha_2": 0.4, "beta": 0.3}, math.sqrt(0.3 + 0.4 * 4 + 0.3 * 1.5)),
        (DGPModel.TGARCH, {"omega": 0.3, "alpha": 0.3, "beta": 0.3}, 0.3 + 0.3 * 2 + 0.3 * math.sqrt(1.5)),
        (DGPModel.LOG_LINEAR_GARCH, {"omega": 0.3, "alpha": 0.3, "beta": 0.3},
         math.sqrt(math.exp(0.3 + 0.3 * math.log(4) + 0.3 * math.log(1.5)))),
    ],
)
def test_dgp_one_step_hand_computed(model, params, expected_sigma):
    # y_{t-1} = -2, sigma_{t-1}^2 = 1.5
    got = conditional_sigma(model, params, y_previous=-2.0, sigma_previous=math.sqrt(1.5))
    assert got == pytest.approx(expected_sigma, rel=1e-12)


@pytest.mark.parametrize("model", list(DGPModel))
def test_dgp_simulate_finite_and_reproducible(model):
    case = next(c for c in TABLE_CASES if c.baseline_model is model and c.is_size)
    a = DGPAssumptions(initial_sigma2=1.0, burn_in=200)
    r1 = simulate_dgp(model, case.baseline_parameters, length=1000, assumptions=a, seed=7)
    r2 = simulate_dgp(model, case.baseline_parameters, length=1000, assumptions=a, seed=7)
    assert np.all(np.isfinite(r1.observations)) and np.all(r1.sigma > 0)
    np.testing.assert_array_equal(r1.observations, r2.observations)
    np.testing.assert_allclose(r1.observations, r1.sigma * r1.innovations)


def test_literal_truncation_not_sign_preserving():
    out = literal_truncate_residuals([-25.0, 25.0, 3.0, -19.0])
    np.testing.assert_allclose(out, [20.0, 20.0, 3.0, -19.0])


def test_table_rows_count():
    assert len(TABLE_CASES) == 45
    assert sum(c.is_size for c in TABLE_CASES) == 5
