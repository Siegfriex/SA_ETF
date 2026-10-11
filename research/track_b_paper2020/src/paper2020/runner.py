"""Glue: build the preregistered protocol spec (config/paper2020_runtime.yaml)
and run Monte Carlo rows / empirical analyses with it."""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import yaml

from .cusum import (
    ArgmaxTiePolicy,
    DegenerateTauPolicy,
    NonPositiveVariancePolicy,
    cusum_squares_test,
)
from .garch_dgp import DGPAssumptions, DGPModel
from .simulation import CASE_BY_ID, SimulationProtocol, run_replication
from .svr_garch import (
    FeatureScalingPolicy,
    SelectionTiePolicy,
    SVRFitPolicy,
    SVRGARCHProtocolSpec,
    SVRParameters,
    TestVarianceInputPolicy,
    VariancePredictionPolicy,
    forecast_test_variance,
    make_residual_estimator,
    select_and_refit,
)
from .cusum import standardized_residuals
from .volatility_proxy import BoundaryPolicy

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = yaml.safe_load((ROOT / "config" / "paper2020_runtime.yaml").read_text())
MASTER_SEED = int(RUNTIME["simulation"]["master_seed"])
VARIANCE_FLOOR = float(np.finfo(float).eps)


def grid(kind: str = "full") -> tuple[SVRParameters, ...]:
    key = "parameter_grid" if kind == "full" else "smoke_parameter_grid"
    g = RUNTIME["svr"][key]
    return tuple(
        SVRParameters(C=c, paper_gamma_squared=g2, epsilon=e)
        for c, g2, e in itertools.product(g["C"], g["paper_gamma_squared"], g["epsilon"])
    )


def make_spec(kind: str = "full", *, clip: bool = False) -> SVRGARCHProtocolSpec:
    """Primary lane: raise on non-positive variance. ``clip=True`` = sensitivity lane."""
    return SVRGARCHProtocolSpec(
        parameter_grid=grid(kind),
        fit_policy=SVRFitPolicy(scaling=FeatureScalingPolicy.X_AND_Y),
        validation_proxy_boundary=BoundaryPolicy.CARRY,
        validation_lag_boundary=BoundaryPolicy.CARRY,
        validation_variance_policy=VariancePredictionPolicy.RAW,
        selection_tie_policy=SelectionTiePolicy.FIRST,
        test_variance_input_policy=TestVarianceInputPolicy.OBSERVED_PROXY,
        test_proxy_boundary=BoundaryPolicy.CARRY,
        test_lag_boundary=BoundaryPolicy.CARRY,
        residual_nonpositive_policy=(
            NonPositiveVariancePolicy.CLIP if clip else NonPositiveVariancePolicy.RAISE
        ),
        variance_floor=VARIANCE_FLOOR,
    )


def dgp_assumptions(model: DGPModel) -> DGPAssumptions:
    """Initial variance: unconditional value of the paper baseline (ASSUMPTION);
    burn-in 1000 makes the start irrelevant. log-GARCH: fixed point
    omega/(1-alpha-beta) in log space => exp(0.3/0.4)."""
    init = {
        DGPModel.GARCH: 0.3 / (1 - 0.3 - 0.3),
        DGPModel.AGARCH: (0.3 + 0.3 * 1.0) / (1 - 0.3 - 0.4),
        DGPModel.GJR_GARCH: 0.3 / (1 - 0.15 - 0.2 - 0.3),
        DGPModel.TGARCH: 1.0,
        DGPModel.LOG_LINEAR_GARCH: float(np.exp(0.3 / 0.4)),
    }[DGPModel(model)]
    return DGPAssumptions(initial_sigma2=init, burn_in=int(RUNTIME["simulation"]["initialization"]["burn_in"]))


def run_row(case_id: str, n: int, reps: int, *, kind: str = "full", clip: bool = False,
            start: int = 0) -> list[dict]:
    case = CASE_BY_ID[case_id]
    protocol = SimulationProtocol(n=n, repetitions=reps, change_index_in_test=None if case.is_size else n // 2)
    estimator = make_residual_estimator(make_spec(kind, clip=clip))
    out = []
    for r in range(start, start + reps):
        t0 = time.perf_counter()
        try:
            res = run_replication(case, protocol, dgp_assumptions(case.baseline_model),
                                  master_seed=MASTER_SEED, replication=r,
                                  residual_estimator=estimator)
            row = res.to_dict() | {"status": "ok"}
        except (ValueError, FloatingPointError) as exc:
            row = {"case_id": case_id, "n": n, "replication": r, "status": "failed", "error": str(exc)[:200]}
        row["seconds"] = round(time.perf_counter() - t0, 3)
        row["grid"] = kind
        row["clip"] = clip
        out.append(row)
    return out


def empirical_analysis(levels: np.ndarray, *, kind: str = "full", train_frac_of_fit: float = 0.7,
                       clip: bool = True) -> dict:
    """§5 protocol: log-returns (x100, ASSUMPTION); first half -> SVR fit
    (split 70/30 into training/validation, ASSUMPTION mirroring §4), second half -> CUSUM test."""
    r = 100 * np.diff(np.log(levels))
    half = r.size // 2
    fit_block, test = r[:half], r[half:]
    n_tr = int(round(train_frac_of_fit * half))
    spec = make_spec(kind, clip=clip)
    fit = select_and_refit(fit_block[:n_tr], fit_block[n_tr:], parameter_grid=spec.parameter_grid,
                           fit_policy=spec.fit_policy, m=spec.m, p=spec.p, q=spec.q,
                           validation_proxy_boundary=spec.validation_proxy_boundary,
                           validation_lag_boundary=spec.validation_lag_boundary,
                           validation_variance_policy=spec.validation_variance_policy,
                           variance_floor=spec.variance_floor, tie_policy=spec.selection_tie_policy)
    fc = forecast_test_variance(fit, test, m=spec.m, p=spec.p, q=spec.q,
                                input_policy=spec.test_variance_input_policy,
                                proxy_boundary=spec.test_proxy_boundary, lag_boundary=spec.test_lag_boundary,
                                prediction_policy=VariancePredictionPolicy.RAW, variance_floor=None)
    n_nonpos = int(np.sum(fc.variance <= 0))
    resid = standardized_residuals(test, fc.variance,
                                   nonpositive_policy=NonPositiveVariancePolicy.CLIP if clip else NonPositiveVariancePolicy.RAISE,
                                   variance_floor=VARIANCE_FLOOR)
    cu = cusum_squares_test(resid, tie_policy=ArgmaxTiePolicy.FIRST, degenerate_tau_policy=DegenerateTauPolicy.RAISE)
    return {"returns": r, "half": half, "fit": fit, "variance": fc.variance, "residuals": resid,
            "cusum": cu, "n_nonpositive_variance": n_nonpos,
            "change_return_index": half + cu.change_index_one_based - 1}
