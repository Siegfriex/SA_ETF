"""Controlled one-factor-at-a-time ablation on *stored* DGP paths.

Every (case, n, replication, dgp_variant) series is generated once, saved as
``results/ablation/dgp/*.npz`` with its SHA-256, and re-read by every estimator
variant, so estimator variants are compared on bit-identical inputs (paired).
DGP variants (burn-in, change location) necessarily use different series and
are therefore *not* paired with the baseline.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cusum import (
    ArgmaxTiePolicy,
    DegenerateTauPolicy,
    NonPositiveVariancePolicy,
    cusum_squares_test,
    standardized_residuals,
)
from .garch_dgp import ChangeSpec, DGPAssumptions, literal_truncate_residuals, simulate_dgp
from .runner import MASTER_SEED, VARIANCE_FLOOR, dgp_assumptions, make_spec
from .simulation import CASE_BY_ID, deterministic_seed
from .svr_garch import (
    FeatureScalingPolicy,
    SVRFitPolicy,
    SVRParameters,
    TestVarianceInputPolicy,
    VariancePredictionPolicy,
    forecast_test_variance,
    select_and_refit,
)

ROOT = Path(__file__).resolve().parents[2]
DGP_DIR = ROOT / "results" / "ablation" / "dgp"


@dataclass(frozen=True)
class DGPVariant:
    change_at: str = "test_mid"   # "test_mid" (B1 choice) | "series_mid" (literal: global n = test start)
    burn_in: int = 1000
    init_scale: float = 1.0       # multiplies the baseline unconditional initial variance


@dataclass(frozen=True)
class EstVariant:
    test_policy: str = "observed"  # "observed" (A) | "recursive" (B)
    scaling: str = "x_and_y"       # e1071 scale=TRUE default
    grid: str = "g27"              # "g27" | "g125"
    kernel: str = "paper"          # "paper": gamma_sk=1/(2 g2) | "e1071_literal": gamma_sk=g2
    scale_factor: float = 1.0      # multiply y by this before estimation (x100 returns analogue)
    nonpos: str = "clip"           # "clip" | "raise"


def _grid(kind: str, kernel: str) -> tuple[SVRParameters, ...]:
    if kind == "g27":
        Cs, G2, Es = (1.0, 10.0, 100.0), (0.1, 0.5, 1.0), (0.1, 0.5, 1.0)
    elif kind == "g125":
        Cs = tuple(np.round(np.geomspace(1, 100, 5), 6))
        G2, Es = tuple(np.round(np.linspace(0.1, 1.0, 5), 6)), tuple(np.round(np.linspace(0.1, 1.0, 5), 6))
    else:
        raise ValueError(kind)
    out = []
    for c, g2, e in itertools.product(Cs, G2, Es):
        # SVRParameters.paper_gamma_squared feeds gamma_sk = 1/(2*value);
        # for the e1071-literal reading (gamma_sk = g2) pass value = 1/(2*g2).
        val = g2 if kernel == "paper" else 1.0 / (2.0 * g2)
        out.append(SVRParameters(C=float(c), paper_gamma_squared=float(val), epsilon=float(e)))
    return tuple(out)


def dgp_path(case_id: str, n: int, rep: int, dv: DGPVariant) -> tuple[np.ndarray, str]:
    tag = f"{case_id}_n{n}_r{rep:04d}_{dv.change_at}_b{dv.burn_in}_i{dv.init_scale:g}"
    f = DGP_DIR / f"{tag}.npz"
    if f.exists():
        z = np.load(f)
        y = z["y"]
        return y, hashlib.sha256(y.tobytes()).hexdigest()
    case = CASE_BY_ID[case_id]
    base = dgp_assumptions(case.baseline_model)
    a = DGPAssumptions(initial_sigma2=base.initial_sigma2 * dv.init_scale, burn_in=dv.burn_in)
    change = None
    if not case.is_size:
        idx = n + n // 2 if dv.change_at == "test_mid" else n
        change = ChangeSpec(idx, case.post_model, case.post_parameters)
    seed = deterministic_seed(MASTER_SEED, case_id, n, rep, "dgp")
    y = simulate_dgp(case.baseline_model, case.baseline_parameters, length=2 * n, assumptions=a,
                     seed=seed, change=change).observations
    DGP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, y=y)
    tmp.replace(f)
    return y, hashlib.sha256(y.tobytes()).hexdigest()


def _lrv_ratio(e: np.ndarray) -> float:
    """Bartlett long-run variance of e^2 (bandwidth floor(n^(1/3))) divided by tau^2."""
    z = e**2 - np.mean(e**2)
    n = z.size
    L = int(np.floor(n ** (1 / 3)))
    g0 = np.dot(z, z) / n
    lrv = g0 + 2 * sum((1 - k / (L + 1)) * np.dot(z[k:], z[:-k]) / n for k in range(1, L + 1))
    return float(lrv / g0)


def run_one(case_id: str, n: int, rep: int, dv: DGPVariant, ev: EstVariant) -> dict:
    case = CASE_BY_ID[case_id]
    y, sha = dgp_path(case_id, n, rep, dv)
    ys = y * ev.scale_factor
    tr, va, te = ys[: 7 * n // 10], ys[7 * n // 10 : n], ys[n:]
    base = make_spec("full", clip=True)
    fit_policy = SVRFitPolicy(scaling=FeatureScalingPolicy(ev.scaling))
    grid = _grid(ev.grid, ev.kernel)
    row = {"case_id": case_id, "n": n, "replication": rep, "dgp_sha256": sha, **dataclasses.asdict(dv),
           **dataclasses.asdict(ev)}
    try:
        fit = select_and_refit(tr, va, parameter_grid=grid, fit_policy=fit_policy, m=5, p=1, q=1,
                               validation_proxy_boundary=base.validation_proxy_boundary,
                               validation_lag_boundary=base.validation_lag_boundary,
                               validation_variance_policy=base.validation_variance_policy,
                               variance_floor=base.variance_floor, tie_policy=base.selection_tie_policy)
        pol = (TestVarianceInputPolicy.OBSERVED_PROXY if ev.test_policy == "observed"
               else TestVarianceInputPolicy.RECURSIVE_PREDICTION)
        fc = forecast_test_variance(fit, te, m=5, p=1, q=1, input_policy=pol,
                                    proxy_boundary=base.test_proxy_boundary, lag_boundary=base.test_lag_boundary,
                                    prediction_policy=VariancePredictionPolicy.RAW, variance_floor=None)
        nonpos = int(np.sum(fc.variance <= 0))
        row["n_nonpos"] = nonpos
        if nonpos and ev.nonpos == "raise":
            row["status"] = "failed"
            return row
        e = standardized_residuals(te, fc.variance, nonpositive_policy=NonPositiveVariancePolicy.CLIP,
                                   variance_floor=VARIANCE_FLOOR * ev.scale_factor**2)
        if case.literal_truncation:
            e = literal_truncate_residuals(e)
        cu = cusum_squares_test(e, tie_policy=ArgmaxTiePolicy.FIRST, degenerate_tau_policy=DegenerateTauPolicy.RAISE)
        # diagnostic: how close sigma_hat^2 is to the 4 most recent observed squares (A absorbs?)
        y2 = np.concatenate((ys[n - 4 : n], te)) ** 2
        realized = np.array([y2[t : t + 4].mean() for t in range(te.size)])
        r = np.corrcoef(fc.variance, realized)[0, 1]
        row.update(status="ok", statistic=cu.statistic, reject=bool(cu.reject), k_hat=cu.change_index_one_based,
                   best=str(fit.best_parameters), r2_sigma_vs_recent4=float(r * r), lrv_ratio=_lrv_ratio(e))
    except (ValueError, FloatingPointError) as exc:
        row.update(status="failed", error=str(exc)[:160])
    return row
