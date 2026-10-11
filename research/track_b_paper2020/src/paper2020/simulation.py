"""Section 4 simulation contract, runner primitives, and checkpoints.

This module intentionally knows nothing about the implementation of the
SVR--GARCH estimator.  The algorithm layer is injected through the
``ResidualEstimator`` interface, making Section 4 experiments independently
testable while preserving a clean adapter boundary for Phase B2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .garch_dgp import (
    ChangeSpec,
    DGPAssumptions,
    DGPModel,
    literal_truncate_residuals,
    simulate_dgp,
)


FloatArray = NDArray[np.float64]
TARGET_N_VALUES = (500, 1000, 2000)
PAPER_REPETITIONS = 1000
PAPER_ALPHA = 0.05
PAPER_CRITICAL_VALUE = 1.3397


class ResidualEstimator(Protocol):
    """Adapter contract shared with the Phase B2 estimator."""

    def __call__(
        self,
        train: FloatArray,
        validation: FloatArray,
        test: FloatArray,
        *,
        seed: int,
    ) -> Sequence[float] | NDArray[np.floating]: ...


class StatisticEvaluator(Protocol):
    """Injectable CUSUM evaluator contract."""

    def __call__(self, residuals: FloatArray) -> object: ...


@dataclass(frozen=True)
class TableCase:
    """One row of Tables 1--6."""

    case_id: str
    table: int
    label: str
    baseline_model: DGPModel
    baseline_parameters: Mapping[str, float]
    post_model: DGPModel | None
    post_parameters: Mapping[str, float] | None
    targets: tuple[float, float, float]
    literal_truncation: bool = False
    post_as_printed: Mapping[str, float] | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_model", DGPModel(self.baseline_model))
        object.__setattr__(
            self, "baseline_parameters", MappingProxyType(dict(self.baseline_parameters))
        )
        if self.post_model is not None:
            object.__setattr__(self, "post_model", DGPModel(self.post_model))
        if self.post_parameters is not None:
            object.__setattr__(
                self, "post_parameters", MappingProxyType(dict(self.post_parameters))
            )
        if self.post_as_printed is not None:
            object.__setattr__(
                self, "post_as_printed", MappingProxyType(dict(self.post_as_printed))
            )
        if len(self.targets) != 3:
            raise ValueError("each table row requires targets for n=500,1000,2000")
        if (self.post_model is None) != (self.post_parameters is None):
            raise ValueError("post_model and post_parameters must both be present or absent")

    @property
    def is_size(self) -> bool:
        return self.post_model is None

    def target_for_n(self, n: int) -> float:
        try:
            return self.targets[TARGET_N_VALUES.index(n)]
        except ValueError as error:
            raise KeyError(f"the paper has no target for n={n}") from error


def _merged(base: Mapping[str, float], patch: Mapping[str, float]) -> dict[str, float]:
    result = dict(base)
    result.update(patch)
    return result


GARCH_BASE = {"omega": 0.3, "alpha": 0.3, "beta": 0.3}
AGARCH_BASE = {"omega": 0.3, "alpha": 0.3, "beta": 0.4, "b": 1.0}
GJR_BASE = {"omega": 0.3, "alpha_1": 0.3, "alpha_2": 0.4, "beta": 0.3}
TGARCH_BASE = {"omega": 0.3, "alpha": 0.3, "beta": 0.3}
LOG_GARCH_BASE = {"omega": 0.3, "alpha": 0.3, "beta": 0.3}


def _row(
    case_id: str,
    table: int,
    label: str,
    model: DGPModel,
    baseline: Mapping[str, float],
    targets: tuple[float, float, float],
    *,
    patch: Mapping[str, float] | None = None,
    post_model: DGPModel | None = None,
    truncation: bool = False,
    post_as_printed: Mapping[str, float] | None = None,
    note: str | None = None,
) -> TableCase:
    if patch is None:
        destination_model = None
        destination_parameters = None
    else:
        destination_model = post_model or model
        if destination_model is model:
            start = dict(baseline)
        else:
            # Model-change rows (Table 6): unlisted post-change parameters carry the
            # printed block-header (pre-change) values where the name is shared
            # (runtime yaml: "unlisted_post_change_parameters: carry baseline values");
            # only names absent from the pre-change model fall back to that model's
            # own Section 4 baseline.  Fix 2026-10-11: previously TGARCH->AGARCH
            # omega=1 inherited AGARCH beta=0.4 instead of the printed beta=0.3.
            start = dict(_default_parameters(destination_model))
            start.update({k: v for k, v in baseline.items() if k in start})
        destination_parameters = _merged(start, patch)
    return TableCase(
        case_id=case_id,
        table=table,
        label=label,
        baseline_model=model,
        baseline_parameters=baseline,
        post_model=destination_model,
        post_parameters=destination_parameters,
        targets=targets,
        literal_truncation=truncation,
        post_as_printed=post_as_printed,
        note=note,
    )


def _default_parameters(model: DGPModel) -> Mapping[str, float]:
    return {
        DGPModel.GARCH: GARCH_BASE,
        DGPModel.AGARCH: AGARCH_BASE,
        DGPModel.GJR_GARCH: GJR_BASE,
        DGPModel.TGARCH: TGARCH_BASE,
        DGPModel.LOG_LINEAR_GARCH: LOG_GARCH_BASE,
    }[model]


# Every numerical row in printed Tables 1--6.  Table 6's printed, extraneous
# ``b=0`` is retained separately while the runnable GJR parameter mapping omits
# it because b does not occur in the displayed GJR equation.
TABLE_CASES: tuple[TableCase, ...] = (
    _row("t1_size", 1, "size", DGPModel.GARCH, GARCH_BASE, (0.023, 0.038, 0.055)),
    _row("t1_omega_1", 1, "omega=1", DGPModel.GARCH, GARCH_BASE, (0.761, 0.826, 0.956), patch={"omega": 1.0}),
    _row("t1_omega_01", 1, "omega=0.1", DGPModel.GARCH, GARCH_BASE, (0.612, 0.792, 0.970), patch={"omega": 0.1}),
    _row("t1_alpha_01_beta_02", 1, "alpha=0.1,beta=0.2", DGPModel.GARCH, GARCH_BASE, (0.355, 0.651, 0.949), patch={"alpha": 0.1, "beta": 0.2}),
    _row("t1_alpha_04_beta_05", 1, "alpha=0.4,beta=0.5", DGPModel.GARCH, GARCH_BASE, (0.649, 0.802, 0.952), patch={"alpha": 0.4, "beta": 0.5}),
    _row("t1_omega_07_alpha_01", 1, "omega=0.7,alpha=0.1", DGPModel.GARCH, GARCH_BASE, (0.871, 0.969, 0.981), patch={"omega": 0.7, "alpha": 0.1}),
    _row("t1_omega_01_beta_01", 1, "omega=0.1,beta=0.1", DGPModel.GARCH, GARCH_BASE, (0.848, 0.952, 0.964), patch={"omega": 0.1, "beta": 0.1}),
    _row("t2_size", 2, "size", DGPModel.AGARCH, AGARCH_BASE, (0.031, 0.040, 0.030)),
    _row("t2_omega_1", 2, "omega=1", DGPModel.AGARCH, AGARCH_BASE, (0.881, 0.951, 0.975), patch={"omega": 1.0}),
    _row("t2_omega_01", 2, "omega=0.1", DGPModel.AGARCH, AGARCH_BASE, (0.260, 0.613, 0.904), patch={"omega": 0.1}),
    _row("t2_alpha_01_beta_01", 2, "alpha=0.1,beta=0.1", DGPModel.AGARCH, AGARCH_BASE, (0.783, 0.939, 0.975), patch={"alpha": 0.1, "beta": 0.1}),
    _row("t2_alpha_06", 2, "alpha=0.6", DGPModel.AGARCH, AGARCH_BASE, (0.762, 0.863, 0.941), patch={"alpha": 0.6}),
    _row("t2_b_0", 2, "b=0", DGPModel.AGARCH, AGARCH_BASE, (0.591, 0.897, 0.976), patch={"b": 0.0}),
    _row("t2_b_3", 2, "b=3", DGPModel.AGARCH, AGARCH_BASE, (0.898, 0.936, 0.957), patch={"b": 3.0}),
    _row("t2_mixed_1", 2, "omega=0.1,alpha=0.5,beta=0.4,b=0", DGPModel.AGARCH, AGARCH_BASE, (0.565, 0.726, 0.846), patch={"omega": 0.1, "alpha": 0.5, "beta": 0.4, "b": 0.0}),
    _row("t2_mixed_2", 2, "omega=0.1,beta=0.6,b=2", DGPModel.AGARCH, AGARCH_BASE, (0.879, 0.926, 0.966), patch={"omega": 0.1, "beta": 0.6, "b": 2.0}),
    _row("t3_size", 3, "size", DGPModel.GJR_GARCH, GJR_BASE, (0.021, 0.029, 0.028)),
    _row("t3_omega_1", 3, "omega=1", DGPModel.GJR_GARCH, GJR_BASE, (0.851, 0.912, 0.947), patch={"omega": 1.0}),
    _row("t3_omega_01", 3, "omega=0.1", DGPModel.GJR_GARCH, GJR_BASE, (0.644, 0.787, 0.866), patch={"omega": 0.1}),
    _row("t3_alpha1_05_alpha2_08", 3, "alpha_1=0.5,alpha_2=0.8", DGPModel.GJR_GARCH, GJR_BASE, (0.418, 0.695, 0.879), patch={"alpha_1": 0.5, "alpha_2": 0.8}),
    _row("t3_alpha2_01_beta_01", 3, "alpha_2=0.1,beta=0.1", DGPModel.GJR_GARCH, GJR_BASE, (0.421, 0.751, 0.888), patch={"alpha_2": 0.1, "beta": 0.1}),
    _row("t3_omega_05_alpha2_06", 3, "omega=0.5,alpha_2=0.6", DGPModel.GJR_GARCH, GJR_BASE, (0.657, 0.863, 0.928), patch={"omega": 0.5, "alpha_2": 0.6}),
    _row("t3_omega_01_beta_07", 3, "omega=0.1,beta=0.7", DGPModel.GJR_GARCH, GJR_BASE, (0.717, 0.857, 0.925), patch={"omega": 0.1, "beta": 0.7}),
    _row("t4_size", 4, "size", DGPModel.TGARCH, TGARCH_BASE, (0.037, 0.049, 0.059)),
    _row("t4_omega_1", 4, "omega=1", DGPModel.TGARCH, TGARCH_BASE, (0.718, 0.795, 0.879), patch={"omega": 1.0}),
    _row("t4_omega_01", 4, "omega=0.1", DGPModel.TGARCH, TGARCH_BASE, (0.647, 0.840, 0.902), patch={"omega": 0.1}),
    _row("t4_alpha_01_beta_02", 4, "alpha=0.1,beta=0.2", DGPModel.TGARCH, TGARCH_BASE, (0.805, 0.886, 0.913), patch={"alpha": 0.1, "beta": 0.2}),
    _row("t4_alpha_04_beta_05", 4, "alpha=0.4,beta=0.5", DGPModel.TGARCH, TGARCH_BASE, (0.735, 0.838, 0.897), patch={"alpha": 0.4, "beta": 0.5}),
    _row("t4_omega_07_alpha_01", 4, "omega=0.7,alpha=0.1", DGPModel.TGARCH, TGARCH_BASE, (0.907, 0.970, 0.994), patch={"omega": 0.7, "alpha": 0.1}),
    _row("t4_omega_01_beta_01", 4, "omega=0.1,beta=0.1", DGPModel.TGARCH, TGARCH_BASE, (0.499, 0.674, 0.772), patch={"omega": 0.1, "beta": 0.1}),
    _row("t5_size", 5, "size", DGPModel.LOG_LINEAR_GARCH, LOG_GARCH_BASE, (0.047, 0.039, 0.037), truncation=True),
    _row("t5_omega_1", 5, "omega=1", DGPModel.LOG_LINEAR_GARCH, LOG_GARCH_BASE, (0.906, 0.984, 0.997), patch={"omega": 1.0}, truncation=True),
    _row("t5_omega_01", 5, "omega=0.1", DGPModel.LOG_LINEAR_GARCH, LOG_GARCH_BASE, (0.228, 0.382, 0.507), patch={"omega": 0.1}, truncation=True),
    _row("t5_alpha_n01_beta_n02", 5, "alpha=-0.1,beta=-0.2", DGPModel.LOG_LINEAR_GARCH, LOG_GARCH_BASE, (0.868, 0.946, 0.976), patch={"alpha": -0.1, "beta": -0.2}, truncation=True),
    _row("t5_alpha_n04_beta_05", 5, "alpha=-0.4,beta=0.5", DGPModel.LOG_LINEAR_GARCH, LOG_GARCH_BASE, (0.917, 0.985, 1.000), patch={"alpha": -0.4, "beta": 0.5}, truncation=True),
    _row("t5_omega_07_beta_05", 5, "omega=0.7,beta=0.5", DGPModel.LOG_LINEAR_GARCH, LOG_GARCH_BASE, (0.820, 0.973, 0.998), patch={"omega": 0.7, "beta": 0.5}, truncation=True),
    _row("t5_omega_01_alpha_n01", 5, "omega=0.1,alpha=-0.1", DGPModel.LOG_LINEAR_GARCH, LOG_GARCH_BASE, (0.862, 0.944, 0.971), patch={"omega": 0.1, "alpha": -0.1}, truncation=True),
    _row("t6_garch_to_log_omega_1", 6, "GARCH->log-GARCH omega=1", DGPModel.GARCH, GARCH_BASE, (0.879, 0.946, 0.956), patch={"omega": 1.0}, post_model=DGPModel.LOG_LINEAR_GARCH, truncation=True),
    _row("t6_garch_to_log_alpha_01_beta_01", 6, "GARCH->log-GARCH alpha=0.1,beta=0.1", DGPModel.GARCH, GARCH_BASE, (0.668, 0.918, 0.969), patch={"alpha": 0.1, "beta": 0.1}, post_model=DGPModel.LOG_LINEAR_GARCH, truncation=True),
    _row("t6_log_to_garch_omega_1", 6, "log-GARCH->GARCH omega=1", DGPModel.LOG_LINEAR_GARCH, LOG_GARCH_BASE, (0.907, 0.970, 0.994), patch={"omega": 1.0}, post_model=DGPModel.GARCH, truncation=True),
    _row("t6_log_to_garch_alpha_01_beta_01", 6, "log-GARCH->GARCH alpha=0.1,beta=0.1", DGPModel.LOG_LINEAR_GARCH, LOG_GARCH_BASE, (0.499, 0.674, 0.772), patch={"alpha": 0.1, "beta": 0.1}, post_model=DGPModel.GARCH, truncation=True),
    _row("t6_tgarch_to_agarch_omega_1", 6, "TGARCH->AGARCH omega=1,b=1", DGPModel.TGARCH, TGARCH_BASE, (0.728, 0.795, 0.879), patch={"omega": 1.0, "b": 1.0}, post_model=DGPModel.AGARCH),
    _row("t6_tgarch_to_agarch_alpha_01_beta_01", 6, "TGARCH->AGARCH alpha=0.1,beta=0.1,b=1", DGPModel.TGARCH, TGARCH_BASE, (0.851, 0.891, 0.919), patch={"alpha": 0.1, "beta": 0.1, "b": 1.0}, post_model=DGPModel.AGARCH),
    _row("t6_agarch_to_gjr_mixed_1", 6, "AGARCH->GJR omega=0.7,alpha_1=0.3,alpha_2=0.7", DGPModel.AGARCH, {"omega": 0.3, "alpha": 0.3, "beta": 0.3, "b": 1.0}, (0.549, 0.861, 0.958), patch={"omega": 0.7, "alpha_1": 0.3, "alpha_2": 0.7}, post_model=DGPModel.GJR_GARCH, post_as_printed={"omega": 0.7, "alpha_1": 0.3, "alpha_2": 0.7, "b": 0.0}, note="Printed b=0 is extraneous to the displayed GJR equation."),
    _row("t6_agarch_to_gjr_mixed_2", 6, "AGARCH->GJR alpha_1=0.1,alpha_2=0.3,beta=0.1", DGPModel.AGARCH, {"omega": 0.3, "alpha": 0.3, "beta": 0.3, "b": 1.0}, (0.802, 0.931, 0.976), patch={"alpha_1": 0.1, "alpha_2": 0.3, "beta": 0.1}, post_model=DGPModel.GJR_GARCH, post_as_printed={"alpha_1": 0.1, "alpha_2": 0.3, "beta": 0.1, "b": 0.0}, note="Printed b=0 is extraneous to the displayed GJR equation."),
)

CASE_BY_ID: Mapping[str, TableCase] = MappingProxyType(
    {case.case_id: case for case in TABLE_CASES}
)


@dataclass(frozen=True)
class SimulationProtocol:
    """Finite-sample protocol for a single paper n value.

    ``change_index_in_test`` must be explicitly selected for power rows because
    the paper says only "middle of the time series" and does not give an
    operational index.  It is zero-based within the test segment.
    """

    n: int
    repetitions: int = PAPER_REPETITIONS
    alpha: float = PAPER_ALPHA
    critical_value: float = PAPER_CRITICAL_VALUE
    change_index_in_test: int | None = None

    def __post_init__(self) -> None:
        if self.n <= 0 or (7 * self.n) % 10 != 0:
            raise ValueError("n must be positive and make 0.7*n an integer")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be between zero and one")
        if self.critical_value <= 0:
            raise ValueError("critical_value must be positive")
        if self.change_index_in_test is not None and not (
            0 <= self.change_index_in_test < self.n
        ):
            raise ValueError("change_index_in_test must lie inside the test segment")

    @property
    def train_count(self) -> int:
        return 7 * self.n // 10

    @property
    def validation_count(self) -> int:
        return 3 * self.n // 10

    @property
    def test_count(self) -> int:
        return self.n

    @property
    def total_count(self) -> int:
        return 2 * self.n

    @property
    def global_change_index(self) -> int | None:
        if self.change_index_in_test is None:
            return None
        return self.n + self.change_index_in_test


@dataclass(frozen=True)
class CUSUMResult:
    statistic: float
    reject: bool
    change_index_one_based: int
    tau_squared: float


@dataclass(frozen=True)
class ReplicationResult:
    case_id: str
    n: int
    replication: int
    dgp_seed: int
    estimator_seed: int
    statistic: float
    reject: bool
    change_index_one_based: int
    oracle_smoke: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_generated_series(
    observations: Sequence[float] | NDArray[np.floating], protocol: SimulationProtocol
) -> tuple[FloatArray, FloatArray, FloatArray]:
    values = np.asarray(observations, dtype=np.float64)
    if values.shape != (protocol.total_count,):
        raise ValueError(f"observations must have shape ({protocol.total_count},)")
    train_end = protocol.train_count
    validation_end = protocol.n
    return (
        values[:train_end].copy(),
        values[train_end:validation_end].copy(),
        values[validation_end:].copy(),
    )


def paper_cusum_squares(
    residuals: Sequence[float] | NDArray[np.floating],
    *,
    critical_value: float = PAPER_CRITICAL_VALUE,
) -> CUSUMResult:
    """Small isolated evaluator matching the Section 3 displayed formulas."""

    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("residuals must be a non-empty finite one-dimensional array")
    squared = values**2
    n = values.size
    path = np.cumsum(squared) - (np.arange(1, n + 1) / n) * squared.sum()
    tau_squared = float(np.mean(values**4) - np.mean(squared) ** 2)
    if not np.isfinite(tau_squared) or tau_squared <= 0:
        raise FloatingPointError("CUSUM normalization is non-positive")
    absolute_path = np.abs(path)
    location = int(np.argmax(absolute_path))
    statistic = float(absolute_path[location] / (np.sqrt(n) * np.sqrt(tau_squared)))
    return CUSUMResult(
        statistic=statistic,
        reject=statistic >= critical_value,
        change_index_one_based=location + 1,
        tau_squared=tau_squared,
    )


def deterministic_seed(
    master_seed: int, case_id: str, n: int, replication: int, stream: str
) -> int:
    """Derive an order- and chunk-independent 64-bit seed."""

    if master_seed < 0 or n <= 0 or replication < 0:
        raise ValueError("master_seed/replication must be nonnegative and n positive")
    payload = f"paper2020|{master_seed}|{case_id}|{n}|{replication}|{stream}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big", signed=False)


def _coerce_cusum_result(result: object, critical_value: float) -> CUSUMResult:
    if isinstance(result, CUSUMResult):
        return result
    statistic = float(getattr(result, "statistic"))
    reject = bool(getattr(result, "reject", statistic >= critical_value))
    location = getattr(result, "change_index_one_based", None)
    if location is None:
        location = getattr(result, "change_index", None)
    if location is None:
        raise TypeError("CUSUM evaluator result must expose a change index")
    tau_squared = float(getattr(result, "tau_squared", np.nan))
    return CUSUMResult(statistic, reject, int(location), tau_squared)


def run_replication(
    case: TableCase,
    protocol: SimulationProtocol,
    assumptions: DGPAssumptions,
    *,
    master_seed: int,
    replication: int,
    residual_estimator: ResidualEstimator | None = None,
    statistic_evaluator: StatisticEvaluator | None = None,
    oracle_smoke: bool = False,
) -> ReplicationResult:
    """Run one deterministic replication.

    Oracle residuals are allowed only behind the explicit ``oracle_smoke`` flag
    and must never be presented as paper results.
    """

    if oracle_smoke and residual_estimator is not None:
        raise ValueError("oracle_smoke and residual_estimator are mutually exclusive")
    if not oracle_smoke and residual_estimator is None:
        raise ValueError("a residual_estimator is required outside oracle smoke mode")
    if not case.is_size and protocol.change_index_in_test is None:
        raise ValueError("power rows require an explicit change_index_in_test decision")

    dgp_seed = deterministic_seed(master_seed, case.case_id, protocol.n, replication, "dgp")
    estimator_seed = deterministic_seed(
        master_seed, case.case_id, protocol.n, replication, "estimator"
    )
    change = None
    if not case.is_size:
        assert protocol.global_change_index is not None
        assert case.post_model is not None and case.post_parameters is not None
        change = ChangeSpec(
            protocol.global_change_index, case.post_model, case.post_parameters
        )
    generated = simulate_dgp(
        case.baseline_model,
        case.baseline_parameters,
        length=protocol.total_count,
        assumptions=assumptions,
        seed=dgp_seed,
        change=change,
    )
    train, validation, test = split_generated_series(generated.observations, protocol)
    if oracle_smoke:
        residuals = generated.innovations[protocol.n :].copy()
    else:
        assert residual_estimator is not None
        residuals = np.asarray(
            residual_estimator(train, validation, test, seed=estimator_seed),
            dtype=np.float64,
        )
    if residuals.shape != (protocol.test_count,):
        raise ValueError(
            f"residual estimator returned shape {residuals.shape}; "
            f"expected ({protocol.test_count},)"
        )
    if case.literal_truncation:
        residuals = literal_truncate_residuals(residuals)

    if statistic_evaluator is None:
        cusum = paper_cusum_squares(residuals, critical_value=protocol.critical_value)
    else:
        cusum = _coerce_cusum_result(
            statistic_evaluator(residuals), protocol.critical_value
        )
    return ReplicationResult(
        case_id=case.case_id,
        n=protocol.n,
        replication=replication,
        dgp_seed=dgp_seed,
        estimator_seed=estimator_seed,
        statistic=cusum.statistic,
        reject=cusum.reject,
        change_index_one_based=cusum.change_index_one_based,
        oracle_smoke=oracle_smoke,
    )


def manifest_payload(
    *,
    cases: Sequence[TableCase],
    n_values: Sequence[int],
    repetitions: int,
    master_seed: int,
    assumptions: DGPAssumptions,
    change_index_in_test: str | int | None,
    estimator_id: str,
    oracle_smoke: bool,
    chunk_size: int,
) -> dict[str, Any]:
    """Build the immutable portion of a checkpoint manifest."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "paper_protocol": {
            "n_values": list(n_values),
            "repetitions": repetitions,
            "alpha": PAPER_ALPHA,
            "critical_value": PAPER_CRITICAL_VALUE,
            "generated_length": "2*n",
            "split": ["0.7*n", "0.3*n", "n"],
        },
        "master_seed": master_seed,
        "assumptions_not_specified_by_paper": asdict(assumptions),
        "change_index_in_test": change_index_in_test,
        "estimator_id": estimator_id,
        "oracle_smoke": oracle_smoke,
        "chunk_size": chunk_size,
        "cases": [
            {
                "case_id": case.case_id,
                "table": case.table,
                "baseline_model": case.baseline_model.value,
                "baseline_parameters": dict(case.baseline_parameters),
                "post_model": None if case.post_model is None else case.post_model.value,
                "post_parameters": None
                if case.post_parameters is None
                else dict(case.post_parameters),
                "literal_truncation": case.literal_truncation,
                "targets": dict(zip(TARGET_N_VALUES, case.targets, strict=True)),
            }
            for case in cases
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = sha256(canonical.encode()).hexdigest()
    return payload


class CheckpointStore:
    """Atomic, manifest-locked JSON chunk storage."""

    def __init__(self, root: Path | str, manifest: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.manifest = dict(manifest)
        self.fingerprint = str(self.manifest["fingerprint"])

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != self.fingerprint:
                raise RuntimeError("checkpoint manifest fingerprint mismatch")
            if existing != self.manifest:
                raise RuntimeError("checkpoint manifest content mismatch")
            return
        _atomic_json_write(self.manifest_path, self.manifest)

    def chunk_path(self, case_id: str, n: int, start: int, stop: int) -> Path:
        return self.root / "chunks" / case_id / f"n{n}_r{start:06d}_{stop:06d}.json"

    def has_chunk(self, case_id: str, n: int, start: int, stop: int) -> bool:
        path = self.chunk_path(case_id, n, start, stop)
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("manifest_fingerprint") != self.fingerprint:
            raise RuntimeError(f"foreign checkpoint chunk: {path}")
        expected = list(range(start, stop))
        observed = [int(row["replication"]) for row in data.get("results", [])]
        if observed != expected:
            raise RuntimeError(f"incomplete checkpoint chunk: {path}")
        return True

    def write_chunk(
        self,
        case_id: str,
        n: int,
        start: int,
        stop: int,
        results: Sequence[ReplicationResult],
    ) -> Path:
        expected = list(range(start, stop))
        observed = [result.replication for result in results]
        if observed != expected:
            raise ValueError(f"chunk replications must equal {expected}")
        if any(result.case_id != case_id or result.n != n for result in results):
            raise ValueError("chunk result case/n mismatch")
        path = self.chunk_path(case_id, n, start, stop)
        if path.exists():
            if self.has_chunk(case_id, n, start, stop):
                return path
        payload = {
            "manifest_fingerprint": self.fingerprint,
            "case_id": case_id,
            "n": n,
            "start": start,
            "stop": stop,
            "results": [result.to_dict() for result in results],
        }
        _atomic_json_write(path, payload)
        return path

    def read_results(self, case_id: str, n: int) -> list[dict[str, Any]]:
        directory = self.root / "chunks" / case_id
        if not directory.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(directory.glob(f"n{n}_r*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("manifest_fingerprint") != self.fingerprint:
                raise RuntimeError(f"foreign checkpoint chunk: {path}")
            rows.extend(data["results"])
        rows.sort(key=lambda row: int(row["replication"]))
        indices = [int(row["replication"]) for row in rows]
        if len(indices) != len(set(indices)):
            raise RuntimeError(f"duplicate checkpoint replications for {case_id}, n={n}")
        return rows


def chunk_ranges(repetitions: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    if repetitions <= 0 or chunk_size <= 0:
        raise ValueError("repetitions and chunk_size must be positive")
    for start in range(0, repetitions, chunk_size):
        yield start, min(start + chunk_size, repetitions)


def summarize_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize an empty result set")
    rejection_count = sum(bool(row["reject"]) for row in rows)
    statistics = np.asarray([row["statistic"] for row in rows], dtype=np.float64)
    return {
        "replications": len(rows),
        "rejections": rejection_count,
        "rejection_rate": rejection_count / len(rows),
        "mean_statistic": float(statistics.mean()),
    }


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)
