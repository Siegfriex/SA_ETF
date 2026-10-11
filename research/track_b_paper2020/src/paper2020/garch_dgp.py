"""Paper-faithful data-generating equations for Lee, Kim & Lee (2020).

The paper does not specify recursion initial values, burn-in, or a log-zero
policy.  This module therefore makes those choices explicit inputs instead of
silently borrowing defaults from a GARCH package.  The five recursions below
are transcribed from printed article page 8 (physical PDF page 37).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import exp, isfinite, log, sqrt
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class DGPModel(StrEnum):
    """The five Section 4 data-generating models."""

    GARCH = "garch"
    AGARCH = "agarch"
    GJR_GARCH = "gjr_garch"
    TGARCH = "tgarch"
    LOG_LINEAR_GARCH = "log_linear_garch"


@dataclass(frozen=True)
class DGPAssumptions:
    """Numerical choices that are absent from the paper.

    ``initial_sigma2`` and ``burn_in`` must be recorded in every experiment
    manifest.  When ``initial_y`` is ``None``, one pre-history innovation is
    drawn from the same RNG and multiplied by ``sqrt(initial_sigma2)``.

    ``log_y2_floor`` is *not* part of the paper.  Leave it ``None`` for the
    literal log-linear equation (which raises on an exact zero); any positive
    floor is a separately disclosed numerical policy.
    """

    initial_sigma2: float
    initial_y: float | None = None
    burn_in: int = 0
    log_y2_floor: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.initial_sigma2) or self.initial_sigma2 <= 0:
            raise ValueError("initial_sigma2 must be finite and positive")
        if self.initial_y is not None and not isfinite(self.initial_y):
            raise ValueError("initial_y must be finite when supplied")
        if self.burn_in < 0:
            raise ValueError("burn_in must be nonnegative")
        if self.log_y2_floor is not None:
            if not isfinite(self.log_y2_floor) or self.log_y2_floor <= 0:
                raise ValueError("log_y2_floor must be finite and positive")


@dataclass(frozen=True)
class ChangeSpec:
    """A one-time DGP/parameter transition.

    ``index`` is zero-based in the returned series and denotes the first
    observation whose conditional scale is computed with the post-change
    equation and parameters.
    """

    index: int
    model: DGPModel | str
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", DGPModel(self.model))
        if self.index < 0:
            raise ValueError("change index must be nonnegative")


@dataclass(frozen=True)
class DGPResult:
    """A generated series and its latent quantities."""

    observations: FloatArray
    sigma: FloatArray
    innovations: FloatArray
    model_per_observation: tuple[DGPModel, ...]
    change_index: int | None

    @property
    def variance(self) -> FloatArray:
        return self.sigma**2


_REQUIRED_PARAMETERS: dict[DGPModel, frozenset[str]] = {
    DGPModel.GARCH: frozenset({"omega", "alpha", "beta"}),
    DGPModel.AGARCH: frozenset({"omega", "alpha", "beta", "b"}),
    DGPModel.GJR_GARCH: frozenset({"omega", "alpha_1", "alpha_2", "beta"}),
    DGPModel.TGARCH: frozenset({"omega", "alpha", "beta"}),
    DGPModel.LOG_LINEAR_GARCH: frozenset({"omega", "alpha", "beta"}),
}


def validate_parameters(model: DGPModel | str, parameters: Mapping[str, float]) -> None:
    """Validate parameter names and the paper's displayed sign constraints."""

    model = DGPModel(model)
    required = _REQUIRED_PARAMETERS[model]
    supplied = frozenset(parameters)
    if supplied != required:
        missing = sorted(required - supplied)
        extra = sorted(supplied - required)
        raise ValueError(f"{model}: parameter mismatch; missing={missing}, extra={extra}")
    if not all(isfinite(float(value)) for value in parameters.values()):
        raise ValueError(f"{model}: all parameters must be finite")

    if model is not DGPModel.LOG_LINEAR_GARCH:
        constrained = required - {"omega", "b"}
        if any(float(parameters[name]) < 0 for name in constrained):
            raise ValueError(f"{model}: displayed alpha/beta parameters must be nonnegative")


def conditional_sigma(
    model: DGPModel | str,
    parameters: Mapping[str, float],
    *,
    y_previous: float,
    sigma_previous: float,
    log_y2_floor: float | None = None,
) -> float:
    """Evaluate one displayed Section 4 conditional-scale recursion."""

    model = DGPModel(model)
    validate_parameters(model, parameters)
    if not isfinite(y_previous):
        raise ValueError("y_previous must be finite")
    if not isfinite(sigma_previous) or sigma_previous <= 0:
        raise ValueError("sigma_previous must be finite and positive")

    omega = float(parameters["omega"])
    if model is DGPModel.GARCH:
        sigma2 = (
            omega
            + float(parameters["alpha"]) * y_previous**2
            + float(parameters["beta"]) * sigma_previous**2
        )
        sigma = _positive_sqrt(sigma2, model)
    elif model is DGPModel.AGARCH:
        sigma2 = (
            omega
            + float(parameters["alpha"]) * (y_previous - float(parameters["b"])) ** 2
            + float(parameters["beta"]) * sigma_previous**2
        )
        sigma = _positive_sqrt(sigma2, model)
    elif model is DGPModel.GJR_GARCH:
        y_positive = max(y_previous, 0.0)
        y_negative = -min(y_previous, 0.0)
        sigma2 = (
            omega
            + float(parameters["alpha_1"]) * y_positive**2
            + float(parameters["alpha_2"]) * y_negative**2
            + float(parameters["beta"]) * sigma_previous**2
        )
        sigma = _positive_sqrt(sigma2, model)
    elif model is DGPModel.TGARCH:
        sigma = (
            omega
            + float(parameters["alpha"]) * abs(y_previous)
            + float(parameters["beta"]) * sigma_previous
        )
        if not isfinite(sigma) or sigma <= 0:
            raise FloatingPointError(f"{model}: recursion produced non-positive sigma={sigma}")
    else:
        y2 = y_previous**2
        if y2 == 0.0:
            if log_y2_floor is None:
                raise FloatingPointError(
                    "log-linear GARCH encountered log(0); the paper gives no floor policy"
                )
            y2 = log_y2_floor
        elif log_y2_floor is not None:
            y2 = max(y2, log_y2_floor)
        log_sigma2 = (
            omega
            + float(parameters["alpha"]) * log(y2)
            + float(parameters["beta"]) * log(sigma_previous**2)
        )
        sigma2 = exp(log_sigma2)
        sigma = _positive_sqrt(sigma2, model)

    return sigma


def _positive_sqrt(value: float, model: DGPModel) -> float:
    if not isfinite(value) or value <= 0:
        raise FloatingPointError(f"{model}: recursion produced non-positive variance={value}")
    return sqrt(value)


def simulate_dgp(
    model: DGPModel | str,
    parameters: Mapping[str, float],
    *,
    length: int,
    assumptions: DGPAssumptions,
    seed: int | None = None,
    innovations: Sequence[float] | NDArray[np.floating] | None = None,
    change: ChangeSpec | None = None,
) -> DGPResult:
    """Generate a series using exactly one of the paper's five recursions.

    Supply either ``seed`` or a full innovation vector, never both.  A supplied
    vector must include burn-in draws and requires an explicit ``initial_y`` so
    no hidden random pre-history value is consumed.
    """

    model = DGPModel(model)
    validate_parameters(model, parameters)
    if length <= 0:
        raise ValueError("length must be positive")
    if change is not None:
        if change.index >= length:
            raise ValueError("change index must be inside the returned series")
        validate_parameters(change.model, change.parameters)

    total_length = assumptions.burn_in + length
    if innovations is not None and seed is not None:
        raise ValueError("supply either seed or innovations, not both")
    if innovations is not None:
        if assumptions.initial_y is None:
            raise ValueError("explicit innovations require an explicit initial_y")
        draws = np.asarray(innovations, dtype=np.float64)
        if draws.shape != (total_length,):
            raise ValueError(f"innovations must have shape ({total_length},)")
        if not np.all(np.isfinite(draws)):
            raise ValueError("innovations must be finite")
        y_previous = float(assumptions.initial_y)
    else:
        if seed is None:
            raise ValueError("seed is required when innovations are not supplied")
        rng = np.random.default_rng(seed)
        if assumptions.initial_y is None:
            y_previous = sqrt(assumptions.initial_sigma2) * float(rng.standard_normal())
        else:
            y_previous = float(assumptions.initial_y)
        draws = np.asarray(rng.standard_normal(total_length), dtype=np.float64)

    sigma_previous = sqrt(assumptions.initial_sigma2)
    observations = np.empty(total_length, dtype=np.float64)
    sigmas = np.empty(total_length, dtype=np.float64)
    models: list[DGPModel] = []

    for raw_index in range(total_length):
        output_index = raw_index - assumptions.burn_in
        use_change = change is not None and output_index >= change.index
        active_model = change.model if use_change else model
        active_parameters = change.parameters if use_change else parameters
        sigma_t = conditional_sigma(
            active_model,
            active_parameters,
            y_previous=y_previous,
            sigma_previous=sigma_previous,
            log_y2_floor=assumptions.log_y2_floor,
        )
        y_t = sigma_t * float(draws[raw_index])
        if not isfinite(y_t):
            raise FloatingPointError(f"non-finite observation at raw index {raw_index}")
        observations[raw_index] = y_t
        sigmas[raw_index] = sigma_t
        if output_index >= 0:
            models.append(active_model)
        y_previous = y_t
        sigma_previous = sigma_t

    start = assumptions.burn_in
    return DGPResult(
        observations=observations[start:].copy(),
        sigma=sigmas[start:].copy(),
        innovations=draws[start:].copy(),
        model_per_observation=tuple(models),
        change_index=None if change is None else change.index,
    )


def literal_truncate_residuals(
    residuals: Sequence[float] | NDArray[np.floating], *, threshold: float = 20.0
) -> FloatArray:
    """Apply the paper-literal, non-sign-preserving truncated residual rule.

    Printed article page 9 defines
    ``e_tilde=e_hat I(|e_hat|<N)+N I(|e_hat|>=N)`` with ``N=20``.  Thus both
    positive and negative extreme residuals map to positive ``N``.
    """

    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("residuals must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("residuals must be finite")
    if not isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    return np.where(np.abs(values) < threshold, values, threshold).astype(np.float64)
