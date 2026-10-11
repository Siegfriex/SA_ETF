"""Sensitivity lane: test-period variance-lag input = RECURSIVE_PREDICTION
(Eq. 9 lag uses sigma_hat_{t-1}^2 like a GARCH filter) instead of the primary
OBSERVED_PROXY. Same seeds, same grid; residual non-positivity -> clip.

Usage: sensitivity_test_policy.py --reps 100 --n 500 [--cases ...] [--workers 20]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CASES = ["t1_size", "t1_omega_1", "t1_omega_07_alpha_01", "t2_size", "t2_omega_1",
                 "t3_size", "t4_size", "t5_size", "t5_omega_1"]


def _one(args):
    case_id, n, start, stop = args
    import time

    from paper2020.garch_dgp import DGPModel  # noqa: F401
    from paper2020.runner import MASTER_SEED, dgp_assumptions, make_spec
    from paper2020.simulation import CASE_BY_ID, SimulationProtocol, run_replication
    from paper2020.svr_garch import TestVarianceInputPolicy, make_residual_estimator

    spec = dataclasses.replace(make_spec("full", clip=True),
                               test_variance_input_policy=TestVarianceInputPolicy.RECURSIVE_PREDICTION)
    est = make_residual_estimator(spec)
    case = CASE_BY_ID[case_id]
    proto = SimulationProtocol(n=n, repetitions=stop - start, change_index_in_test=None if case.is_size else n // 2)
    rows = []
    for r in range(start, stop):
        t0 = time.perf_counter()
        try:
            res = run_replication(case, proto, dgp_assumptions(case.baseline_model), master_seed=MASTER_SEED,
                                  replication=r, residual_estimator=est)
            row = res.to_dict() | {"status": "ok"}
        except (ValueError, FloatingPointError) as exc:
            row = {"case_id": case_id, "n": n, "replication": r, "status": "failed", "error": str(exc)[:200]}
        row["seconds"] = round(time.perf_counter() - t0, 3)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out_dir = ROOT / "results" / "mc" / "recursive"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(c, a.n, s, min(a.reps, s + 25)) for c in a.cases for s in range(0, a.reps, 25)]
    tasks = [t for t in tasks if not (out_dir / f"{t[0]}_n{t[1]}_{t[2]:04d}_{t[3]:04d}.jsonl").exists()]
    with ProcessPoolExecutor(a.workers) as ex:
        for t, rows in zip(tasks, ex.map(_one, tasks)):
            (out_dir / f"{t[0]}_n{t[1]}_{t[2]:04d}_{t[3]:04d}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows))
    print("done", len(tasks))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
