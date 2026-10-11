"""OFAT ablation (n=500, representative H0/H1 rows) around BOTH test policies A and B.

Usage: run_ablation.py --reps 300 [--workers 18]
Outputs results/ablation/rows.jsonl (per replication) and reports/ABLATION_RESULTS.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

CASES = ["t1_size", "t5_size", "t1_omega_07_alpha_01", "t2_omega_1", "t4_omega_01"]


def variants():
    from paper2020.ablation import DGPVariant as D, EstVariant as E

    out = []
    for pol in ("observed", "recursive"):
        b = E(test_policy=pol)
        out += [
            ("baseline", D(), b, True),
            ("scaling=x_only", D(), E(test_policy=pol, scaling="x_only"), True),
            ("scaling=none", D(), E(test_policy=pol, scaling="none"), True),
            ("grid=g125", D(), E(test_policy=pol, grid="g125"), True),
            ("kernel=e1071_literal", D(), E(test_policy=pol, kernel="e1071_literal"), True),
            ("scale_factor=100", D(), E(test_policy=pol, scale_factor=100.0), True),
            ("nonpos=raise", D(), E(test_policy=pol, nonpos="raise"), True),
            ("change_at=series_mid", D(change_at="series_mid"), b, False),
            ("burn_in=0", D(burn_in=0), b, False),
            ("burn_in=0,init_x4", D(burn_in=0, init_scale=4.0), b, False),
        ]
    return out


def _task(args):
    case_id, rep, name, dv, ev, paired = args
    from paper2020.ablation import run_one
    from paper2020.simulation import CASE_BY_ID

    if dv.change_at != "test_mid" and CASE_BY_ID[case_id].is_size:
        return None
    t0 = time.perf_counter()
    row = run_one(case_id, 500, rep, dv, ev)
    row.update(variant=name, paired_with_baseline=paired, seconds=round(time.perf_counter() - t0, 3))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--out", default="results/ablation/rows.jsonl")
    a = ap.parse_args()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["case_id"], r["replication"], r["variant"], r["test_policy"]))
    tasks = [(c, r, name, dv, ev, p) for (name, dv, ev, p) in variants() for c in CASES for r in range(a.reps)
             if (c, r, name, ev.test_policy) not in done]
    print(len(tasks), "tasks", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, out.open("a") as fh:
        for i, row in enumerate(ex.map(_task, tasks, chunksize=8), 1):
            if row is not None:
                fh.write(json.dumps(row) + "\n")
            if i % 2000 == 0:
                fh.flush()
                print(f"{i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
