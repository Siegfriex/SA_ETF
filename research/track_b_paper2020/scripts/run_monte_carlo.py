"""Resumable parallel Monte Carlo for Tables 1-6.

Usage: run_monte_carlo.py --n 500 --reps 1000 [--lane clip|raise] [--chunk 50] [--workers 22]
Each (case, n, chunk) is written to results/mc/<lane>/<case>_n<n>_<start>_<stop>.jsonl;
existing chunk files are skipped, so rerunning the same command resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _task(args: tuple[str, int, int, int, bool, str]) -> str:
    case_id, n, start, stop, clip, out = args
    from paper2020.runner import run_row

    rows = run_row(case_id, n, stop - start, clip=clip, start=start)
    tmp = Path(out + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--lane", choices=["clip", "raise"], default="clip")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--cases", nargs="*")
    a = ap.parse_args()
    from paper2020.simulation import TABLE_CASES

    out_dir = ROOT / "results" / "mc" / a.lane
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [c.case_id for c in TABLE_CASES if not a.cases or c.case_id in a.cases]
    tasks = []
    for n in a.n:
        for cid in cases:
            for s in range(0, a.reps, a.chunk):
                e = min(a.reps, s + a.chunk)
                out = out_dir / f"{cid}_n{n}_{s:04d}_{e:04d}.jsonl"
                if not out.exists():
                    tasks.append((cid, n, s, e, a.lane == "clip", str(out)))
    print(f"{len(tasks)} chunks to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_task, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 20 == 0 or i == len(futs):
                print(f"done {i}/{len(futs)}", flush=True)


if __name__ == "__main__":
    main()
