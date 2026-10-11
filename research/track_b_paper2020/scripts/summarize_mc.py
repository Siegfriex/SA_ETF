"""Aggregate results/mc/<lane>/*.jsonl into a paper-vs-replication comparison table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper2020.simulation import TABLE_CASES  # noqa: E402


def load(lane: str) -> pd.DataFrame:
    files = sorted((ROOT / "results" / "mc" / lane).glob("*.jsonl"))
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines() if line]
    return pd.DataFrame(rows)


def summarize(lane: str) -> pd.DataFrame:
    df = load(lane)
    if df.empty:
        return df
    out = []
    for case in TABLE_CASES:
        for n, target in zip((500, 1000, 2000), case.targets):
            g = df[(df.case_id == case.case_id) & (df.n == n)]
            if g.empty:
                continue
            ok = g[g.status == "ok"]
            out.append({
                "table": case.table, "case_id": case.case_id, "label": case.label, "n": n,
                "reps_attempted": len(g), "reps_ok": len(ok), "reps_failed": int((g.status != "ok").sum()),
                "reject_rate_ok": round(ok.reject.mean(), 3) if len(ok) else None,
                # failures counted as non-rejections (conservative, full denominator)
                "reject_rate_all": round(ok.reject.sum() / len(g), 3),
                "paper": target,
                "diff_ok_minus_paper": round(ok.reject.mean() - target, 3) if len(ok) else None,
                "mean_stat": round(ok.statistic.mean(), 3) if len(ok) else None,
                "status": "COMPLETE_1000" if len(g) >= 1000 else f"PARTIAL_{len(g)}",
            })
    return pd.DataFrame(out)


def main() -> None:
    for lane in ("clip", "raise", "recursive"):
        s = summarize(lane)
        if s.empty:
            continue
        s.to_csv(ROOT / "results" / f"mc_summary_{lane}.csv", index=False)
        print(f"== lane {lane}: {len(s)} rows")
        print(s.drop(columns=["case_id"]).to_string(index=False))


if __name__ == "__main__":
    main()
