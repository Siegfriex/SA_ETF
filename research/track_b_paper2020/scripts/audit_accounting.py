"""P0 statistical-accounting audit of results/mc/<lane>/*.jsonl.

Writes reports/ORIGINAL_VS_REPRODUCED.csv (135 paper cells x lanes) and
results/accounting_audit.json (numbers quoted in STATISTICAL_ACCOUNTING_AUDIT.md).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper2020.simulation import TABLE_CASES  # noqa: E402

LANES = ("clip", "raise", "recursive")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(lane: str) -> tuple[pd.DataFrame, dict]:
    rows, chunk_issues = [], []
    for f in sorted((ROOT / "results" / "mc" / lane).glob("*.jsonl")):
        m = re.search(r"_n(\d+)_(\d{4})_(\d{4})\.jsonl$", f.name)
        lines = [json.loads(x) for x in f.read_text().splitlines() if x]
        start, stop = int(m.group(2)), int(m.group(3))
        if sorted(r["replication"] for r in lines) != list(range(start, stop)):
            chunk_issues.append(f.name)
        rows += lines
    df = pd.DataFrame(rows)
    dup = int(df.duplicated(["case_id", "n", "replication"]).sum()) if len(df) else 0
    return df, {"files": len(list((ROOT / "results" / "mc" / lane).glob("*.jsonl"))), "rows": len(df),
                "duplicate_case_n_rep": dup, "incomplete_or_misnumbered_chunks": chunk_issues}


def main() -> None:
    audit, data = {"lanes": {}}, {}
    for lane in LANES:
        df, meta = load(lane)
        data[lane] = df
        audit["lanes"][lane] = meta
    cells = []
    for case in TABLE_CASES:
        for n, target in zip((500, 1000, 2000), case.targets):
            rec = {"table": case.table, "case_id": case.case_id, "label": case.label, "n": n,
                   "kind": "size" if case.is_size else "power", "paper": target}
            for lane in LANES:
                g = data[lane]
                g = g[(g.case_id == case.case_id) & (g.n == n)] if len(g) else g
                if len(g) == 0:
                    rec[f"{lane}_status"] = "NOT_RUN"
                    continue
                ok = g[g.status == "ok"]
                k, m = int(ok.reject.sum()), len(ok)
                lo, hi = wilson(k, m)
                rec.update({
                    f"{lane}_reps": len(g), f"{lane}_failed": len(g) - m,
                    f"{lane}_fail_rate": round((len(g) - m) / len(g), 4),
                    f"{lane}_rejections": k,
                    f"{lane}_rate_cond_ok": round(k / m, 4) if m else None,
                    f"{lane}_ci_lo": round(lo, 4), f"{lane}_ci_hi": round(hi, 4),
                    f"{lane}_paper_in_ci": bool(lo <= target <= hi),
                    f"{lane}_status": "COMPLETE_1000" if len(g) >= 1000 else f"PARTIAL_{len(g)}",
                })
            cells.append(rec)
    cells = pd.DataFrame(cells)
    cells.to_csv(ROOT / "reports" / "ORIGINAL_VS_REPRODUCED.csv", index=False)

    summ = {}
    for lane in LANES:
        st = cells[f"{lane}_status"]
        run = cells[st != "NOT_RUN"]
        summ[lane] = {"cells_run": int(len(run)), "cells_not_run": int((st == "NOT_RUN").sum()),
                      "cells_complete_1000": int((st == "COMPLETE_1000").sum()),
                      "paper_inside_wilson95": int(run[f"{lane}_paper_in_ci"].sum()),
                      "by_kind": {k: {"cells": int(len(g)), "paper_inside_ci": int(g[f"{lane}_paper_in_ci"].sum()),
                                      "repl_below_ci_of_paper": int((g[f"{lane}_ci_hi"] < g.paper).sum()),
                                      "repl_above": int((g[f"{lane}_ci_lo"] > g.paper).sum())}
                                  for k, g in run.groupby("kind")}}
    audit["cells"] = summ

    # raise failures vs clip outcomes on the same (case, n, rep)
    r, c = data["raise"], data["clip"]
    fails = r[r.status != "ok"][["case_id", "n", "replication"]]
    j = fails.merge(c, on=["case_id", "n", "replication"], how="left")
    audit["raise_failures"] = {"count": int(len(fails)), "of": int(len(r)),
                               "by_case": fails.case_id.value_counts().to_dict(),
                               "clip_stat_on_same_reps": {"mean": round(float(j.statistic.mean()), 4),
                                                          "max": round(float(j.statistic.max()), 4),
                                                          "clip_rejects": int(j.reject.sum())}}
    # same dgp seed across lanes?
    key = ["case_id", "n", "replication"]
    seeds = c[key + ["dgp_seed"]].merge(data["recursive"][key + ["dgp_seed"]], on=key, suffixes=("_A", "_B"))
    audit["seed_identity_clip_vs_recursive"] = {"joined": int(len(seeds)),
                                                "dgp_seed_equal": int((seeds.dgp_seed_A == seeds.dgp_seed_B).sum())}
    # paired McNemar A (clip) vs B (recursive), n=500, reps 0-99
    pa = c[(c.n == 500) & (c.status == "ok")][key + ["reject"]].merge(
        data["recursive"][(data["recursive"].n == 500) & (data["recursive"].status == "ok")][key + ["reject"]],
        on=key, suffixes=("_A", "_B"))
    mc = {}
    for kind, sub in (("size", pa[pa.case_id.str.endswith("_size")]), ("power", pa[~pa.case_id.str.endswith("_size")])):
        b = int((sub.reject_A & ~sub.reject_B).sum())
        cc = int((~sub.reject_A & sub.reject_B).sum())
        mc[kind] = {"pairs": int(len(sub)), "A_only_reject": b, "B_only_reject": cc,
                    "mcnemar_exact_p": float(binomtest(b, b + cc, 0.5).pvalue) if b + cc else None}
    audit["paired_A_vs_B_n500"] = mc
    (ROOT / "results" / "accounting_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    print(json.dumps(audit, indent=1, default=str)[:3500])


if __name__ == "__main__":
    main()
