"""results/ablation/rows.jsonl -> reports/ABLATION_RESULTS.csv (one row per policy x variant x case)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from audit_accounting import wilson  # noqa: E402
from paper2020.simulation import CASE_BY_ID  # noqa: E402


def main() -> None:
    d = pd.read_json(ROOT / "results" / "ablation" / "rows.jsonl", lines=True)
    key = ["case_id", "replication"]
    out = []
    for (pol, var, case), g in d.groupby(["test_policy", "variant", "case_id"], sort=False):
        ok = g[g.status == "ok"].astype({"reject": bool})
        k, m = int(ok.reject.sum()), len(ok)
        lo, hi = wilson(k, m)
        base = d[(d.test_policy == pol) & (d.variant == "baseline") & (d.case_id == case) & (d.status == "ok")].astype({"reject": bool})
        j = ok[key + ["reject", "statistic", "dgp_sha256"]].merge(
            base[key + ["reject", "statistic", "dgp_sha256"]], on=key, suffixes=("", "_base"))
        same_input = bool(len(j) and (j.dgp_sha256 == j.dgp_sha256_base).all())
        b = int((j.reject & ~j.reject_base).sum())
        c = int((~j.reject & j.reject_base).sum())
        tgt = CASE_BY_ID[case].targets[0]
        out.append({
            "test_policy": pol, "variant": var, "case_id": case,
            "kind": "size" if CASE_BY_ID[case].is_size else "power", "paper_n500": tgt,
            "reps": len(g), "failed": len(g) - m, "rejections": k,
            "reject_rate": round(k / m, 4) if m else None, "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "paper_in_ci": bool(lo <= tgt <= hi) if m else None,
            "mean_stat": round(ok.statistic.mean(), 4) if m else None,
            "paired_same_dgp_input": same_input,
            "mean_dstat_vs_baseline": round(float((j.statistic - j.statistic_base).mean()), 4) if same_input else None,
            "max_abs_dstat_vs_baseline": round(float((j.statistic - j.statistic_base).abs().max()), 6) if same_input else None,
            "only_variant_rejects": b if same_input else None, "only_baseline_rejects": c if same_input else None,
            "mcnemar_p": (float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0) if same_input else None,
            "mean_r2_sigma_vs_recent4": round(ok.r2_sigma_vs_recent4.mean(), 4) if m else None,
            "mean_lrv_ratio": round(ok.lrv_ratio.mean(), 4) if m else None,
            "mean_nonpos": round(g.n_nonpos.fillna(0).mean(), 3) if "n_nonpos" in g else None,
            "sec_per_rep": round(g.seconds.mean(), 3),
        })
    res = pd.DataFrame(out)
    res.to_csv(ROOT / "reports" / "ABLATION_RESULTS.csv", index=False)
    piv = res.pivot_table(index=["test_policy", "variant"], columns="case_id", values="reject_rate", sort=False)
    print(piv.round(3).to_string())
    print(res.groupby(["test_policy", "variant"], sort=False)[["mean_lrv_ratio", "mean_r2_sigma_vs_recent4", "sec_per_rep"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
