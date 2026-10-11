"""results/empirical_summary.json + sensitivity_sp500_training.json -> reports/EMPIRICAL_ORIGINAL_VS_REPRODUCED.csv"""
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
emp = json.loads((ROOT / "results/empirical_summary.json").read_text())
sens = json.loads((ROOT / "results/sensitivity_sp500_training.json").read_text())
rows = []
lanes = {"raise": ("A observed-proxy", "raise"), "clip": ("A observed-proxy", "clip"), "recursive_clip": ("B recursive", "clip")}
for name, d in emp.items():
    for lane, (pol, np_) in lanes.items():
        v = d[lane]
        rows.append({"series": name, "data_label": "SOURCE_SUBSTITUTED (Yahoo)", "training": f"first half ({v['fit_start']}..{v['fit_end']})",
                     "test_policy": pol, "nonpos_policy": np_, "paper_T": d["paper"]["stat"], "repl_T": v["statistic"],
                     "critical_5pct": 1.3397, "repl_reject": v["reject_5pct"], "paper_date": d["paper"]["date"],
                     "repl_date": v["change_date"], "date_match": v["change_date"] == d["paper"]["date"],
                     "n_returns": v["n_returns"], "n_test": v["n_test"], "nonpos_var": v["n_nonpositive_variance"]})
for k, pol in (("observed_proxy", "A observed-proxy"), ("recursive_prediction", "B recursive")):
    v = sens[k]
    rows.append({"series": "sp500", "data_label": "SOURCE_SUBSTITUTED (Yahoo)", "training": "paper example 2012-04-02..2014-04-16 (SUSPECTED_SOURCE_INCONSISTENCY with 'first half')",
                 "test_policy": pol, "nonpos_policy": "clip", "paper_T": 2.4751, "repl_T": v["statistic"], "critical_5pct": 1.3397,
                 "repl_reject": v["reject"], "paper_date": "2015-08-18", "repl_date": v["change_date"],
                 "date_match": v["change_date"] == "2015-08-18", "n_returns": None, "n_test": v["n_test"], "nonpos_var": v["nonpos"]})
pd.DataFrame(rows).to_csv(ROOT / "reports/EMPIRICAL_ORIGINAL_VS_REPRODUCED.csv", index=False)
print(pd.DataFrame(rows)[["series", "training", "test_policy", "nonpos_policy", "paper_T", "repl_T", "repl_reject", "paper_date", "repl_date"]].to_string(index=False))
