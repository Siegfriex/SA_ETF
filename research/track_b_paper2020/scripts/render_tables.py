"""Render results/mc_summary_<lane>.csv -> reports/MC_TABLES.md (paper vs replication, per table)."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DESC = {"clip": "OBSERVED_PROXY test lags · non-positive variance clipped to machine eps (executed lane; preregistered as sensitivity)",
        "raise": "OBSERVED_PROXY · non-positive variance -> replication failed (preregistered primary)",
        "recursive": "RECURSIVE_PREDICTION test lags · clip (sensitivity lane, not preregistered as primary)"}
lines = ["# Monte Carlo tables — paper vs replication\n", "값 = 5% 수준 기각률. `ok` = 성공 반복 기준, `all` = 실패를 비기각으로 센 전체 분모 기준. status 는 실행 반복 수.\n"]
for lane, d in DESC.items():
    p = ROOT / "results" / f"mc_summary_{lane}.csv"
    if not p.exists():
        continue
    s = pd.read_csv(p)
    lines.append(f"\n## lane `{lane}` — {d}\n")
    for t, g in s.groupby("table"):
        lines.append(f"\n### Table {t}\n")
        w = g.pivot_table(index="label", columns="n", values=["paper", "reject_rate_ok", "reject_rate_all", "reps_attempted", "reps_failed"], sort=False)
        rows = ["| case | n | paper | repl (ok) | repl (all) | reps | failed |", "|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            rows.append(f"| {r.label} | {r.n} | {r.paper:.3f} | {r.reject_rate_ok:.3f} | {r.reject_rate_all:.3f} | {r.reps_attempted} | {r.reps_failed} |")
        lines += rows
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "MC_TABLES.md").write_text("\n".join(lines) + "\n")
print("ok")
