"""figures/** 를 훑어 reports/VISUAL_INDEX.md 생성 (C). notebook 내 참조와 claim 매핑을 같이 적는다. Deterministic."""
import json, re
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
figs = sorted((ROOT / "figures").rglob("*.png"))
nb_refs = {}
for nb in sorted((ROOT / "notebooks").rglob("*.ipynb")):
    if "00_raw_eda" in nb.parts or "_template" in nb.parts or ".ipynb_checkpoints" in nb.parts:
        continue
    txt = nb.read_text(encoding="utf-8")
    for f in figs:
        rel = f.relative_to(ROOT).as_posix()
        if rel in txt or f.name in txt:
            nb_refs.setdefault(rel, []).append(nb.relative_to(ROOT).as_posix())
cem = pd.read_csv(ROOT / "reports/CLAIM_EVIDENCE_MATRIX.csv", dtype=str, keep_default_na=False)
claim_of = {}
for _, r in cem.iterrows():
    for p in re.split(r"[;\s]+", r.get("final_figure", "") or ""):
        if p.startswith("figures/"):
            claim_of.setdefault(p, []).append(r["claim_id"])
lines = ["# VISUAL INDEX — Phase 00.5 Visualization Atlas", "",
         f"총 figure {len(figs)}장. 열: 경로 · 그룹 · notebook 참조 · 지지 claim. (자동 생성: scripts/build_visual_index.py)", ""]
groups = {}
for f in figs:
    rel = f.relative_to(ROOT).as_posix(); g = "/".join(rel.split("/")[1:-1]) or "root"
    groups.setdefault(g, []).append(rel)
for g in sorted(groups):
    lines += [f"## {g} ({len(groups[g])})", "", "| figure | notebook | claim |", "|---|---|---|"]
    for rel in groups[g]:
        lines.append(f"| [{Path(rel).name}](../{rel}) | {', '.join(Path(n).name for n in nb_refs.get(rel, [])) or '—'} | {', '.join(claim_of.get(rel, [])) or '—'} |")
    lines.append("")
(ROOT / "reports/VISUAL_INDEX.md").write_text("\n".join(lines), encoding="utf-8")
print("figures", len(figs), "groups", len(groups))
