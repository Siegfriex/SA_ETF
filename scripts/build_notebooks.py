"""Template(SSOT) → notebooks/00_raw_eda/00_{slot:02d}_{ticker}_raw_eda.ipynb. Deterministic.

usage: python scripts/build_notebooks.py [--root ROOT]
ETF_UNIVERSE_CSV env 로 universe 경로 override 가능.
"""
import argparse, hashlib, os, sys
from pathlib import Path
import nbformat
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
args = ap.parse_args()
ROOT = Path(args.root)
uni = Path(os.environ.get("ETF_UNIVERSE_CSV", ROOT / "config" / "universe.csv"))
U = pd.read_csv(uni, dtype=str, keep_default_na=False)
tpl = ROOT / "notebooks" / "_template" / "00_etf_raw_eda.ipynb"
out_dir = ROOT / "notebooks" / "00_raw_eda"; out_dir.mkdir(parents=True, exist_ok=True)
tpl_nb = nbformat.read(tpl, as_version=4)
written = []
for _, r in U.iterrows():
    slot, tk = int(r["slot"]), r["ticker"]
    nb = nbformat.reads(nbformat.writes(tpl_nb), as_version=4)
    for i, c in enumerate(nb.cells):
        src = c.source.replace("__TICKER__", tk).replace("__SLOT__", str(slot))
        if i == 0:
            src = src.replace(f"# 00_raw_eda · {tk}", f"# 00_raw_eda · {tk} {r.get('name', '')}", 1)
        c.source = src
        c.id = hashlib.sha1(f"{tk}-{i}".encode()).hexdigest()[:12]
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None
    p = out_dir / f"00_{slot:02d}_{tk}_raw_eda.ipynb"
    nbformat.write(nb, p); written.append(p)
keep = {p.name for p in written}
for p in sorted(out_dir.glob("*.ipynb")):
    if p.name not in keep:
        p.unlink(); print("removed stale", p.name)
for p in written:
    print("wrote", p.relative_to(ROOT))
