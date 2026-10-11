"""A 노트북 빌더 공용 도구 — 셀 원본은 각 build_*.py 가 SSOT 다."""
from pathlib import Path

import nbformat as nbf

SETUP = r'''
import sys, json, warnings
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config" / "research_a_contract.yaml").exists())
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="obj.round has no effect")   # 표시용 round 가 날짜 열을 건너뛸 때의 경고 (값 영향 없음)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.kfont import setup_korean_font
from src.r2h.research_a import contract as K, data as D, qa as Q
from src.r2h.research_a import rolling_correlation as RC, robustness as RB, visualization as V
from src.r2h.research_a import correlation_events as CE, price_events as PE, event_matching as EM

setup_korean_font()
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 80)
plt.rcParams.update({"figure.dpi": 105, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})

C = K.load_contract(ROOT)
VERSION = K.input_data_version(D.read_manifest(C))
REF = C.reference
SPEC = D.etf_manifest(C).set_index("ticker")
LAB = {t: f"{t} {SPEC.at[t, 'name']}" for t in SPEC.index}
FIG_DIR = ROOT / "reports" / "r2h" / "research_a" / "figures" / "{nb}"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def savefig(fig, name):
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", dpi=110)
    return p

assert D.verify_raw_sha(C)["match"].all(), "raw sha mismatch"
r = D.returns(C, "dq_trim_canonical")      # 실제 데이터 · 일간 로그수익률(logdiff) · 261240 DQ 2일 NaN
rr = D.returns(C, "raw")                   # DQ 처리 없음 (민감도용)
close = D.load_close_panel(C)              # KRW, 20종 공통 거래일
print("ROOT:", ROOT, "| input_version:", VERSION[:16], "| sha:", K.git_sha(ROOT)[:10], "| r:", r.shape, r.index[0].date(), "~", r.index[-1].date())
'''


class Book:
    def __init__(self, nb_id: str):
        self.cells = []
        self.nb_id = nb_id

    def md(self, s):
        self.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

    def code(self, s):
        self.cells.append(nbf.v4.new_code_cell(s.strip("\n")))

    def setup(self):
        self.code(SETUP.replace("{nb}", self.nb_id))

    def write(self, path: Path, phase: str):
        nb = nbf.v4.new_notebook(cells=self.cells)
        nb.metadata["kernelspec"] = {"name": "hongik_26_2", "display_name": "Python (hongik 26-2 .venv CUDA)", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        nb.metadata["r2h"] = {"track": "A", "phase": phase, "generator": f"notebooks/research_a/{Path(path).stem.split('_')[0]}"}
        nbf.write(nb, path)
        print(path)
