"""ETF 일봉 raw 수집 — fdr.DataReader(ticker, "2019-01-01") → <RAW_DIR>/{ticker}.csv (immutable).

- 이미 있는 raw 는 덮어쓰지 않는다 (--force 가 없으면 skip). 재수집이 필요하면 새 파일명으로 남기도록 C 결정.
- 한 ticker 실패가 전체를 멈추지 않는다. 실패는 MANIFEST 에 status=FAIL_DATA 로 기록.
- MANIFEST.csv 는 git 추적 대상 (raw 는 gitignored).

usage: python scripts/collect_raw.py [--force]
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os
import hashlib
import sys

import pandas as pd
import FinanceDataReader as fdr

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("ETF_RAW_DIR", os.environ.get("ETF_SHARED_DATA", "/home/sieg/projects-wsl/hongik_univ_26_2/SA/ETF_EDA_SCAFFOLD/data") + "/raw"))
START = "2019-01-01"
KST = timezone(timedelta(hours=9))
ADJUSTED_POLICY = ("FDR Close 는 분배 소급조정으로 추정(069500/KS200 누적비 1.155 ≈ TR 1.163), "
                   "O/H/L 조정 일관성 미보장, 과거 값은 수집일에 묶임 — C-0012 DEC-3")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(force=False):
    u = pd.read_csv(ROOT / "config" / "universe.csv", dtype=str, keep_default_na=False)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    recs = []
    for _, row in u.iterrows():
        t = row["ticker"]
        path = RAW_DIR / f"{t}.csv"
        rec = {"slot": row["slot"], "ticker": t, "source": f"FinanceDataReader {fdr.__version__} DataReader",
               "request_start": START, "adjusted_policy": ADJUSTED_POLICY}
        try:
            if path.exists() and not force:
                rec["fetched_at"] = datetime.fromtimestamp(path.stat().st_mtime, KST).isoformat(timespec="seconds")
                rec["action"] = "kept_existing"
            else:
                d = fdr.DataReader(t, START)
                d.index.name = "Date"
                d.to_csv(path)
                rec["fetched_at"] = datetime.now(KST).isoformat(timespec="seconds")
                rec["action"] = "fetched"
            d = pd.read_csv(path, parse_dates=["Date"])
            rec.update({
                "rows": len(d), "first": d["Date"].min().date().isoformat(),
                "last": d["Date"].max().date().isoformat(),
                "columns": "|".join(d.columns),
                "n_zero_volume": int((d["Volume"] == 0).sum()) if "Volume" in d else -1,
                "sha256": sha256(path), "status": "OK",
            })
        except Exception as e:  # 한 ETF 실패가 batch 를 멈추지 않는다
            rec.update({"status": "FAIL_DATA", "error": f"{type(e).__name__}: {e}"})
        recs.append(rec)
        print(t, rec.get("status"), rec.get("rows"), rec.get("first"), rec.get("last"))
    m = pd.DataFrame(recs)
    cols = ["slot", "ticker", "rows", "first", "last", "n_zero_volume", "sha256", "source", "fetched_at",
            "adjusted_policy", "request_start", "columns", "action", "status"] + (["error"] if "error" in m else [])
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    out = ROOT / "data" / "raw" / "MANIFEST.csv"
    m[cols].to_csv(out, index=False)
    print(f"MANIFEST → {out} · OK {int((m['status'] == 'OK').sum())}/{len(m)}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
