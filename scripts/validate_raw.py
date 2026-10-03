"""raw deterministic validation + 공통 날짜 정렬 패널.

검사 (ETF 별): 행수, 중복 날짜, 날짜 역순, 결측, OHLC 논리 위반, 0/결측 거래량, 비정상 가격 점프(|r|>15%),
FDR Change 와 Close pct_change 의 불일치, 5일 초과 날짜 공백, 공통 캘린더 대비 누락일.
출력:
- reports/raw_validation.csv (git 추적, 작은 표)
- <ETF_EDA_SCAFFOLD>/data/interim/panel_close.csv, panel_logret.csv (공통 날짜 wide panel, gitignored, A/D 공용 입력)

usage: python scripts/validate_raw.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHARED = Path("/home/sieg/projects-wsl/hongik_univ_26_2/SA/ETF_EDA_SCAFFOLD")
RAW_DIR = SHARED / "data" / "raw"
JUMP = 0.15


def main():
    u = pd.read_csv(ROOT / "config" / "universe.csv", dtype=str, keep_default_na=False)
    frames, recs = {}, []
    for _, row in u.iterrows():
        t = row["ticker"]
        rec = {"slot": row["slot"], "ticker": t, "name": row["name"]}
        p = RAW_DIR / f"{t}.csv"
        if not p.exists():
            rec["status"] = "FAIL_DATA"
            recs.append(rec)
            continue
        d = pd.read_csv(p)
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        rec["rows"] = len(d)
        rec["n_nat_date"] = int(d["Date"].isna().sum())
        rec["n_dup_date"] = int(d["Date"].duplicated().sum())
        rec["is_monotonic"] = bool(d["Date"].is_monotonic_increasing)
        rec["n_missing_ohlcv"] = int(d[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum())
        bad = (d["High"] < d[["Open", "Close"]].max(axis=1)) | (d["Low"] > d[["Open", "Close"]].min(axis=1)) | (d["High"] < d["Low"])
        rec["n_ohlc_violation"] = int(bad.sum())
        rec["n_zero_volume"] = int((d["Volume"] == 0).sum())
        # 거래 없는 날 FDR 은 O/H/L=0 을 줄 수 있다 → 별도 집계
        rec["n_zero_open"] = int((d["Open"] == 0).sum())
        r = d["Close"].pct_change()
        jumps = d.loc[r.abs() > JUMP, "Date"].dt.date.astype(str).tolist()
        rec["n_jump_gt15pct"] = len(jumps)
        rec["jump_dates"] = ";".join(jumps)
        if "Change" in d:
            rec["max_abs_change_mismatch"] = float((d["Change"] - r).abs().max())
        gaps = d["Date"].diff().dt.days
        rec["n_gap_gt5d"] = int((gaps > 5).sum())
        rec["first"] = d["Date"].min().date().isoformat()
        rec["last"] = d["Date"].max().date().isoformat()
        flags = []
        if rec["n_dup_date"] or rec["n_nat_date"] or not rec["is_monotonic"]:
            flags.append("date_integrity")
        if rec["n_ohlc_violation"]:
            flags.append("ohlc_violation")
        if rec["n_zero_volume"]:
            flags.append("zero_volume")
        if jumps:
            flags.append("price_jump_check_distribution_or_split")
        rec["flags"] = ";".join(flags)
        rec["status"] = "OK" if not flags else "OK_WITH_FLAGS"
        recs.append(rec)
        frames[t] = d.set_index("Date")["Close"]
    rep = pd.DataFrame(recs)
    close = pd.DataFrame(frames)
    all_dates = close.index
    rep["n_missing_vs_union_calendar"] = [int(close[t].isna().sum()) if t in close else -1 for t in rep["ticker"]]
    (ROOT / "reports").mkdir(exist_ok=True)
    rep.to_csv(ROOT / "reports" / "raw_validation.csv", index=False)
    common = close.dropna()
    out = SHARED / "data" / "interim"
    out.mkdir(parents=True, exist_ok=True)
    common.to_csv(out / "panel_close.csv")
    np.log(common).diff().dropna().to_csv(out / "panel_logret.csv")
    print(rep[["ticker", "rows", "n_dup_date", "n_ohlc_violation", "n_zero_volume", "n_zero_open",
               "n_jump_gt15pct", "n_missing_vs_union_calendar", "status"]].to_string(index=False))
    print(f"union dates {len(all_dates)} · common dates {len(common)} → {out}/panel_close.csv, panel_logret.csv")


if __name__ == "__main__":
    main()
