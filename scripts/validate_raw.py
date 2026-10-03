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


def spike_reversal_flags(close, k=6.0, reversal=0.7, window=60, market_k=1.0):
    """단일 ETF 고립 spike-reversal (C-0007, A-F01): raw 수정 없이 flag 만.

    조건 (모두 충족):
    - |r_t| > k · robust sd_t   (robust sd = 1.4826·MAD, r_{t-window..t-1} — shift(1), 당일 제외)
    - r_{t+1} 이 반대 부호이고 |r_{t+1}| >= reversal · |r_t|
    - 같은 날 universe 의 median |r| 이 그 날짜 기준 market_k · (universe median |r| 의 shift(1) rolling median) 이하
      → 시장 공통 충격이 아닌 고립 사건
    """
    lr = np.log(close).diff()
    med = lr.shift(1).rolling(window, min_periods=20).median()
    mad = (lr.shift(1) - med).abs().rolling(window, min_periods=20).median()
    rsd = 1.4826 * mad.replace(0, np.nan)
    uni_abs = lr.abs().median(axis=1)
    uni_base = uni_abs.shift(1).rolling(window, min_periods=20).median()
    quiet_market = uni_abs <= market_k * 3 * uni_base  # 그날 universe 전체가 평시의 3배 이내
    nxt = lr.shift(-1)
    recs = []
    for t in close.columns:
        big = (lr[t] - med[t]).abs() > k * rsd[t]
        rev = (np.sign(nxt[t]) == -np.sign(lr[t])) & (nxt[t].abs() >= reversal * lr[t].abs())
        hit = big & rev & quiet_market
        for d in lr.index[hit.fillna(False)]:
            i = lr.index.get_loc(d)
            d2 = lr.index[i + 1]
            detail = (f"r_t={lr.at[d, t]:+.4f} r_t+1={lr.at[d2, t]:+.4f} robust_sd={rsd.at[d, t]:.4f} "
                      f"universe_med_abs={uni_abs.at[d]:.4f}")
            for dd in (d, d2):
                recs.append({"ticker": t, "date": dd.date().isoformat(), "flag": "isolated_spike_reversal",
                             "detail": detail, "pair_start": d.date().isoformat(), "action": "flag_only_raw_unchanged"})
    return pd.DataFrame(recs, columns=["ticker", "date", "flag", "detail", "pair_start", "action"])


def ohlc_flags(raw_dir, tickers):
    """OHLC 논리 위반일 → dq_class=ohlc_close_gt_high (C-0014, 원인 UNRESOLVED, clip 금지)."""
    recs = []
    for t in tickers:
        d = pd.read_csv(raw_dir / f"{t}.csv", parse_dates=["Date"])
        hi = d[["Open", "Close"]].max(axis=1) - d["High"]
        lo = d["Low"] - d[["Open", "Close"]].min(axis=1)
        for i in d.index[(hi > 0) | (lo > 0) | (d["High"] < d["Low"])]:
            recs.append({"ticker": t, "date": d.at[i, "Date"].date().isoformat(), "flag": "ohlc_violation",
                         "detail": f"O={d.at[i,'Open']} H={d.at[i,'High']} L={d.at[i,'Low']} C={d.at[i,'Close']}",
                         "pair_start": "", "action": "flag_only_raw_unchanged_no_clip",
                         "dq_class": "ohlc_close_gt_high"})
    return pd.DataFrame(recs)


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
    (ROOT / "reports").mkdir(exist_ok=True)
    dq = spike_reversal_flags(close)
    # C-0012 DEC-1 → C-0014 FACT_CORRECTION: 확인된 사례는 price_anomaly_unverified, 나머지 spike-reversal 은 candidate
    confirmed = {("261240", "2019-03-14"), ("261240", "2019-03-15")}
    dq["dq_class"] = ["price_anomaly_unverified" if (t, d) in confirmed else "spike_reversal_candidate"
                      for t, d in zip(dq["ticker"], dq["date"])]
    dq = pd.concat([dq, ohlc_flags(RAW_DIR, list(close.columns))], ignore_index=True)
    dq.to_csv(ROOT / "reports" / "dq_flags.csv", index=False)
    rep["n_dq_spike_reversal"] = [int((dq["ticker"] == t).sum()) for t in rep["ticker"]]
    print(f"dq_flags (isolated spike-reversal): {len(dq)} rows → reports/dq_flags.csv")
    if len(dq):
        print(dq.to_string(index=False))
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
