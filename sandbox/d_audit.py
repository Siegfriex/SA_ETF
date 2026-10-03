"""D 독립 raw audit + threshold fragility harness (non-canonical, B 코드 import 안 함).

usage:
  python sandbox/d_audit.py audit   [RAW_DIR] [OUT_CSV]     # 20 ETF raw audit 표
  python sandbox/d_audit.py fragility [RAW_DIR] [OUT_CSV]   # 정의별 event set Jaccard
RAW_DIR 기본값: <ROOT>/data/raw (읽기 전용으로만 연다)
"""
from pathlib import Path
import hashlib
import itertools
import sys

import numpy as np
import pandas as pd

ROOT = Path("/home/sieg/projects-wsl/hongik_univ_26_2/SA/ETF_EDA_SCAFFOLD")
RAW = ROOT / "data" / "raw"
DATE_COLS = ["Date", "date", "날짜", "일자"]


def read_raw(path):
    """원본을 손대지 않고 읽는다. 정렬·중복제거 전 상태를 그대로 돌려준다."""
    df = pd.read_csv(path)
    dcol = next((c for c in DATE_COLS if c in df.columns), df.columns[0])
    df = df.rename(columns={dcol: "Date"})
    df["Date_raw"] = df["Date"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def audit_one(path, calendar=None):
    df = read_raw(path)
    n_in = len(df)
    n_nat = int(df["Date"].isna().sum())
    n_dup = int(df["Date"].dropna().duplicated().sum())
    monotonic = bool(df["Date"].dropna().is_monotonic_increasing)
    d = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    d = d.set_index("Date")
    c = d["Close"].astype(float)
    r = np.log(c).diff()
    vol = d["Volume"] if "Volume" in d else pd.Series(np.nan, index=d.index)

    # 거래일 캘린더 대비 gap: calendar = universe 합집합 날짜. 없으면 영업일 대비
    span = d.index[[0, -1]]
    ref = calendar[(calendar >= span[0]) & (calendar <= span[1])] if calendar is not None \
        else pd.bdate_range(*span)
    missing_vs_cal = int(len(ref.difference(d.index)))

    # stale: 종가·거래량이 바뀌지 않은 연속 구간
    same = c.eq(c.shift(1))
    runs = same.groupby((~same).cumsum()).sum()
    stale_max = int(runs.max()) if len(runs) else 0

    # OHLC 일관성
    ohlc = {"Open", "High", "Low", "Close"}.issubset(d.columns)
    if ohlc:
        bad = (d["High"] < d[["Open", "Close"]].max(axis=1)) | (d["Low"] > d[["Open", "Close"]].min(axis=1)) \
            | (d["High"] < d["Low"]) | (d[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        n_ohlc_bad = int(bad.sum())
        n_open0 = int((d["Open"] == 0).sum())
    else:
        n_ohlc_bad = n_open0 = -1

    # 가격 점프: shift(1) robust z (MAD) 기준 |z|>8, 그리고 분할·병합 의심(종가비 0.45~0.55 / 1.8~2.2 등 정수배)
    med = r.rolling(60, min_periods=20).median().shift(1)
    mad = (r - med).abs().rolling(60, min_periods=20).median().shift(1) * 1.4826
    rz = (r - med) / mad.replace(0, np.nan)
    ratio = c / c.shift(1)
    split_like = ratio.between(0.18, 0.55) | ratio.between(1.8, 5.5)
    jumps = rz.abs() > 8

    return {
        "ticker": Path(path).stem,
        "sha256_12": hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12],
        "rows_in": n_in, "rows_clean": len(d), "nat_rows": n_nat, "dup_date_rows": n_dup,
        "monotonic_raw": monotonic,
        "first": d.index[0].date(), "last": d.index[-1].date(),
        "missing_vs_calendar": missing_vs_cal,
        "close_na": int(c.isna().sum()),
        "vol_na": int(vol.isna().sum()), "vol_zero": int((vol == 0).sum()),
        "vol_zero_pct": round(float((vol == 0).mean()), 4),
        "stale_close_max_run": stale_max,
        "ohlc_bad": n_ohlc_bad, "open_zero": n_open0,
        "jump_rz8_n": int(jumps.sum()),
        "jump_dates": ";".join(str(x.date()) for x in rz.abs().nlargest(3).index),
        "split_like_n": int(split_like.sum()),
        "ret_sd_bp": round(float(r.std() * 1e4), 1),
        "excess_kurt": round(float(r.kurt()), 2),
        "has_adj_close": any("adj" in col.lower() for col in d.columns),
    }


def event_sets(c):
    """동일 ETF 에서 '큰 상승일' 정의 5종. 각 정의는 같은 개수(top-q) 가 아니라 원래 threshold 로 둔다."""
    r = c.pct_change()
    lr = np.log(c).diff()
    vol_incl = lr.rolling(20).std()
    vol_lag = vol_incl.shift(1)
    med = lr.rolling(60, min_periods=20).median().shift(1)
    mad = (lr - med).abs().rolling(60, min_periods=20).median().shift(1) * 1.4826
    pctl = lr.rolling(250, min_periods=120).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    return {
        "abs_3pct": r >= 0.03,
        "z_incl_2": lr / vol_incl >= 2,
        "z_lag_2": lr / vol_lag >= 2,
        "pctl250_97.5": pctl >= 0.975,
        "mad_z_3": (lr - med) / mad.replace(0, np.nan) >= 3,
    }


def fragility_one(path):
    d = read_raw(path).dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    c = d.set_index("Date")["Close"].astype(float)
    sets = {k: set(v[v.fillna(False)].index) for k, v in event_sets(c).items()}
    rows = []
    for a, b in itertools.combinations(sets, 2):
        u = sets[a] | sets[b]
        rows.append({"ticker": Path(path).stem, "def_a": a, "def_b": b, "n_a": len(sets[a]), "n_b": len(sets[b]),
                     "jaccard": round(len(sets[a] & sets[b]) / len(u), 3) if u else np.nan})
    return rows


def raw_files(raw_dir):
    return sorted(p for p in Path(raw_dir).glob("*.csv") if "Close" in pd.read_csv(p, nrows=0).columns)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "audit"
    raw_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else RAW
    files = raw_files(raw_dir)
    if not files:
        sys.exit(f"[STOP] raw csv 없음: {raw_dir}")
    if mode == "audit":
        cal = pd.DatetimeIndex(sorted(set().union(*[set(read_raw(p)["Date"].dropna()) for p in files])))
        out = pd.DataFrame([audit_one(p, cal) for p in files])
    else:
        out = pd.DataFrame([row for p in files for row in fragility_one(p)])
    dest = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).parent / f"{mode}.csv"
    out.to_csv(dest, index=False)
    print(out.to_string(index=False))
