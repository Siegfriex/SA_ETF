
from pathlib import Path
import json
import numpy as np
import pandas as pd

ALIASES = {
    "date": ["date","Date","날짜","일자","datetime","Datetime","timestamp"],
    "open": ["open","Open","시가"],
    "high": ["high","High","고가"],
    "low": ["low","Low","저가"],
    "close": ["close","Close","종가","adj_close","Adj Close"],
    "volume": ["volume","Volume","거래량"],
    "turnover": ["turnover","value","Value","거래대금","trading_value"],
}

def read_universe(root):
    return pd.read_csv(Path(root)/"config"/"etf_universe.csv")

def canonicalize_columns(df):
    rename = {}
    for canon, aliases in ALIASES.items():
        for a in aliases:
            if a in df.columns:
                rename[a] = canon
                break
    return df.rename(columns=rename).copy()

def load_raw(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".parquet",".pq"]:
        df = pd.read_parquet(path)
    elif ext in [".xlsx",".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    df = canonicalize_columns(df)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df

def engineer_basic(df):
    x = df.copy()
    if "close" in x:
        x["ret_1d"] = x["close"].pct_change()
        x["log_ret_1d"] = np.log(x["close"]).diff()
        x["abs_ret_1d"] = x["ret_1d"].abs()
        x["roll_vol_20"] = x["log_ret_1d"].rolling(20).std()
        x["roll_vol_60"] = x["log_ret_1d"].rolling(60).std()
        peak = x["close"].cummax()
        x["drawdown"] = x["close"]/peak - 1
    if set(["high","low","close"]).issubset(x.columns):
        x["range_pct"] = (x["high"]-x["low"]) / x["close"].replace(0, np.nan)
    if "volume" in x:
        med = x["volume"].rolling(20, min_periods=10).median()
        mad = (x["volume"]-med).abs().rolling(20, min_periods=10).median()
        x["volume_ratio_20"] = x["volume"] / med.replace(0, np.nan)
        x["volume_robust_z_20"] = (x["volume"]-med) / (1.4826*mad.replace(0,np.nan))
    if "turnover" in x:
        med = x["turnover"].rolling(20, min_periods=10).median()
        x["turnover_ratio_20"] = x["turnover"] / med.replace(0, np.nan)
    return x

def quality_summary(df):
    return pd.DataFrame([{
        "column": c,
        "dtype": str(df[c].dtype),
        "n": len(df),
        "missing_n": int(df[c].isna().sum()),
        "missing_pct": float(df[c].isna().mean()),
        "unique_n": int(df[c].nunique(dropna=True)),
    } for c in df.columns])

def save_profile(profile, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2, default=str)
