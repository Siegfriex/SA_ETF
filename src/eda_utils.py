"""ETF raw EDA 공통 유틸 (Phase 00).

규약:
- rolling baseline 은 shift(1) — 당일 값은 자기 baseline 에 들어가지 않는다.
- 양(+)/음(-)/절대 후보 컬럼은 분리한다. 최종 label/threshold 는 정하지 않는다.
- full-sample quantile 후보(q95/q05)는 look-ahead 이므로 EDA 전용.
- price policy 는 provenance 에 문자열로 기록한다 (alias 자동선택 금지).
"""
from pathlib import Path
import json
import math
import os

import numpy as np
import pandas as pd

DEFAULT_RAW_DIR = "/home/sieg/projects-wsl/hongik_univ_26_2/SA/ETF_EDA_SCAFFOLD/data/raw"
Z_LEVELS = (2.0, 2.5, 3.0)
PRICE_POLICY_UNADJ = ("FDR Close — 조정가격 수익률(추정): FDR Close 는 분배 소급조정으로 추정(069500/KS200 누적비 1.155 ≈ TR 1.163), "
                      "O/H/L 조정 일관성 미보장 — KNOWN_LIMITATION")
LIMITATION_PRICE = "adjusted_close_estimated"
PRICE_POLICY_ADJ = "Adj Close (adjusted, explicitly requested)"

ALIASES = {
    "date": ["date", "Date", "날짜", "일자", "datetime", "Datetime", "timestamp"],
    "open": ["open", "Open", "시가"],
    "high": ["high", "High", "고가"],
    "low": ["low", "Low", "저가"],
    "close": ["close", "Close", "종가"],
    "adj_close": ["adj_close", "Adj Close", "AdjClose"],
    "volume": ["volume", "Volume", "거래량"],
    "turnover": ["turnover", "value", "Value", "거래대금", "trading_value"],
    "change": ["change", "Change"],
}


# ---------------------------------------------------------------- paths / universe
def raw_dir():
    return Path(os.environ.get("ETF_RAW_DIR", DEFAULT_RAW_DIR))


def raw_path(ticker):
    return raw_dir() / f"{ticker}.csv"


def find_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for q in [p, *p.parents]:
        if (q / "config" / "universe.csv").exists():
            return q
    raise FileNotFoundError("config/universe.csv not found walking up from " + str(p))


def universe_path(root):
    return Path(os.environ.get("ETF_UNIVERSE_CSV", Path(root) / "config" / "universe.csv"))


def read_universe(root):
    return pd.read_csv(universe_path(root), dtype=str, keep_default_na=False)


# ---------------------------------------------------------------- load
def canonicalize_columns(df):
    rename = {}
    for canon, aliases in ALIASES.items():
        for a in aliases:
            if a in df.columns and a not in rename:
                rename[a] = canon
                break
    return df.rename(columns=rename).copy()


def load_raw(path, use_adjusted=False):
    """Return (df, provenance). provenance 는 정렬/중복제거 *이전* 상태를 기록한다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    df = canonicalize_columns(df)
    prov = {"source_file": str(path), "rows_raw": int(len(df))}
    if "date" not in df.columns:
        raise ValueError("no date column")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    prov["n_unparseable_dates"] = int(df["date"].isna().sum())
    valid = df["date"].dropna()
    prov["was_sorted"] = bool(valid.is_monotonic_increasing)
    prov["n_duplicate_dates"] = int(valid.duplicated().sum())
    prov["duplicate_dates"] = [str(d.date()) for d in valid[valid.duplicated()].unique()][:20]
    df = (df.dropna(subset=["date"]).sort_values("date", kind="mergesort")
            .drop_duplicates("date", keep="last").reset_index(drop=True))
    prov["rows_after"] = int(len(df))
    for c in ("open", "high", "low", "close", "adj_close", "volume", "turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    has_adj = "adj_close" in df.columns
    if use_adjusted:
        if not has_adj:
            raise ValueError("use_adjusted=True but no adjusted close column")
        df["price"] = df["adj_close"]
        prov["price_policy"] = PRICE_POLICY_ADJ
    else:
        df["price"] = df["close"]
        prov["price_policy"] = PRICE_POLICY_UNADJ
    prov["has_adj_close"] = bool(has_adj)
    vol_ok = "volume" in df.columns and df["volume"].notna().any() and (df["volume"].fillna(0) != 0).any()
    prov["has_volume"] = bool(vol_ok)
    prov["n_zero_volume_days"] = int((df["volume"] == 0).sum()) if "volume" in df.columns else None
    return df, prov


# ---------------------------------------------------------------- features
def engineer_basic(df, price_col="price"):
    """Return (x, flags). 모든 rolling baseline 은 shift(1)."""
    x = df.copy()
    if price_col not in x.columns:
        price_col = "close"
    p = x[price_col].where(x[price_col] > 0)
    x["ret_1d"] = p.pct_change()
    x["log_ret"] = np.log(p).diff()
    x["abs_ret"] = x["log_ret"].abs()
    for n in (20, 60):
        x[f"roll_vol_{n}"] = x["log_ret"].rolling(n).std().shift(1)
    x["ret_z20"] = x["log_ret"] / x["roll_vol_20"].replace(0, np.nan)
    x["drawdown"] = p / p.cummax() - 1
    if {"high", "low"}.issubset(x.columns):
        x["range_pct"] = (x["high"] - x["low"]) / p
    for k in Z_LEVELS:
        tag = f"{k:g}".replace(".", "_")
        x[f"pos_z{tag}"] = x["ret_z20"] >= k
        x[f"neg_z{tag}"] = x["ret_z20"] <= -k
        x[f"abs_z{tag}"] = x["ret_z20"].abs() >= k
    # full-sample quantiles: LOOK-AHEAD, EDA only
    q95, q05 = x["log_ret"].quantile(0.95), x["log_ret"].quantile(0.05)
    aq95 = x["abs_ret"].quantile(0.95)
    x["pos_q95"] = x["log_ret"] >= q95
    x["neg_q05"] = x["log_ret"] <= q05
    x["abs_q95"] = x["abs_ret"] >= aq95
    flags = {"q95_lookahead": True, "q95": float(q95), "q05": float(q05), "abs_q95": float(aq95)}

    has_vol = "volume" in x.columns and x["volume"].notna().any() and (x["volume"].fillna(0) != 0).any()
    flags["has_volume"] = bool(has_vol)
    flags["n_zero_volume_days"] = int((x["volume"] == 0).sum()) if "volume" in x.columns else 0
    if has_vol:
        v = x["volume"].astype(float)
        vs = v.shift(1)
        med = vs.rolling(20, min_periods=10).median()
        mad = (vs - med).abs().rolling(20, min_periods=10).median()
        mad0 = (mad == 0)
        flags["n_mad_zero"] = int(mad0.sum())
        x["volume_ratio_20"] = v / med.replace(0, np.nan)
        x["volume_robust_z_20"] = (v - med) / (1.4826 * mad.where(~mad0))
    else:
        flags["n_mad_zero"] = 0
    return x, flags


def quality_summary(df):
    return pd.DataFrame([{
        "column": c, "dtype": str(df[c].dtype), "n": len(df),
        "missing_n": int(df[c].isna().sum()), "missing_pct": float(df[c].isna().mean()),
        "unique_n": int(df[c].nunique(dropna=True)),
    } for c in df.columns])


def ohlc_violations(df):
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        return pd.DataFrame()
    bad = ((df["high"] < df[["open", "close", "low"]].max(axis=1))
           | (df["low"] > df[["open", "close", "high"]].min(axis=1))
           | (df[["open", "high", "low", "close"]] <= 0).any(axis=1))
    return df.loc[bad, ["date", "open", "high", "low", "close"]]


def date_gaps(df, min_days=5):
    d = df["date"].diff().dt.days
    g = df.loc[d > min_days, ["date"]].copy()
    g["prev_date"] = df["date"].shift(1)[d > min_days]
    g["gap_days"] = d[d > min_days].astype(int)
    return g.reset_index(drop=True)


def acf(series, nlags=20):
    s = pd.Series(series).dropna().to_numpy(dtype=float)
    s = s - s.mean()
    den = (s * s).sum()
    return [float((s[k:] * s[:-k]).sum() / den) if den > 0 else float("nan") for k in range(1, nlags + 1)]


def drawdown_episodes(price, dates, top=5):
    price = pd.Series(np.asarray(price, float)); dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    peak = price.cummax(); dd = price / peak - 1
    eps, i, n = [], 0, len(price)
    while i < n:
        if dd.iloc[i] < 0:
            start = i - 1 if i > 0 else 0
            j = i
            while j < n and dd.iloc[j] < 0:
                j += 1
            seg = dd.iloc[i:j]
            t = int(seg.idxmin())
            eps.append({"peak_date": dates[start].date(), "trough_date": dates[t].date(),
                        "depth": float(seg.min()),
                        "recovery_date": dates[j].date() if j < n else None,
                        "recovery_days": int((dates[j] - dates[t]).days) if j < n else None})
            i = j
        else:
            i += 1
    out = pd.DataFrame(eps, columns=["peak_date", "trough_date", "depth", "recovery_date", "recovery_days"])
    return out.sort_values("depth").head(top).reset_index(drop=True)


def jaccard(a, b):
    a = pd.Series(a).fillna(False).astype(bool); b = pd.Series(b).fillna(False).astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


# ---------------------------------------------------------------- captions
def _fmt(v):
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "NaN"
        return f"{v:,.4g}"
    return str(v)


WHY = {
    "01": "행 수·중복·결측·날짜 공백은 이후 모든 rolling 창과 정렬(merge)의 기준이 된다. 감사되지 않은 행은 전처리 단계에서 조용히 왜곡을 만든다.",
    "02": "전체 이력의 수준·거래량·낙폭을 한 화면에서 보면 구조 변화(regime)나 이상 구간이 있는지, 정규화 창을 어디서 끊어야 할지 판단할 근거가 된다.",
    "03": "수익률 분포의 꼬리(첨도·왜도·분위수)는 z-score 같은 정규 가정 기반 정규화가 얼마나 맞지 않는지를 직접 보여준다.",
    "04": "변동성 군집(|r|, r^2 의 자기상관)이 있으면 고정 임계값보다 시간가변 baseline 정규화가 필요하다는 근거가 된다.",
    "05": "거래량 baseline(shift(1))과 수익률 크기의 관계는 거래량을 극단 후보의 보조 신호로 쓸 수 있는지에 대한 근거가 된다.",
    "06": "낙폭 깊이·회복 기간·극단일 이후 forward return 은 이벤트 창 길이와 사후 관측 구간 설계의 근거가 된다.",
    "07": "후보 정의마다 선택되는 날이 얼마나 겹치는지는 극단 정의가 정의 선택에 얼마나 민감한지를 보여준다 (최종 정의는 여기서 정하지 않는다).",
    "08": "벤치마크·peer 와의 동조성은 개별 ETF 신호와 시장 공통 신호를 분리(초과수익 정규화)해야 하는지에 대한 근거가 된다.",
    "09": "여러 feature 가 소수의 축으로 요약되는지는 다변량 정규화·차원 축소가 필요한지에 대한 진단이다.",
}

STATIC_UNKNOWN = {
    "01": ["원천(FDR) 이 휴장일 외의 누락을 어떻게 처리하는지는 이 데이터만으로 확인할 수 없다."],
    "02": ["가격 점프가 분배금/분할 때문인지 실제 가격 변동인지 이 데이터만으로 구분할 수 없다."],
    "03": ["분포 모양이 기간에 따라 안정적인지는 단일 전체표본 통계로는 알 수 없다."],
    "04": ["표본이 크면 Ljung-Box 는 작은 자기상관도 유의하게 만든다 — 유의성은 크기의 증거가 아니다."],
    "05": ["거래량 급증의 원인(설정/환매, LP 호가, 리밸런싱)은 이 데이터에 없다."],
    "06": ["forward return 은 겹치는 표본(overlapping)이라 독립 관측 수가 실제보다 적다. 인과가 아니다."],
    "07": ["q95/q05 후보는 전체표본 분위수(look-ahead) 이므로 실시간 탐지에는 쓸 수 없다."],
    "08": ["benchmark 는 universe 내 대리 ticker 이며 공식 추종지수 자체가 아닐 수 있다."],
    "09": ["PCA 는 선형·전체표본 진단이며, 축의 의미를 확정하지 않는다."],
}

NEXT_Q = {
    "01": "감사에서 걸린 행(중복/공백/OHLC 위반)을 전처리에서 제거·보간·flag 중 무엇으로 처리할 것인가?",
    "02": "전체 기간을 하나의 정규화 창으로 볼 수 있는가, 아니면 구간을 나눠야 하는가?",
    "03": "꼬리 두께를 감안할 때 z-score 대신 robust/분위수 기반 정규화가 필요한가?",
    "04": "baseline 창 길이(20 vs 60)를 무엇으로 고를 것인가?",
    "05": "거래량 robust z 를 극단 후보의 보조 조건으로 쓸 가치가 있는가?",
    "06": "극단일 이후 며칠까지를 이벤트 창으로 볼 것인가?",
    "07": "후보 정의 간 불일치가 큰 날들은 어떤 공통 특성을 갖는가?",
    "08": "극단 후보를 원수익률이 아니라 benchmark 초과수익으로 정의해야 하는가?",
    "09": "PC1 이 주로 변동성 축인가, 방향 축인가 — 이것이 ETF 간에 일관적인가?",
}

PLACEHOLDER = "> A/C 해석 통합 대기 (ETF_CARDS)"


def caption_see(section, stats):
    lines = [f"- `{k}` = {_fmt(v)}" for k, v in stats.items()]
    return "\n".join(lines) if lines else "- (계산된 수치 없음)"


def caption(section, stats, extra_unknown=None):
    """Four-part Markdown. WHAT WE SEE 는 계산된 수치만."""
    unk = list(STATIC_UNKNOWN.get(section, [])) + list(extra_unknown or [])
    return (f"#### {section} · WHAT WE SEE\n{caption_see(section, stats)}\n\n"
            f"#### WHY IT MATTERS\n{WHY.get(section, '')}\n\n"
            f"#### WHAT WE DO NOT KNOW\n" + "\n".join(f"- {u}" for u in unk) + "\n\n"
            f"#### NEXT QUESTION\n- {NEXT_Q.get(section, '')}\n\n{PLACEHOLDER}\n")


# ---------------------------------------------------------------- profile
def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if o is None or isinstance(o, (int, str)):
        return o
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return str(o)


def save_profile(profile, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(profile), f, ensure_ascii=False, indent=2, allow_nan=False)
    return path


# ---------------------------------------------------------------- C-0007 additions
DQ_EXCLUDE_CLASSES = ("price_anomaly_unverified",)  # C-0014: 포함/제외 병기 대상 (OHLC 위반은 수익률 제외 대상 아님)
EXCLUDE_WINDOW = ("2026-07-20", "2026-08-10")


def kurt_trim(r, k=3):
    """excess kurtosis after dropping the top-k |r| days (DEC-2)."""
    from scipy import stats as _st
    r = pd.Series(r).dropna()
    if len(r) <= k + 3:
        return np.nan
    return float(_st.kurtosis(r.drop(r.abs().nlargest(k).index)))


def core_stats(x):
    """ann_vol, ex_kurt, p01, p99, max_dd on a (possibly subset) engineered frame."""
    from scipy import stats as _st
    r = x["log_ret"].dropna()
    p = x["price"].dropna()
    return {"n": int(len(r)), "ann_vol": float(r.std() * np.sqrt(252)) if len(r) > 1 else np.nan,
            "ex_kurt": float(_st.kurtosis(r)) if len(r) > 3 else np.nan,
            "ex_kurt_trim3": kurt_trim(r, 3),
            "p01": float(r.quantile(.01)) if len(r) else np.nan, "p99": float(r.quantile(.99)) if len(r) else np.nan,
            "max_dd": float((p / p.cummax() - 1).min()) if len(p) else np.nan}


def read_event_anchors(root):
    p = Path(root) / "config" / "event_anchors.csv"
    if not p.exists():
        return pd.DataFrame(columns=["anchor_id", "start", "end", "label", "source"])
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def read_dq_flags(root, ticker):
    p = Path(root) / "reports" / "dq_flags.csv"
    if not p.exists():
        return pd.DataFrame(columns=["ticker", "date", "flag", "detail"])
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    return d[d["ticker"] == ticker]


def stats_variants(x, root, ticker):
    """Return (table, dq_note). Versions: full / by year / excl window / DQ incl vs excl."""
    rows = {"full": core_stats(x)}
    for y, g in x.groupby(x["date"].dt.year):
        rows[f"year_{y}"] = core_stats(g)
    lo, hi = pd.Timestamp(EXCLUDE_WINDOW[0]), pd.Timestamp(EXCLUDE_WINDOW[1])
    rows[f"excl_{EXCLUDE_WINDOW[0]}~{EXCLUDE_WINDOW[1]}"] = core_stats(x[(x["date"] < lo) | (x["date"] > hi)])
    dq = read_dq_flags(root, ticker)
    rows["dq_included"] = rows["full"]
    if len(dq) and "dq_class" in dq.columns:
        dq = dq[dq["dq_class"].isin(DQ_EXCLUDE_CLASSES)]
    if len(dq):
        bad = set(pd.to_datetime(dq["date"], errors="coerce").dropna())
        rows["dq_excluded"] = core_stats(x[~x["date"].isin(bad)])
        note = f"{",".join(DQ_EXCLUDE_CLASSES)} rows={len(dq)}, distinct dates={len(bad)}"
    else:
        rows["dq_excluded"] = rows["full"]
        note = "no_dq_flags"
    return pd.DataFrame(rows).T, note


def extreme_definitions(x, window=60, k=3.0):
    """Three separate candidate definitions (EDA-only). Returns dict name -> bool Series, and score Series."""
    ar = x["ret_1d"].abs()
    lr = x["log_ret"]
    base = lr.shift(1)
    med = base.rolling(window, min_periods=window // 2).median()
    mad = (base - med).abs().rolling(window, min_periods=window // 2).median()
    rz = (lr - med).abs() / (1.4826 * mad.replace(0, np.nan))
    q01, q99 = x["ret_1d"].quantile(.01), x["ret_1d"].quantile(.99)
    defs = {"abs_pct_q95": ar >= ar.quantile(.95),
            "robust_z_shift1_w60_k3": rz >= k,
            "pctile_q01_q99": (x["ret_1d"] <= q01) | (x["ret_1d"] >= q99)}
    scores = {"abs_pct_q95": ar, "robust_z_shift1_w60_k3": rz, "pctile_q01_q99": ar}
    return {n: s.fillna(False).astype(bool) for n, s in defs.items()}, scores
