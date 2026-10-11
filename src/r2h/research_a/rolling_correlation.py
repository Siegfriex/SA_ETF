"""Phase A2 — KOSPI200 기준 pair rolling 상관과 그 변화 (결정적, 난수 없음).

시점 규약
- rho_t 는 [t-w+1, t] 창의 수익률로 계산 → t 종가 이후 관측 가능 (available_at = t).
- 지속성(persistence)으로 확정한 사건은 candidate_at(첫 신호일)과 confirmed_at(확정일)을 분리한다.
  confirmed_at > candidate_at 이면 그 사이의 정보는 '미래'이므로 candidate_at 시점에는 알 수 없었다.

NaN 규약
- 창 안에서 두 시계열이 모두 관측된 날만 쓴다(pairwise complete). n_eff < min_periods 이면 rho = NaN.
- 창 안 한쪽 분산이 0 이면(상수 수익률) rho 는 정의되지 않으므로 NaN 으로 두고 n_const 로 센다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata

from .data import rank_safe

PERIODS = {"dev_2019_2024": ("2019-01-01", "2024-12-31"), "val_2025": ("2025-01-01", "2025-12-31"),
           "holdout_2026": ("2026-01-01", "2026-12-31")}


def period_of(idx: pd.DatetimeIndex) -> pd.Series:
    out = pd.Series("", index=idx, dtype=object)
    for k, (a, b) in PERIODS.items():
        out[(idx >= a) & (idx <= b)] = k
    return out


# ---------------------------------------------------------------- rolling correlation
def _win(a: np.ndarray, w: int) -> np.ndarray:
    """행 t = t 에서 끝나는 길이 w 창. 앞쪽 w-1 행은 NaN 으로 채워 index 를 원 시계열과 맞춘다."""
    return sliding_window_view(np.concatenate([np.full(w - 1, np.nan), a]), w)


def rolling_corr(x: pd.Series, y: pd.Series, window: int, method: str = "pearson",
                 min_periods: int | None = None, rank_round: bool = True) -> pd.DataFrame:
    """반환 DataFrame(index = x.index): rho, n_eff, const (분산 0 창 여부).

    spearman: 각 창 *안에서* 순위를 매긴다(전체표본 순위가 아님 — 그러면 미래 정보가 창 안 순위에 섞인다).
    rank_round=True 면 A1 계약대로 rank_safe(1e-12 반올림) 후 순위 → logdiff/logratio 산식 차이에 둔감.
    """
    if not x.index.equals(y.index):
        raise ValueError("x, y index must be identical (align before calling)")
    mp = window if min_periods is None else min_periods
    xa, ya = x.to_numpy(float), y.to_numpy(float)
    if method == "spearman" and rank_round:
        xa, ya = rank_safe(pd.Series(xa)).to_numpy(), rank_safe(pd.Series(ya)).to_numpy()
    X, Y = _win(xa, window).copy(), _win(ya, window).copy()
    valid = ~np.isnan(X) & ~np.isnan(Y)
    X[~valid] = np.nan; Y[~valid] = np.nan
    n = valid.sum(axis=1)
    if method == "spearman":
        X = rankdata(X, axis=1, nan_policy="omit"); Y = rankdata(Y, axis=1, nan_policy="omit")
        X[~valid] = np.nan; Y[~valid] = np.nan
    elif method != "pearson":
        raise ValueError(method)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = X - np.nanmean(X, axis=1, keepdims=True)
        ym = Y - np.nanmean(Y, axis=1, keepdims=True)
        sxx, syy = np.nansum(xm * xm, axis=1), np.nansum(ym * ym, axis=1)
        rho = np.nansum(xm * ym, axis=1) / np.sqrt(sxx * syy)
    const = (sxx == 0) | (syy == 0)
    rho[(n < mp) | const] = np.nan
    return pd.DataFrame({"rho": np.clip(rho, -1, 1), "n_eff": n, "const": const & (n >= mp)}, index=x.index)


def rolling_panel(r: pd.DataFrame, ref: str, tickers: list[str], windows=(30, 60, 120),
                  methods=("pearson", "spearman"), min_periods_frac: float = 1.0) -> pd.DataFrame:
    """long 형식: Date · ticker · window · method · rho · n_eff · const."""
    out = []
    for t in tickers:
        for w in windows:
            for m in methods:
                rc = rolling_corr(r[t], r[ref], w, m, min_periods=int(np.ceil(w * min_periods_frac)))
                rc = rc.assign(ticker=t, window=w, method=m).reset_index()
                out.append(rc)
    return pd.concat(out, ignore_index=True)


def null_band(window: int, z: float = 1.96) -> float:
    """독립·정규 가정에서 rho 의 근사 표본 오차대 ±z/sqrt(w). 겹치는 창·두꺼운 꼬리에서는 과소평가될 수 있다(근사)."""
    return z / np.sqrt(window)


# ---------------------------------------------------------------- 0선 교차와 지속성
def zero_crossings(rho: pd.Series) -> pd.DataFrame:
    """부호가 바뀐 날(직전 유효값 대비). rho==0 정확히는 부호 없음으로 보고 건너뛴다."""
    s = np.sign(rho.dropna())
    s = s[s != 0]
    ch = s.ne(s.shift(1)) & s.shift(1).notna()
    idx = s.index[ch]
    return pd.DataFrame({"candidate_at": idx, "new_sign": s[ch].astype(int).to_numpy(),
                         "rho_at": rho.reindex(idx).to_numpy()})


@dataclass(frozen=True)
class RegimeRule:
    band: float          # |rho| 가 이 값을 넘어야 부호 상태로 인정 (0 이면 순수 부호)
    persist: int         # band 밖 같은 부호가 연속 이 일수 이상이면 확정

    @property
    def id(self) -> str:
        return f"b{self.band:.3f}_k{self.persist}"


def sign_regimes(rho: pd.Series, rule: RegimeRule) -> pd.DataFrame:
    """히스테리시스 상태기계. 상태 s∈{+1,-1}: rho 가 ±band 밖에서 persist 일 연속 같은 부호면 그 부호로 확정.

    반환: 확정된 *부호 전환* 사건 (첫 확정은 초기 상태이므로 사건이 아님 → is_initial=True 로 구분)
      candidate_at  새 부호 run 의 첫날 (band 밖으로 나간 첫날)
      zero_cross_at 그 run 직전의 0선 교차일 (가장 가까운 과거; 없으면 NaT)
      confirmed_at  persist 일째 — 이 날 종가에 비로소 '지속'이 확인됨 (candidate_at 시점엔 미래 정보)
      prev_peak_abs 직전 확정 상태 동안 |rho| 최대 (이전 관계의 강도)
    """
    v = rho.to_numpy(float)
    idx = rho.index
    state, run_sign, run_len, run_start = 0, 0, 0, None
    last_cross, prev_val = None, np.nan
    state_since, peak = None, 0.0
    ev = []
    for i, x in enumerate(v):
        if np.isnan(x):
            run_sign, run_len = 0, 0
            continue
        if not np.isnan(prev_val) and np.sign(x) != 0 and np.sign(prev_val) != 0 and np.sign(x) != np.sign(prev_val):
            last_cross = idx[i]
        prev_val = x
        if state != 0 and np.sign(x) == state:
            peak = max(peak, abs(x))
        sg = int(np.sign(x)) if abs(x) > rule.band else 0
        if sg != 0 and sg == run_sign:
            run_len += 1
        elif sg != 0:
            run_sign, run_len, run_start = sg, 1, idx[i]
        else:
            run_sign, run_len = 0, 0
        if run_len == rule.persist and run_sign != state:
            ev.append({"candidate_at": run_start, "zero_cross_at": last_cross if state != 0 else pd.NaT,
                       "confirmed_at": idx[i], "from_sign": state, "to_sign": run_sign,
                       "prev_peak_abs": peak if state != 0 else np.nan, "state_since": state_since,
                       "is_initial": state == 0})
            state, state_since, peak = run_sign, idx[i], abs(x)
    out = pd.DataFrame(ev, columns=["candidate_at", "zero_cross_at", "confirmed_at", "from_sign", "to_sign",
                                    "prev_peak_abs", "state_since", "is_initial"])
    if len(out):
        pos = {d: k for k, d in enumerate(idx)}
        out["confirm_lag_days"] = [pos[c] - pos[a] for a, c in zip(out["confirmed_at"], out["candidate_at"])]
        out["confirm_lag_days"] = -out["confirm_lag_days"]
    else:
        out["confirm_lag_days"] = pd.Series(dtype=int)
    return out


def crossing_summary(rho: pd.Series, rules: list[RegimeRule]) -> pd.DataFrame:
    """raw 0선 교차 수 vs 규칙별 확정 부호 전환 수. 둘의 비율이 '잡음 교차'의 비중을 보여준다."""
    zc = zero_crossings(rho)
    nyears = rho.dropna().shape[0] / 252
    rows = [{"rule": "raw_zero_cross", "n_events": len(zc), "per_year": len(zc) / nyears if nyears else np.nan}]
    for ru in rules:
        sr = sign_regimes(rho, ru)
        n = int((~sr["is_initial"]).sum()) if len(sr) else 0
        rows.append({"rule": ru.id, "n_events": n, "per_year": n / nyears if nyears else np.nan})
    return pd.DataFrame(rows)


def time_near_zero(rho: pd.Series, band: float) -> float:
    r = rho.dropna()
    return float((r.abs() <= band).mean()) if len(r) else np.nan


# ---------------------------------------------------------------- rho 변화량
def fisher_change(rho: pd.Series, window: int) -> pd.DataFrame:
    """비겹침 창 비교: z_t = [atanh(rho_t) - atanh(rho_{t-w})] / sqrt(2/(w-3)).
    두 창이 겹치지 않으므로 (독립·정규 근사에서) 표준정규. 수익률 꼬리·변동성 군집이 있으면 |z| 가 과대될 수 있다."""
    a = np.arctanh(rho.clip(-0.999999, 0.999999))
    d = a - a.shift(window)
    z = d / np.sqrt(2.0 / (window - 3))
    return pd.DataFrame({"rho": rho, "rho_lag_w": rho.shift(window), "delta_rho": rho - rho.shift(window), "fisher_z": z})


def large_change_events(rho: pd.Series, window: int, z_thr: float = 2.58, min_gap: int | None = None) -> pd.DataFrame:
    """|fisher_z| > z_thr 인 구간을 하나의 사건으로 묶는다 (연속 초과일 = 한 사건, 같은 방향).
    candidate_at = 첫 초과일, peak_at = |z| 최대일. 모두 t 종가에 관측 가능한 양만 사용(미래 없음)."""
    fc = fisher_change(rho, window)
    z = fc["fisher_z"]
    hit = z.abs() > z_thr
    sgn = np.sign(z).where(hit, 0)
    grp = (sgn != sgn.shift(1)).cumsum()
    rows = []
    for _, g in fc[hit].groupby(grp[hit]):
        k = g["fisher_z"].abs().idxmax()
        rows.append({"candidate_at": g.index[0], "end_at": g.index[-1], "peak_at": k, "direction": int(np.sign(g["fisher_z"].iloc[0])),
                     "peak_z": float(g.loc[k, "fisher_z"]), "delta_rho_at_peak": float(g.loc[k, "delta_rho"]), "n_days": len(g)})
    out = pd.DataFrame(rows, columns=["candidate_at", "end_at", "peak_at", "direction", "peak_z", "delta_rho_at_peak", "n_days"])
    if min_gap and len(out):  # 같은 방향 사건이 min_gap 일 안에 다시 시작하면 병합
        keep, last = [], None
        for i, row in out.iterrows():
            if last is not None and row.direction == out.loc[last, "direction"] and \
                    (row.candidate_at - out.loc[last, "end_at"]).days <= min_gap * 7 / 5:
                out.loc[last, "end_at"] = row.end_at
                if abs(row.peak_z) > abs(out.loc[last, "peak_z"]):
                    out.loc[last, ["peak_at", "peak_z", "delta_rho_at_peak"]] = row[["peak_at", "peak_z", "delta_rho_at_peak"]]
                out.loc[last, "n_days"] += row.n_days
            else:
                keep.append(i); last = i
        out = out.loc[keep].reset_index(drop=True)
    return out


# ---------------------------------------------------------------- 비동기 거래 (lead-lag)
def lead_lag_corr(r: pd.DataFrame, ref: str, tickers: list[str], lags=range(-3, 4), method: str = "pearson",
                  period_mask: pd.Series | None = None) -> pd.DataFrame:
    """corr(r_asset_{t+L}, r_ref_t). L=+1: 자산의 '다음날' 수익률과 오늘 KOSPI — 자산이 KOSPI 를 하루 늦게 반영하면 커진다.
    L=-1: 자산의 '어제' 수익률과 오늘 KOSPI — KOSPI 가 해외(전일 미국장)를 하루 늦게 반영하면 커진다."""
    rows = []
    base = r if period_mask is None else r[period_mask.reindex(r.index).fillna(False).astype(bool)]
    for t in tickers:
        for L in lags:
            a = r[t].shift(-L).reindex(base.index)
            b = base[ref]
            if method == "spearman":
                a, b = rank_safe(a), rank_safe(b)
            rows.append({"ticker": t, "lag": L, "rho": a.corr(b, method=method), "n": int(pd.concat([a, b], axis=1).dropna().shape[0])})
    return pd.DataFrame(rows)


def aggregated_corr(r: pd.DataFrame, ref: str, tickers: list[str], k: int = 2) -> pd.DataFrame:
    """비겹침 k 일 합산 수익률 상관. 하루 어긋난 정보가 k 일 블록 안에 같이 들어가므로 비동기 편향이 줄어든다."""
    blk = np.arange(len(r)) // k
    rk = r.groupby(blk).sum(min_count=k)
    return pd.DataFrame({"ticker": tickers, f"rho_{k}d": [rk[t].corr(rk[ref]) for t in tickers],
                         "rho_1d": [r[t].corr(r[ref]) for t in tickers], "n_blocks": len(rk)})


def dimson_corr(r: pd.DataFrame, ref: str, t: str, L: int = 1) -> float:
    """Dimson(1979) 식 합산 공분산: sum_{l=-L..L} cov(r_a,t+l, r_ref,t) / (sd_a·sd_ref). 상관 크기 상한 1 을 보장하지 않는 진단량."""
    a, b = r[t], r[ref]
    cov = sum(a.shift(-l).cov(b) for l in range(-L, L + 1))
    return float(cov / (a.std() * b.std()))


# ---------------------------------------------------------------- 가격·변동성 동시 표시용
def price_context(close: pd.DataFrame, r: pd.DataFrame, tickers: list[str], vol_window: int = 20) -> dict[str, pd.DataFrame]:
    cum = np.exp(r[tickers].fillna(0).cumsum()) - 1
    vol = r[tickers].rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    norm = close[tickers] / close[tickers].iloc[0]
    return {"cum_ret": cum, "vol": vol, "norm_price": norm}


# ---------------------------------------------------------------- 기존 결과 정합
def avg_pairwise_rolling(r: pd.DataFrame, cols: list[str], window: int, method: str = "pearson") -> pd.Series:
    """d0122 · universe_final 의 '평균 pairwise' rolling 상관을 A 의 pair 함수로 재구성 (완전 창, pairwise complete)."""
    acc, cnt = None, None
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rc = rolling_corr(r[cols[i]], r[cols[j]], window, method, min_periods=1, rank_round=(method == "spearman"))["rho"]
            rc = rc.where(np.arange(len(rc)) >= window - 1)
            acc = rc.fillna(0) if acc is None else acc + rc.fillna(0)
            cnt = rc.notna().astype(int) if cnt is None else cnt + rc.notna().astype(int)
    return (acc / cnt.replace(0, np.nan)).rename(f"avg_{method}_w{window}")
