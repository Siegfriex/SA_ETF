"""Phase A2·A3 — 민감도·강건성 진단. 각 함수는 '결론이 무엇에 따라 바뀌는가'를 표로 돌려준다 (결정적)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import rolling_correlation as RC
from .data import rank_safe


# ---------------------------------------------------------------- 창·방법·기간 요약
def window_summary(r: pd.DataFrame, ref: str, tickers: list[str], windows=(30, 60, 120, 250), methods=("pearson", "spearman"),
                   min_periods_frac: float = 0.9) -> pd.DataFrame:
    per = RC.period_of(r.index)
    rows = []
    for t in tickers:
        for w in windows:
            for m in methods:
                rho = RC.rolling_corr(r[t], r[ref], w, m, min_periods=int(np.ceil(w * min_periods_frac)))["rho"]
                band = RC.null_band(w)
                row = {"ticker": t, "window": w, "method": m, "n_valid": int(rho.notna().sum()), "mean": rho.mean(), "sd": rho.std(),
                       "min": rho.min(), "min_at": rho.idxmin(), "max": rho.max(), "max_at": rho.idxmax(),
                       "share_pos": float((rho > 0).mean()), "null_band": band, "share_within_null_band": RC.time_near_zero(rho, band),
                       "n_zero_cross": len(RC.zero_crossings(rho)),
                       "n_regime_b0.1_k10": int((~RC.sign_regimes(rho, RC.RegimeRule(0.1, 10))["is_initial"]).sum()),
                       "mean_abs_daily_change": rho.diff().abs().mean()}
                for k in RC.PERIODS:
                    row[f"mean_{k}"] = rho[per == k].mean()
                rows.append(row)
    return pd.DataFrame(rows)


def static_corr_by_period(r: pd.DataFrame, ref: str, tickers: list[str]) -> pd.DataFrame:
    per = RC.period_of(r.index)
    rows = []
    for t in tickers:
        row = {"ticker": t, "full_pearson": r[t].corr(r[ref]), "full_spearman": rank_safe(r[t]).corr(rank_safe(r[ref]), method="spearman")}
        for k in RC.PERIODS:
            m = per == k
            row[f"{k}_pearson"] = r.loc[m, t].corr(r.loc[m, ref]); row[f"{k}_n"] = int(m.sum())
        for y in sorted(set(r.index.year)):
            m = r.index.year == y
            row[f"y{y}"] = r.loc[m, t].corr(r.loc[m, ref])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- DQ·이상치 민감도
def rolling_variant_diff(ra: pd.DataFrame, rb: pd.DataFrame, ref: str, ticker: str, windows=(30, 60, 120),
                         method: str = "pearson", min_periods_frac: float = 0.9, tol: float = 0.01) -> pd.DataFrame:
    """두 수익률 변형(예: raw vs dq_trim) 의 rolling rho 차이: 영향받는 기간·최대 차이·부호가 달라진 날 수."""
    rows = []
    for w in windows:
        mp = int(np.ceil(w * min_periods_frac))
        a = RC.rolling_corr(ra[ticker], ra[ref], w, method, min_periods=mp)["rho"]
        b = RC.rolling_corr(rb[ticker], rb[ref], w, method, min_periods=mp)["rho"]
        d = (a - b)
        hit = d.abs() > tol
        rows.append({"ticker": ticker, "window": w, "method": method, "max_abs_diff": float(d.abs().max()),
                     "max_abs_diff_at": d.abs().idxmax() if d.notna().any() else pd.NaT,
                     "n_days_diff_gt_tol": int(hit.sum()), "first_affected": d.index[hit][0] if hit.any() else pd.NaT,
                     "last_affected": d.index[hit][-1] if hit.any() else pd.NaT,
                     "n_sign_disagree": int(((np.sign(a) != np.sign(b)) & a.notna() & b.notna()).sum()),
                     "n_nan_a": int(a.isna().sum()), "n_nan_b": int(b.isna().sum()),
                     "n_zero_cross_a": len(RC.zero_crossings(a)), "n_zero_cross_b": len(RC.zero_crossings(b))})
    return pd.DataFrame(rows)


def mask_days(r: pd.DataFrame, cells: dict[str, list]) -> pd.DataFrame:
    """지정 (ticker → 날짜들) 셀만 NaN. 행은 지우지 않는다 (다른 ETF 의 같은 날 수익률은 그대로 남는다)."""
    out = r.copy()
    for t, ds in cells.items():
        out.loc[pd.to_datetime(list(ds)), t] = np.nan
    return out


def zero_return_diagnostics(r: pd.DataFrame, ref: str, ticker: str, windows=(30, 60, 120)) -> pd.DataFrame:
    """반복 0 수익률의 영향: (a) 창 안 0 비율과 |rho| 의 관계 (b) 0 을 NaN 으로 둔 변형 (c) Pearson vs Spearman."""
    z = (r[ticker] == 0)
    r_nz = r.copy(); r_nz.loc[z, ticker] = np.nan
    rows = []
    for w in windows:
        mp = int(np.ceil(w * 0.9))
        p = RC.rolling_corr(r[ticker], r[ref], w, "pearson", min_periods=mp)["rho"]
        s = RC.rolling_corr(r[ticker], r[ref], w, "spearman", min_periods=mp)["rho"]
        # 0 을 NaN 으로 둬도 날짜 index 는 그대로 — 창 길이(달력)는 같고 유효 관측 n_eff 만 줄어든다 (min_periods 0.5w)
        pnz_full = RC.rolling_corr(r_nz[ticker], r_nz[ref], w, "pearson", min_periods=int(w * 0.5))
        pnz = pnz_full["rho"]
        zshare = z.astype(float).rolling(w).mean()
        both = pd.concat([zshare, p.abs()], axis=1).dropna()
        rows.append({"ticker": ticker, "window": w, "share_zero_mean_in_window": float(zshare.mean()),
                     "share_zero_max_in_window": float(zshare.max()),
                     "spearman_corr_zero_share_vs_abs_rho": float(both.corr(method="spearman").iloc[0, 1]),
                     "n_cross_pearson": len(RC.zero_crossings(p)), "n_cross_spearman": len(RC.zero_crossings(s)),
                     "n_cross_pearson_zero_as_nan": len(RC.zero_crossings(pnz)),
                     "mean_abs_diff_pearson_spearman": float((p - s).abs().mean()),
                     "n_sign_disagree_pearson_spearman": int(((np.sign(p) != np.sign(s)) & p.notna() & s.notna()).sum()),
                     "n_valid": int(p.notna().sum()), "n_valid_zero_as_nan": int(pnz.notna().sum()),
                     "n_eff_mean_zero_as_nan": float(pnz_full["n_eff"][pnz.notna()].mean()),
                     "n_eff_min_zero_as_nan": int(pnz_full["n_eff"][pnz.notna()].min()),
                     "index_preserved": bool(pnz.index.equals(p.index))})
    return pd.DataFrame(rows)


def rank_policy_sensitivity(r_logdiff: pd.DataFrame, r_logratio: pd.DataFrame, ref: str, tickers: list[str],
                            windows=(30, 60, 120)) -> pd.DataFrame:
    """rolling Spearman: (i) logdiff 반올림 vs 미반올림 (ii) 산식 간(logdiff vs logratio) 미반올림 차이."""
    rows = []
    for t in tickers:
        for w in windows:
            mp = int(np.ceil(w * 0.9))
            a = RC.rolling_corr(r_logdiff[t], r_logdiff[ref], w, "spearman", mp, rank_round=True)["rho"]
            b = RC.rolling_corr(r_logdiff[t], r_logdiff[ref], w, "spearman", mp, rank_round=False)["rho"]
            c = RC.rolling_corr(r_logratio[t], r_logratio[ref], w, "spearman", mp, rank_round=False)["rho"]
            ok = a.notna() & b.notna() & c.notna()
            rows.append({"ticker": t, "window": w, "max_abs_round_vs_unround": float((a - b)[ok].abs().max()),
                         "max_abs_formula_unrounded": float((b - c)[ok].abs().max()),
                         "n_sign_flip_formula_unrounded": int((np.sign(b) != np.sign(c))[ok].sum()),
                         "n_sign_flip_round": int((np.sign(a) != np.sign(b))[ok].sum())})
    return pd.DataFrame(rows)


def min_periods_sensitivity(r: pd.DataFrame, ref: str, tickers: list[str], windows=(30, 60, 120)) -> pd.DataFrame:
    rows = []
    for t in tickers:
        for w in windows:
            a = RC.rolling_corr(r[t], r[ref], w, min_periods=w)["rho"]
            b = RC.rolling_corr(r[t], r[ref], w, min_periods=int(np.ceil(0.9 * w)))["rho"]
            ok = a.notna() & b.notna()
            rows.append({"ticker": t, "window": w, "n_valid_full": int(a.notna().sum()), "n_valid_0.9": int(b.notna().sum()),
                         "max_abs_diff_common": float((a - b)[ok].abs().max()) if ok.any() else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 기계적 동시성: 진입일 영향
def entering_day_influence(x: pd.Series, y: pd.Series, window: int, min_periods_frac: float = 0.9) -> pd.DataFrame:
    """rho_t (창 t-w+1..t) vs rho_t^{(-t)} (같은 창에서 '오늘' 하나만 뺀 t-w+1..t-1) vs rho_{t-1} (t-w..t-1).
    오늘 하루를 빼면 0선 교차가 사라진다면 → 그 교차는 '오늘의 큰 수익률 1개'가 만든 기계적 사건이다."""
    mp = int(np.ceil(window * min_periods_frac))
    now = RC.rolling_corr(x, y, window, min_periods=mp)["rho"]
    excl = RC.rolling_corr(x, y, window - 1, min_periods=mp - 1)["rho"].shift(1)
    prev = now.shift(1)
    return pd.DataFrame({"rho": now, "rho_excl_today": excl, "rho_prev": prev, "infl_today": now - excl})


def classify_crossing_drivers(x: pd.Series, y: pd.Series, window: int, dates: pd.Series) -> pd.DataFrame:
    """각 0선 교차일에 대해: entering_day_driven(오늘 빼면 부호가 이전과 같음) / not."""
    inf = entering_day_influence(x, y, window)
    d = pd.to_datetime(dates)
    sub = inf.reindex(d)
    sub["entering_day_driven"] = np.sign(sub["rho_excl_today"]) == np.sign(sub["rho_prev"])
    sub["abs_ret_asset"] = x.abs().reindex(d).to_numpy()
    sub["abs_ret_ref"] = y.abs().reindex(d).to_numpy()
    sub["product_today"] = (x * y).reindex(d).to_numpy()
    return sub.reset_index().rename(columns={"index": "date", "Date": "date"})


# ---------------------------------------------------------------- 기존 결과 정합
def d0122_parity(r: pd.DataFrame, existing: pd.DataFrame, cols: list[str], group: str, windows=(60, 120, 250)) -> pd.DataFrame:
    """analysis/robustness/d0122 rolling_corr_window_estimator_series.csv 의 평균 pairwise 시계열 재현.
    d0122 는 logratio 산식 + pandas corr(pairwise complete) + 완전 창.
    Spearman 은 d0122 가 '창 전체 열별 순위 → 쌍별 상관'(win.rank().corr()) 이고 A 는 '쌍별 완전관측 → 순위' 라서,
    창 안에 NaN(261240 DQ 2일)이 있을 때만 정의가 다르다 → NaN 창과 그 밖을 나눠 보고한다."""
    nanwin = r[cols].isna().any(axis=1)
    rows = []
    for w in windows:
        inwin = nanwin.astype(int).rolling(w, min_periods=1).max().astype(bool)
        for m in ("pearson", "spearman"):
            mine = RC.avg_pairwise_rolling(r, cols, w, m)
            col = f"{group}_{m}_w{w}"
            ex = existing[col].reindex(mine.index)
            ok = ex.notna() & mine.notna()
            d = (mine - ex)[ok].abs()
            rows.append({"series": col, "n_compared": int(ok.sum()), "max_abs_diff": float(d.max()),
                         "max_abs_diff_outside_nan_windows": float(d[~inwin[ok]].max()),
                         "max_abs_diff_in_nan_windows": float(d[inwin[ok]].max()) if inwin[ok].any() else 0.0,
                         "n_nan_windows": int(inwin[ok].sum())})
    return pd.DataFrame(rows)
