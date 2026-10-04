"""Phase 00.5 ETF Final Notebook — 통계 사전계산 + figure 함수 (B, C2-B1).

정의(모든 notebook 공통, 바꾸면 parity 표가 깨진다):
- r_t = log(Close_t / Close_{t-1}), FDR Close (분배 소급조정 추정, DEC-3)
- ann_vol = std(r, ddof=1)·√252 · ex_kurt = pandas kurt (Fisher, bias 보정) · trim3 = |r| 상위 3일 제거 후 kurt
- rz60 = (r_t − med_{t-1}) / (1.4826·MAD_{t-1}), med·MAD 는 직전 60일 창(min 40), MAD = median|r − 창 중앙값|, shift(1) (당일 제외)
- extreme = |rz60| > 3, 부호별(+/−) 분리 · vol_ratio = volume / 직전 20일 median volume (shift(1))
- benchmark: universe.csv benchmark_proxy_ticker (빈 값이면 069500) + market reference 069500 (069500 자신은 102110)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as sps

from src.kfont import setup_korean_font

ROOT = Path(__file__).resolve().parents[1]
MARKET = "069500"
ANN = np.sqrt(252)
C_UP, C_DN, C_MAIN, C_REF, C_GREY = "#c0392b", "#1f5fa8", "#2c3e50", "#e67e22", "#95a5a6"


def raw_dir():
    return Path(os.environ.get("ETF_RAW_DIR", ROOT / "data" / "raw"))


def universe():
    return pd.read_csv(ROOT / "config" / "universe.csv", dtype=str, keep_default_na=False)


def load_panel():
    U = universe()
    close, vol = {}, {}
    for tk in U["ticker"]:
        d = pd.read_csv(raw_dir() / f"{tk}.csv", parse_dates=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
        d = d.set_index("Date")
        close[tk], vol[tk] = d["Close"].astype(float), d["Volume"].astype(float)
    close, vol = pd.DataFrame(close), pd.DataFrame(vol)
    return close, vol, np.log(close).diff()


def anchors():
    a = pd.read_csv(ROOT / "config" / "event_anchors.csv", dtype=str, keep_default_na=False)
    a["start"], a["end"] = pd.to_datetime(a["start"]), pd.to_datetime(a["end"])
    return a


def robust_z(r, win=60, minp=40):
    med = r.rolling(win, min_periods=minp).median()
    mad = r.rolling(win, min_periods=minp).apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
    return (r - med.shift(1)) / (1.4826 * mad.shift(1))


def kurt_trim(r, k=3):
    r = r.dropna()
    return float(r.drop(r.abs().nlargest(k).index).kurt())


def ols(y, x):
    d = pd.concat([y, x], axis=1).dropna()
    yy, xx = d.iloc[:, 0].values, d.iloc[:, 1].values
    b, a = np.polyfit(xx, yy, 1)
    res = yy - (a + b * xx)
    corr = float(np.corrcoef(xx, yy)[0, 1])
    rho = float(sps.spearmanr(xx, yy).statistic)
    return {"beta": float(b), "alpha": float(a), "corr": corr, "spearman": rho, "r2": corr ** 2,
            "resid_sd_bp": float(res.std(ddof=1) * 1e4), "n": int(len(d))}


def dd_series(p):
    return p / p.cummax() - 1


def dd_episodes(p, top=5):
    dd = dd_series(p)
    eps, i, n = [], 0, len(p)
    vals, idx = dd.values, dd.index
    while i < n:
        if vals[i] < 0:
            j = i
            while j < n and vals[j] < 0:
                j += 1
            seg = dd.iloc[i:j]
            t = seg.idxmin()
            eps.append({"peak": idx[i - 1] if i > 0 else idx[i], "trough": t, "depth": float(seg.min()),
                        "recovered": idx[j] if j < n else pd.NaT,
                        "dd_days": int((idx.get_loc(t) - (i - 1))),
                        "rec_days": int(j - idx.get_loc(t)) if j < n else None})
            i = j
        else:
            i += 1
    return pd.DataFrame(eps).sort_values("depth").head(top).reset_index(drop=True)


def refs_for(tk, U):
    row = U.set_index("ticker").loc[tk]
    proxy = row["benchmark_proxy_ticker"] or MARKET
    market = MARKET if tk != MARKET else "102110"
    return proxy, market


def compute(tk, close=None, vol=None, R=None):
    if close is None:
        close, vol, R = load_panel()
    U = universe()
    row = U.set_index("ticker").loc[tk].to_dict()
    proxy, market = refs_for(tk, U)
    r = R[tk].dropna()
    p = close[tk]
    yr = r.groupby(r.index.year).std() * ANN
    rz = robust_z(r)
    vr = vol[tk] / vol[tk].rolling(20, min_periods=10).median().shift(1)
    ext = rz.abs() > 3
    dd = dd_series(p)
    S = {
        "ticker": tk, "name": row["name"], "proxy": proxy, "market": market,
        "proxy_name": U.set_index("ticker").loc[proxy, "name"], "market_name": U.set_index("ticker").loc[market, "name"],
        "n_obs": int(r.size), "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        "ann_vol": float(r.std() * ANN), "vol_2026": float(yr.get(2026, np.nan)),
        "vol_2019_25": float(r[r.index.year < 2026].std() * ANN),
        "mean_ann": float(r.mean() * 252), "skew": float(r.skew()), "ex_kurt": float(r.kurt()),
        "ex_kurt_trim3": kurt_trim(r), "q01": float(r.quantile(.01)), "q99": float(r.quantile(.99)),
        "max_dd": float(dd.min()), "max_dd_date": str(dd.idxmin().date()), "last_dd": float(dd.iloc[-1]),
        "zero_ret_ratio": float((r == 0).mean()),
        "n_ext_pos": int((ext & (rz > 0)).sum()), "n_ext_neg": int((ext & (rz < 0)).sum()),
        "vol_by_year": {int(k): float(v) for k, v in yr.items()},
        "max_up": float(r.max()), "max_up_date": str(r.idxmax().date()),
        "max_dn": float(r.min()), "max_dn_date": str(r.idxmin().date()),
        "vol_med": float(vol[tk].median()), "vr_q99": float(vr.quantile(.99)),
        "corr_absr_vr": float(pd.concat([r.abs(), np.log(vr)], axis=1).replace([np.inf, -np.inf], np.nan).dropna().corr().iloc[0, 1]),
    }
    S["vol_ratio_26"] = S["vol_2026"] / S["vol_2019_25"]
    for lab, ref in (("proxy", proxy), ("mkt", market)):
        o = ols(r, R[ref])
        for k, v in o.items():
            S[f"{lab}_{k}"] = v
    # 2019-21 vs 2022-26 corr with market (regime sign)
    for lab, msk in (("p1", r.index.year <= 2021), ("p2", r.index.year >= 2022)):
        S[f"mkt_corr_{lab}"] = float(r[msk].corr(R[market].reindex(r.index)[msk]))
    rc = r.rolling(120).corr(R[market].reindex(r.index))
    S["roll_corr_min"], S["roll_corr_max"] = float(rc.min()), float(rc.max())
    S["roll_corr_min_date"], S["roll_corr_max_date"] = str(rc.idxmin().date()), str(rc.idxmax().date())
    # relative cumulative vs market (log)
    rel = (r - R[market].reindex(r.index)).cumsum()
    S["rel_cum_end"] = float(rel.iloc[-1])
    S["cum_logret"] = float(r.sum())
    # vol on extreme days
    vr_e = vr.reindex(r.index)
    S["vr_med_ext_neg"] = float(vr_e[ext & (rz < 0)].median()) if (ext & (rz < 0)).any() else np.nan
    S["vr_med_ext_pos"] = float(vr_e[ext & (rz > 0)].median()) if (ext & (rz > 0)).any() else np.nan
    S["vr_med_all"] = float(vr_e.median())
    # anchors participation
    A = anchors()
    S["anchor_ret"] = {a.anchor_id: float(r[(r.index >= a.start) & (r.index <= a.end)].sum()) for a in A.itertuples()}
    S["top_ext"] = [(str(d.date()), float(r[d]), float(rz[d])) for d in rz.abs().nlargest(5).index]
    ep = dd_episodes(p)
    S["episodes"] = [{"peak": str(e.peak.date()), "trough": str(e.trough.date()), "depth": e.depth,
                      "recovered": (str(e.recovered.date()) if pd.notna(e.recovered) else "미회복"),
                      "dd_days": int(e.dd_days), "rec_days": (None if pd.isna(e.rec_days) else int(e.rec_days))} for e in ep.itertuples()]
    S["jul31"] = float(r.get(pd.Timestamp("2026-07-31"), np.nan))
    S["jul31_102110"] = float(R["102110"].get(pd.Timestamp("2026-07-31"), np.nan))
    if tk == "148070":  # proxy 153130 극단 2일 제외 β (artifact 확인)
        drop = pd.to_datetime(["2024-01-30", "2024-01-31"])
        S["proxy_beta_x2"] = ols(r.drop(drop, errors="ignore"), R[proxy].drop(drop, errors="ignore"))["beta"]
    if tk == "261240":  # price_anomaly_unverified 2019-03-14/15 trim 버전
        rt = r.drop(pd.to_datetime(["2019-03-14", "2019-03-15"]), errors="ignore")
        S["trim_anom"] = {"ann_vol": float(rt.std() * ANN), "ex_kurt": float(rt.kurt()),
                          "mkt_corr": float(rt.corr(R[market].reindex(rt.index))),
                          "vol_ratio_26": float(rt[rt.index.year == 2026].std() / rt[rt.index.year < 2026].std())}
    return S


# ------------------------------------------------------------------ figures
class Ctx:
    def __init__(self, tk, outdir=None):
        setup_korean_font()
        self.close, self.vol, self.R = load_panel()
        self.tk = tk
        self.U = universe()
        self.row = self.U.set_index("ticker").loc[tk]
        self.proxy, self.market = refs_for(tk, self.U)
        self.r = self.R[tk].dropna()
        self.p = self.close[tk]
        self.rz = robust_z(self.r)
        self.vr = (self.vol[tk] / self.vol[tk].rolling(20, min_periods=10).median().shift(1)).reindex(self.r.index)
        self.A = anchors()
        self.outdir = Path(outdir or ROOT / "figures" / "etf" / tk)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.label = f"{tk} {self.row['name']}"

    def nm(self, t):
        return f"{t} {self.U.set_index('ticker').loc[t, 'name']}"

    def save(self, fig, name):
        fig.savefig(self.outdir / f"{name}.png", dpi=110, bbox_inches="tight")
        plt.show() if matplotlib.get_backend().lower() not in ("agg",) else None
        plt.close(fig)


def _years(ax, idx):
    for y in range(idx[0].year, idx[-1].year + 1):
        if y % 2 == 0:
            ax.axvspan(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31"), color="#f2f2f2", zorder=0)


def _anchors(ax, A, label=True):
    for a in A.itertuples():
        ax.axvspan(a.start - pd.Timedelta(days=2), a.end + pd.Timedelta(days=2), color="#f5b041", alpha=.35, zorder=0)
        if label:
            ax.text(a.start, 1.0, a.anchor_id, transform=ax.get_xaxis_transform(), fontsize=8, va="bottom", color="#a04000")


def fig01_price_drawdown(c):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    a1.semilogy(c.p.index, c.p.values, color=C_MAIN, lw=1)
    a1.set_ylabel("Close (로그축, 원)"); a1.set_title(f"{c.label} — 가격(로그축)과 고점 대비 낙폭")
    _anchors(a1, c.A)
    dd = dd_series(c.p)
    a2.fill_between(dd.index, dd.values * 100, 0, color=C_DN, alpha=.5)
    a2.set_ylabel("Drawdown (%)"); a2.axhline(0, color="k", lw=.5)
    t = dd.idxmin(); a2.annotate(f"MDD {dd.min():.1%}\n{t.date()}", (t, dd.min() * 100), fontsize=8,
                                 xytext=(10, 10), textcoords="offset points")
    c.save(fig, "01_price_drawdown")


def fig02_return_ts(c):
    fig, ax = plt.subplots(figsize=(12, 3.8))
    _years(ax, c.r.index)
    ax.bar(c.r.index, c.r.values * 100, width=1.5, color=np.where(c.r.values >= 0, C_UP, C_DN))
    sd = c.r.rolling(60, min_periods=40).std().shift(1) * 100
    ax.plot(sd.index, 3 * sd, color="k", lw=.7, label="±3σ (직전 60일, shift1)")
    ax.plot(sd.index, -3 * sd, color="k", lw=.7)
    ax.set_ylabel("일간 로그수익률 (%)"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{c.label} — 일간 수익률과 조건부 ±3σ 밴드 (회색 띠 = 짝수 연도)")
    c.save(fig, "02_return_ts")


def fig03_distribution(c):
    r = c.r * 100
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4))
    a1.hist(r, bins=120, density=True, color=C_GREY, alpha=.8, label="실현 분포")
    xs = np.linspace(r.min(), r.max(), 400)
    a1.plot(xs, sps.norm.pdf(xs, r.mean(), r.std()), color=C_UP, lw=1.2, label="정규(같은 μ,σ)")
    a1.set_yscale("log"); a1.set_ylim(1e-4, None); a1.set_xlabel("일간 로그수익률 (%)"); a1.set_ylabel("밀도 (로그축)")
    a1.legend(fontsize=8); a1.set_title("분포 vs 정규")
    (osm, osr), (sl, ic, _) = sps.probplot(r.values, dist="norm")
    a2.scatter(osm, osr, s=4, color=C_MAIN); a2.plot(osm, sl * osm + ic, color=C_UP, lw=1)
    a2.set_xlabel("정규 이론 분위수"); a2.set_ylabel("표본 분위수 (%)"); a2.set_title("Q-Q (정규)")
    ar = np.sort(r.abs().values)[::-1]; surv = np.arange(1, len(ar) + 1) / len(ar)
    a3.loglog(ar, surv, ".", ms=3, color=C_MAIN, label="실현 |r| 생존함수")
    xs = np.logspace(np.log10(max(ar[-1], 1e-3)), np.log10(ar[0]), 200)
    a3.loglog(xs, 2 * sps.norm.sf(xs, 0, r.std()), color=C_UP, lw=1, label="정규 생존함수")
    a3.set_ylim(1 / len(ar) / 2, 1); a3.set_xlabel("|r| (%)"); a3.set_ylabel("P(|R| > x)"); a3.legend(fontsize=8)
    a3.set_title("꼬리: 생존함수 (log-log)")
    fig.suptitle(f"{c.label} — 수익률 분포·Q-Q·꼬리", y=1.02)
    c.save(fig, "03_distribution_tail")


def fig04_rolling_vol(c):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1.1]})
    for w, col in ((20, C_GREY), (60, C_MAIN)):
        v = c.r.rolling(w).std() * ANN * 100
        a1.plot(v.index, v, color=col, lw=.9 if w == 20 else 1.4, label=f"{w}일 rolling (연환산)")
    a1.axhline(c.r.std() * ANN * 100, color=C_UP, ls="--", lw=1, label="전체기간")
    _anchors(a1, c.A); a1.set_ylabel("연환산 변동성 (%)"); a1.legend(fontsize=8)
    a1.set_title(f"{c.label} — rolling 변동성")
    yv = c.r.groupby(c.r.index.year).std() * ANN * 100
    a2.bar(yv.index.astype(str), yv.values, color=[C_UP if y == 2026 else C_GREY for y in yv.index])
    a2.set_title("연도별 연환산 변동성 (%)"); a2.tick_params(axis="x", rotation=60)
    c.save(fig, "04_rolling_vol")


def fig05_dd_episodes(c):
    ep = dd_episodes(c.p, top=5)
    fig, ax = plt.subplots(figsize=(12, 4))
    dd = dd_series(c.p) * 100
    ax.plot(dd.index, dd, color=C_GREY, lw=.8)
    for i, e in ep.iterrows():
        end = e.recovered if pd.notna(e.recovered) else dd.index[-1]
        ax.axvspan(e.peak, end, color=plt.cm.Reds(0.65 - 0.1 * i), alpha=.3)
        rec = "미회복" if pd.isna(e.rec_days) else f"{int(e.rec_days)}일"
        ax.annotate(f"#{i+1} {e.depth:.1%}\n하락 {int(e.dd_days)}일 / 회복 {rec}", (e.trough, e.depth * 100), fontsize=8,
                    xytext=(6, -14 - 12 * (i % 2)), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.8))
        ax.plot([e.trough], [e.depth * 100], "v", color="k", ms=5)
    ax.set_ylabel("Drawdown (%)"); ax.set_title(f"{c.label} — 상위 낙폭 에피소드 (깊이·하락기간·회복기간, 거래일)")
    c.save(fig, "05_drawdown_episodes")


def fig06_volume(c):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    vr = c.vr
    a1.semilogy(vr.index, vr.values, color=C_GREY, lw=.6)
    big = vr > vr.quantile(.99)
    a1.scatter(vr.index[big], vr[big], s=10, color=C_UP, label="상위 1% 거래량 배수")
    a1.axhline(1, color="k", lw=.5); a1.set_ylabel("거래량 / 직전20일 median (로그축)"); a1.legend(fontsize=8)
    a1.set_title(f"{c.label} — 거래량 배수 (shift1 baseline)")
    ok = vr.notna() & np.isfinite(np.log(vr))
    a2.scatter(c.r[ok] * 100, np.log2(vr[ok]), s=3, alpha=.4, color=C_MAIN)
    e = (c.rz.abs() > 3) & ok
    a2.scatter(c.r[e] * 100, np.log2(vr[e]), s=12, color=C_UP, label="|rz60|>3")
    a2.set_xlabel("일간 수익률 (%)"); a2.set_ylabel("log2 거래량 배수"); a2.legend(fontsize=8)
    a2.set_title("수익률–거래량 joint")
    c.save(fig, "06_volume_joint")


def fig07_benchmark_scatter(c):
    refs = [(c.proxy, "proxy"), (c.market, "market")] if c.proxy != c.market else [(c.proxy, "proxy=market")]
    fig, axes = plt.subplots(1, len(refs), figsize=(6.5 * len(refs), 5), squeeze=False)
    for ax, (ref, lab) in zip(axes[0], refs):
        x, y = c.R[ref].reindex(c.r.index) * 100, c.r * 100
        o = ols(c.r, c.R[ref])
        ax.scatter(x, y, s=4, alpha=.35, color=C_MAIN)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        ax.plot(xs, o["alpha"] * 100 + o["beta"] * xs, color=C_UP, lw=1.3)
        ax.axhline(0, color="k", lw=.4); ax.axvline(0, color="k", lw=.4)
        ax.set_xlabel(f"{c.nm(ref)} 일간 수익률 (%)"); ax.set_ylabel(f"{c.tk} 일간 수익률 (%)")
        ax.set_title(f"{lab}: β={o['beta']:.2f} · ρ={o['corr']:.3f} · R²={o['r2']:.2f} · 잔차sd={o['resid_sd_bp']:.0f}bp", fontsize=9)
    fig.suptitle(f"{c.label} — benchmark 대비 동일일 수익률 산점도와 OLS", y=1.02)
    c.save(fig, "07_benchmark_scatter")


def fig08_rolling_corr_beta(c):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    for ref, col in ((c.market, C_MAIN), (c.proxy, C_REF)):
        x = c.R[ref].reindex(c.r.index)
        rc = c.r.rolling(120).corr(x)
        rb = c.r.rolling(120).cov(x) / x.rolling(120).var()
        a1.plot(rc.index, rc, color=col, lw=1.1, label=c.nm(ref))
        a2.plot(rb.index, rb, color=col, lw=1.1, label=c.nm(ref))
        if c.proxy == c.market:
            break
    for a in (a1, a2):
        a.axhline(0, color="k", lw=.5); _anchors(a, c.A, label=False); a.legend(fontsize=8, loc="lower left")
    a1.set_ylabel("rolling 120일 상관"); a2.set_ylabel("rolling 120일 β")
    a1.set_title(f"{c.label} — benchmark 관계의 시간변화 (120 거래일 창)")
    c.save(fig, "08_rolling_corr_beta")


def fig09_relative(c):
    m = c.R[c.market].reindex(c.r.index)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    a1.plot(c.r.index, c.r.cumsum() * 100, color=C_MAIN, label=c.label)
    a1.plot(m.index, m.cumsum() * 100, color=C_REF, label=c.nm(c.market))
    a1.set_ylabel("누적 로그수익률 (%)"); a1.legend(fontsize=8)
    rel = (c.r - m).cumsum() * 100
    a2.fill_between(rel.index, rel, 0, where=rel >= 0, color=C_UP, alpha=.4)
    a2.fill_between(rel.index, rel, 0, where=rel < 0, color=C_DN, alpha=.4)
    a2.set_ylabel(f"누적 차이 (%p, vs {c.market})")
    a1.set_title(f"{c.label} — 시장 reference 대비 누적 상대수익 (로그, 단순 차)")
    c.save(fig, "09_relative_return")


def fig10_extreme_timeline(c):
    fig, ax = plt.subplots(figsize=(12, 4))
    rz = c.rz.clip(-15, 15)
    ax.plot(rz.index, rz, color=C_GREY, lw=.4)
    pos, neg = c.rz > 3, c.rz < -3
    ax.scatter(rz.index[pos], rz[pos], s=14, color=C_UP, label=f"+ 극단 {int(pos.sum())}일")
    ax.scatter(rz.index[neg], rz[neg], s=14, color=C_DN, label=f"− 극단 {int(neg.sum())}일")
    ax.axhline(3, color="k", ls=":", lw=.7); ax.axhline(-3, color="k", ls=":", lw=.7)
    _anchors(ax, c.A)
    ax.set_ylabel("rz60 (±15 에서 clip)"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"{c.label} — robust-z 극단일 타임라인 (shift1 baseline, 주황 = 공통충격 anchor)")
    c.save(fig, "10_extreme_timeline")


def fig11_event_window(c):
    fig, ax = plt.subplots(figsize=(10, 4))
    m = c.R[c.market]
    for a, col in zip(c.A.itertuples(), plt.cm.tab10.colors):
        i0 = c.r.index.searchsorted(a.start)
        w = c.r.iloc[max(0, i0 - 5): i0 + 11]
        k = np.arange(len(w)) - min(5, i0)
        ax.plot(k, w.cumsum().values * 100, color=col, lw=1.6, label=f"{a.anchor_id} {a.start.date()} {a.label}")
        wm = m.reindex(w.index)
        ax.plot(k, wm.cumsum().values * 100, color=col, lw=.8, ls="--")
    ax.axvline(0, color="k", lw=.6); ax.axhline(0, color="k", lw=.4)
    ax.set_xlabel("anchor 시작일 기준 거래일 (t=0)"); ax.set_ylabel("누적 로그수익률 (%)")
    ax.legend(fontsize=7); ax.set_title(f"{c.label} — 공통충격 anchor 전후 경로 (실선 = 본 ETF, 점선 = {c.market})")
    c.save(fig, "11_event_window")


FIGS = [("01_price_drawdown", fig01_price_drawdown), ("02_return_ts", fig02_return_ts),
        ("03_distribution_tail", fig03_distribution), ("04_rolling_vol", fig04_rolling_vol),
        ("05_drawdown_episodes", fig05_dd_episodes), ("06_volume_joint", fig06_volume),
        ("07_benchmark_scatter", fig07_benchmark_scatter), ("08_rolling_corr_beta", fig08_rolling_corr_beta),
        ("09_relative_return", fig09_relative), ("10_extreme_timeline", fig10_extreme_timeline),
        ("11_event_window", fig11_event_window)]
