"""A0 analysis harness — read-only over B's raw daily files.

Usage: python a0_harness.py <raw_dir> <out_dir> [universe_csv]
Produces common-calendar matrices + per-ETF profile + cross/shock tables for A1–A4.
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

RAW = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
UNI = Path(sys.argv[3]) if len(sys.argv) > 3 else None
FIG = OUT / "fig"; FIG.mkdir(exist_ok=True)

def read_one(p):
    p = Path(p)
    df = pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p)
    cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols)
    dcol = next((c for c in ["date", "unnamed: 0", "index"] if c in df.columns), df.columns[0])
    df = df.rename(columns={dcol: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return df.set_index("date")

files = sorted([f for f in glob.glob(str(RAW / "**" / "*"), recursive=True)
                if f.endswith((".csv", ".parquet"))])
data = {}
for f in files:
    tk = Path(f).stem.split("_")[0]
    try:
        d = read_one(f)
        if "close" in d and len(d) > 50:
            data[tk] = d
    except Exception as e:
        print("skip", f, e)
print("loaded", len(data), "tickers:", list(data))

names = {}
if UNI and UNI.exists():
    u = pd.read_csv(UNI, dtype=str)
    tcol = next(c for c in u.columns if c.lower() in ("ticker", "symbol", "code"))
    ncol = next((c for c in u.columns if c.lower() == "name"), None)
    if ncol:
        names = dict(zip(u[tcol].str.zfill(6), u[ncol]))

close = pd.DataFrame({k: v["close"] for k, v in data.items()}).sort_index()
vol_ = pd.DataFrame({k: v["volume"] for k, v in data.items() if "volume" in v}).sort_index()
close = close.where(close > 0)
lr = np.log(close).diff()
close.to_parquet(OUT / "close.parquet"); lr.to_parquet(OUT / "logret.parquet"); vol_.to_parquet(OUT / "volume.parquet")

# common window = all ETFs trading
common_start = close.apply(lambda s: s.first_valid_index()).max()
lrc = lr.loc[lr.index > common_start].dropna(how="any")

# ---------- per-ETF profile ----------
rows = []
for k, d in data.items():
    r = lr[k].dropna()
    c = close[k].dropna()
    dd = c / c.cummax() - 1
    trough = dd.idxmin()
    peak = c.loc[:trough].idxmax()
    rec = c.loc[trough:][c.loc[trough:] >= c.loc[peak]]
    v = d["volume"] if "volume" in d else pd.Series(dtype=float)
    rng = ((d["high"] - d["low"]) / d["close"]) if {"high", "low"} <= set(d) else pd.Series(dtype=float)
    ar = r.abs()
    acf_abs = [ar.autocorr(l) for l in (1, 5, 20)]
    acf_r1 = r.autocorr(1)
    q01, q99 = r.quantile([.01, .99])
    rows.append(dict(
        ticker=k, name=names.get(k, ""), start=c.index[0].date(), end=c.index[-1].date(), n_obs=len(r),
        cum_ret=c.iloc[-1] / c.iloc[0] - 1, ann_ret=r.mean() * 252, ann_vol=r.std() * np.sqrt(252),
        skew=stats.skew(r), ex_kurt=stats.kurtosis(r), q01=q01, q99=q99, tail_ratio=abs(q99 / q01) if q01 else np.nan,
        max_abs=r.abs().max(), date_max_up=r.idxmax().date(), max_up=r.max(), date_max_dn=r.idxmin().date(), max_dn=r.min(),
        n_gt_4sd=int((r.abs() > 4 * r.std()).sum()),
        acf_r1=acf_r1, acf_abs1=acf_abs[0], acf_abs5=acf_abs[1], acf_abs20=acf_abs[2],
        max_dd=dd.min(), dd_peak=peak.date(), dd_trough=trough.date(),
        dd_recovered=(rec.index[0].date() if len(rec) else None),
        dd_days_to_trough=int((c.loc[peak:trough]).shape[0]),
        last_dd=dd.iloc[-1],
        vol_ratio_hi_lo=(r.rolling(60).std().quantile(.95) / r.rolling(60).std().quantile(.05)),
        med_volume=v.median() if len(v) else np.nan, zero_vol_days=int((v == 0).sum()) if len(v) else np.nan,
        vol_cv_log=np.log1p(v).std() / np.log1p(v).mean() if len(v) else np.nan,
        corr_absret_logvolchg=(pd.concat([ar, np.log1p(v).diff()], axis=1, sort=True).dropna().corr(method="spearman").iloc[0, 1] if len(v) else np.nan),
        med_range=rng.median() if len(rng) else np.nan,
    ))
prof = pd.DataFrame(rows).set_index("ticker")

# ---------- exclusion-window variant (D-009: 2026-07-20~08-10 common mega-shock) ----------
EXW = (pd.Timestamp("2026-07-20"), pd.Timestamp("2026-08-10"))
lrx = lr.loc[(lr.index < EXW[0]) | (lr.index > EXW[1])]
for k in data:
    r = lrx[k].dropna()
    prof.loc[k, "ann_vol_exw"] = r.std() * np.sqrt(252)
    prof.loc[k, "ex_kurt_exw"] = stats.kurtosis(r)
    prof.loc[k, "skew_exw"] = stats.skew(r)
    prof.loc[k, "acf_abs1_exw"] = r.abs().autocorr(1)
# spike-reversal data-quality suspects: |r_t|>6sd and r_t+1 reverses >=70%
sus = []
for k in data:
    r = lr[k].dropna(); sd = r.std(); nx = r.shift(-1)
    m = (r.abs() > 6 * sd) & (np.sign(nx) == -np.sign(r)) & (nx.abs() >= 0.7 * r.abs())
    for dt in r[m].index:
        sus.append(dict(ticker=k, date=dt.date(), ret=r.loc[dt], next_ret=nx.loc[dt], z=r.loc[dt] / sd,
                        xs_median_same_day=lr.loc[dt].median()))
pd.DataFrame(sus).to_csv(OUT / "dq_spike_reversal.csv", index=False)
prof["n_dq_spike_reversal"] = pd.Series(pd.DataFrame(sus).groupby("ticker").size() if sus else {}, dtype=float).reindex(prof.index).fillna(0)
prof.to_csv(OUT / "profile.csv")

# ---------- extreme windows per ETF (top 5 |r| days, with fwd 1/5/20 log ret) ----------
ext = []
for k in data:
    r = lr[k].dropna(); sd = r.std()
    for dt in r.abs().nlargest(8).index:
        i = r.index.get_loc(dt)
        ext.append(dict(ticker=k, date=dt.date(), ret=r.loc[dt], z=r.loc[dt] / sd,
                        fwd1=r.iloc[i + 1:i + 2].sum(), fwd5=r.iloc[i + 1:i + 6].sum(), fwd20=r.iloc[i + 1:i + 21].sum(),
                        prior20=r.iloc[max(0, i - 20):i].sum(),
                        xs_median_same_day=lr.loc[dt].median(), n_same_sign_gt2sd=int(((lr.loc[dt] / lr.std()).abs() > 2).sum())))
pd.DataFrame(ext).to_csv(OUT / "extremes.csv", index=False)

# ---------- cross-ETF ----------
sp = lrc.corr(method="spearman"); pe = lrc.corr()
sp.to_csv(OUT / "corr_spearman.csv"); pe.to_csv(OUT / "corr_pearson.csv")
lrcx = lrc.loc[(lrc.index < EXW[0]) | (lrc.index > EXW[1])]
lrcx.corr(method="spearman").to_csv(OUT / "corr_spearman_exw.csv")
# calm vs stress: stress = top-10% days by cross-sectional mean |z| (trailing-scale)
zc = lrc / lrc.rolling(252, min_periods=60).std().shift(1)
stress_score = zc.abs().mean(axis=1)
thr = stress_score.quantile(.9)
lrc.loc[stress_score >= thr].corr(method="spearman").to_csv(OUT / "corr_spearman_stress.csv")
lrc.loc[(stress_score < thr) & stress_score.notna()].corr(method="spearman").to_csv(OUT / "corr_spearman_calm.csv")
# lag-1 cross corr (foreign-underlying timing): corr(r_i,t , r_j,t-1)
lag1 = pd.DataFrame({j: lrc.corrwith(lrc[j].shift(1), method="spearman") for j in lrc.columns})
lag1.to_csv(OUT / "corr_lag1_rows_t_cols_tminus1.csv")
dist = np.sqrt(np.clip(2 * (1 - sp.values), 0, None)); np.fill_diagonal(dist, 0)
Z = linkage(squareform(dist, checks=False), "average")
for kk in (3, 4, 5, 6):
    prof[f"cluster_k{kk}"] = pd.Series(fcluster(Z, kk, "maxclust"), index=sp.index)
prof.to_csv(OUT / "profile.csv")
lab = [f"{t} {names.get(t, '')[:14]}" for t in sp.index]
fig, ax = plt.subplots(figsize=(10, 6)); dendrogram(Z, labels=lab, orientation="right", ax=ax)
ax.set_title(f"Spearman-corr dendrogram (common window from {common_start.date()})"); fig.tight_layout(); fig.savefig(FIG / "dendrogram.png", dpi=110)
order = dendrogram(Z, no_plot=True)["leaves"]
fig, ax = plt.subplots(figsize=(11, 9)); m = sp.iloc[order, order]
im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1); ax.set_xticks(range(len(m))); ax.set_yticks(range(len(m)))
ax.set_xticklabels([lab[i] for i in order], rotation=90, fontsize=7); ax.set_yticklabels([lab[i] for i in order], fontsize=7)
fig.colorbar(im); ax.set_title("Spearman corr (cluster order)"); fig.tight_layout(); fig.savefig(FIG / "corr_heatmap.png", dpi=110)

# return-vol plane
fig, ax = plt.subplots(figsize=(9, 6)); ax.scatter(prof.ann_vol, prof.ann_ret)
for t, rr in prof.iterrows(): ax.annotate(f"{t} {names.get(t, '')[:10]}", (rr.ann_vol, rr.ann_ret), fontsize=7)
ax.set_xlabel("ann vol (log)"); ax.set_ylabel("ann mean log ret"); ax.set_title("Return–volatility plane"); fig.tight_layout(); fig.savefig(FIG / "ret_vol_plane.png", dpi=110)

# PCA on standardized common-window returns (diagnostic)
X = (lrc - lrc.mean()) / lrc.std()
ev, evec = np.linalg.eigh(np.cov(X.T.values)); idx = ev.argsort()[::-1]; ev, evec = ev[idx], evec[:, idx]
pca = pd.DataFrame(evec[:, :3], index=lrc.columns, columns=["PC1", "PC2", "PC3"])
pca.loc["explained"] = ev[:3] / ev.sum(); pca.to_csv(OUT / "pca_loadings.csv")
# PCA robustness: drop top-1% |xs mean| days
keep = X.abs().mean(axis=1) < X.abs().mean(axis=1).quantile(.99)
ev2 = np.sort(np.linalg.eigvalsh(np.cov(X[keep].T.values)))[::-1]
Xx = X.loc[(X.index < EXW[0]) | (X.index > EXW[1])]
ev3 = np.sort(np.linalg.eigvalsh(np.cov(Xx.T.values)))[::-1]
json.dump({"pc_explained": (ev[:5] / ev.sum()).round(4).tolist(),
           "pc_explained_drop_top1pct_days": (ev2[:5] / ev2.sum()).round(4).tolist(),
           "pc_explained_excl_2026-07-20_08-10": (ev3[:5] / ev3.sum()).round(4).tolist(),
           "common_start": str(common_start.date()), "n_common_days": int(len(lrc))},
          open(OUT / "pca_summary.json", "w"), indent=2)

# rolling mean pairwise corr (126d)
def mean_offdiag(c):
    a = c.values; n = len(a); return (a.sum() - n) / (n * n - n)
roll = pd.Series({lrc.index[i]: mean_offdiag(lrc.iloc[i - 126:i].corr()) for i in range(126, len(lrc), 5)})
roll.to_csv(OUT / "rolling_mean_corr_126.csv")

# ---------- universe / time (shock) ----------
zs = lr / lr.rolling(252, min_periods=60).std().shift(1)   # trailing-only scale (D-002)
xs = pd.DataFrame({
    "xs_mean": lr.mean(axis=1), "xs_median": lr.median(axis=1), "xs_disp": lr.std(axis=1),
    "xs_iqr": lr.quantile(.75, axis=1) - lr.quantile(.25, axis=1),
    "n_up_gt2z": (zs > 2).sum(axis=1), "n_dn_gt2z": (zs < -2).sum(axis=1), "n_avail": lr.notna().sum(axis=1),
    "frac_pos": (lr > 0).sum(axis=1) / lr.notna().sum(axis=1),
})
xs.to_csv(OUT / "xs_daily.csv")
# common vs specific shock days
common = xs[(xs.n_up_gt2z + xs.n_dn_gt2z) >= 0.5 * xs.n_avail].copy()
common.to_csv(OUT / "common_shock_days.csv")
# per-date participants for common shock days
common.assign(up=[",".join(zs.columns[(zs.loc[d] > 2).values]) for d in common.index],
              dn=[",".join(zs.columns[(zs.loc[d] < -2).values]) for d in common.index]).to_csv(OUT / "common_shock_days.csv")
spec = []
for k in lr.columns:
    s = zs[k]; others = zs.drop(columns=k)
    hit = s[(s.abs() > 3) & (others.abs().median(axis=1) < 1)]
    for dt, v in hit.items():
        spec.append(dict(ticker=k, date=dt.date(), z=v, ret=lr.loc[dt, k], others_median_abs_z=others.loc[dt].abs().median()))
pd.DataFrame(spec).to_csv(OUT / "specific_shock_days.csv", index=False)

fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
axs[0].plot(xs.index, xs.xs_mean.rolling(1).mean(), lw=.6); axs[0].set_title("Cross-sectional mean log return")
axs[1].plot(xs.index, xs.xs_disp, lw=.6); axs[1].set_title("Cross-sectional dispersion (std across ETFs)")
axs[2].plot(roll.index, roll.values); axs[2].set_title("Rolling 126d mean pairwise corr (common window)")
fig.tight_layout(); fig.savefig(FIG / "universe_time.png", dpi=110)

# per-ETF small multiples: normalized price + drawdown
n = len(data); cols = 4; rws = int(np.ceil(n / cols))
fig, axs = plt.subplots(rws, cols, figsize=(18, 3 * rws), squeeze=False)
for a, k in zip(axs.flat, data):
    c = close[k].dropna(); a.plot(c.index, c / c.iloc[0], lw=.7); a2 = a.twinx(); a2.fill_between(c.index, c / c.cummax() - 1, 0, color="tab:red", alpha=.2)
    a.set_title(f"{k} {names.get(k, '')[:18]}", fontsize=8); a.tick_params(labelsize=6); a2.tick_params(labelsize=6)
fig.tight_layout(); fig.savefig(FIG / "small_multiples_price_dd.png", dpi=100)
print("done ->", OUT)
print(prof[["name", "start", "n_obs", "ann_vol", "max_dd", "ex_kurt", "acf_abs1", "cluster_k4"]].to_string())
print(open(OUT / "pca_summary.json").read())
print("common shock days:", len(common), " specific shock rows:", len(spec))
