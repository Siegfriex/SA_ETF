"""Phase 00.5 Universe-level EDA (A, C2-A1). 결정적: raw → tables(reports/tables/universe_*.csv) + figures(figures/universe/*.png).
usage: python analysis/scripts/universe_final.py   (env ETF_RAW_DIR 로 raw 경로 override)
정의: r = log(Close_t/Close_{t-1}) (FDR Close 그대로). rolling baseline 은 shift(1).
  rz3_D : med=rolling60(min20).median.shift1 ; MAD=|r-med|.rolling60(min20).median.shift1 ; |r-med|/(1.4826 MAD)>3   (D 식)
  rz3_A : med,MAD 모두 창 내부 중앙값 기준, min_periods 40, shift1                                            (A 식)
  q99   : ETF별 전체기간 |r| 99% 분위 초과 (look-ahead 포함 · 서술용)
  abs3  : |r| > 3%
  dedup : KOSPI200 동일 exposure 4종(069500,102110,122630,114800)을 1개로 셈
"""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.kfont import setup_korean_font
FONT = setup_korean_font()

RAW = Path(os.environ.get("ETF_RAW_DIR", ROOT / "data" / "raw"))
FIG = ROOT / "figures" / "universe"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "reports" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
U = pd.read_csv(ROOT / "config" / "universe.csv", dtype=str, keep_default_na=False)
TK = U.ticker.tolist()
SHORT = {r.ticker: r["name"].replace("KODEX ", "").replace("TIGER ", "").replace("PLUS ", "").replace("KIWOOM ", "") for _, r in U.iterrows()}
LAB = {t: f"{t} {SHORT[t]}" for t in TK}
SCOPE = dict(zip(U.ticker, U.asset_scope))
SC_COL = {"domestic_equity": "#2a6fdb", "foreign_equity": "#e07b00", "bond": "#2f9e44", "commodity": "#b5179e", "fx": "#6b7280"}
SC_KO = {"domestic_equity": "국내주식", "foreign_equity": "해외주식", "bond": "채권", "commodity": "원자재", "fx": "FX"}
K200 = ["069500", "102110", "122630", "114800"]
DQ_DAYS = {"261240": ["2019-03-14", "2019-03-15"]}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
OUT = {"figures": [], "tables": [], "checks": {}}


def save(fig, name):
    p = FIG / name; fig.savefig(p, bbox_inches="tight"); plt.close(fig); OUT["figures"].append(str(p.relative_to(ROOT)))


def tsave(df, name, index=True):
    p = TAB / name; df.to_csv(p, index=index, float_format="%.6g"); OUT["tables"].append(str(p.relative_to(ROOT)))


# ---------- panel ----------
close = pd.concat({t: pd.read_csv(RAW / f"{t}.csv", parse_dates=["Date"]).set_index("Date")["Close"] for t in TK}, axis=1).sort_index()
R = np.log(close / close.shift(1)).iloc[1:]
Rt = R.copy()  # trim: DQ price anomaly 날 NaN
for t, ds in DQ_DAYS.items():
    Rt.loc[pd.to_datetime(ds), t] = np.nan
year = R.index.year

# ---------- 1. vol ranking ----------
vol = (R.std() * np.sqrt(252)).rename("ann_vol")
vol_t = (Rt.std() * np.sqrt(252)).rename("ann_vol_trimdq")
v = pd.concat([vol, vol_t], axis=1); v["asset_scope"] = v.index.map(SCOPE); v = v.sort_values("ann_vol")
tsave(v, "universe_vol_rank.csv")
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh([LAB[t] for t in v.index], v.ann_vol * 100, color=[SC_COL[SCOPE[t]] for t in v.index])
ax.set_xscale("log"); ax.set_xlabel("연환산 변동성 (%, 로그축) — std(일간 로그수익률)×√252, 2019-01-03~2026-10-02")
for i, (t, x) in enumerate(v.ann_vol.items()):
    ax.text(x * 100 * 1.05, i, f"{x*100:.2f}%", va="center", fontsize=8)
for k, c in SC_COL.items():
    ax.bar(0, 0, color=c, label=SC_KO[k])
ax.legend(loc="lower right"); ax.set_title(f"변동성 스케일: 최대/최소 = {v.ann_vol.max()/v.ann_vol.min():.0f}배")
save(fig, "01_vol_ranking.png")
OUT["checks"]["vol_max_min_ratio"] = float(v.ann_vol.max() / v.ann_vol.min())

# ---------- 2. ETF x year vol heatmap ----------
vy = (R.groupby(year).std() * np.sqrt(252)).T
order_scope = sorted(TK, key=lambda t: (list(SC_COL).index(SCOPE[t]), TK.index(t)))
vy = vy.loc[order_scope]
tsave(vy, "universe_vol_by_year.csv")
rel = vy.div(vy.loc[:, [y for y in vy.columns if y < 2026]].median(axis=1), axis=0)
fig, axs = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1, 1]})
for ax, M, ttl, fmt, cm in [(axs[0], vy * 100, "연도별 연환산 변동성 (%)", "{:.0f}", "viridis"), (axs[1], rel, "연도 vol / 2019-25 연도 vol 중앙값 (배)", "{:.1f}", "RdBu_r")]:
    kw = dict(vmin=0, vmax=2, cmap=cm) if cm == "RdBu_r" else dict(cmap=cm, norm=matplotlib.colors.LogNorm(vmin=0.2, vmax=140))
    im = ax.imshow(M.values, aspect="auto", **kw)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([LAB[t] for t in M.index], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M.values[i, j]), ha="center", va="center", fontsize=7, color="k" if cm == "RdBu_r" else "w")
    ax.set_title(ttl); ax.grid(False); fig.colorbar(im, ax=ax, shrink=.7)
save(fig, "02_vol_year_heatmap.png")
v26 = (R[year == 2026].std() / R[year < 2026].std()).rename("vol_ratio_2026_vs_2019_25")
tsave(v26.to_frame().assign(asset_scope=lambda d: d.index.map(SCOPE)), "universe_vol_ratio_2026.csv")
OUT["checks"]["vol_ratio_2026"] = v26.round(2).to_dict()

# ---------- 3. Q01-Q99 interval ----------
q = R.quantile([.01, .05, .25, .5, .75, .95, .99]).T
q = q.loc[v.index]
tsave(q, "universe_quantiles.csv")
fig, ax = plt.subplots(figsize=(9, 7))
y = np.arange(len(q))
ax.hlines(y, q[.01] * 100, q[.99] * 100, color=[SC_COL[SCOPE[t]] for t in q.index], lw=1.5)
ax.hlines(y, q[.05] * 100, q[.95] * 100, color=[SC_COL[SCOPE[t]] for t in q.index], lw=4)
ax.hlines(y, q[.25] * 100, q[.75] * 100, color="k", lw=7, alpha=.6)
ax.scatter(R.min()[q.index] * 100, y, marker="|", c="r", s=60, label="전체기간 최저")
ax.scatter(R.max()[q.index] * 100, y, marker="|", c="g", s=60, label="전체기간 최고")
ax.set_yticks(y); ax.set_yticklabels([LAB[t] for t in q.index], fontsize=8)
ax.set_xlabel("일간 로그수익률 (%) — 얇은선 Q01–Q99 · 굵은선 Q05–Q95 · 검정 IQR"); ax.set_xlim(-25, 25); ax.axvline(0, c="k", lw=.5)
ax.legend(loc="lower right"); ax.set_title("ETF별 일간수익률 분위 구간 (x축 ±25% 절단)")
save(fig, "03_quantile_interval.png")

# ---------- 4. distribution comparison (standardized) ----------
Z = (Rt - Rt.median()) / (1.4826 * (Rt - Rt.median()).abs().median())
fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
data = [Z[t].dropna().clip(-15, 15).values for t in v.index]
parts = axs[0].violinplot(data, orientation="horizontal", showextrema=False, widths=.9)
for pc, t in zip(parts["bodies"], v.index):
    pc.set_facecolor(SC_COL[SCOPE[t]]); pc.set_alpha(.6)
axs[0].set_yticks(range(1, len(v) + 1)); axs[0].set_yticklabels([LAB[t] for t in v.index], fontsize=8)
axs[0].set_xlabel("robust 표준화 수익률 (r−median)/(1.4826·MAD), ±15 clip"); axs[0].set_title("스케일 제거 후 분포 모양")
xs = np.linspace(0, 15, 300)
from scipy.stats import norm
for t in v.index:
    a = np.abs(Z[t].dropna().values)
    axs[1].plot(xs, [(a > x).mean() for x in xs], color=SC_COL[SCOPE[t]], lw=1, alpha=.8)
axs[1].plot(xs, 2 * norm.sf(xs), "k--", lw=2, label="정규분포 2·SF(z)")
axs[1].set_yscale("log"); axs[1].set_ylim(1e-4, 1); axs[1].set_xlabel("|robust z| 임계값"); axs[1].set_ylabel("P(|z| > x) 경험적 (로그)")
axs[1].set_title("꼬리 생존함수 (색=자산군)"); axs[1].legend()
save(fig, "04_distribution_standardized.png")
tail = pd.Series({t: (np.abs(Z[t].dropna()) > 5).mean() for t in TK}, name="p_absrz_gt5")
OUT["checks"]["p_absrz_gt5_normal"] = float(2 * norm.sf(5))

# ---------- 5. Spearman corr heatmap (cluster order) ----------
S = Rt.rank().corr(method="pearson") if False else Rt.corr(method="spearman")
P = Rt.corr()
tsave(S, "universe_corr_spearman.csv"); tsave(P, "universe_corr_pearson.csv")
Dm = np.clip(1 - S.values, 0, None); np.fill_diagonal(Dm, 0)
Lk = linkage(squareform(Dm, checks=False), "average")
ordr = [TK[i] for i in leaves_list(Lk)]
fig, ax = plt.subplots(figsize=(11, 9.5))
M = S.loc[ordr, ordr]
im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(20)); ax.set_xticklabels([LAB[t] for t in ordr], rotation=90, fontsize=8)
ax.set_yticks(range(20)); ax.set_yticklabels([LAB[t] for t in ordr], fontsize=8)
for i in range(20):
    for j in range(20):
        ax.text(j, i, f"{M.values[i,j]:.2f}", ha="center", va="center", fontsize=5.5, color="w" if abs(M.values[i, j]) > .6 else "k")
for lab in ax.get_yticklabels() + ax.get_xticklabels():
    lab.set_color(SC_COL[SCOPE[lab.get_text()[:6]]])
ax.grid(False); fig.colorbar(im, shrink=.7, label="Spearman ρ")
ax.set_title("Spearman 상관 (261240 DQ 2일 제외) · 순서 = average linkage on 1−ρ · 라벨색 = 자산군")
save(fig, "05_corr_spearman_clustered.png")

# ---------- 6. dendrogram ----------
fig, ax = plt.subplots(figsize=(12, 5.5))
dendrogram(Lk, labels=[LAB[t] for t in TK], leaf_rotation=90, ax=ax, color_threshold=0.7 * Lk[:, 2].max())
for lab in ax.get_xticklabels():
    lab.set_color(SC_COL[SCOPE[lab.get_text()[:6]]])
ax.set_ylabel("병합 거리 (1 − Spearman ρ, average)"); ax.set_title("통계적 군집 vs 자산군(라벨 색) — 1−ρ 이므로 음의 상관 쌍은 가장 멀다")
save(fig, "06_dendrogram.png")
clus = pd.DataFrame({k: fcluster(Lk, k, "maxclust") for k in [3, 5, 8]}, index=TK); clus.columns = [f"avg_k{k}" for k in clus.columns]
clus["asset_scope"] = clus.index.map(SCOPE); clus["category"] = U.set_index("ticker").category
tsave(clus, "universe_clusters.csv")
# signed-distance artifact check: 1-|rho|
Da = np.clip(1 - np.abs(S.values), 0, None); np.fill_diagonal(Da, 0)
cl_abs = fcluster(linkage(squareform(Da, checks=False), "average"), 5, "maxclust")
OUT["checks"]["cluster_abs_k5_114800_with_069500"] = bool(cl_abs[TK.index("114800")] == cl_abs[TK.index("069500")])
OUT["checks"]["cluster_signed_k5_114800_with_261240"] = bool(clus.loc["114800", "avg_k5"] == clus.loc["261240", "avg_k5"])

# ---------- 7. period-difference corr (Fisher z) ----------
pre, post = Rt[year < 2026], Rt[year == 2026]
Cpre, Cpost = pre.corr(), post.corr()
Zd = np.arctanh(Cpost.clip(-.999, .999)) - np.arctanh(Cpre.clip(-.999, .999))
se = np.sqrt(1 / (len(post) - 3) + 1 / (len(pre) - 3))
Zs = Zd / se
Zs = pd.DataFrame(np.where(np.eye(20, dtype=bool), 0.0, Zs.values), index=Zs.index, columns=Zs.columns)
fig, axs = plt.subplots(1, 2, figsize=(17, 7.5))
for ax, M, ttl, lim, lbl in [(axs[0], (Cpost - Cpre).loc[ordr, ordr], f"Δρ = ρ(2026, n={len(post)}) − ρ(2019-25, n={len(pre)})", .6, "Δ Pearson ρ"),
                              (axs[1], Zs.loc[ordr, ordr], "Fisher z 차이 통계량 (|z|>3.4 ≈ Bonferroni 190쌍 α=.05)", 8, "z")]:
    im = ax.imshow(M.values, cmap="PuOr_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(20)); ax.set_xticklabels([LAB[t] for t in ordr], rotation=90, fontsize=7)
    ax.set_yticks(range(20)); ax.set_yticklabels([LAB[t] for t in ordr], fontsize=7)
    ax.grid(False); ax.set_title(ttl, fontsize=10); fig.colorbar(im, ax=ax, shrink=.7, label=lbl)
    if lim == 8:
        for i in range(20):
            for j in range(20):
                if abs(M.values[i, j]) > 3.4:
                    ax.text(j, i, "•", ha="center", va="center", fontsize=8)
save(fig, "07_period_diff_corr.png")
iu = np.triu_indices(20, 1)
pd_tab = pd.DataFrame({"a": [TK[i] for i in iu[0]], "b": [TK[j] for j in iu[1]], "rho_2019_25": Cpre.values[iu], "rho_2026": Cpost.values[iu], "fisher_z_stat": Zs.values[iu]})
pd_tab = pd_tab.reindex(pd_tab.fisher_z_stat.abs().sort_values(ascending=False).index)
tsave(pd_tab, "universe_corr_period_diff.csv", index=False)
OUT["checks"]["n_pairs_absz_gt_3.4"] = int((pd_tab.fisher_z_stat.abs() > 3.4).sum())

# ---------- 8. rolling avg pairwise corr + dispersion ----------
EQ = [t for t in TK if SCOPE[t] == "domestic_equity" and t not in ("102110", "122630", "114800")]
ALLD = [t for t in TK if t not in ("102110", "122630", "114800")]
def avg_pair(cols, w=120):
    X = Rt[cols]
    out = []
    idx = X.index
    for i in range(w - 1, len(X)):
        c = X.iloc[i - w + 1:i + 1].corr().values
        out.append(np.nanmean(c[np.triu_indices(len(cols), 1)]))
    return pd.Series(out, index=idx[w - 1:])
rc_eq = avg_pair(EQ); rc_all = avg_pair(ALLD)
Zv = Rt[EQ] / Rt[EQ].rolling(250, min_periods=120).std().shift(1)
disp = Zv.std(axis=1).rolling(20).mean()
mkt_vol = R["069500"].rolling(20).std() * np.sqrt(252)
roll = pd.DataFrame({"avgcorr_domeq9_120d": rc_eq, "avgcorr_dedup17_120d": rc_all, "xs_disp_volnorm_20d": disp, "vol20_069500": mkt_vol})
tsave(roll, "universe_rolling_state.csv")
fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
axs[0].plot(rc_eq, label="국내주식 9종(KOSPI200 중복 제외) 평균 pairwise ρ", c="#2a6fdb")
axs[0].plot(rc_all, label="전체 17종(K200 dedup) 평균 pairwise ρ", c="#6b7280")
axs[0].set_ylabel("120일 평균 ρ"); axs[0].legend(fontsize=8)
axs[1].plot(disp, c="#b5179e"); axs[1].set_ylabel("vol-정규화 횡단면\n분산 (20일 평균)")
axs[2].plot(mkt_vol * 100, c="k"); axs[2].set_ylabel("069500 20일\n연환산 vol (%)")
for ax in axs:
    for s, e in [("2020-03-19", "2020-03-24"), ("2024-08-05", "2024-08-05"), ("2026-03-04", "2026-03-04"), ("2025-04-07", "2025-04-10")]:
        ax.axvspan(pd.Timestamp(s) - pd.Timedelta(days=2), pd.Timestamp(e) + pd.Timedelta(days=2), color="r", alpha=.25)
axs[0].set_title("관계는 상태변수: 평균 상관·횡단면 분산·시장 변동성 (빨강 = shock anchor)")
save(fig, "08_rolling_corr_dispersion.png")
OUT["checks"]["corr_rc_eq_vs_mktvol"] = float(pd.concat([rc_eq, mkt_vol.rolling(120).mean()], axis=1).dropna().corr(method="spearman").iloc[0, 1])

# ---------- 9. PCA ----------
def pca(cols):
    X = Rt[cols].dropna(); X = (X - X.mean()) / X.std()
    w, V = np.linalg.eigh(np.cov(X.values.T)); o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    if V[:, 0][cols.index("069500")] < 0: V[:, 0] *= -1
    return w / w.sum(), pd.DataFrame(V[:, :3], index=cols, columns=["PC1", "PC2", "PC3"])
ev_all, L_all = pca(TK)
DED = [t for t in TK if t not in ("102110", "122630", "114800")]
ev_d, L_d = pca(DED)
EXEQ = [t for t in TK if SCOPE[t] != "domestic_equity"] + ["069500"]
ev_x, _ = pca(EXEQ)
pcs = pd.DataFrame({"all20": pd.Series(ev_all[:6]), "dedup17": pd.Series(ev_d[:6]), "nondomestic8+069500": pd.Series(ev_x[:6])}); pcs.index = [f"PC{i+1}" for i in range(6)]
tsave(pcs, "universe_pca_evr.csv"); tsave(L_all.join(L_d, rsuffix="_dedup"), "universe_pca_loadings.csv")
fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
for c, m in zip(pcs.columns, ["o", "s", "^"]):
    axs[0].plot(range(1, 7), pcs[c] * 100, marker=m, label=f"{c} (PC1 {pcs[c].iloc[0]*100:.1f}%)")
axs[0].set_xlabel("주성분"); axs[0].set_ylabel("설명분산 비중 (%)"); axs[0].legend(); axs[0].set_title("PCA scree: universe 구성에 따라 PC1 비중이 바뀐다 (CL-13)")
o2 = L_all.sort_values("PC1").index
axs[1].barh([LAB[t] for t in o2], L_all.loc[o2, "PC1"], color=[SC_COL[SCOPE[t]] for t in o2], alpha=.85, label="PC1 (all20)")
axs[1].scatter(L_all.loc[o2, "PC2"], range(20), c="k", marker="x", label="PC2 (all20)")
axs[1].axvline(0, c="k", lw=.5); axs[1].legend(fontsize=8); axs[1].set_xlabel("loading (표준화 수익률)"); axs[1].tick_params(axis="y", labelsize=8)
axs[1].set_title("PC1/PC2 loading")
save(fig, "09_pca_scree_loading.png")
OUT["checks"]["pca_pc1"] = pcs.iloc[0].round(3).to_dict()

# ---------- 10. shock definitions ----------
def rz_D(r):
    med = r.rolling(60, min_periods=20).median().shift(1)
    mad = (r - med).abs().rolling(60, min_periods=20).median().shift(1) * 1.4826
    return (r - med) / mad
def rz_A(r):
    med = r.rolling(60, min_periods=40).median().shift(1)
    mad = r.rolling(60, min_periods=40).apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True).shift(1)
    return (r - med) / (1.4826 * mad)
RZD, RZA = rz_D(Rt), rz_A(Rt)
q99 = Rt.abs().quantile(.99)
FLAG = {"rz3_D": RZD.abs() > 3, "rz3_A": RZA.abs() > 3, "q99": Rt.abs() > q99, "abs3": Rt.abs() > .03}
def dedup_count(F):
    k200 = F[K200].any(axis=1).astype(int)
    return F.drop(columns=K200).sum(axis=1) + k200
valid = RZA.notna().sum(axis=1) >= 15  # A 식 warm-up 이후만 공정 비교
sens = []
for name, F in FLAG.items():
    F = F & valid.values[:, None]
    n, nd = F.sum(axis=1), dedup_count(F)
    for k in [3, 5, 8]:
        sens.append({"definition": name, "k": k, "n_days": int((n >= k).sum()), "n_days_dedup": int((nd >= k).sum())})
sens = pd.DataFrame(sens); tsave(sens, "universe_shock_definition_sensitivity.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
pv = sens.pivot(index="k", columns="definition", values="n_days")[list(FLAG)]
pvd = sens.pivot(index="k", columns="definition", values="n_days_dedup")[list(FLAG)]
pv.T.plot.bar(ax=axs[0], rot=0); axs[0].set_yscale("log"); axs[0].set_ylabel("공통 극단일 수 (로그)")
axs[0].set_title("정의 × k: 동반 ETF 수 ≥ k 인 날 수 (raw 집계)")
for ctn in axs[0].containers:
    axs[0].bar_label(ctn, fontsize=7)
red = (1 - pvd / pv) * 100
red.T.plot.bar(ax=axs[1], rot=0); axs[1].set_ylabel("dedup 후 감소율 (%)"); axs[1].set_title("same-exposure dedup 효과: KOSPI200 4종 → 1")
for ctn in axs[1].containers:
    axs[1].bar_label(ctn, fmt="%.0f", fontsize=7)
save(fig, "10_shock_definition_sensitivity_dedup.png")
OUT["checks"]["shock_k5"] = {r.definition: [r.n_days, r.n_days_dedup] for r in sens[sens.k == 5].itertuples()}

# jaccard between definitions at k>=5 dedup
sets = {nm: set(R.index[(dedup_count(F & valid.values[:, None]) >= 5)]) for nm, F in FLAG.items()}
jac = pd.DataFrame({a: {b: len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b])) for b in sets} for a in sets})
tsave(jac, "universe_shock_def_jaccard_k5dedup.csv")

# ---------- 11. breadth timeline ----------
nD = (FLAG["rz3_D"] & valid.values[:, None]).sum(axis=1); nDd = dedup_count(FLAG["rz3_D"] & valid.values[:, None])
nabs = dedup_count(FLAG["abs3"])
br = pd.DataFrame({"n_rz3D": nD, "n_rz3D_dedup": nDd, "n_abs3_dedup": nabs, "n_q99_dedup": dedup_count(FLAG["q99"])})
tsave(br, "universe_breadth_daily.csv")
fig, axs = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
axs[0].vlines(br.index, 0, br.n_rz3D, color="#9aa5b1", lw=.8, label="rz3_D raw 동반 수")
axs[0].vlines(br.index, 0, br.n_rz3D_dedup, color="#2a6fdb", lw=.8, label="rz3_D dedup 동반 수")
axs[0].axhline(5, c="r", ls="--", lw=.8, label="k=5"); axs[0].legend(fontsize=8, ncol=3); axs[0].set_ylabel("동반 ETF 수")
axs[1].vlines(br.index, 0, br.n_abs3_dedup, color="#e07b00", lw=.8, label="abs3 dedup"); axs[1].axhline(5, c="r", ls="--", lw=.8)
axs[1].legend(fontsize=8); axs[1].set_ylabel("동반 ETF 수")
for d0, txt in [("2020-03-19", "2020-03"), ("2024-08-05", "2024-08-05"), ("2025-04-07", "2025-04-07/10\n(FRAGILE)"), ("2026-03-04", "2026-03-04")]:
    axs[0].annotate(txt, (pd.Timestamp(d0), 17), fontsize=8, ha="center", color="r")
axs[0].set_title("Shock breadth timeline: rz3(D식)는 고변동 국면에서 기준선이 넓어져 2026 후반을 과소계수 (CL-21) · abs3 는 2026 에 몰림")
save(fig, "11_breadth_timeline.png")

# ---------- 12. anchor participation + sign matrix ----------
anchors = {"2020-03-19": "A1", "2020-03-20": "A1", "2020-03-23": "A1", "2020-03-24": "A1", "2024-08-05": "A2", "2026-03-04": "A3", "2025-04-07": "F1(FRAGILE)", "2025-04-10": "F1(FRAGILE)"}
ad = pd.to_datetime(sorted(anchors))
part = (RZD.loc[ad].abs() > 3).astype(int)
sign = np.sign(Rt.loc[ad].where(Rt.loc[ad].abs() >= 0.001, 0.0)) * np.sign(Rt.loc[ad, "069500"]).values[:, None]  # |r|<0.1% 는 0 (경제적 최소 크기)
rr = Rt.loc[ad] * 100
anc = []
for d in ad:
    row = {"date": d.date(), "anchor": anchors[str(d.date())], "r_069500_pct": rr.loc[d, "069500"]}
    for nm, F in FLAG.items():
        row[f"n_{nm}"] = int(F.loc[d].sum()); row[f"n_{nm}_dedup"] = int(dedup_count(F.loc[[d]]).iloc[0])
    row["n_same_sign_as_069500"] = int((sign.loc[d] > 0).sum()); row["n_opposite"] = int((sign.loc[d] < 0).sum())
    anc.append(row)
anc = pd.DataFrame(anc); tsave(anc, "universe_shock_anchor_survival.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(17, 6.5))
ox = order_scope
im = axs[0].imshow(rr[ox].values, cmap="RdBu", vmin=-10, vmax=10, aspect="auto")
for i in range(len(ad)):
    for j, t in enumerate(ox):
        txt = f"{rr.iloc[i][t]:.1f}" + ("*" if part.iloc[i][t] else "")
        axs[0].text(j, i, txt, ha="center", va="center", fontsize=6.5, color="w" if abs(rr.iloc[i][t]) > 6 else "k")
axs[0].set_title("anchor일 ETF별 로그수익률 (%) · * = |rz_D|>3 참여", fontsize=10)
im2 = axs[1].imshow(sign[ox].values, cmap="PiYG", vmin=-1, vmax=1, aspect="auto")
axs[1].set_title("부호 행렬: 069500 과 같은 방향(+1) / 반대(−1) / |r|<0.1% = 0(흰색)", fontsize=10)
for ax in axs:
    ax.set_xticks(range(20)); ax.set_xticklabels([LAB[t] for t in ox], rotation=90, fontsize=7)
    ax.set_yticks(range(len(ad))); ax.set_yticklabels([f"{d.date()} {anchors[str(d.date())]}" for d in ad], fontsize=8); ax.grid(False)
    for lab in ax.get_xticklabels():
        lab.set_color(SC_COL[SCOPE[lab.get_text()[:6]]])
fig.colorbar(im, ax=axs[0], shrink=.7); fig.colorbar(im2, ax=axs[1], shrink=.7)
save(fig, "12_shock_participation_sign.png")

# participation rate by ETF on top common days (rz3_D dedup>=5)
topd = br.index[br.n_rz3D_dedup >= 5]
pr = pd.DataFrame({"participation_rate": (RZD.loc[topd].abs() > 3).mean(), "same_sign_rate_vs_069500": (np.sign(Rt.loc[topd]).mul(np.sign(Rt.loc[topd, "069500"]), axis=0) > 0).mean(),
                   "mean_r_pct_on_down_days": Rt.loc[topd][Rt.loc[topd, "069500"] < 0].mean() * 100})
pr["asset_scope"] = pr.index.map(SCOPE); tsave(pr, "universe_shock_participation_by_etf.csv")
fig, ax = plt.subplots(figsize=(10, 6))
for t, rw in pr.iterrows():
    ax.scatter(rw.participation_rate, rw.same_sign_rate_vs_069500, s=60, c=SC_COL[SCOPE[t]])
    ax.annotate(LAB[t], (rw.participation_rate, rw.same_sign_rate_vs_069500), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.axhline(.5, c="k", lw=.5); ax.set_xlabel(f"참여율: 공통충격일(rz3_D dedup≥5, n={len(topd)}) 중 |rz|>3 비율"); ax.set_ylabel("069500 과 같은 부호 비율")
ax.set_title("누가 공통충격에 참여하고, 어느 방향인가")
save(fig, "13_shock_participation_scatter.png")
OUT["checks"]["n_common_days_rz3D_dedup5"] = int(len(topd))

# ---------- 13. key pair claims ----------
def pair(a, b, X=Rt):
    xy = X[[a, b]].dropna()
    return {"pearson": xy.corr().iloc[0, 1], "spearman": xy.corr("spearman").iloc[0, 1], "n": len(xy)}
claims = {
    "069500~102110": pair("069500", "102110"),
    "143860~229200": pair("143860", "229200"), "143860~069500": pair("143860", "069500"),
    "161510~091170": pair("161510", "091170"), "161510~069500": pair("161510", "069500"),
    "114800~069500": pair("114800", "069500"), "261240~069500": pair("261240", "069500"),
    "305540~117680": pair("305540", "117680"), "305540~091230": pair("305540", "091230"),
    "261220~069500_2019_25": pair("261220", "069500", pre), "261220~069500_2026": pair("261220", "069500", post),
    "261220~219480_2019_25": pair("261220", "219480", pre), "261220~219480_2026": pair("261220", "219480", post),
}
cl = pd.DataFrame(claims).T
cl.loc["069500-102110_diff_sd_bp", "pearson"] = (R["069500"] - R["102110"]).std() * 1e4
beta = lambda a, b: np.cov(Rt[[a, b]].dropna().values.T)[0, 1] / Rt[b].var()
cl.loc["beta_122630_on_069500", "pearson"] = beta("122630", "069500"); cl.loc["beta_114800_on_069500", "pearson"] = beta("114800", "069500")
# 305540 regime check
for lab_, m in [("2019_21", (year <= 2021)), ("2022_25", (year >= 2022) & (year <= 2025)), ("2026", year == 2026)]:
    cl.loc[f"305540~117680_{lab_}", "pearson"] = Rt.loc[m, ["305540", "117680"]].corr().iloc[0, 1]
    cl.loc[f"305540~091230_{lab_}", "pearson"] = Rt.loc[m, ["305540", "091230"]].corr().iloc[0, 1]
tsave(cl, "universe_claim_checks.csv")
# bootstrap CI (block) for corr differences
rng = np.random.default_rng(20261004)
def block_boot_diff(a, b, c, B=1000, bl=20):
    X = Rt[[a, b, c]].dropna().values; n = len(X); out = []
    for _ in range(B):
        st = rng.integers(0, n - bl, n // bl + 1); idx = np.concatenate([np.arange(s, s + bl) for s in st])[:n]
        Y = X[idx]; Cm = np.corrcoef(Y.T, ) if False else pd.DataFrame(Y).corr("spearman").values
        out.append(Cm[0, 1] - Cm[0, 2])
    return np.percentile(out, [2.5, 50, 97.5])
bci = {"rho(143860,229200)-rho(143860,069500)": block_boot_diff("143860", "229200", "069500", B=300),
       "rho(161510,091170)-rho(161510,069500)": block_boot_diff("161510", "091170", "069500", B=300)}
bci = pd.DataFrame(bci, index=["ci_lo", "median", "ci_hi"]).T; tsave(bci, "universe_claim_bootstrap_ci.csv")
OUT["checks"]["bootstrap_ci"] = bci.round(3).to_dict(orient="index")
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
for ax, (a, b, c) in zip(axs, [("143860", "229200", "069500"), ("161510", "091170", "069500"), ("305540", "117680", "091230")]):
    s1 = Rt[a].rolling(120, min_periods=100).corr(Rt[b]); s2 = Rt[a].rolling(120, min_periods=100).corr(Rt[c])
    ax.plot(s1, label=f"ρ({SHORT[a]}, {SHORT[b]})"); ax.plot(s2, label=f"ρ({SHORT[a]}, {SHORT[c]})")
    ax.fill_between(s1.index, s1, s2, where=s1 > s2, color="C0", alpha=.15); ax.fill_between(s1.index, s1, s2, where=s1 <= s2, color="C1", alpha=.15)
    ax.legend(fontsize=8); ax.set_ylim(-.2, 1); ax.set_title(f"{a} 의 경험적 peer (120일 rolling Pearson)", fontsize=10)
save(fig, "14_empirical_peer_rolling.png")
# WTI sign flip
fig, ax = plt.subplots(figsize=(13, 4))
for b_, c_ in [("069500", "#2a6fdb"), ("219480", "#e07b00"), ("261240", "#6b7280")]:
    ax.plot(Rt["261220"].rolling(120, min_periods=100).corr(Rt[b_]), c=c_, label=f"ρ(WTI 261220, {LAB[b_]})")
ax.axhline(0, c="k", lw=.6); ax.legend(fontsize=8); ax.set_title("관계가 깨지는 때: WTI–주식 상관의 부호 반전 (CL-31) · 120일 rolling")
save(fig, "15_wti_equity_sign_flip.png")

json.dump(OUT, open(TAB / "universe_run_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(OUT["checks"], ensure_ascii=False, indent=1, default=str))
print(len(OUT["figures"]), "figures")
