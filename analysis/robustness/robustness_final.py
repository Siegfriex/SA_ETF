"""C2-D1 (Phase 00.5, D): DOCX claim 의 통계 증거 + 정의 민감도 → reports/ROBUSTNESS_MATRIX.csv, analysis/robustness/*.csv, figures/robustness/*.png.
결정적: block bootstrap seed=20261004, block=10, B=1000. raw = ETF_RAW_DIR (기본 <ROOT>/data/raw). FDR Close 로그수익률.
usage: python analysis/robustness/robustness_final.py
"""
import itertools, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.kfont import setup_korean_font
setup_korean_font()

RAW = Path(os.environ.get("ETF_RAW_DIR", ROOT / "data" / "raw"))
OUT = ROOT / "analysis" / "robustness"; FIG = ROOT / "figures" / "robustness"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
SEED, BLOCK, B = 20261004, 10, 1000
ANOM = {"261240": ["2019-03-14", "2019-03-15"]}          # C-0012 price_anomaly_unverified
K200 = ["069500", "102110", "122630", "114800"]
U = pd.read_csv(ROOT / "config" / "universe.csv", dtype=str, keep_default_na=False).set_index("ticker")
NAME = U["name"].to_dict(); SCOPE = U["asset_scope"].to_dict()


def panel():
    cols = {}
    for t in U.index:
        d = pd.read_csv(RAW / f"{t}.csv", parse_dates=["Date"]).set_index("Date")["Close"].astype(float)
        cols[t] = np.log(d).diff()
    return pd.DataFrame(cols).iloc[1:]


R_RAW = panel()
R = R_RAW.copy()
for t, days in ANOM.items():
    R.loc[pd.to_datetime(days), t] = np.nan
N = len(R)
rng = np.random.default_rng(SEED)


def block_idx(n):
    starts = rng.integers(0, n - BLOCK + 1, size=int(np.ceil(n / BLOCK)))
    return (starts[:, None] + np.arange(BLOCK)).ravel()[:n]


IDX = [block_idx(N) for _ in range(B)]


def boot(df, f):
    """df: 정렬된 DataFrame (행=날짜). f(DataFrame)->float. moving-block bootstrap 95% percentile CI."""
    a = df.to_numpy()
    est = f(df)
    vals = []
    for ix in IDX:
        ix = ix[ix < len(a)] if len(a) < N else ix
        vals.append(f(pd.DataFrame(a[ix % len(a)], columns=df.columns)))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return est, lo, hi


rows = []


def add(cid, claim, test, variant, stat, lo, hi, n, verdict, fig, note=""):
    rows.append(dict(claim_id=cid, claim=claim, test=test, variant=variant, statistic=round(float(stat), 4),
                     ci_low=None if lo is None else round(float(lo), 4), ci_high=None if hi is None else round(float(hi), 4),
                     n=n, verdict=verdict, figure=fig, note=note))


pear = lambda x, y: lambda d: d[x].corr(d[y])
spear = lambda x, y: lambda d: d[x].rank().corr(d[y].rank())
beta = lambda y, x: lambda d: np.cov(d[y], d[x])[0, 1] / np.var(d[x], ddof=1)
FOREST = []   # (label, est, lo, hi, ref)

# ── 1. 대표 claim ───────────────────────────────────────────
d = R[["069500", "102110"]].dropna()
e, lo, hi = boot(d, pear("069500", "102110"))
add("DOCX-01", "KODEX200–TIGER200 corr≈0.999", "Pearson ρ + block bootstrap", "full", e, lo, hi, len(d), "ROBUST" if lo > 0.99 else "FRAGILE", "forest_claims.png")
FOREST.append(("069500–102110 ρ", e, lo, hi, 0.999))
e, lo, hi = boot(d, lambda x: (x["069500"] - x["102110"]).std() * 1e4)
add("DOCX-02", "동일지수 ETF 일간차 sd ≈ 8bp", "sd(r1−r2) bp", "full", e, lo, hi, len(d), "ROBUST" if 5 <= e <= 12 else "FRAGILE", "", "noise floor of same exposure")
for y, ref, cid in [("122630", 1.99, "DOCX-03"), ("114800", -1.02, "DOCX-04")]:
    d = R[[y, "069500"]].dropna()
    e, lo, hi = boot(d, beta(y, "069500"))
    add(cid, f"{NAME[y]} beta≈{ref}", "OLS beta + block bootstrap", "full", e, lo, hi, len(d),
        "ROBUST" if lo <= ref <= hi or abs(e - ref) < 0.02 else "CONTRADICTED", "forest_claims.png")
    FOREST.append((f"{y} β vs 069500", e, lo, hi, ref))
    for per, (a, b) in {"2019-21": ("2019", "2021"), "2022-25": ("2022", "2025"), "2026": ("2026", "2026")}.items():
        dd = d.loc[a:b]
        add(cid, f"{NAME[y]} beta 기간 안정성", "OLS beta", per, beta(y, "069500")(dd), None, None, len(dd), "ROBUST", "")
for tgt, alt, cid, lab in [("143860", "229200", "DOCX-05", "헬스케어: ρ(KOSDAQ150) − ρ(KOSPI200)"),
                           ("161510", "091170", "DOCX-06", "고배당: ρ(은행) − ρ(KOSPI200)")]:
    d = R[[tgt, alt, "069500"]].dropna()
    for meth, fn in [("Spearman", spear), ("Pearson", pear)]:
        e, lo, hi = boot(d, lambda x, fn=fn: fn(tgt, alt)(x) - fn(tgt, "069500")(x))
        add(cid, lab + " > 0", f"Δρ ({meth}) block bootstrap", "full", e, lo, hi, len(d), "ROBUST" if lo > 0 else "FRAGILE", "forest_claims.png",
            f"ρ_alt={fn(tgt, alt)(d):.3f}, ρ_K200={fn(tgt, '069500')(d):.3f}")
        if meth == "Spearman":
            FOREST.append((f"{tgt} Δρ (Spearman)", e, lo, hi, 0))
        for per, (a, b) in {"2019-21": ("2019", "2021"), "2022-25": ("2022", "2025"), "2026": ("2026", "2026")}.items():
            dd = d.loc[a:b]
            add(cid, lab + " 기간별", f"Δρ ({meth})", per, fn(tgt, alt)(dd) - fn(tgt, "069500")(dd), None, None, len(dd),
                "ROBUST" if fn(tgt, alt)(dd) - fn(tgt, "069500")(dd) > 0 else "FRAGILE", "")

# 2026 vol 비율
vr = []
for t in U.index:
    x = R[t].dropna(); a, b = x.loc[:"2025"], x.loc["2026"]
    ratio = b.std() / a.std()
    bv = []
    for ix in IDX[:B]:
        ia, ib = ix[ix < len(a)], ix[ix < len(b)]   # 각 기간 내부 block 재표집
        ib = ib if len(ib) > 20 else block_idx(len(b))[:len(b)]
        bv.append(b.to_numpy()[ib % len(b)].std() / a.to_numpy()[ia % len(a)].std())
    lo, hi = np.percentile(bv, [2.5, 97.5])
    bf = stats.levene(a, b, center="median")
    vr.append(dict(ticker=t, name=NAME[t], scope=SCOPE[t], ratio=ratio, lo=lo, hi=hi, bf_stat=bf.statistic, bf_p=bf.pvalue, n26=len(b)))
vr = pd.DataFrame(vr); vr.to_csv(OUT / "vol_ratio_2026.csv", index=False)
for _, v in vr.iterrows():
    exp_up = v.scope == "domestic_equity"
    ok = (v.lo > 1) if exp_up else (v.lo <= 1.2)
    add("DOCX-07", "2026 vol 확대는 국내주식 중심", "σ2026/σ2019-25 + block CI; Brown–Forsythe p", v.ticker, v.ratio, v.lo, v.hi, int(v.n26),
        "ROBUST" if ok else "FRAGILE", "vol_ratio_2026.png", f"{v['name']} BF p={v.bf_p:.2g}" + ("; 반례: 비주식인데 2026 vol 유의 확대" if not ok else ""))
dom = vr[vr.scope == "domestic_equity"].ratio; fo = vr[vr.scope == "foreign_equity"].ratio
mw = stats.mannwhitneyu(dom, fo, alternative="greater")
add("DOCX-07", "2026 vol 확대: 국내주식 > 해외주식", "Mann–Whitney (ETF 단위 ratio)", f"dom n={len(dom)} vs for n={len(fo)}", dom.median() - fo.median(), None, None, len(dom) + len(fo),
    "ROBUST" if mw.pvalue < 0.05 else "FRAGILE", "vol_ratio_2026.png", f"p={mw.pvalue:.3g}; 단 국내 12종 중 KOSPI200 4종 중복")

# 261240 폭락일 상승률 — 정의별
k = R["069500"]
med = k.rolling(60, min_periods=20).median().shift(1)
mad = (k - med).abs().rolling(60, min_periods=20).median().shift(1) * 1.4826
crash_defs = {"069500 ≤ q05": k <= k.quantile(.05), "069500 ≤ −2%": np.expm1(k) <= -0.02,
              "069500 rz≤−3 (D식)": (k - med) / mad <= -3, "069500 ≤ −3%": np.expm1(k) <= -0.03}
for lab, m in crash_defs.items():
    x = R.loc[m.fillna(False), "261240"].dropna()
    n, s = len(x), int((x > 0).sum())
    ci = stats.binomtest(s, n).proportion_ci(method="wilson")
    add("DOCX-08", "달러선물은 주식 폭락일 86% 상승", "up-rate (Wilson CI)", lab, s / n, ci.low, ci.high, n,
        "DEFINITION_DEPENDENT", "", f"mean {x.mean()*100:+.2f}%")

_d8 = [r_ for r_ in rows if r_["claim_id"] == "DOCX-08"]
add("DOCX-08", "달러선물은 주식 폭락일 86% 상승", "DOCX 86% 재현 여부", "정의 4종 범위", min(r_["statistic"] for r_ in _d8), min(r_["ci_low"] for r_ in _d8),
    max(r_["ci_high"] for r_ in _d8), len(_d8), "DEFINITION_DEPENDENT", "",
    "86% 는 어느 정의로도 정확히 재현 안 됨: 0.72(≤−3%)~0.91(rz≤−3). '대부분(72–91%) 상승' 으로 정의 병기 권고; ci 칸=정의 간 min/max")

# kurtosis 처리별
kt = []
for t in ["261240", "153130"]:
    x = R_RAW[t].dropna()
    v = {"전체": x, "trim3 (상위|r| 3일 제거)": x.drop(x.abs().nlargest(3).index),
         ("anomaly 2일 제거" if t in ANOM else "anomaly 제거 (해당 없음)"): x.drop(pd.to_datetime(ANOM.get(t, [])), errors="ignore"),
         "winsor 1%": x.clip(x.quantile(.01), x.quantile(.99))}
    for lab, y in v.items():
        kk = stats.kurtosis(y, fisher=True, bias=False)
        kt.append(dict(ticker=t, variant=lab, ex_kurt=kk, ann_vol=y.std() * np.sqrt(252), n=len(y)))
        add("DOCX-09" if t == "261240" else "DOCX-10", f"{NAME[t]} kurtosis 300대는 극단 소수일 산물", "excess kurtosis (unbiased)", lab, kk, None, None, len(y),
            "ROBUST", "kurtosis_variants.png", f"ann_vol={y.std()*np.sqrt(252)*100:.2f}%; unbiased(G2) 추정 — universe_profile 과 소수점 차이 가능")
kt = pd.DataFrame(kt); kt.to_csv(OUT / "kurtosis_variants.csv", index=False)


# ── 2/3. shock 정의 grid + idio 민감도 ─────────────────────
def rz_D(r):
    m = r.rolling(60, min_periods=20).median().shift(1)
    s = (r - m).abs().rolling(60, min_periods=20).median().shift(1) * 1.4826
    return (r - m) / s.replace(0, np.nan)


def rz_A(r):
    m = r.rolling(60, min_periods=40).median().shift(1)
    s = r.rolling(60, min_periods=40).apply(lambda z: np.nanmedian(np.abs(z - np.nanmedian(z))), raw=True).shift(1)
    return (r - m) / (1.4826 * s).replace(0, np.nan)


RZD, RZA = rz_D(R), rz_A(R)
EXT = {"rz3 (D식 MAD: 잔차 rolling median, mp20)": RZD.abs() > 3, "rz3 (A식 MAD: 창내 MAD, mp40)": RZA.abs() > 3,
       "q99 (ETF별 전체기간 |r|)": R.abs() >= R.abs().quantile(.99), "abs3 (|단순수익률|≥3%)": np.expm1(R).abs() >= 0.03}


def cnt(e, dedup):
    if dedup:
        e = e.drop(columns=K200[1:]).assign(**{K200[0]: e[K200].any(axis=1)})
    return e.sum(axis=1)


grid = []
for (lab, e), kk, dd in itertools.product(EXT.items(), [3, 5, 8], [False, True]):
    grid.append(dict(definition=lab, k=kk, dedup_k200=dd, n_days=int((cnt(e, dd) >= kk).sum())))
grid = pd.DataFrame(grid); grid.to_csv(OUT / "shock_definition_grid.csv", index=False)
g5 = grid[(grid.k == 5) & (~grid.dedup_k200)].set_index("definition").n_days
for lab, ref in [("rz3 (D식 MAD: 잔차 rolling median, mp20)", 52), ("rz3 (A식 MAD: 창내 MAD, mp40)", 61),
                 ("q99 (ETF별 전체기간 |r|)", 22), ("abs3 (|단순수익률|≥3%)", 175)]:
    add("DOCX-11", "k≥5 공통충격 후보 52/22/175 (정의 민감)", "count days with ≥5 ETFs extreme", lab, g5[lab], None, None, N,
        "DEFINITION_DEPENDENT", "shock_definition_grid.png", f"DOCX/CL-32 기준값 {ref}; 재현={'YES' if g5[lab]==ref else 'NO (diff '+str(g5[lab]-ref)+')'}")
add("DOCX-11", "정의 간 개수 배율", "max/min k≥5", "4 정의", g5.max() / g5.min(), None, None, N, "DEFINITION_DEPENDENT", "shock_definition_grid.png")

# idio: ETF 단독 |rz|>3 (D식) & universe median |rz| < c ; 경제 최소 크기 |r|≥1% 필터 병기
ext = RZD.abs() > 3; nn = ext.sum(axis=1); medrz = RZD.abs().median(axis=1)
base = None; idio_rows = []
for c in [0.75, 1.0, 1.25]:
    solo = ext[(nn == 1) & (medrz < c)]
    cases = {(dte, row.idxmax()) for dte, row in solo.iterrows()}
    big = {(dte, t) for dte, t in cases if abs(np.expm1(R.loc[dte, t])) >= 0.01}
    idio_rows.append(dict(threshold=c, n_cases=len(cases), n_cases_ge1pct=len(big)))
    if c == 1.0:
        base, base_big = cases, big
    globals()[f"_idio_{c}"] = (cases, big)
for r_ in idio_rows:
    cs, bg = globals()[f"_idio_{r_['threshold']}"]
    r_["survive_from_1.0"] = len(base & cs) / len(base); r_["survive_from_1.0_ge1pct"] = len(base_big & bg) / max(len(base_big), 1)
    add("CL-idio", "ETF 고유(idiosyncratic) 충격 사례는 calm 기준에 강건", "universe median|rz|<c, solo |rz|>3", f"c={r_['threshold']}",
        r_["survive_from_1.0"], None, None, r_["n_cases"], "ROBUST" if r_["survive_from_1.0"] >= 0.8 else "FRAGILE", "",
        f"n={r_['n_cases']}, |r|≥1% n={r_['n_cases_ge1pct']}, ≥1% 생존 {r_['survive_from_1.0_ge1pct']:.2f}")
pd.DataFrame(idio_rows).to_csv(OUT / "idio_threshold_sensitivity.csv", index=False)
top = sorted(base_big, key=lambda z: -abs(RZD.loc[z]))
pd.DataFrame([dict(date=dte.date(), ticker=t, name=NAME[t], ret_pct=round(np.expm1(R.loc[dte, t]) * 100, 2), rz=round(RZD.loc[dte, t], 1),
                   in_c075=(dte, t) in globals()["_idio_0.75"][0])
              for dte, t in top]).to_csv(OUT / "idio_cases_c1_ge1pct.csv", index=False)

# ── 4. cluster / PCA ARI ────────────────────────────────────
X = R.dropna()


def clus(df, meth, link, kk):
    c = df.rank().corr() if meth == "spearman" else df.corr()
    dist = np.sqrt(np.clip(2 * (1 - c.values), 0, None)) if link == "ward" else np.clip(1 - c.values, 0, None)
    np.fill_diagonal(dist, 0)
    return fcluster(linkage(squareform(dist, checks=False), method=link), kk, criterion="maxclust"), c


ari = []
for kk in [3, 5]:
    ref, _ = clus(X, "pearson", "average", kk)
    for (per, (a, b)), meth, link in itertools.product([("full", (None, None)), ("2019-21", ("2019", "2021")), ("2022-25", ("2022", "2025")), ("2026", ("2026", "2026"))],
                                                       ["pearson", "spearman"], ["average", "complete", "ward"]):
        df = X.loc[a:b] if a else X
        lab, c = clus(df, meth, link, kk)
        ev = np.linalg.eigvalsh(c.values)[::-1]
        ari.append(dict(k=kk, period=per, method=meth, linkage=link, ari_vs_full_pearson_avg=adjusted_rand_score(ref, lab),
                        pc1_share=ev[0] / ev.sum(), sizes=str(sorted(np.bincount(lab)[1:].tolist(), reverse=True))))
ari = pd.DataFrame(ari); ari.to_csv(OUT / "cluster_ari.csv", index=False)
for kk in [3, 5]:
    s_ = ari[(ari.k == kk) & (ari.linkage != "ward")]
    add("DOCX-12", "cluster 구조 강건성 (ward 제외)", "ARI vs full/Pearson/average", f"k={kk} min over 16 variants (average/complete)",
        s_.ari_vs_full_pearson_avg.min(), s_.ari_vs_full_pearson_avg.quantile(.25), s_.ari_vs_full_pearson_avg.median(), len(s_),
        "ROBUST" if s_.ari_vs_full_pearson_avg.median() >= 0.8 else "FRAGILE", "cluster_ari_heatmap.png",
        "ci 칸=Q1/median; 불안정의 주원인은 ward(분산 기준 → 큰 주식 덩어리를 쪼갬)와 2026 단독 기간")
for kk in [3, 5]:
    s = ari[ari.k == kk]
    add("DOCX-12", "통계 cluster ≠ 경제 taxonomy; cluster 구조 자체의 강건성", "ARI vs full/Pearson/average", f"k={kk} min over 24 variants",
        s.ari_vs_full_pearson_avg.min(), s.ari_vs_full_pearson_avg.quantile(.25), s.ari_vs_full_pearson_avg.median(), len(s),
        "ROBUST" if s.ari_vs_full_pearson_avg.median() >= 0.8 else "FRAGILE", "cluster_ari_heatmap.png", "ci_low/ci_high 칸 = Q1/median (CI 아님)")
    add("CL-13", "PC1 비중은 기간/방법 의존", "PC1 share range", f"k-indep", s.pc1_share.max() - s.pc1_share.min(), s.pc1_share.min(), s.pc1_share.max(), len(s),
        "DEFINITION_DEPENDENT", "cluster_ari_heatmap.png", "ci_low/ci_high 칸 = min/max") if kk == 3 else None
X2 = X.drop(columns=["122630", "114800", "102110"])
ev = np.linalg.eigvalsh(X2.corr().values)[::-1]; ev0 = np.linalg.eigvalsh(X.corr().values)[::-1]
add("CL-13", "PC1 비중 universe 구성 종속", "PC1 share full vs dedup(−102110,122630,114800)", "Pearson full", ev0[0] / ev0.sum(), ev[0] / ev.sum(), None, len(X), "DEFINITION_DEPENDENT", "", "ci_low 칸 = dedup 값")

M = pd.DataFrame(rows); M.to_csv(ROOT / "reports" / "ROBUSTNESS_MATRIX.csv", index=False)

# ── figures ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1]))
groups = [[f for f in FOREST if "ρ" in f[0] and "Δ" not in f[0]], [f for f in FOREST if "β" in f[0]], [f for f in FOREST if "Δρ" in f[0]]]
titles = ["동일지수 상관 (ρ)", "geared beta (vs 069500)", "경험적 peer 우위 Δρ (Spearman)"]
for ax, g, ti in zip(axes, groups, titles):
    for i, (lab, e, lo, hi, ref) in enumerate(g):
        ax.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt="o", color="#1f5fa8", capsize=4)
        ax.plot(ref, i, marker="|", ms=18, color="#c0392b")
        ax.annotate(f"{e:.4f} [{lo:.4f}, {hi:.4f}]", (e, i), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    ax.set_yticks(range(len(g))); ax.set_yticklabels([x[0] for x in g]); ax.set_ylim(-0.7, len(g) - 0.3)
    ax.set_title(ti); ax.grid(axis="x", alpha=.3)
axes[2].axvline(0, color="grey", lw=.8)
fig.suptitle("주요 claim 통계량 + moving-block bootstrap 95% CI (block=10, B=1000) · 빨간 눈금 = DOCX 보고값/귀무값", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "forest_claims.png", dpi=130); plt.close(fig)

v = vr.sort_values("ratio")
fig, ax = plt.subplots(figsize=(9, 7))
col = v.scope.map({"domestic_equity": "#c0392b", "foreign_equity": "#1f5fa8", "bond": "#7f8c8d", "commodity": "#d68910", "fx": "#27ae60"})
ax.hlines(range(len(v)), v.lo, v.hi, color=col, lw=2); ax.scatter(v.ratio, range(len(v)), color=col, zorder=3)
ax.axvline(1, color="grey", ls="--"); ax.set_yticks(range(len(v))); ax.set_yticklabels(v.ticker + " " + v["name"], fontsize=8)
ax.set_xlabel("σ(2026) / σ(2019–2025)  (점=추정, 선=block bootstrap 95% CI)")
ax.set_title(f"2026 변동성 배율 — 빨강=국내주식, 파랑=해외주식 · Mann–Whitney 국내>해외 p={mw.pvalue:.2g}")
fig.tight_layout(); fig.savefig(FIG / "vol_ratio_2026.png", dpi=130); plt.close(fig)

piv = grid.assign(col=grid.k.astype(str) + np.where(grid.dedup_k200, " dedup", "")).pivot(index="definition", columns="col", values="n_days")
piv = piv[["3", "3 dedup", "5", "5 dedup", "8", "8 dedup"]]
fig, ax = plt.subplots(figsize=(10, 3.8))
im = ax.imshow(np.log10(piv.values), cmap="YlOrRd", aspect="auto")
for (i, j), val in np.ndenumerate(piv.values):
    ax.text(j, i, str(val), ha="center", va="center", fontsize=10)
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(["k≥" + c for c in piv.columns]); ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=9)
ax.set_title("공통충격 후보일 수: 정의 × 동반 ETF 수 k × KOSPI200 4종 dedup (색=log10)")
fig.tight_layout(); fig.savefig(FIG / "shock_definition_grid.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
for ax, kk in zip(axes, [3, 5]):
    s = ari[ari.k == kk].assign(row=lambda z: z.period + " · " + z.method).pivot(index="row", columns="linkage", values="ari_vs_full_pearson_avg")
    im = ax.imshow(s.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for (i, j), val in np.ndenumerate(s.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="w" if val < .6 else "k", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(s.columns); ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=8)
    ax.set_title(f"k={kk}: ARI vs (전체기간·Pearson·average)")
fig.colorbar(im, ax=axes, shrink=.8, label="Adjusted Rand Index")
fig.savefig(FIG / "cluster_ari_heatmap.png", dpi=130, bbox_inches="tight"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, t in zip(axes, ["261240", "153130"]):
    s = kt[kt.ticker == t]
    b = ax.bar(range(len(s)), s.ex_kurt, color=["#c0392b", "#1f5fa8", "#d68910", "#27ae60"])
    ax.set_yscale("log"); ax.set_ylim(0.5, s.ex_kurt.max() * 8); ax.set_xticks(range(len(s))); ax.set_xticklabels(s.variant, fontsize=8)
    for i, (kv, vv) in enumerate(zip(s.ex_kurt, s.ann_vol)):
        ax.text(i, kv, f"κ={kv:.1f}\nσ={vv*100:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{t} {NAME[t]}: excess kurtosis 처리별 (log 축)")
fig.tight_layout(); fig.savefig(FIG / "kurtosis_variants.png", dpi=130); plt.close(fig)

print(M.groupby("verdict").size().to_string()); print(grid[grid.k == 5].to_string(index=False)); print(pd.DataFrame(idio_rows).to_string(index=False))
print(kt.to_string(index=False)); print(vr[["ticker", "ratio", "lo", "hi", "bf_p"]].round(3).to_string(index=False))
