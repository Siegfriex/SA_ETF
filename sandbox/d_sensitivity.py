"""D leave-window-out + cluster/PCA 민감도 harness (non-canonical, B·A 코드/패널 import 안 함).

usage:
  python sandbox/d_sensitivity.py lwo      # 2026-07-20~08-10 제외 전후 kurt / corr / PCA 변화
  python sandbox/d_sensitivity.py grid     # scaler × 기간 × 창 제외 × lev/inv × trim → cluster ARI, PC1 비중
  python sandbox/d_sensitivity.py lag      # 해외 ETF 거래시간 비동기: lag-0 vs lag-1 상관
산출: sandbox/{lwo_*.csv, grid.csv, lag.csv}
"""
from pathlib import Path
import itertools
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

RAW = Path("/home/sieg/projects-wsl/hongik_univ_26_2/SA/ETF_EDA_SCAFFOLD/data/raw")
OUT = Path(__file__).parent
WINDOW = ("2026-07-20", "2026-08-10")
LEV_INV = ["122630", "114800"]
TRIM = {"261240": ["2019-03-14", "2019-03-15"]}          # C-0012 closing_print_anomaly
TRIM_TOP3 = ["153130"]                                     # 상위 |r| 3일 제외
FOREIGN = ["133690", "192090", "219480", "132030", "261220", "261240"]
PERIODS = {"2019-21": ("2019", "2021"), "2022-24": ("2022", "2024"), "2025-26": ("2025", "2026")}
K = 5


def panel():
    cols = {}
    for p in sorted(RAW.glob("*.csv")):
        if "listing" in p.name or p.name == "MANIFEST.csv":
            continue
        d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"].astype(float)
        cols[p.stem] = np.log(d).diff()
    return pd.DataFrame(cols).iloc[1:]


def drop_window(r):
    return r.drop(r.loc[WINDOW[0]:WINDOW[1]].index)


def trim(r):
    r = r.copy()
    for t, days in TRIM.items():
        r.loc[r.index.intersection(pd.to_datetime(days)), t] = np.nan
    for t in TRIM_TOP3:
        r.loc[r[t].abs().nlargest(3).index, t] = np.nan
    return r


def scale(r, how):
    if how == "none":
        return r
    if how == "standard":
        return (r - r.mean()) / r.std()
    if how == "robust":
        return (r - r.median()) / (r.quantile(.75) - r.quantile(.25))
    if how == "rank":
        return r.rank(pct=True)
    raise ValueError(how)


def clusters(r, k=K):
    c = r.corr(method="pearson", min_periods=60)
    dist = np.sqrt(np.clip(2 * (1 - c.values), 0, None))
    np.fill_diagonal(dist, 0)
    return pd.Series(fcluster(linkage(squareform(dist, checks=False), "average"), k, "maxclust"), index=c.index)


def pc1_share(r):
    x = r.dropna()
    x = x - x.mean()
    s = np.linalg.svd(x.values, compute_uv=False) ** 2
    return float(s[0] / s.sum())


def pc1_loadings(r):
    x = r.dropna()
    x = (x - x.mean())
    _, _, vt = np.linalg.svd(x.values, full_matrices=False)
    v = pd.Series(vt[0], index=x.columns)
    return v * np.sign(v.get("069500", 1))


def lwo():
    r = panel()
    w = drop_window(r)
    k = pd.DataFrame({"kurt_all": r.kurt(), "kurt_lwo": w.kurt(),
                      "sd_bp_all": r.std() * 1e4, "sd_bp_lwo": w.std() * 1e4})
    k["kurt_drop_pct"] = (1 - k.kurt_lwo / k.kurt_all) * 100
    k.round(2).to_csv(OUT / "lwo_kurt.csv")
    ca, cw = r.corr(), w.corr()
    iu = np.triu_indices(len(ca), 1)
    pairs = pd.DataFrame({"a": ca.index[iu[0]], "b": ca.columns[iu[1]],
                          "corr_all": ca.values[iu], "corr_lwo": cw.values[iu]})
    pairs["delta"] = pairs.corr_lwo - pairs.corr_all
    pairs.round(4).sort_values("delta", key=abs, ascending=False).to_csv(OUT / "lwo_corr.csv", index=False)
    la, lw = pc1_loadings(r), pc1_loadings(w)
    pcs = pd.DataFrame({"pc1_load_all": la, "pc1_load_lwo": lw})
    pcs.round(4).to_csv(OUT / "lwo_pca.csv")
    ari = adjusted_rand_score(clusters(r), clusters(w))
    # 비교 기준: 같은 길이(16일)의 무작위 연속 창을 빼면 얼마나 바뀌는가
    rng = np.random.default_rng(0)
    n = len(r.loc[WINDOW[0]:WINDOW[1]])
    null = []
    for s in rng.integers(0, len(r) - n, 200):
        rr = r.drop(r.index[s:s + n])
        null.append((rr.corr().values[iu] - ca.values[iu]).__abs__().mean())
    print(f"window rows={n}  PC1 share all={pc1_share(r):.3f} lwo={pc1_share(w):.3f}  cluster ARI(all vs lwo)={ari:.3f}")
    print(f"mean |Δcorr| window={pairs.delta.abs().mean():.4f}  random {n}-day windows: median={np.median(null):.4f} p95={np.quantile(null,.95):.4f}")
    print(k.round(2).sort_values("kurt_drop_pct", ascending=False).to_string())
    print(pairs.round(3).sort_values("delta", key=abs, ascending=False).head(10).to_string(index=False))
    print(pcs.round(3).to_string())


def grid():
    base_r = panel()
    ref = clusters(base_r)
    rows = []
    for sc, per, win, lev, tr in itertools.product(["none", "standard", "robust", "rank"],
                                                    ["all"] + list(PERIODS), [False, True], [True, False], [False, True]):
        r = base_r
        if per != "all":
            r = r.loc[PERIODS[per][0]:PERIODS[per][1]]
        if win:
            r = drop_window(r)
        if not lev:
            r = r.drop(columns=LEV_INV)
        if tr:
            r = trim(r)
        r = scale(r, sc)
        cl = clusters(r)
        rows.append({"scaler": sc, "period": per, "drop_window": win, "lev_inv": lev, "trim": tr,
                     "ari_vs_base": round(adjusted_rand_score(ref[cl.index], cl), 3),
                     "pc1_share": round(pc1_share(r), 3),
                     "clusters": ";".join("|".join(sorted(g.index)) for _, g in cl.groupby(cl))})
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "grid.csv", index=False)
    print(g.groupby("scaler")["ari_vs_base"].describe().round(3).to_string())
    print(g.groupby("period")["ari_vs_base"].describe().round(3).to_string())
    print(g.groupby(["lev_inv"])["pc1_share"].describe().round(3).to_string())
    print("base clusters:", ";".join("|".join(sorted(x.index)) for _, x in ref.groupby(ref)))


def lag():
    r = panel()
    rows = []
    for t in FOREIGN:
        rows.append({"ticker": t, "corr_lag0_069500": r[t].corr(r["069500"]),
                     "corr_t_vs_069500_t-1": r[t].corr(r["069500"].shift(1)),
                     "corr_t_vs_069500_t+1": r[t].corr(r["069500"].shift(-1)),
                     "autocorr_lag1": r[t].autocorr(1)})
    out = pd.DataFrame(rows).round(3)
    out.to_csv(OUT / "lag.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    {"lwo": lwo, "grid": grid, "lag": lag}[sys.argv[1] if len(sys.argv) > 1 else "lwo"]()
