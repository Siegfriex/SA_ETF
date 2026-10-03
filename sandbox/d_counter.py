"""DEC-8 부호 축 + A 카드 반례 a~d 공격 (non-canonical, raw 직접).

usage: python sandbox/d_counter.py → sandbox/counter_*.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import d_sensitivity as ds
import d_shock as sh

OUT = Path(__file__).parent
PER = {"2019-21": ("2019", "2021"), "2022-24": ("2022", "2024"), "2025-26": ("2025", "2026"), "excl_2026": ("2019", "2025")}
# DEC-8: 국내 대형주와 반대로 움직이는 것이 정상인 ETF
SIGN = {"114800": -1}


def raw_volume():
    v = {}
    for p in sorted(ds.RAW.glob("*.csv")):
        if "listing" in p.name or p.name == "MANIFEST.csv":
            continue
        v[p.stem] = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Volume"].astype(float)
    return pd.DataFrame(v)


def sign_axis(r, rz):
    ext = rz.abs() > 3
    mkt = np.sign(r["069500"])
    raw_same = (np.sign(rz).mul(mkt, axis=0) > 0) & ext                     # 부호 그대로, 시장과 같은 방향만
    adj = rz.copy()
    for t, s in SIGN.items():
        adj[t] = adj[t] * s
    adj_same = (np.sign(adj).mul(mkt, axis=0) > 0) & ext                    # −1x 반전 후 같은 방향
    rows = []
    for name, e in {"abs_z": ext, "same_sign_raw": raw_same, "same_sign_adj": adj_same}.items():
        n = e.sum(axis=1)
        for k in [3, 5, 8]:
            rows.append({"axis": name, "k": k, "n_common_days": int((n >= k).sum())})
    tab = pd.DataFrame(rows)
    # 114800 오분류: |z| 공통일(k=5) 중 114800 이 극단인 날, 같은 부호(raw) 기준이면 동반에서 빠지는 비율
    common = ext.sum(axis=1) >= 5
    inv_ext = ext["114800"] & common
    dropped = inv_ext & ~raw_same["114800"]
    # 같은 부호(raw) 기준에서 114800 이 '혼자 반대' 로 남는 날 = 고유 후보로 잘못 분류
    solo_wrong = dropped & (raw_same.drop(columns="114800").sum(axis=1) >= 4)
    mis = {"abs_z_common_days_k5": int(common.sum()), "114800_extreme_on_common": int(inv_ext.sum()),
           "114800_dropped_by_same_sign_raw": int(dropped.sum()),
           "dropped_pct": round(dropped.sum() / max(inv_ext.sum(), 1) * 100, 1),
           "misread_as_idiosyncratic": int(solo_wrong.sum())}
    return tab, mis


def a_305540(r):
    rows = []
    for per, (lo, hi) in ({"all": ("2019", "2026")} | PER).items():
        x = r.loc[lo:hi]
        rows.append({"period": per, "n": len(x), "corr_305540_091230_semis": x["305540"].corr(x["091230"]),
                     "corr_305540_117680_steel": x["305540"].corr(x["117680"]),
                     "spearman_semis": x["305540"].corr(x["091230"], method="spearman"),
                     "spearman_steel": x["305540"].corr(x["117680"], method="spearman")})
    t = pd.DataFrame(rows).round(3)
    t["steel_minus_semis"] = (t.corr_305540_117680_steel - t.corr_305540_091230_semis).round(3)
    return t


def b_261240(r):
    rt = sh.returns()                                                       # trim 적용
    med = r["069500"].rolling(60, min_periods=20).median().shift(1)
    mad = (r["069500"] - med).abs().rolling(60, min_periods=20).median().shift(1) * 1.4826
    crisis = ((r["069500"] - med) / mad) < -3
    rows = []
    for per, (lo, hi) in ({"all": ("2019", "2026")} | PER).items():
        c = crisis.loc[lo:hi]
        days = c[c].index
        for t in ["261240", "148070", "132030", "153130", "114800"]:
            for lab, src in [("raw", r), ("trim", rt)]:
                x = np.expm1(src.loc[days, t]).dropna()
                rows.append({"period": per, "ticker": t, "version": lab, "n_crisis": len(days),
                             "mean_pct": round(x.mean() * 100, 3), "median_pct": round(x.median() * 100, 3),
                             "hit_rate_pos": round((x > 0).mean(), 3)})
    return pd.DataFrame(rows)


def c_async(r):
    rows = []
    two = r.rolling(2).sum().iloc[1::2]                                     # 겹치지 않는 2일 수익률
    for t in ds.FOREIGN:
        rows.append({"ticker": t, "corr_1d": r[t].corr(r["069500"]), "corr_2d": two[t].corr(two["069500"]),
                     "corr_1d_lead": r[t].corr(r["069500"].shift(-1)), "corr_1d_lag": r[t].corr(r["069500"].shift(1))})
    t = pd.DataFrame(rows).round(3)
    t["d_2d_minus_1d"] = (t.corr_2d - t.corr_1d).round(3)
    # 비교 기준: 국내 ETF 는 비동기가 없으므로 2일 상관 증가가 0 근처여야 함
    for d in ["091230", "091170", "229200"]:
        t.loc[len(t)] = [d + "(domestic ref)", round(r[d].corr(r["069500"]), 3), round(two[d].corr(two["069500"]), 3),
                         np.nan, np.nan, round(two[d].corr(two["069500"]) - r[d].corr(r["069500"]), 3)]
    return t


def d_volume(r):
    v = raw_volume()
    ratio = (v / v.rolling(20, min_periods=10).median().shift(1)).reindex(r.index)
    rows = []
    for t in r.columns:
        q05, q95 = r[t].quantile(.05), r[t].quantile(.95)
        dn = ratio.loc[r[t] <= q05, t].dropna()
        up = ratio.loc[r[t] >= q95, t].dropna()
        rows.append({"ticker": t, "n_down": len(dn), "n_up": len(up), "vol_ratio_down_med": dn.median(),
                     "vol_ratio_up_med": up.median(), "down_over_up": dn.median() / up.median()})
    out = pd.DataFrame(rows).round(3)
    # bootstrap 신뢰구간 (단순 iid, 자기상관 미반영 → 하한 과대평가 가능)
    rng = np.random.default_rng(0)
    lo_hi = []
    for t in r.columns:
        q05, q95 = r[t].quantile(.05), r[t].quantile(.95)
        dn = ratio.loc[r[t] <= q05, t].dropna().values
        up = ratio.loc[r[t] >= q95, t].dropna().values
        b = [np.median(rng.choice(dn, len(dn))) / np.median(rng.choice(up, len(up))) for _ in range(500)]
        lo_hi.append(np.quantile(b, [.025, .975]))
    out["ci_lo"], out["ci_hi"] = np.round(np.array(lo_hi)[:, 0], 3), np.round(np.array(lo_hi)[:, 1], 3)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    r = ds.panel()
    _, rz = sh.extremes(sh.returns())
    tab, mis = sign_axis(sh.returns(), rz)
    tab.to_csv(OUT / "counter_sign_axis.csv", index=False)
    print(tab.pivot(index="axis", columns="k", values="n_common_days").to_string()); print(mis)
    for name, f in [("a_305540", a_305540), ("b_261240", b_261240), ("c_async", c_async), ("d_volume", d_volume)]:
        t = f(r)
        t.to_csv(OUT / f"counter_{name}.csv", index=False)
        print(f"--- {name}")
        if name == "b_261240":
            print(t.pivot_table(index=["ticker", "version"], columns="period", values="mean_pct").round(3).to_string())
            print(t.pivot_table(index=["ticker", "version"], columns="period", values="hit_rate_pos").round(2).to_string())
            print("n_crisis:", t.groupby("period").n_crisis.first().to_dict())
        else:
            print(t.to_string(index=False))
