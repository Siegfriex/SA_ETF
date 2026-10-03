"""C-0017: common vs idiosyncratic shock 판정의 정의 민감도 (non-canonical, raw 직접).

정의 3종 × 동반 기준 k=3,5,8 → 일자별 동시 극단 ETF 수, 공통일 Jaccard, anchor 생존, asset_scope 참여율, 단독 극단일.
usage: python sandbox/d_shock.py   → sandbox/shock_sens.csv, shock_daily.csv, shock_idio.csv
"""
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

import d_sensitivity as ds

ROOT = Path("/home/sieg/projects-wsl/hongik_univ_26_2/SA/ETF_EDA_SCAFFOLD")
OUT = Path(__file__).parent
KS = [3, 5, 8]
ANCHORS = {"2020-03": ("2020-03-19", "2020-03-24"), "2024-08-05": ("2024-08-05", "2024-08-05"),
           "2026-03-04": ("2026-03-04", "2026-03-04"), "2026-07/08": ("2026-07-28", "2026-08-03")}
# KOSPI200 을 배율·브랜드만 바꿔 담은 4종 → exposure 기준으로는 1개
K200 = ["069500", "102110", "122630", "114800"]


def returns():
    r = ds.panel()
    for t, days in ds.TRIM.items():               # C-0012 closing_print_anomaly 만 제거
        r.loc[r.index.intersection(pd.to_datetime(days)), t] = np.nan
    return r


def extremes(r):
    med = r.rolling(60, min_periods=20).median().shift(1)
    mad = (r - med).abs().rolling(60, min_periods=20).median().shift(1) * 1.4826
    rz = (r - med) / mad.replace(0, np.nan)
    simple = np.expm1(r)
    return {"rz3_shift1": rz.abs() > 3,
            "q99_per_etf": r.abs() >= r.abs().quantile(.99),
            "abs_3pct": simple.abs() >= 0.03}, rz


def count(e, dedup=False):
    if dedup:
        e = e.drop(columns=K200[1:]).assign(**{K200[0]: e[K200].any(axis=1)})
    return e.sum(axis=1)


def main():
    r = returns()
    ext, rz = extremes(r)
    u = pd.read_csv(ROOT / "config" / "universe.csv", dtype=str).set_index("ticker")
    scope = u["asset_scope"]

    daily = pd.DataFrame({f"n_{d}": count(e) for d, e in ext.items()} |
                         {f"n_{d}_dedup": count(e, True) for d, e in ext.items()})
    daily.to_csv(OUT / "shock_daily.csv")

    rows = []
    sets = {}
    for d, k, dd in itertools.product(ext, KS, [False, True]):
        col = f"n_{d}" + ("_dedup" if dd else "")
        s = set(daily.index[daily[col] >= k])
        sets[(d, k, dd)] = s
        row = {"definition": d, "k": k, "dedup_k200": dd, "n_common_days": len(s)}
        for a, (lo, hi) in ANCHORS.items():
            row[f"anchor_{a}_max_n"] = int(daily.loc[lo:hi, col].max())
        rows.append(row)
    sens = pd.DataFrame(rows)

    jac = []
    for k, dd in itertools.product(KS, [False, True]):
        for a, b in itertools.combinations(ext, 2):
            x, y = sets[(a, k, dd)], sets[(b, k, dd)]
            jac.append({"k": k, "dedup_k200": dd, "def_a": a, "def_b": b,
                        "jaccard": round(len(x & y) / len(x | y), 3) if x | y else np.nan})
    jac = pd.DataFrame(jac)

    # anchor 생존: 3 정의 모두에서 k 이상이면 SURVIVES(k)
    surv = []
    for a, (lo, hi) in ANCHORS.items():
        row = {"anchor": a}
        for k, dd in itertools.product(KS, [False, True]):
            col = lambda d: f"n_{d}" + ("_dedup" if dd else "")
            ok = [daily.loc[lo:hi, col(d)].max() >= k for d in ext]
            row[f"k{k}{'_dedup' if dd else ''}"] = "SURVIVES" if all(ok) else f"FRAGILE({sum(ok)}/3)"
        surv.append(row)
    surv = pd.DataFrame(surv)

    # 공통일 asset_scope 참여율과 방향: rz 정의, k=5
    common = sorted(sets[("rz3_shift1", 5, False)])
    e = ext["rz3_shift1"].loc[common]
    sign = np.sign(r.loc[common]).mul(np.sign(r.loc[common, "069500"]), axis=0)   # 국내 대형주와 같은 방향=+1
    part = pd.DataFrame({"participation": e.T.groupby(scope).mean().mean(axis=1),
                         "same_dir_as_069500": sign.T.groupby(scope).mean().mean(axis=1)}).round(3)

    # idiosyncratic: rz 정의에서 그 ETF 만 극단(|rz|>3) + universe median |r| 가 평소 수준(하위 75%) + DQ 일 제외
    n = ext["rz3_shift1"].sum(axis=1)
    med_abs = r.abs().median(axis=1)
    calm = med_abs <= med_abs.quantile(.75)
    solo = ext["rz3_shift1"][(n == 1) & calm]
    idio = []
    for day, row in solo.iterrows():
        t = row.idxmax()
        idio.append({"date": day.date(), "ticker": t, "name": u.loc[t, "name"], "asset_scope": scope[t],
                     "ret_pct": round(float(np.expm1(r.loc[day, t]) * 100), 2), "rz": round(float(rz.loc[day, t]), 1),
                     "universe_median_abs_pct": round(float(med_abs[day] * 100), 2)})
    idio = pd.DataFrame(idio).sort_values("rz", key=abs, ascending=False)
    idio.to_csv(OUT / "shock_idio.csv", index=False)

    sens.to_csv(OUT / "shock_sens.csv", index=False)
    jac.to_csv(OUT / "shock_jaccard.csv", index=False)
    surv.to_csv(OUT / "shock_anchor_survival.csv", index=False)
    part.to_csv(OUT / "shock_participation.csv")
    pd.set_option("display.width", 250)
    print(sens.to_string(index=False)); print(jac.to_string(index=False)); print(surv.to_string(index=False))
    print(f"common days (rz3, k=5): {len(common)}"); print(part.to_string())
    print("idio per ticker:", idio.ticker.value_counts().to_dict()); print(idio.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
