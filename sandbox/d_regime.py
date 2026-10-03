"""C 지시: ① CL-26 block bootstrap(10일 블록) ② 2026 vs 2019-25 corr 차이(Fisher z) · vol 비율.

usage: python sandbox/d_regime.py → sandbox/regime_cl26_block.csv, regime_corr_fisher.csv, regime_vol.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import d_counter as dc
import d_sensitivity as ds

OUT = Path(__file__).parent
BLOCK, REPS, SEED = 10, 500, 0


def ratio_stat(r, ratio):
    q05, q95 = r.quantile(.05), r.quantile(.95)
    dn, up = ratio[r <= q05].dropna(), ratio[r >= q95].dropna()
    return dn.median() / up.median()


def cl26_block(r):
    v = dc.raw_volume()
    ratio = (v / v.rolling(20, min_periods=10).median().shift(1)).reindex(r.index)
    rng = np.random.default_rng(SEED)
    n = len(r)
    starts = [rng.integers(0, n - BLOCK, int(np.ceil(n / BLOCK))) for _ in range(REPS)]
    idx = [np.concatenate([np.arange(s, s + BLOCK) for s in st])[:n] for st in starts]   # 같은 블록을 20종에 공유
    rows = []
    for t in r.columns:
        x, y = r[t].values, ratio[t].values
        b = [ratio_stat(pd.Series(x[i]), pd.Series(y[i])) for i in idx]
        lo, hi = np.nanquantile(b, [.025, .975])
        rows.append({"ticker": t, "down_over_up": round(ratio_stat(r[t], ratio[t]), 3),
                     "block_ci_lo": round(lo, 3), "block_ci_hi": round(hi, 3)})
    out = pd.DataFrame(rows)
    iid = pd.read_csv(OUT / "counter_d_volume.csv", dtype={"ticker": str})[["ticker", "ci_lo", "ci_hi"]]
    out = out.merge(iid.rename(columns={"ci_lo": "iid_ci_lo", "ci_hi": "iid_ci_hi"}), on="ticker")
    side = lambda lo, hi: np.where(lo > 1, "down>up", np.where(hi < 1, "up>down", "includes_1"))
    out["iid_call"] = side(out.iid_ci_lo, out.iid_ci_hi)
    out["block_call"] = side(out.block_ci_lo, out.block_ci_hi)
    out["iid_width"] = (out.iid_ci_hi - out.iid_ci_lo).round(3)
    out["block_width"] = (out.block_ci_hi - out.block_ci_lo).round(3)
    return out


def fisher(r, a, b, label):
    ca, cb = a.corr(), b.corr()
    iu = np.triu_indices(len(ca), 1)
    za, zb = np.arctanh(ca.values[iu]), np.arctanh(cb.values[iu])
    se = np.sqrt(1 / (len(a) - 3) + 1 / (len(b) - 3))
    z = (zb - za) / se
    return pd.DataFrame({"comparison": label, "a": ca.index[iu[0]], "b": ca.columns[iu[1]],
                         "corr_2019_25": ca.values[iu], "corr_2026": cb.values[iu], "fisher_z": z})


def main():
    pd.set_option("display.width", 250)
    r = ds.panel()
    rt = ds.trim(r)

    c = cl26_block(r)
    c.to_csv(OUT / "regime_cl26_block.csv", index=False)
    print(c.to_string(index=False))
    print(pd.crosstab(c.iid_call, c.block_call))

    base = rt.loc["2019":"2025"]
    y26 = rt.loc["2026"]
    y26_lwo = ds.drop_window(y26)
    # 비교 기준: 2019-25 안의 각 연도를 나머지 연도와 비교하면 몇 쌍이 |z|>1.96 인가 (연도 간 자연 변동)
    f = pd.concat([fisher(rt, base, y26, "2026_vs_2019-25"), fisher(rt, base, y26_lwo, "2026lwo_vs_2019-25")]
                  + [fisher(rt, rt.loc["2019":"2025"].drop(rt.loc[str(y)].index), rt.loc[str(y)], f"{y}_vs_rest")
                     for y in range(2019, 2026)])
    f.round(4).to_csv(OUT / "regime_corr_fisher.csv", index=False)
    s = f.groupby("comparison").agg(n_pairs=("fisher_z", "size"), sig_pairs=("fisher_z", lambda z: int((z.abs() > 1.96).sum())),
                                    up=("fisher_z", lambda z: int((z > 1.96).sum())), down=("fisher_z", lambda z: int((z < -1.96).sum())),
                                    mean_corr_base=("corr_2019_25", "mean"), mean_corr_target=("corr_2026", "mean"))
    print(s.round(3).to_string())
    print(f[f.comparison == "2026_vs_2019-25"].sort_values("fisher_z", key=abs, ascending=False).head(10).round(3).to_string(index=False))

    vol = pd.DataFrame({"sd_bp_2019_25": base.std() * 1e4, "sd_bp_2026": y26.std() * 1e4, "sd_bp_2026_lwo": y26_lwo.std() * 1e4})
    v20 = rt.rolling(20).std()
    vol["vol20_med_2019_25_bp"] = v20.loc["2019":"2025"].median() * 1e4
    vol["vol20_med_2026_bp"] = v20.loc["2026"].median() * 1e4
    vol["ratio_sd"] = vol.sd_bp_2026 / vol.sd_bp_2019_25
    vol["ratio_sd_lwo"] = vol.sd_bp_2026_lwo / vol.sd_bp_2019_25
    vol["ratio_vol20_median"] = vol.vol20_med_2026_bp / vol.vol20_med_2019_25_bp
    # 비교 기준: 2019-25 각 연도 sd / 2019-25 sd 의 최대값
    yr = pd.DataFrame({y: rt.loc[str(y)].std() for y in range(2019, 2026)}).div(base.std(), axis=0)
    vol["max_ratio_any_year_2019_25"] = yr.max(axis=1)
    vol = vol.round(3).sort_values("ratio_vol20_median", ascending=False)
    vol.to_csv(OUT / "regime_vol.csv")
    print(vol.to_string())


if __name__ == "__main__":
    main()
