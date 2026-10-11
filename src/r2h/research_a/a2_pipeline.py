"""Phase A2 산출물 일괄 생성 — KOSPI200 기준 pair rolling 상관, 교차·부호전환 후보, 민감도 표 (결정적).

usage: .venv/bin/python -m src.r2h.research_a.a2_pipeline      (작업 트리 루트)
출력: reports/r2h/research_a/a2/*.csv · A2_RUN_MANIFEST.json · data/processed/research_a/<run_id>/rolling_correlations.parquet
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import qa
from . import robustness as RB
from . import rolling_correlation as RC
from .contract import KST, git_branch, git_sha, input_data_version, load_contract, make_run_id, runtime_info, sha256_file, write_json
from .data import etf_manifest, read_manifest, read_universe, returns

WINDOWS = (30, 60, 120)
WINDOWS_ALL = (30, 60, 120, 250)
METHODS = ("pearson", "spearman")
MP_FRAC = 0.9
RULES = [RC.RegimeRule(b, k) for b in (0.0, 0.1, 0.2) for k in (5, 10, 20)]
BONDS = ["148070", "153130"]
OTHERS = ["229200", "219480", "133690", "132030", "261220", "261240"]


def rules_with_null(w: int) -> list[RC.RegimeRule]:
    return RULES + [RC.RegimeRule(round(RC.null_band(w), 3), 10)]


def pair_manifest(c) -> pd.DataFrame:
    m = etf_manifest(c).set_index("ticker")
    rows = []
    for ref in (c.reference, "102110"):
        for t in c.base_panel:
            if t == ref or (ref == "102110" and t == c.reference):
                continue
            rows.append({"pair_id": f"{t}~{ref}", "series_id": t, "reference_series_id": ref, "name": m.at[t, "name"],
                         "asset_scope": m.at[t, "asset_scope"], "comparison_group": m.at[t, "a_comparison_group"],
                         "session_sync": m.at[t, "a_session_sync"], "currency_hedge": m.at[t, "currency_hedge"],
                         "reference_role": "primary" if ref == c.reference else "control_not_independent",
                         "notebook": "A01" if t in BONDS + ["102110"] else "A012"})
    return pd.DataFrame(rows)


def run_a2(root: str | Path | None = None, run_id: str | None = None) -> dict:
    c = load_contract(root)
    version = input_data_version(read_manifest(c))
    run_id = run_id or make_run_id(version, phase="A2")
    rep = c.report_dir / "a2"; rep.mkdir(parents=True, exist_ok=True)
    out = c.derived_dir(run_id); out.mkdir(parents=True, exist_ok=True)
    ref = c.reference
    r = returns(c, "dq_trim_canonical")
    rr = returns(c, "raw")
    rlr = returns(c, "dq_trim_canonical", formula="logratio")
    comp = [t for t in c.base_panel if t != ref]                 # 102110 포함 (대조)

    # ---- 1. rolling 상관 long 테이블 (ref × variant × ticker × window × method)
    longs = []
    for refx in (ref, "102110"):
        for vname, rv in (("dq_trim_canonical", r), ("raw", rr)):
            tick = [t for t in c.base_panel if t not in (refx,) and not (refx == "102110" and t == ref)]
            lp = RC.rolling_panel(rv, refx, tick, WINDOWS_ALL, METHODS, MP_FRAC)
            longs.append(lp.assign(reference_series_id=refx, variant=vname))
    roll = pd.concat(longs, ignore_index=True)
    roll["run_id"] = run_id
    p_roll = out / "rolling_correlations.parquet"
    roll.to_parquet(p_roll, index=False)

    def rho(t, w, m="pearson", refx=ref, variant="dq_trim_canonical"):
        s = roll[(roll.ticker == t) & (roll.window == w) & (roll.method == m) & (roll.reference_series_id == refx) & (roll.variant == variant)]
        return s.set_index("Date")["rho"]

    # ---- 2. 0선 교차 (C1 후보) + 진입일 영향
    cr = []
    for t in comp:
        for w in WINDOWS:
            for m in METHODS:
                s = rho(t, w, m)
                zc = RC.zero_crossings(s)
                if not len(zc):
                    continue
                drv = RB.classify_crossing_drivers(r[t], r[ref], w, zc["candidate_at"]) if m == "pearson" else None
                zc = zc.assign(ticker=t, window=w, method=m, abs_rho_at=zc["rho_at"].abs(),
                               within_null_band_next5=[bool((s.loc[d:].iloc[:5].abs() <= RC.null_band(w)).all()) for d in zc["candidate_at"]])
                if drv is not None:
                    zc["entering_day_driven"] = drv["entering_day_driven"].to_numpy()
                    zc["infl_today"] = drv["infl_today"].to_numpy()
                cr.append(zc)
    crossings = pd.concat(cr, ignore_index=True)
    crossings["period"] = RC.period_of(pd.DatetimeIndex(crossings["candidate_at"])).to_numpy()

    # ---- 3. 부호전환 regime 후보 (C3 규칙 grid)
    rg = []
    for t in comp:
        for w in WINDOWS:
            for m in METHODS:
                s = rho(t, w, m)
                for ru in rules_with_null(w):
                    sr = RC.sign_regimes(s, ru)
                    if len(sr):
                        rg.append(sr.assign(ticker=t, window=w, method=m, rule=ru.id, band=ru.band, persist=ru.persist))
    regimes = pd.concat(rg, ignore_index=True)
    regimes["period"] = RC.period_of(pd.DatetimeIndex(regimes["candidate_at"])).to_numpy()

    # 규칙별 개수 요약 (initial 제외)
    reg_n = (regimes[~regimes.is_initial].groupby(["ticker", "window", "method", "rule"]).size().rename("n_sign_changes").reset_index())
    cross_n = crossings.groupby(["ticker", "window", "method"]).size().rename("n_zero_cross").reset_index()

    # ---- 4. 창·방법 요약, 기간별 정적 상관, 대조(102110) 비교
    win = RB.window_summary(r, ref, comp, WINDOWS_ALL, METHODS, MP_FRAC)
    stat = RB.static_corr_by_period(r, ref, comp)
    ctrl = []
    for t in [x for x in comp if x != "102110"]:
        for w in WINDOWS:
            a, b = rho(t, w), rho(t, w, refx="102110")
            ok = a.notna() & b.notna()
            ctrl.append({"ticker": t, "window": w, "max_abs_diff_ref069500_vs_102110": float((a - b)[ok].abs().max()),
                         "mean_abs_diff": float((a - b)[ok].abs().mean()),
                         "n_cross_069500": len(RC.zero_crossings(a)), "n_cross_102110": len(RC.zero_crossings(b)),
                         "n_sign_disagree_days": int((np.sign(a) != np.sign(b))[ok].sum())})
    control = pd.DataFrame(ctrl)

    # ---- 5. 데이터 품질 민감도
    dq = []
    dq.append(RB.rolling_variant_diff(rr, r, ref, "261240").assign(variant_a="raw", variant_b="dq_trim_canonical", target="261240 price_anomaly_unverified 2019-03-14·15"))
    sp_cells = {"153130": ["2024-01-30", "2024-01-31"], "219480": ["2025-04-08", "2025-04-09"]}
    rsp = RB.mask_days(r, sp_cells)
    for t in sp_cells:
        dq.append(RB.rolling_variant_diff(r, rsp, ref, t).assign(variant_a="dq_trim_canonical", variant_b="mask_spike_candidates", target=f"{t} spike_reversal_candidate"))
    lm = qa.large_moves(c, c.base_panel)
    lm_cells = {}
    for t, d in zip(lm.ticker, lm.date):
        if t in ("069500", "102110", "261220"):
            lm_cells.setdefault(t, []).append(d)
    rlm = RB.mask_days(r, {k: v for k, v in lm_cells.items() if k == "261220"})
    dq.append(RB.rolling_variant_diff(r, rlm, ref, "261220").assign(variant_a="dq_trim_canonical", variant_b="mask_261220_gt15pct", target="261220 |r|>15% 8일"))
    rk = RB.mask_days(r, {ref: lm_cells.get(ref, [])})
    for t in comp:
        if t == "102110":
            continue
        dq.append(RB.rolling_variant_diff(r, rk, ref, t).assign(variant_a="dq_trim_canonical", variant_b="mask_069500_2026-07-31",
                                                               target="069500 2026-07-31 +24% (KOSPI 쪽 대형 변동)"))
    dqs = pd.concat(dq, ignore_index=True)
    zero = RB.zero_return_diagnostics(r, ref, "153130")
    rankp = RB.rank_policy_sensitivity(r, rlr, ref, comp)
    mpp = RB.min_periods_sensitivity(r, ref, comp)

    # ---- 6. 비동기 거래: lead-lag · 2일 합산 · Dimson, 기간별
    ll = RC.lead_lag_corr(r, ref, comp).assign(period="full")
    per = RC.period_of(r.index)
    for k in RC.PERIODS:
        ll = pd.concat([ll, RC.lead_lag_corr(r, ref, comp, period_mask=per == k).assign(period=k)], ignore_index=True)
    agg = RC.aggregated_corr(r, ref, comp, 2).merge(RC.aggregated_corr(r, ref, comp, 5)[["ticker", "rho_5d"]], on="ticker")
    agg["dimson_L1"] = [RC.dimson_corr(r, ref, t, 1) for t in comp]
    agg["session_sync"] = agg["ticker"].map(c.session_sync_of)

    # ---- 7. 기존 결과 정합 (d0122 평균 pairwise 60/120/250 재현)
    U = read_universe(c); SC = dict(zip(U.ticker, U.asset_scope)); TK = U.ticker.tolist()
    DED = [t for t in TK if t not in ("102110", "122630", "114800")]; EQ = [t for t in DED if SC[t] == "domestic_equity"]
    ex = pd.read_csv(c.root / "analysis/robustness/d0122/rolling_corr_window_estimator_series.csv", index_col=0, parse_dates=True)
    legacy = pd.concat([RB.d0122_parity(rlr, ex, EQ, "domeq9"), RB.d0122_parity(rlr, ex, DED, "dedup17")], ignore_index=True)

    tables = {"etf_pair_manifest.csv": pair_manifest(c), "correlation_crossings.csv": crossings,
              "correlation_regime_candidates.csv": regimes, "regime_counts_by_rule.csv": reg_n, "crossing_counts.csv": cross_n,
              "window_sensitivity.csv": win, "static_corr_by_period.csv": stat, "control_102110_comparison.csv": control,
              "data_quality_sensitivity.csv": dqs, "zero_return_diagnostics_153130.csv": zero, "rank_policy_sensitivity.csv": rankp,
              "min_periods_sensitivity.csv": mpp, "lead_lag_corr.csv": ll, "async_aggregated_corr.csv": agg, "legacy_parity_d0122.csv": legacy}
    for name, df in tables.items():
        qa.write_table(df, rep / name, run_id, version)
    checks = {"legacy_d0122_max_abs_diff": float(legacy["max_abs_diff"].max()),
              "legacy_d0122_reproduced_1e-5_outside_nan_windows": bool((legacy["max_abs_diff_outside_nan_windows"] < 1e-5).all()),
              "legacy_d0122_pearson_reproduced_all": bool((legacy.loc[legacy.series.str.contains("pearson"), "max_abs_diff"] < 1e-5).all()),
              "dq_261240_full_rho_raw": float(rr["261240"].corr(rr[ref])), "dq_261240_full_rho_trim": float(r["261240"].corr(r[ref])),
              "rank_round_rolling_max_effect": float(rankp["max_abs_round_vs_unround"].max()),
              "n_rolling_rows": int(len(roll))}
    manifest = {"phase": "A2", "run_id": run_id, "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
                "input_data_version": version, "git_sha": git_sha(c.root), "git_branch": git_branch(c.root), "runtime": runtime_info(),
                "params": {"windows": WINDOWS, "windows_sensitivity": WINDOWS_ALL, "methods": METHODS, "min_periods_frac": MP_FRAC,
                           "regime_rules": [ru.id for ru in RULES] + ["null_band(w)_k10"], "periods": RC.PERIODS,
                           "returns": "dq_trim_canonical logdiff (primary), raw (DQ 민감도)", "rank_policy": "rank_safe round 12"},
                "derived_files": {str(p_roll.relative_to(c.root)): {"sha256": sha256_file(p_roll), "rows": len(roll)}},
                "report_tables": {n: sha256_file(rep / n) for n in tables}, "checks": checks}
    write_json(manifest, rep / "A2_RUN_MANIFEST.json")
    return {"run_id": run_id, "version": version, "tables": tables, "roll": roll, "checks": checks, "manifest": manifest,
            "r": r, "rr": rr}


if __name__ == "__main__":
    res = run_a2()
    print(res["run_id"]); print(json.dumps(res["checks"], indent=1, default=str))
