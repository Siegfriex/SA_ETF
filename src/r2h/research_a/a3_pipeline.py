"""Phase A3 산출물 — 상관 사건 C × 가격 사건 P 정렬, 우연 수준 비교, 민감도, 사례 색인 (결정적; seed 고정).

usage: .venv/bin/python -m src.r2h.research_a.a3_pipeline     (A2 산출물 불필요 — 같은 함수로 다시 계산)
출력: reports/r2h/research_a/a3/*.csv · A3_RUN_MANIFEST.json · data/processed/research_a/<run_id>/*.parquet

탐색적 검증이다: Ground Truth 가 확정되지 않았으므로 '일치율'은 탐지 성능이 아니다. 근접은 인과가 아니다.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import correlation_events as CE
from . import event_matching as EM
from . import price_events as PE
from . import qa
from . import robustness as RB
from . import rolling_correlation as RC
from .contract import KST, git_branch, git_sha, input_data_version, load_contract, make_run_id, runtime_info, sha256_file, write_json
from .data import dq_flag_matrix, etf_manifest, read_manifest, returns

SEED, B = 20261011, 1000
LAGS = (0, 5, 10, 20)
WINDOWS = (30, 60, 120)
PRIMARY_RULE = RC.RegimeRule(0.1, 10)
SENS_RULES = [RC.RegimeRule(0.0, 5), RC.RegimeRule(0.2, 10), RC.RegimeRule(0.1, 20)]
P_TEST_DEFS = ["P1_vol_scaled", "P2_quantile_exp", "P2_quantile_full", "P3_rz_D", "P3_rz_A", "P3_rz_eda",
               "P4_cum_h5", "P4_cum_h10", "P4_cum_h20", "P5_legacy_abs3", "P5_legacy_q99", "P5_common_shock"]


def _pos(idx, dates):
    return pd.Series(np.arange(len(idx)), index=idx).reindex(pd.to_datetime(dates)).dropna().astype(int).to_numpy()


def c_subsets(ce: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """검정 대상 C 정의 묶음. key = 'C3_b0.100_k10' 형식."""
    out = {}
    for (d, ps), g in ce.groupby(["definition_id", "parameter_set"]):
        p = json.loads(ps)
        if d == "C3_sign_regime":
            key = f"C3_b{p['band']:.3f}_k{p['persist']}"
        elif d == "C4_multi_window":
            sp = p["source_params"]
            key = "C4_from_" + ("C3_b{:.3f}_k{}".format(sp["band"], sp["persist"]) if "band" in sp else "C2")
        else:
            key = d.split("_")[0] + "_" + "_".join(d.split("_")[1:])
        out[key] = g
    return out


def run_a3(root: str | Path | None = None, run_id: str | None = None, B: int = B) -> dict:
    c = load_contract(root)
    version = input_data_version(read_manifest(c))
    run_id = run_id or make_run_id(version, phase="A3")
    rep = c.report_dir / "a3"; rep.mkdir(parents=True, exist_ok=True)
    out = c.derived_dir(run_id); out.mkdir(parents=True, exist_ok=True)
    ref = c.reference
    r = returns(c, "dq_trim_canonical"); rr = returns(c, "raw")
    idx = r.index; n = len(idx)
    man = etf_manifest(c)
    ac = dict(zip(man.ticker, man.a_comparison_group))
    comp = [t for t in c.base_panel if t not in (ref, "102110")]

    # 품질 표시: 경로 영향 DQ flag + 261240 이상 2일 → 창 안에 있으면 dq_in_window
    flags = dq_flag_matrix(c, idx, list(r.columns))
    path_dq = sum((m for k, m in flags.items() if k != "ohlc_close_gt_high"), start=pd.DataFrame(False, index=idx, columns=r.columns))
    qmap = {t: path_dq[t] | path_dq[ref] for t in r.columns}

    # ---- 1. 사건 생성
    rules = [PRIMARY_RULE] + SENS_RULES
    ce = CE.build_correlation_events(r, ref, comp + ["102110"], ac, WINDOWS, ("pearson", "spearman"), 0.9, rules, 2.58, 10,
                                     version[:16], run_id, quality=qmap)
    ce_ctrl = CE.build_correlation_events(r, "102110", comp, ac, WINDOWS, ("pearson",), 0.9, [PRIMARY_RULE], 2.58, 10,
                                          version[:16], run_id, quality={t: path_dq[t] | path_dq["102110"] for t in comp})
    pe = PE.build_price_events(r, list(r.columns), ac, version[:16], run_id, quality={t: path_dq[t] for t in r.columns})
    pe_raw = PE.build_price_events(rr, ["261240", ref], ac, version[:16], run_id + "-raw")
    # parquet 은 열 타입이 하나여야 한다: window 는 C4 에서 '30+60' 같은 문자열이 있으므로 저장본만 문자열로
    ce.astype({"window": str}).to_parquet(out / "correlation_events.parquet", index=False)
    pe.astype({"window": str}).to_parquet(out / "price_events.parquet", index=False)

    # ---- 2. 정렬 + 귀무비교 grid (pearson primary, spearman 은 C3 primary 만)
    csub = c_subsets(ce[ce.method == "pearson"])
    keep_c = ["C1_zero_cross", "C2_large_change", f"C3_b0.100_k10", "C3_b0.000_k5", "C3_b0.200_k10", "C3_b0.100_k20",
              "C4_from_C3_b0.100_k10", "C4_from_C2"]
    tests, aligns = [], []
    for ckey in keep_c:
        if ckey not in csub:
            continue
        cg = csub[ckey]
        for t in comp:
            for w in (sorted(cg.window.unique(), key=str) if ckey.startswith("C4") else WINDOWS):
                C = cg[(cg.series_id == t) & (cg.window == w)]
                if not len(C):
                    continue
                for target in ("market", "own", "universe"):
                    for pdef in P_TEST_DEFS:
                        if (pdef == "P5_common_shock") != (target == "universe"):
                            continue
                        sid = {"market": ref, "own": t, "universe": "UNIVERSE"}[target]
                        for d in (+1, -1):
                            Pd = pe[(pe.series_id == sid) & (pe.definition_id == pdef) & (pe.direction == d)]
                            for ctime in ("candidate_at", "confirmed_at"):
                                cp = _pos(idx, C[ctime]); pp = _pos(idx, Pd["candidate_at"])
                                for L in LAGS:
                                    res = EM.circular_shift_test_fast(cp, pp, n, L, B=B, seed=SEED)
                                    row = {"c_def": ckey, "ticker": t, "window": w, "p_target": target, "p_def": pdef, "p_direction": d,
                                           "c_time": ctime, "max_lag": L, **res}
                                    if ctime == "candidate_at":
                                        m = EM.match_events(C, Pd, idx, L, c_time=ctime)
                                        row.update({f"m_{k}": v for k, v in EM.summarize_matches(m, len(C), len(Pd)).items()})
                                        if L == 10 and pdef in ("P1_vol_scaled", "P5_common_shock", "P4_cum_h10", "P3_rz_D"):
                                            aligns.append(m.assign(c_def=ckey, ticker=t, window=w, p_target=target, p_def=pdef, p_direction=d, max_lag=L))
                                    tests.append(row)
    T = pd.DataFrame(tests)
    T = refine_and_correct(T, ce, pe, idx, n)
    align = pd.concat(aligns, ignore_index=True)

    # ---- 3. 같은 날 일치의 기계적 성분: C1·C3 candidate 일에 '오늘 하루'를 빼면 교차가 사라지는가
    mech = []
    for t in comp:
        for w in WINDOWS:
            for ckey in ("C1_zero_cross", "C3_b0.100_k10"):
                C = csub[ckey]; C = C[(C.series_id == t) & (C.window == w)]
                if not len(C):
                    continue
                inf = RB.entering_day_influence(r[t], r[ref], w).reindex(C["candidate_at"])
                pm = pe[(pe.series_id == ref) & (pe.definition_id == "P1_vol_scaled")]["candidate_at"]
                po = pe[(pe.series_id == t) & (pe.definition_id == "P1_vol_scaled")]["candidate_at"]
                if ckey.startswith("C1"):
                    driven = np.sign(inf["rho_excl_today"].to_numpy()) == np.sign(inf["rho_prev"].to_numpy())
                else:   # C3: 새 run 첫날 — 오늘을 빼면 |rho| 가 band 안으로 돌아가는가 (=오늘이 band 밖으로 밀었음)
                    driven = ~(np.abs(inf["rho_excl_today"].to_numpy()) > PRIMARY_RULE.band) | \
                             (np.sign(inf["rho_excl_today"].to_numpy()) != C["direction"].to_numpy())
                mech.append(pd.DataFrame({"c_def": ckey, "ticker": t, "window": w, "candidate_at": C["candidate_at"].to_numpy(),
                                          "entering_day_driven": driven, "infl_today": inf["infl_today"].to_numpy(),
                                          "same_day_P1_market": C["candidate_at"].isin(set(pm)).to_numpy(),
                                          "same_day_P1_own": C["candidate_at"].isin(set(po)).to_numpy()}))
    mech = pd.concat(mech, ignore_index=True)
    mech_sum = (mech.groupby(["c_def", "window"]).agg(n=("ticker", "size"), share_entering_day_driven=("entering_day_driven", "mean"),
                                                      share_same_day_P1_any=("same_day_P1_market", lambda s: float((s | mech.loc[s.index, "same_day_P1_own"]).mean())))
                .reset_index())
    sd = mech[mech.same_day_P1_market | mech.same_day_P1_own]
    mech_sum = mech_sum.merge(sd.groupby(["c_def", "window"]).agg(n_same_day_P1=("ticker", "size"),
                                                                 share_driven_given_same_day=("entering_day_driven", "mean")).reset_index(),
                              on=["c_def", "window"], how="left")

    # ---- 3b. 기계적 성분 제거 재검정: 069500 P1 사건일의 수익률을 두 시리즈 모두에서 NaN 으로 두고 C3 를 다시 만든 뒤 검정
    mech_rm = mechanical_removed_retest(r, ref, comp, pe, idx, ac, version, run_id)

    # ---- 4. 사건 개수·정의 간 중첩 (가격 사건)
    pcount = pe.groupby(["series_id", "definition_id", "direction"]).size().rename("n").reset_index()
    jac = []
    for sid in [ref] + comp:
        sets = {d: set(pe[(pe.series_id == sid) & (pe.definition_id == d) & (pe.direction == +1)]["candidate_at"]) for d in P_TEST_DEFS[:-1]}
        for a in sets:
            for b in sets:
                if a < b:
                    u = sets[a] | sets[b]
                    jac.append({"series_id": sid, "def_a": a, "def_b": b, "direction": +1, "n_a": len(sets[a]), "n_b": len(sets[b]),
                                "jaccard": len(sets[a] & sets[b]) / len(u) if u else np.nan})
    pjac = pd.DataFrame(jac)
    # robust-z 정의 차이: 사건 수·날짜가 얼마나 바뀌는가
    rz = pjac[pjac.def_a.str.startswith("P3") & pjac.def_b.str.startswith("P3")]

    # ---- 5. 노출 중복: 102110 을 기준으로 바꿔도 상관 사건이 같은가 / 가격 사건 이중계산
    exp_rows = []
    for t in comp:
        for w in WINDOWS:
            a = ce[(ce.series_id == t) & (ce.reference_series_id == ref) & (ce.window == w) & (ce.method == "pearson") &
                   (ce.definition_id == "C3_sign_regime") & (ce.parameter_set == json.dumps({"band": 0.1, "persist": 10}))]
            b = ce_ctrl[(ce_ctrl.series_id == t) & (ce_ctrl.window == w) & (ce_ctrl.definition_id == "C3_sign_regime")]
            pa, pb = set(_pos(idx, a.candidate_at)), set(_pos(idx, b.candidate_at))
            near = sum(any(abs(x - y) <= 2 for y in pb) for x in pa)
            exp_rows.append({"ticker": t, "window": w, "n_C3_ref069500": len(pa), "n_C3_ref102110": len(pb),
                             "n_069500_with_102110_within2d": near})
    expo = pd.DataFrame(exp_rows)
    p69 = set(pe[(pe.series_id == ref) & (pe.definition_id == "P1_vol_scaled")]["candidate_at"])
    p21 = set(pe[(pe.series_id == "102110") & (pe.definition_id == "P1_vol_scaled")]["candidate_at"])
    expo_p = {"P1_069500": len(p69), "P1_102110": len(p21), "P1_both": len(p69 & p21), "P1_union": len(p69 | p21),
              "naive_sum_double_count": len(p69) + len(p21)}

    # ---- 6. 기간별 (dev / val 2025 / holdout 2026) — 사건률과 근접 비율
    per = RC.period_of(idx)
    prow = []
    for k, (a_, b_) in RC.PERIODS.items():
        msk = (idx >= a_) & (idx <= b_); ndays = int(msk.sum())
        for t in comp:
            for w in WINDOWS:
                C = csub["C3_b0.100_k10"]; C = C[(C.series_id == t) & (C.window == w) & (C.candidate_at >= a_) & (C.candidate_at <= b_)]
                Pm = pe[(pe.series_id == ref) & (pe.definition_id == "P1_vol_scaled") & (pe.candidate_at >= a_) & (pe.candidate_at <= b_)]
                cp, pp = _pos(idx, C.candidate_at), _pos(idx, Pm.candidate_at)
                prow.append({"period": k, "ticker": t, "window": w, "n_days": ndays, "n_C3": len(C), "C3_per_year": len(C) / ndays * 252,
                             "n_P1_market_bothdir": len(Pm), "n_C3_with_P1_within10": EM.proximity_count(cp, pp, 10)})
    period_tab = pd.DataFrame(prow)

    # ---- 7. DQ 민감도 (사건 수준): 261240 raw vs trim, C3 primary
    dq_rows = []
    for w in WINDOWS:
        for vname, rv in (("raw", rr), ("dq_trim_canonical", r)):
            s = RC.rolling_corr(rv["261240"], rv[ref], w, min_periods=int(np.ceil(0.9 * w)))["rho"]
            sr = RC.sign_regimes(s, PRIMARY_RULE); zc = RC.zero_crossings(s)
            dq_rows.append({"ticker": "261240", "window": w, "variant": vname, "n_zero_cross": len(zc),
                            "n_C3": int((~sr.is_initial).sum()) if len(sr) else 0,
                            "zero_cross_2019": int((zc.candidate_at.dt.year == 2019).sum()) if len(zc) else 0})
    pr = pe_raw[(pe_raw.series_id == "261240") & (pe_raw.candidate_at.between("2019-03-01", "2019-06-30"))]
    pt = pe[(pe.series_id == "261240") & (pe.candidate_at.between("2019-03-01", "2019-06-30"))]
    dq_ev = pd.DataFrame(dq_rows)
    dq_price = pd.DataFrame({"variant": ["raw", "dq_trim_canonical"], "n_price_events_261240_2019Q2": [len(pr), len(pt)],
                             "defs": [";".join(sorted(pr.definition_id.unique())), ";".join(sorted(pt.definition_id.unique()))]})

    # ---- 8. 사례 색인 — ETF 쌍별 독립 선정 (select_cases)
    case_idx = select_cases(ce, pe, align, idx, comp, ref)

    # ---- 9. 결과 저장
    tables = {"correlation_event_candidates.csv": ce, "price_event_candidates.csv": pe, "event_alignment.csv": align,
              "event_null_tests.csv": T, "mechanical_cooccurrence.csv": mech, "mechanical_summary.csv": mech_sum,
              "price_event_counts.csv": pcount, "price_def_jaccard.csv": pjac, "robust_z_definition_overlap.csv": rz,
              "exposure_dedup_check.csv": expo, "period_results.csv": period_tab, "dq_event_sensitivity.csv": dq_ev,
              "dq_price_event_sensitivity.csv": dq_price, "case_study_index.csv": case_idx,
              "mechanical_removed_retest.csv": mech_rm}
    for name, df in tables.items():
        qa.write_table(df, rep / name, run_id, version)
    prim = T[(T.c_def == "C3_b0.100_k10") & (T.c_time == "candidate_at")]
    metrics = {
        "n_correlation_events": int(len(ce)), "n_price_events": int(len(pe)), "n_tests": int(len(T)),
        "n_tests_testable": int(T.testable.sum()), "n_simulated_shifts_total": int(T.B_used.sum()),
        "share_tests_p_lt_0.05": float((T.p_value < 0.05).mean()),
        "share_testable_p_lt_0.05": float((T.loc[T.testable, "p_value"] < 0.05).mean()),
        "n_refined_B20000": int((T.B_used > B).sum()), "min_p_value": float(T.p_value.min()),
        "min_q_global": float(T.q_value_bh.min()), "n_tests_q_lt_0.10": int((T.q_value_bh < 0.10).sum()),
        "n_primary_family": int(T.primary_family.sum()), "min_q_primary": float(T.q_value_bh_primary.min()),
        "n_primary_q_lt_0.10": int((T.q_value_bh_primary < 0.10).sum()),
        "mech_removed_share_p_lt_0.05_by_lag": mech_rm.groupby("max_lag").apply(lambda g: float((g.p_value_masked < 0.05).mean())).to_dict(),
        "mech_kept_share_p_lt_0.05_by_lag": mech_rm.groupby("max_lag").apply(lambda g: float((g.p_value_orig < 0.05).mean())).to_dict(),
        "primary_C3_share_p_lt_0.05": float((prim.p_value < 0.05).mean()),
        "primary_C3_share_p_lt_0.05_lag0": float((prim[prim.max_lag == 0].p_value < 0.05).mean()),
        "primary_C3_share_p_lt_0.05_lag20": float((prim[prim.max_lag == 20].p_value < 0.05).mean()),
        "confirmed_at_share_p_lt_0.05": float((T[(T.c_def == "C3_b0.100_k10") & (T.c_time == "confirmed_at")].p_value < 0.05).mean()),
        "mech_C1_entering_day_driven": float(mech[mech.c_def == "C1_zero_cross"].entering_day_driven.mean()),
        "common_shock_days_reproduced": int((pe.definition_id == "P5_common_shock").sum()), "exposure_P1": expo_p,
    }
    manifest = {"phase": "A3", "run_id": run_id, "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
                "input_data_version": version, "git_sha": git_sha(c.root), "git_branch": git_branch(c.root), "runtime": runtime_info(),
                "params": {"seed": SEED, "B": B, "lags": LAGS, "windows": WINDOWS, "primary_rule": PRIMARY_RULE.id,
                           "sens_rules": [x.id for x in SENS_RULES], "z_thr_C2": 2.58, "C4_tol": 10, "p_defs": P_TEST_DEFS,
                           "null": "circular shift of P positions, u~U{2L+1..n-2L-1}, T=#C with P within ±L, one-sided",
                           "multiple_testing": "BH over all rows of event_null_tests.csv"},
                "derived_files": {str(p.relative_to(c.root)): {"sha256": sha256_file(p)} for p in
                                  [out / "correlation_events.parquet", out / "price_events.parquet"]},
                "report_tables": {nm: sha256_file(rep / nm) for nm in tables}, "metrics": metrics,
                "code_hash": code_hash()}
    write_json(manifest, rep / "A3_RUN_MANIFEST.json")
    return {"run_id": run_id, "tables": tables, "metrics": metrics, "manifest": manifest, "ce": ce, "pe": pe, "r": r, "rr": rr,
            "expo_p": expo_p}


# ================================================================ 후처리 함수 (사례·통계)
def code_hash() -> str:
    """A3 결과에 영향을 주는 모듈 소스의 해시 — checkpoint 재사용 시 코드 버전 일치 확인용."""
    import hashlib
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in ["rolling_correlation.py", "correlation_events.py", "price_events.py", "event_matching.py", "robustness.py",
              "data.py", "a3_pipeline.py"]:
        h.update((here / f).read_bytes())
    return h.hexdigest()


def refine_and_correct(T: pd.DataFrame, ce: pd.DataFrame, pe: pd.DataFrame, idx, n: int, B_refine: int = 20000,
                       refine_below: float = 0.01) -> pd.DataFrame:
    """(1) 검정 가능 여부: n_C≥1 이고 n_P≥1 이어야 귀무분포가 퇴화하지 않는다 → 그 밖은 BH 가족에서 제외(p=1 로 남겨 기록).
    (2) Monte Carlo 해상도: B=1000 의 최소 p = 1/1001. BH 가 수만 개 가설에서 이 해상도로는 원리적으로 통과할 수 없으므로
        p ≤ refine_below 인 검정은 같은 귀무·통계량으로 B_refine 회 다시 추정한다 (기준 변경이 아니라 추정 정밀도 향상).
    (3) BH 를 두 가족에 적용: global(검정 가능 전부), primary(사전 지정 규칙 C3_b0.100_k10 × candidate_at × lag 10)."""
    T = T.copy()
    T["testable"] = (T["n_C"] > 0) & (T["n_P"] > 0)
    T["B_used"] = T["B"]
    T["p_value_B1000"] = T["p_value"]
    low = T.index[T.testable & (T.p_value <= refine_below)]
    for i in low:
        row = T.loc[i]
        C = _c_events_for(ce, row); Pd = _p_events_for(pe, row)
        res = EM.circular_shift_test_fast(_pos(idx, C[row.c_time]), _pos(idx, Pd["candidate_at"]), n, int(row.max_lag),
                                          B=B_refine, seed=SEED + 1)
        T.loc[i, ["p_value", "T_null_mean", "T_null_p95", "ratio_obs_null"]] = [res["p_value"], res["T_null_mean"], res["T_null_p95"], res["ratio_obs_null"]]
        T.loc[i, "B_used"] = B_refine
    T["q_value_bh"] = np.nan
    T.loc[T.testable, "q_value_bh"] = EM.bh_qvalues(T.loc[T.testable, "p_value"].to_numpy())
    T["primary_family"] = T.testable & (T.c_def == "C3_b0.100_k10") & (T.c_time == "candidate_at") & (T.max_lag == 10)
    T["q_value_bh_primary"] = np.nan
    T.loc[T.primary_family, "q_value_bh_primary"] = EM.bh_qvalues(T.loc[T.primary_family, "p_value"].to_numpy())
    return T


def _ckey_filter(ce: pd.DataFrame, ckey: str) -> pd.DataFrame:
    return c_subsets(ce[ce.method == "pearson"])[ckey]


_CSUB_CACHE: dict = {}


def _c_events_for(ce: pd.DataFrame, row) -> pd.DataFrame:
    key = id(ce)
    if key not in _CSUB_CACHE:
        _CSUB_CACHE.clear(); _CSUB_CACHE[key] = c_subsets(ce[ce.method == "pearson"])
    g = _CSUB_CACHE[key][row.c_def]
    return g[(g.series_id == row.ticker) & (g.window.astype(str) == str(row.window))]


def _p_events_for(pe: pd.DataFrame, row) -> pd.DataFrame:
    sid = {"market": "069500", "own": row.ticker, "universe": "UNIVERSE"}[row.p_target]
    return pe[(pe.series_id == sid) & (pe.definition_id == row.p_def) & (pe.direction == row.p_direction)]


def mechanical_removed_retest(r: pd.DataFrame, ref: str, comp: list[str], pe: pd.DataFrame, idx, ac: dict, version: str,
                              run_id: str, B: int = 2000) -> pd.DataFrame:
    """leave-event-days-out: 069500 P1 사건일(양방향)의 수익률을 *두 시리즈 모두* NaN 으로 둔다 → 그 날의 큰 곱 r_a·r_ref 가
    창에 들어가지 않으므로, C3 가 그 날 생기는 기계적 경로가 끊긴다. 같은 P 집합과의 근접을 다시 검정한다.
    P 는 원래 날짜에 있으므로 lag 0 에서도 '그날 수익률 없이 상관이 이미 바뀌고 있었는가'를 본다."""
    pdays = pd.DatetimeIndex(pe[(pe.series_id == ref) & (pe.definition_id == "P1_vol_scaled")]["candidate_at"].unique())
    rm = r.copy()
    rm.loc[rm.index.isin(pdays), :] = np.nan
    ce0 = CE.build_correlation_events(r, ref, comp, ac, WINDOWS, ("pearson",), 0.9, [PRIMARY_RULE], 2.58, 10, version[:16], run_id)
    ce1 = CE.build_correlation_events(rm, ref, comp, ac, WINDOWS, ("pearson",), 0.9, [PRIMARY_RULE], 2.58, 10, version[:16], run_id + "-mask")
    rows = []
    n = len(idx)
    for t in comp:
        for w in WINDOWS:
            for d in (+1, -1):
                Pd = pe[(pe.series_id == ref) & (pe.definition_id == "P1_vol_scaled") & (pe.direction == d)]
                pp = _pos(idx, Pd.candidate_at)
                sel = lambda e: e[(e.series_id == t) & (e.window == w) & (e.definition_id == "C3_sign_regime")]
                c0, c1 = sel(ce0), sel(ce1)
                for L in LAGS:
                    a = EM.circular_shift_test_fast(_pos(idx, c0.candidate_at), pp, n, L, B=B, seed=SEED + 2)
                    b = EM.circular_shift_test_fast(_pos(idx, c1.candidate_at), pp, n, L, B=B, seed=SEED + 2)
                    rows.append({"ticker": t, "window": w, "p_direction": d, "max_lag": L, "n_C3_orig": len(c0), "n_C3_masked": len(c1),
                                 "n_P1_days_masked": len(pdays), "T_obs_orig": a["T_obs"], "T_obs_masked": b["T_obs"],
                                 "p_value_orig": a["p_value"], "p_value_masked": b["p_value"], "B": B})
    return pd.DataFrame(rows)


def select_cases(ce: pd.DataFrame, pe: pd.DataFrame, align: pd.DataFrame, idx, comp: list[str], ref: str,
                 tol: int = 20) -> pd.DataFrame:
    """ETF 쌍별로 사례 유형을 독립 판정한다. 같은 (ticker, 날짜) 는 한 유형에만 쓴다. 유형이 존재하지 않으면 found=False 행으로 기록.
    기준 사건: C = C3_b0.100_k10 (pearson), P = 069500·자산 자신의 P1_vol_scaled 및 공통충격 P5."""
    pos = pd.Series(np.arange(len(idx)), index=idx)
    C3 = ce[(ce.method == "pearson") & (ce.definition_id == "C3_sign_regime") & (ce.parameter_set == json.dumps({"band": 0.1, "persist": 10}))]
    Call = ce[(ce.method == "pearson") & (ce.definition_id.isin(["C1_zero_cross", "C2_large_change", "C3_sign_regime"]))]
    rows, used = [], set()

    def near(d, dates, k=tol):
        if not len(dates):
            return np.array([], int)
        dp = pos.reindex(pd.to_datetime(dates)).dropna().astype(int).to_numpy()
        return dp[np.abs(dp - pos[pd.Timestamp(d)]) <= k] - pos[pd.Timestamp(d)]

    def add(t, w, ctype, cd, pdate, lag, reason, found=True):
        key = (t, pd.Timestamp(cd if pd.notna(cd) else pdate).date() if (pd.notna(cd) or pd.notna(pdate)) else None)
        if found and key in used:
            return False
        if found:
            used.add(key)
        rows.append({"ticker": t, "window": w, "case_type": ctype, "c_date": cd, "p_date": pdate, "lag": lag, "found": found, "reason": reason})
        return True

    for t in comp:
        Pm = pe[(pe.series_id.isin([ref, t])) & (pe.definition_id == "P1_vol_scaled")]
        Pcs = pe[pe.definition_id == "P5_common_shock"]
        a = align[(align.ticker == t) & (align.c_def == "C3_b0.100_k10") & (align.p_def == "P1_vol_scaled") & (align.window == 60)]
        for mt, why in (("co_occur", "C3 candidate 와 P1 이 같은 거래일"), ("corr_leads", "C3 candidate 가 P1 보다 먼저"),
                        ("price_leads", "P1 이 C3 candidate 보다 먼저")):
            sub = a[a.match_type == mt].sort_values("lag", key=lambda s: s.abs())
            ok = False
            for _, rw in sub.iterrows():
                if add(t, 60, mt, rw.c_date, rw.p_date, rw.lag, f"{why} (w60, {rw.p_target} P1, lag {rw.lag:+.0f})"):
                    ok = True; break
            if not ok:
                add(t, 60, mt, pd.NaT, pd.NaT, np.nan, "해당 유형 사건 없음 (w60 C3 × P1, ±10)", found=False)
        # corr_only: C3 (어느 창이든) 인데 ±tol 안에 069500·자산의 P1/P3_rz_D/P4_cum_h10 및 공통충격 없음
        Pany = pe[(pe.series_id.isin([ref, t, "UNIVERSE"])) & pe.definition_id.isin(["P1_vol_scaled", "P3_rz_D", "P4_cum_h10", "P5_common_shock"])]
        ok = False
        for _, e in C3[C3.series_id == t].sort_values("score", ascending=False).iterrows():
            if not len(near(e.candidate_at, Pany.candidate_at)):
                if add(t, e.window, "corr_only", e.candidate_at, pd.NaT, np.nan,
                       f"w{e.window} C3 (직전 강도 |ρ|={e.score:.2f}) — ±{tol}일 안 P1·P3_rz_D·P4_h10·공통충격 없음"):
                    ok = True; break
        if not ok:
            add(t, None, "corr_only", pd.NaT, pd.NaT, np.nan, "모든 C3 주변에 가격 사건 존재 또는 C3 없음", found=False)
        # price_only: 공통충격(우선) 또는 069500 P1 인데 이 쌍의 C1·C2·C3(모든 창) 이 ±tol 안에 없음
        ok = False
        for _, e in pd.concat([Pcs.sort_values("score", ascending=False), Pm[Pm.series_id == ref]]).iterrows():
            if not len(near(e.candidate_at, Call[Call.series_id == t].candidate_at)):
                if add(t, None, "price_only", pd.NaT, e.candidate_at, np.nan,
                       f"{e.definition_id} {e.series_id} ({'+' if e.direction > 0 else '−'}) — 이 쌍의 C1·C2·C3 가 ±{tol}일 안 없음"):
                    ok = True; break
        if not ok:
            add(t, None, "price_only", pd.NaT, pd.NaT, np.nan, f"모든 공통충격·P1 주변 ±{tol}일에 이 쌍의 상관 사건 존재", found=False)
        # window_dependent: w30 C3 는 있는데 ±tol 안 w120 C3 없음 (또는 반대)
        ok = False
        c30, c120 = C3[(C3.series_id == t) & (C3.window == 30)], C3[(C3.series_id == t) & (C3.window == 120)]
        for _, e in c30.sort_values("score", ascending=False).iterrows():
            if not len(near(e.candidate_at, c120.candidate_at)):
                if add(t, 30, "window_dependent", e.candidate_at, pd.NaT, np.nan, f"w30 C3 있으나 ±{tol}일 안 w120 C3 없음"):
                    ok = True; break
        if not ok:
            add(t, None, "window_dependent", pd.NaT, pd.NaT, np.nan, "w30·w120 C3 가 서로 겹치거나 C3 자체가 없음", found=False)
    # DQ 영향 사례 (쌍 고유)
    for t, d, why in (("261240", "2019-03-14", "price_anomaly_unverified 2일 — 30일 ρ 최대 0.59 차이 (raw vs trim)"),
                      ("153130", "2024-01-30", "spike_reversal_candidate 2일 — 단기채 저변동 분모"),
                      ("219480", "2025-04-08", "spike_reversal_candidate 2일 — 2025-04 급락·반등 창과 겹침")):
        add(t, 30, "dq_affected", pd.Timestamp(d), pd.NaT, np.nan, why)
    out = pd.DataFrame(rows)
    out.insert(0, "case_id", [f"CASE{i + 1:02d}" if f else f"NONE{i + 1:02d}" for i, f in enumerate(out.found)])
    return out


def load_a3_checkpoint(root: str | Path | None = None, strict: bool = True) -> dict:
    """A3 결과 재사용: 입력 데이터 버전과 코드 해시가 현재와 같을 때만 checkpoint 를 쓴다 (다르면 RuntimeError → run_a3 재실행).
    반환 dict: manifest, 그리고 표 이름(확장자 제외) → DataFrame. ticker·series 열은 문자열(선행 0 보존)."""
    c = load_contract(root)
    rep = c.report_dir / "a3"
    man = json.loads((rep / "A3_RUN_MANIFEST.json").read_text())
    cur_v = input_data_version(read_manifest(c)); cur_h = code_hash()
    status = {"input_version_match": man["input_data_version"] == cur_v, "code_hash_match": man.get("code_hash") == cur_h,
              "table_hash_match": all(sha256_file(rep / n) == h for n, h in man["report_tables"].items() if (rep / n).exists()),
              "all_tables_present": all((rep / n).exists() for n in man["report_tables"])}
    if strict and not all(status.values()):
        raise RuntimeError(f"A3 checkpoint mismatch: {status} — run_a3() 로 다시 계산하라")
    strcols = {"ticker": str, "series_id": str, "reference_series_id": str, "window": str}
    out = {"manifest": man, "status": status}
    for n in man["report_tables"]:
        df = pd.read_csv(rep / n, dtype=strcols, low_memory=False)
        for col in ("candidate_at", "confirmed_at", "available_at", "event_end", "c_date", "p_date"):
            if col in df:
                df[col] = pd.to_datetime(df[col])
        out[n.removesuffix(".csv")] = df
    return out


if __name__ == "__main__":
    res = run_a3()
    print(res["run_id"]); print(json.dumps(res["metrics"], indent=1, default=str))
