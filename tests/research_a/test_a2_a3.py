"""Track A Phase A2·A3 — 단위(synthetic) + 통합(checkpoint) 테스트.

usage: .venv/bin/python -m pytest tests/research_a -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.r2h.research_a import contract as K, data as D  # noqa: E402
from src.r2h.research_a import rolling_correlation as RC, correlation_events as CE, price_events as PE  # noqa: E402
from src.r2h.research_a import event_matching as EM, robustness as RB  # noqa: E402

C = K.load_contract(ROOT)
HAS_RAW = (C.raw_dir / "069500.csv").exists()
real = pytest.mark.skipif(not HAS_RAW, reason="raw not available")
A3_MAN = C.report_dir / "a3" / "A3_RUN_MANIFEST.json"
has_a3 = pytest.mark.skipif(not A3_MAN.exists(), reason="A3 checkpoint not available")


def _pair(n=400, rho=0.5, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(z[:, 0], idx), pd.Series(z[:, 1], idx)


# ---------------------------------------------------------------- rolling correlation
def test_rolling_pearson_matches_pandas():
    x, y = _pair()
    a = RC.rolling_corr(x, y, 60)["rho"]
    assert np.nanmax(np.abs(a - x.rolling(60).corr(y))) < 1e-12


def test_rolling_spearman_ranks_within_window():
    x, y = _pair(seed=1)
    a = RC.rolling_corr(x, y, 30, "spearman")["rho"]
    i = 200
    assert np.isclose(a.iloc[i], stats.spearmanr(x.iloc[i - 29:i + 1], y.iloc[i - 29:i + 1])[0])


def test_rolling_no_lookahead():
    x, y = _pair(seed=2)
    a = RC.rolling_corr(x, y, 60)["rho"]
    x2 = x.copy(); x2.iloc[300:] = np.random.default_rng(9).normal(size=100)   # 미래를 바꿔도
    b = RC.rolling_corr(x2, y, 60)["rho"]
    assert np.allclose(a.iloc[:300], b.iloc[:300], equal_nan=True)               # 과거 rho 는 그대로


def test_rolling_nan_and_constant_policy():
    x, y = _pair(n=100, seed=3)
    x.iloc[10:12] = np.nan
    out = RC.rolling_corr(x, y, 30, min_periods=30)
    assert out["rho"].iloc[29:41].isna().all() and out["n_eff"].iloc[35] == 28   # 완전창 규칙이면 NaN, 0 으로 대체 안 함
    out9 = RC.rolling_corr(x, y, 30, min_periods=27)
    assert out9["rho"].iloc[35] == out9["rho"].iloc[35]                          # 90% 규칙이면 계산
    xc = pd.Series(1.0, index=x.index)
    oc = RC.rolling_corr(xc, y, 30)
    assert oc["rho"].iloc[40:].isna().all() and oc["const"].iloc[40:].all()


def test_inputs_not_mutated():
    x, y = _pair(seed=4); x0, y0 = x.copy(), y.copy()
    RC.rolling_corr(x, y, 30, "spearman"); RC.fisher_change(RC.rolling_corr(x, y, 30)["rho"], 30)
    assert x.equals(x0) and y.equals(y0)


# ---------------------------------------------------------------- crossings / persistence
def _rho_path():
    idx = pd.bdate_range("2021-01-01", periods=60)
    v = np.r_[np.full(15, 0.4), np.full(3, -0.05), np.full(4, 0.3), np.full(20, -0.4), np.full(18, 0.02)]
    return pd.Series(v, idx)


def test_zero_crossings_count():
    zc = RC.zero_crossings(_rho_path())
    assert len(zc) == 4 and list(zc.new_sign) == [-1, 1, -1, 1]   # 마지막 +0.02 도 교차로 셈 (band 없음)


def test_sign_regime_hysteresis_and_confirmed_after_candidate():
    sr = RC.sign_regimes(_rho_path(), RC.RegimeRule(0.1, 10))
    ev = sr[~sr.is_initial]
    assert len(ev) == 1                                     # 짧은 −0.05 담금(3일, band 안)은 전환이 아님
    e = ev.iloc[0]
    assert e.to_sign == -1 and e.confirm_lag_days == 9 and e.confirmed_at > e.candidate_at


def test_fisher_change_large_event():
    idx = pd.bdate_range("2021-01-01", periods=200)
    rho = pd.Series(np.r_[np.full(100, -0.5), np.full(100, 0.5)], idx)
    ev = RC.large_change_events(rho, 60)
    assert len(ev) == 1 and ev.direction.iloc[0] == 1


# ---------------------------------------------------------------- price events
def test_price_events_sign_separation_and_posthoc_fields():
    idx = pd.bdate_range("2020-01-01", periods=400)
    r = pd.Series(np.random.default_rng(5).normal(0, 0.01, 400), idx)
    r.iloc[300] = 0.08; r.iloc[350] = -0.08
    ev = PE.finalize(pd.DataFrame(PE.series_price_events(r, "X", "test")))
    p1 = ev[ev.definition_id == "P1_vol_scaled"]
    assert (p1[p1.candidate_at == idx[300]].direction == 1).all() and (p1[p1.candidate_at == idx[350]].direction == -1).all()
    p4 = ev[ev.definition_id == "P4_cum_h10"]
    assert (p4.available_at > p4.candidate_at).all()        # 누적 사건은 사후 라벨: 시작일에는 알 수 없음
    assert (ev[ev.definition_id == "P1_vol_scaled"].available_at == ev[ev.definition_id == "P1_vol_scaled"].candidate_at).all()


def test_robust_z_definitions_are_distinct_and_shifted():
    idx = pd.bdate_range("2020-01-01", periods=200)
    r = pd.Series(np.random.default_rng(6).standard_t(3, 200) * 0.01, idx)
    zd, za, ze = PE.rz_D(r), PE.rz_A(r), PE.rz_eda(r)
    assert not np.allclose(zd.dropna().iloc[-50:], za.dropna().iloc[-50:])
    r2 = r.copy(); r2.iloc[150] = 1.0                       # 당일 큰 값이 자기 기준선(med·MAD)에 들어가지 않는다
    for f in (PE.rz_D, PE.rz_A, PE.rz_eda):
        z1, z2 = f(r), f(r2)
        assert np.allclose(z1.iloc[:150], z2.iloc[:150], equal_nan=True)           # 과거 무영향
        assert z2.iloc[150] > 50                                                     # 기준선이 오염되지 않아 z 가 매우 큼


# ---------------------------------------------------------------- matching / null
def test_one_to_one_matching_and_types():
    idx = pd.bdate_range("2020-01-01", periods=100)
    Cev = pd.DataFrame({"event_id": ["c1", "c2", "c3"], "candidate_at": idx[[10, 12, 60]], "quality_flag": [None] * 3})
    Pev = pd.DataFrame({"event_id": ["p1", "p2"], "candidate_at": idx[[11, 90]], "quality_flag": [None, "dq"]})
    m = EM.match_events(Cev, Pev, idx, 5)
    s = EM.summarize_matches(m, 3, 2)
    assert s["n_matched"] == 1 and s["n_corr_only"] == 2 and s["n_price_only"] == 1
    assert m.loc[m.p_event_id == "p1", "lag"].iloc[0] in (1, -1)
    assert m.loc[m.p_event_id == "p2", "dq_uncertain"].iloc[0]


def test_null_test_fast_equals_reference_and_pvalue_rule():
    idx = pd.bdate_range("2019-01-01", periods=1000)
    rng = np.random.default_rng(3)
    cpos = np.sort(rng.choice(1000, 15, replace=False)); ppos = np.sort(rng.choice(1000, 30, replace=False))
    a = EM.circular_shift_test(pd.DataFrame({"candidate_at": idx[cpos]}), pd.DataFrame({"candidate_at": idx[ppos]}), idx, 5, B=200, seed=11)
    b = EM.circular_shift_test_fast(cpos, ppos, 1000, 5, B=200, seed=11)
    assert a["T_obs"] == b["T_obs"] and a["p_value"] == b["p_value"]
    assert b["p_value"] >= 1 / 201                          # (1+count)/(B+1) 하한


def test_bh_qvalues_monotone():
    p = np.array([0.001, 0.01, 0.02, 0.5, 0.9])
    q = EM.bh_qvalues(p)
    assert np.all(q >= p) and np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_entering_day_influence_identity():
    x, y = _pair(seed=7)
    inf = RB.entering_day_influence(x, y, 30, min_periods_frac=1.0)
    i = 100
    ref = np.corrcoef(x.iloc[i - 29:i], y.iloc[i - 29:i])[0, 1]   # t-w+1..t-1
    assert np.isclose(inf["rho_excl_today"].iloc[i], ref)


# ---------------------------------------------------------------- integration on real data / checkpoint
@real
def test_real_261240_dq_reproduces_A1_and_rolling_window_of_effect():
    rr, r = D.returns(C, "raw"), D.returns(C, "dq_trim_canonical")
    assert abs(rr["261240"].corr(rr["069500"]) + 0.3084) < 1e-3 and abs(r["261240"].corr(r["069500"]) + 0.4577) < 1e-3
    d = RB.rolling_variant_diff(rr, r, "069500", "261240").set_index("window")
    assert str(d.loc[30, "last_affected"].date()) == "2019-04-25" and d.loc[30, "max_abs_diff"] > 0.5


@real
def test_real_common_shock_parity_40_days():
    r = D.returns(C, "dq_trim_canonical")
    cs, _ = PE.common_shock_events(r)
    assert len(cs) == 40                                    # SHOCK_MAP_FINAL rz3_D dedup k≥5


@real
def test_real_153130_zero_return_diagnostics():
    r = D.returns(C, "dq_trim_canonical")
    z = RB.zero_return_diagnostics(r, "069500", "153130").set_index("window")
    assert z.loc[30, "n_cross_pearson"] == 137 and z.loc[30, "n_cross_pearson_zero_as_nan"] == 97
    assert z["index_preserved"].all()


@has_a3
def test_a3_checkpoint_consistency():
    man = json.loads(A3_MAN.read_text())
    assert man["input_data_version"] == K.input_data_version(D.read_manifest(C))
    T = pd.read_csv(C.report_dir / "a3" / "event_null_tests.csv", low_memory=False)
    assert len(T) == man["metrics"]["n_tests"] and T["p_value"].min() >= 1 / (T["B_used"].max() + 1) - 1e-12
    assert (T.loc[T.testable, "q_value_bh"] >= T.loc[T.testable, "p_value"] - 1e-12).all()
    cases = pd.read_csv(C.report_dir / "a3" / "case_study_index.csv", dtype={"ticker": str})
    f = cases[cases.found]
    assert not f.duplicated(["ticker", "c_date", "p_date"]).any()
    for col in CE.EVENT_COLS:
        assert col in pd.read_csv(C.report_dir / "a3" / "price_event_candidates.csv", nrows=2).columns
