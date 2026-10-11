"""Track A Phase A1 — 데이터 계약 테스트.

usage: .venv/bin/python -m pytest tests/research_a -q   (작업 트리 루트)
- synthetic: 함수 의미 검증 (raw 불필요)
- real: 공유 raw 가 있을 때만 — 계약의 expected 값과 canonical 결과 재현
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.r2h.research_a import contract as K, data as D, qa as Q  # noqa: E402

C = K.load_contract(ROOT)
HAS_RAW = (C.raw_dir / "069500.csv").exists()
real = pytest.mark.skipif(not HAS_RAW, reason="shared raw not available")


# ---------------------------------------------------------------- synthetic
def test_log_returns_definition_and_formulas():
    close = pd.DataFrame({"a": [100.0, 110.0, 99.0, 99.0]}, index=pd.bdate_range("2020-01-01", periods=4))
    r = D.log_returns(close)
    assert len(r) == 3 and r.index[0] == close.index[1]          # 첫 행 제외
    assert np.isclose(r["a"].iloc[0], np.log(1.1)) and r["a"].iloc[2] == 0.0
    assert np.allclose(r, D.log_returns(close, "logratio"), atol=1e-15)


def test_log_returns_rejects_nonpositive():
    with pytest.raises(ValueError):
        D.log_returns(pd.DataFrame({"a": [1.0, 0.0, 2.0]}))


def test_rank_safe_restores_float_ties():
    # 같은 가격비를 두 산식으로 만들면 1ulp 차이로 동률이 깨질 수 있다 → 반올림이 동률을 되살린다
    pairs = [(np.log(b) - np.log(a), np.log(b / a)) for a in range(9000, 9400) for b in (a + 5, a + 10)]
    broken = [p for p in pairs if p[0] != p[1]]
    assert broken, "산식 차이로 동률이 깨지는 예가 있어야 테스트가 의미를 가진다"
    x = pd.Series(broken[0])
    assert x.nunique() == 2 and D.rank_safe(x).nunique() == 1


def test_input_data_version_order_invariant_and_sensitive():
    m = pd.DataFrame({"ticker": ["b", "a"], "sha256": ["2", "1"]})
    assert K.input_data_version(m) == K.input_data_version(m.iloc[::-1])
    assert K.input_data_version(m) != K.input_data_version(m.assign(sha256=["2", "x"]))


def test_contract_roles_and_fields_are_separate():
    roles = C.roles
    assert len(roles) == C.expected["n_tickers"]
    assert roles["069500"] == "reference" and roles["102110"] == "control_not_independent"
    assert set(C.base_panel) == {t for t, r in roles.items() if r != "not_in_base_panel"}
    # 102110 은 069500 과 같은 exposure — 사건 집계에서 중복 금지
    assert C.exposure_group_of("102110") == C.exposure_group_of("069500") == "KOSPI200"
    # 수집되지 않은 ETF 를 만들지 않는다 (국고채5년 등)
    assert all(t in roles for t in C.base_panel)
    sync = C.spec["session_sync"]
    assert sorted(sum(sync.values(), [])) == sorted(roles)          # 20종 모두 정확히 1개 분류


def test_robust_z_registry_unresolved():
    reg = C.spec["robust_z_registry"]
    assert reg["status"] == "unresolved" and {"rz_D", "rz_A"} <= set(reg["definitions"])


# ---------------------------------------------------------------- real data
@real
def test_raw_sha_matches_manifest():
    assert D.verify_raw_sha(C)["match"].all()


@real
def test_raw_integrity_expected_shape():
    t = Q.raw_integrity(C)
    e = C.expected
    assert len(t) == e["n_tickers"] and t["ok"].all()
    assert (t["rows"] == e["rows_per_ticker"]).all()
    assert (t["first"] == e["first_date"]).all() and (t["last"] == e["last_date"]).all()


@real
def test_interim_parity_index_columns_values():
    p = Q.interim_parity(C)
    assert p["close_shape_interim"] == (C.expected["panel_close_rows"], 20)
    assert p["logret_shape_interim"] == (C.expected["panel_logret_rows"], 20)
    assert p["close_index_equal"] and p["close_columns_equal"] and p["close_max_abs_diff"] == 0.0
    assert p["logret_index_equal"] and p["logret_columns_equal"]
    assert p["logret_nan_interim"] == p["logret_nan_rebuilt"] == 0
    assert p["logret_max_abs_diff"] < 1e-15


@real
def test_dq_trim_variant_only_two_cells():
    raw, trim = D.returns(C, "raw"), D.returns(C, "dq_trim_canonical")
    assert raw.index.equals(trim.index)                              # 행 삭제 없음
    diff = raw.isna() != trim.isna()
    assert int(diff.to_numpy().sum()) == 2 and diff["261240"].sum() == 2


@real
def test_dq_flags_count_and_path_split():
    imp = Q.dq_impact(C)
    assert len(imp) == C.expected["dq_flag_rows"]
    assert imp["in_panel_index"].all()
    assert (~imp.loc[imp.dq_class == "ohlc_close_gt_high", "affects_logret_path"]).all()
    assert imp["nan_in_dq_trim_canonical"].sum() == 2


@real
def test_change_column_is_derived_from_close():
    assert Q.change_column_check(C, C.base_panel)["max_abs_diff"].max() < 1e-12


@real
def test_existing_closure_tables_reproduced():
    ep = Q.existing_parity(C, C.base_panel)
    assert ep.loc[ep["expected_to_match"], "match_6g"].all()


@real
def test_etf_manifest_has_spec_and_role_fields():
    m = D.etf_manifest(C)
    assert len(m) == 20 and m["in_original_universe"].all()
    for col in ["asset_scope", "benchmark", "currency_hedge", "leverage_inverse", "a_analysis_role", "a_session_sync"]:
        assert col in m.columns and (m[col] != "").all()


@real
def test_existing_raw_validation_reproduced():
    e = Q.existing_qa_parity(C)
    assert e["match"].all(), e[~e["match"]]


@real
def test_large_moves_not_dq_except_261240():
    lm = Q.large_moves(C, C.base_panel)
    assert set(lm.loc[lm.has_dq_flag, "ticker"]) == {"261240"}
    assert (lm.loc[~lm.has_dq_flag, "ticker"].isin(["069500", "102110", "261220"])).all()
