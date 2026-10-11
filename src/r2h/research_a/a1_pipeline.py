"""Phase A1 산출물 일괄 생성 (결정적, 난수 없음).

usage: .venv/bin/python -m src.r2h.research_a.a1_pipeline   (작업 트리 루트에서)

출력
- data/processed/research_a/<run_id>/   파생 패널 (gitignored) — close / logret raw / logret dq_trim_canonical / dq flag long
- reports/r2h/research_a/a1/            QA 표 (git 추적, 작은 표) + A1_RUN_MANIFEST.json (파생 파일 sha256 포함)
raw·interim·기존 reports 는 읽기만 한다.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import qa
from .contract import (KST, git_branch, git_sha, input_data_version, load_contract, make_run_id, runtime_info,
                       sha256_file, write_json)
from .data import etf_manifest, load_close_panel, read_dq_flags, read_manifest, returns, verify_raw_sha

BASE_COMMIT = "d5c2f792022d3c8efdbc52bdecd246b04bf32c45"


def run_a1(root: str | Path | None = None, run_id: str | None = None) -> dict:
    c = load_contract(root)
    man = read_manifest(c)
    version = input_data_version(man)
    run_id = run_id or make_run_id(version)
    B = c.base_panel
    rep = c.report_dir / "a1"
    out = c.derived_dir(run_id)
    out.mkdir(parents=True, exist_ok=True)

    sha = verify_raw_sha(c)
    if not sha["match"].all():
        raise RuntimeError(f"raw sha mismatch: {sha.loc[~sha['match'], 'ticker'].tolist()} — 입력 버전이 계약과 다르다")

    # ---- 파생 데이터 (versioned, gitignored)
    close = load_close_panel(c)
    derived = {
        "close_panel.csv": close,
        "logret_raw.csv": returns(c, "raw"),
        "logret_dq_trim_canonical.csv": returns(c, "dq_trim_canonical"),
    }
    files = {}
    for name, df in derived.items():
        p = out / name
        df.to_csv(p, float_format="%.17g")      # 17 유효숫자 = float64 왕복 손실 없음
        files[str(p.relative_to(c.root))] = {"sha256": sha256_file(p), "rows": len(df), "cols": df.shape[1]}
    dql = read_dq_flags(c)
    p = out / "dq_flags_long.csv"; dql.to_csv(p, index=False)
    files[str(p.relative_to(c.root))] = {"sha256": sha256_file(p), "rows": len(dql), "cols": dql.shape[1]}

    # ---- QA 표 (git 추적)
    impact = qa.dq_impact(c)
    tables = {
        "etf_manifest.csv": etf_manifest(c),
        "raw_sha_check.csv": sha.drop(columns=["path"]),
        "raw_integrity.csv": qa.raw_integrity(c),
        "calendar_alignment_base_panel.csv": qa.calendar_alignment(c, B),
        "change_column_check.csv": qa.change_column_check(c, man["ticker"].tolist()),
        "ohlc_consistency.csv": qa.ohlc_consistency(c, man["ticker"].tolist()),
        "dq_impact.csv": impact,
        "dq_summary.csv": qa.dq_summary(impact),
        "dq_corr_sensitivity_base_panel.csv": qa.dq_corr_sensitivity(c, B),
        "existing_parity_base_panel.csv": qa.existing_parity(c, B),
        "pair_coverage_base_panel.csv": qa.pair_coverage(c, B),
        "existing_qa_parity.csv": qa.existing_qa_parity(c),
        "large_moves_base_panel.csv": qa.large_moves(c, B),
    }
    for name, df in tables.items():
        qa.write_table(df, rep / name, run_id, version)
    parity = qa.interim_parity(c)

    ep = tables["existing_parity_base_panel.csv"]
    checks = {
        "raw_sha_all_match": bool(sha["match"].all()),
        "raw_integrity_all_ok": bool(tables["raw_integrity.csv"]["ok"].all()),
        "base_panel_calendar_identical": bool((tables["calendar_alignment_base_panel.csv"][["only_in_ticker", "only_in_reference"]] == 0).all().all()),
        "interim_close_exact": parity["close_index_equal"] and parity["close_columns_equal"] and parity["close_max_abs_diff"] == 0.0,
        "interim_logret_match_1e-15": parity["logret_index_equal"] and parity["logret_columns_equal"]
                                      and parity["logret_nan_rebuilt"] == 0 and parity["logret_max_abs_diff"] < 1e-15,
        "existing_tables_reproduced": bool(ep.loc[ep["expected_to_match"], "match_6g"].all()),
        "existing_raw_validation_reproduced": bool(tables["existing_qa_parity.csv"]["match"].all()),
        "dq_flag_rows": int(len(dql)),
    }
    manifest = {
        "phase": "A1", "run_id": run_id, "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "input_data_version": version, "git_sha": git_sha(c.root), "git_branch": git_branch(c.root),
        "base_commit": BASE_COMMIT, "contract_version": c.spec["contract_version"],
        "raw_dir": str(c.raw_dir), "interim_dir": str(c.interim_dir), "runtime": runtime_info(),
        "derived_dir": str(out.relative_to(c.root)), "derived_files": files,
        "report_tables": sorted(tables), "interim_parity": {k: (list(v) if isinstance(v, tuple) else v) for k, v in parity.items()},
        "checks": checks,
        "audit_discrepancies": {
            "finance_datareader_in_shared_venv": importlib.util.find_spec("FinanceDataReader") is not None,
            "note": "감사 기록은 'FDR 없음'. A1 은 FDR 을 import/사용하지 않으며 raw 재수집도 하지 않는다",
        },
    }
    write_json(manifest, rep / "A1_RUN_MANIFEST.json")
    return {"run_id": run_id, "version": version, "checks": checks, "tables": tables, "parity": parity, "manifest": manifest}


if __name__ == "__main__":
    res = run_a1()
    print(res["run_id"])
    for k, v in res["checks"].items():
        print(f"{k:40s} {v}")
