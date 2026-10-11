"""Track A 데이터 QA — 검증 함수는 모두 '표를 반환'하고, 판정 bool 은 표의 열로 둔다.

노트북과 테스트가 같은 함수를 부르므로, 노트북의 숫자와 테스트의 assert 가 어긋날 수 없다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contract import Contract
from .data import (dq_flag_matrix, load_close_panel, load_interim, load_raw_ticker, log_returns, rank_safe,
                   read_dq_flags, read_manifest, returns)


# ---------------------------------------------------------------- 1. raw 무결성
def raw_integrity(c: Contract, tickers: list[str] | None = None) -> pd.DataFrame:
    """ticker 별 행수·기간·중복·역순·결측·0이하 가격·5일 초과 공백. 정렬 전 원 상태 기준."""
    m = read_manifest(c).set_index("ticker")
    tickers = tickers or m.index.tolist()
    rows = []
    for t in tickers:
        d = load_raw_ticker(c, t)
        gaps = d["Date"].diff().dt.days
        rows.append({
            "ticker": t, "rows": len(d), "rows_manifest": int(m.loc[t, "rows"]),
            "first": d["Date"].min().date().isoformat(), "last": d["Date"].max().date().isoformat(),
            "n_dup_date": int(d["Date"].duplicated().sum()), "is_monotonic": bool(d["Date"].is_monotonic_increasing),
            "n_nan_close": int(d["Close"].isna().sum()), "n_nonpos_close": int((d["Close"] <= 0).sum()),
            "n_gap_gt5d": int((gaps > 5).sum()), "max_gap_days": int(gaps.max()),
            "n_zero_volume": int((d["Volume"] == 0).sum()),
        })
    out = pd.DataFrame(rows)
    out["ok"] = ((out["rows"] == out["rows_manifest"]) & (out["n_dup_date"] == 0) & out["is_monotonic"]
                 & (out["n_nan_close"] == 0) & (out["n_nonpos_close"] == 0))
    return out


def calendar_alignment(c: Contract, tickers: list[str]) -> pd.DataFrame:
    """각 ticker 날짜 집합 vs 기준(069500) 날짜 집합. 공통 거래일이 정말 '공통'인지 확인."""
    ref = set(load_raw_ticker(c, c.reference)["Date"])
    rows = []
    for t in tickers:
        s = set(load_raw_ticker(c, t)["Date"])
        rows.append({"ticker": t, "n_dates": len(s), "only_in_ticker": len(s - ref), "only_in_reference": len(ref - s),
                     "n_common_with_reference": len(s & ref)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. canonical interim 과의 정합
def interim_parity(c: Contract) -> dict[str, object]:
    """raw 에서 재구성한 패널 vs canonical interim. NaN 을 건너뛰는 max|Δ| 만으로는 날짜 어긋남을 못 잡으므로
    index·열 순서·NaN 수를 각각 따로 확인한다."""
    close_i, lr_i = load_interim(c)
    close_r = load_close_panel(c, tickers=list(close_i.columns))
    lr_r = log_returns(close_r)
    return {
        "close_shape_interim": close_i.shape, "close_shape_rebuilt": close_r.shape,
        "close_index_equal": bool(close_i.index.equals(close_r.index)),
        "close_columns_equal": list(close_i.columns) == list(close_r.columns),
        "close_max_abs_diff": float((close_i - close_r).abs().to_numpy().max()),
        "logret_shape_interim": lr_i.shape, "logret_shape_rebuilt": lr_r.shape,
        "logret_index_equal": bool(lr_i.index.equals(lr_r.index)),
        "logret_columns_equal": list(lr_i.columns) == list(lr_r.columns),
        "logret_nan_interim": int(lr_i.isna().sum().sum()), "logret_nan_rebuilt": int(lr_r.isna().sum().sum()),
        "logret_max_abs_diff": float((lr_i - lr_r).abs().to_numpy().max()),
        "logret_first": str(lr_i.index[0].date()), "logret_last": str(lr_i.index[-1].date()),
    }


def change_column_check(c: Contract, tickers: list[str]) -> pd.DataFrame:
    """raw Change 열이 Close.pct_change() 와 같은가. 같다면 Change 는 Close 파생값 →
    가격조정 여부의 '독립' 증거로 쓸 수 없다."""
    rows = []
    for t in tickers:
        d = load_raw_ticker(c, t)
        diff = (d["Change"] - d["Close"].pct_change()).abs()
        rows.append({"ticker": t, "max_abs_diff": float(diff.max()), "n_compared": int(diff.notna().sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. 가격조정 증거 (판정하지 않음)
def ohlc_consistency(c: Contract, tickers: list[str]) -> pd.DataFrame:
    """Close>High / Close<Low 건수·기간·크기. Close 만 조정되고 H/L 은 아니면 이런 위반이 생길 수 있으나,
    반올림·데이터 공급 오류로도 생긴다 — '정황'으로만 기록."""
    rows = []
    for t in tickers:
        d = load_raw_ticker(c, t)
        hi, lo = d["Close"] > d["High"], d["Close"] < d["Low"]
        v = d[hi | lo]
        ratio = (v["Close"] / np.where(hi[hi | lo], v["High"], v["Low"])) if len(v) else pd.Series(dtype=float)
        rows.append({"ticker": t, "n_close_gt_high": int(hi.sum()), "n_close_lt_low": int(lo.sum()),
                     "first": v["Date"].min().date().isoformat() if len(v) else "",
                     "last": v["Date"].max().date().isoformat() if len(v) else "",
                     "max_rel_excess": float((ratio - 1).abs().max()) if len(v) else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. DQ flag → 수익률 입력 영향
def dq_impact(c: Contract, tickers: list[str] | None = None) -> pd.DataFrame:
    """flag 1건마다: 그 날 수익률 r 이 패널에 있는가, 값, 그리고 Close→r 경로에 닿는지(affects_logret_path).
    OHLC 위반은 O/H/L 문제라 Close 기반 r 에 직접 닿지 않는다. spike/anomaly 는 그 날 Close 자체가 의심 대상이다."""
    d = read_dq_flags(c)
    if tickers:
        d = d[d["ticker"].isin(tickers)]
    r = returns(c, "raw")
    trim = c.dq_trim_cells
    out = d[["ticker", "date", "flag", "dq_class"]].copy()
    out["in_panel_index"] = out["date"].isin(r.index)
    out["logret"] = [float(r.at[dt, t]) if dt in r.index else np.nan for t, dt in zip(out["ticker"], out["date"])]
    out["affects_logret_path"] = out["dq_class"].ne("ohlc_close_gt_high")
    out["nan_in_dq_trim_canonical"] = [dt.strftime("%Y-%m-%d") in trim.get(t, []) for t, dt in zip(out["ticker"], out["date"])]
    out["in_base_panel"] = out["ticker"].isin(c.base_panel)
    return out.reset_index(drop=True)


def dq_summary(impact: pd.DataFrame) -> pd.DataFrame:
    return (impact.groupby(["dq_class", "affects_logret_path"])
            .agg(n_flags=("ticker", "size"), n_tickers=("ticker", "nunique"),
                 n_in_base_panel=("in_base_panel", "sum"), n_nan_in_trim=("nan_in_dq_trim_canonical", "sum"))
            .reset_index())


def dq_corr_sensitivity(c: Contract, tickers: list[str]) -> pd.DataFrame:
    """기준 ETF 와의 전체표본 Pearson ρ — raw / dq_trim_canonical / 경로 영향 flag 전부 mask.
    '삭제 결정'이 아니라 결론이 flag 에 얼마나 민감한지 보는 민감도 표다."""
    ref = c.reference
    r_raw = returns(c, "raw")
    r_trim = returns(c, "dq_trim_canonical")
    flags = dq_flag_matrix(c, r_raw.index, list(r_raw.columns))
    mask_all = sum((m for k, m in flags.items() if k != "ohlc_close_gt_high"),
                   start=pd.DataFrame(False, index=r_raw.index, columns=r_raw.columns))
    r_all = r_raw.mask(mask_all)
    rows = []
    for t in tickers:
        if t == ref:
            continue
        rows.append({"ticker": t, "rho_raw": r_raw[t].corr(r_raw[ref]), "rho_dq_trim_canonical": r_trim[t].corr(r_trim[ref]),
                     "rho_mask_all_path_flags": r_all[t].corr(r_all[ref]),
                     "n_masked_path_flags": int(mask_all[t].sum() + mask_all[ref].sum())})
    out = pd.DataFrame(rows)
    out["max_abs_delta"] = (out[["rho_dq_trim_canonical", "rho_mask_all_path_flags"]].sub(out["rho_raw"], axis=0)).abs().max(axis=1)
    return out


# ---------------------------------------------------------------- 5. 기존 canonical 결과 재현
def existing_parity(c: Contract, tickers: list[str]) -> pd.DataFrame:
    """A 로더 결과가 기존 closure 표(universe_final.py 산출)와 같은가.
    기존 표는 float_format='%.6g' 로 저장됐으므로 상대 1e-5 수준이 허용 오차다."""
    tab = c.root / "reports" / "tables"
    r = returns(c, "dq_trim_canonical")                       # A 기본 (logdiff)
    r_src = returns(c, "dq_trim_canonical", formula="logratio")  # 원 스크립트(universe_final) 산식
    rows = []
    vol = pd.read_csv(tab / "universe_vol_rank.csv", index_col=0, dtype={0: str})
    vol.index = vol.index.astype(str).str.zfill(6)
    mine_vol = r.std() * np.sqrt(252)
    for t in tickers:
        rows.append({"check": "ann_vol_trimdq", "ticker": t, "existing": float(vol.loc[t, "ann_vol_trimdq"]),
                     "a_loader": float(mine_vol[t])})
    pear = pd.read_csv(tab / "universe_corr_pearson.csv", index_col=0)
    pear.index = pear.index.astype(str).str.zfill(6); pear.columns = pear.columns.astype(str).str.zfill(6)
    sp = pd.read_csv(tab / "universe_corr_spearman.csv", index_col=0)
    sp.index = sp.index.astype(str).str.zfill(6); sp.columns = sp.columns.astype(str).str.zfill(6)
    ref = c.reference
    for t in tickers:
        if t == ref:
            continue
        rows.append({"check": "pearson_vs_ref_trimdq", "ticker": t, "existing": float(pear.loc[t, ref]),
                     "a_loader": float(r[t].corr(r[ref]))})
        rows.append({"check": "spearman_vs_ref_trimdq[logratio]", "ticker": t, "existing": float(sp.loc[t, ref]),
                     "a_loader": float(r_src[t].corr(r_src[ref], method="spearman"))})
        rows.append({"check": "spearman_vs_ref_trimdq[logdiff,rank_safe]", "ticker": t, "existing": float(sp.loc[t, ref]),
                     "a_loader": float(rank_safe(r[t]).corr(rank_safe(r[ref]), method="spearman"))})
        rows.append({"check": "spearman_vs_ref_trimdq[logdiff,unrounded]", "ticker": t, "existing": float(sp.loc[t, ref]),
                     "a_loader": float(r[t].corr(r[ref], method="spearman"))})
    out = pd.DataFrame(rows)
    out["abs_diff"] = (out["existing"] - out["a_loader"]).abs()
    out["rel_diff"] = out["abs_diff"] / out["existing"].abs().clip(lower=1e-12)
    out["match_6g"] = out["rel_diff"] <= 1e-5
    out["expected_to_match"] = ~out["check"].str.contains("unrounded")  # unrounded 는 동률 민감도를 보이려는 대조행
    return out


# ---------------------------------------------------------------- 6. 기본 패널 공통 기간
def pair_coverage(c: Contract, tickers: list[str]) -> pd.DataFrame:
    """기준 ETF 와 각 비교 ETF 의 공통 관측 수·기간·0 수익률 비율.
    0 수익률이 많으면(단기채 틱 크기 등) 상관·z-score 분모가 왜곡될 수 있다."""
    r = returns(c, "raw")
    ref = c.reference
    rows = []
    for t in tickers:
        both = r[[ref, t]].dropna()
        rows.append({"ticker": t, "n_common_obs": len(both), "first": str(both.index[0].date()),
                     "last": str(both.index[-1].date()), "share_zero_return": float((r[t] == 0).mean()),
                     "ann_vol": float(r[t].std() * np.sqrt(252))})
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, path: str | Path, run_id: str, input_version: str) -> Path:
    """모든 A 표는 run_id·input_data_version 열을 달고 저장한다 (어느 실행·어느 raw 에서 나왔는지)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.insert(0, "input_data_version", input_version[:16])
    out.insert(0, "run_id", run_id)
    out.to_csv(path, index=False, float_format="%.10g")
    return path


# ---------------------------------------------------------------- 7. 기존 QA 표(raw_validation.csv)와 대조
JUMP_SIMPLE = 0.15   # validate_raw.py JUMP — 단순수익률 |Close.pct_change()| > 15%


def existing_qa_parity(c: Contract) -> pd.DataFrame:
    """validate_raw.py 의 정의를 그대로 재구현해 reports/raw_validation.csv 와 ticker×항목 단위로 비교.
    A 의 ohlc_consistency(Close 만 H/L 밖) 와 기존 n_ohlc_violation(Open 포함·H<L 포함)은 정의가 다르므로 둘 다 싣는다."""
    v = pd.read_csv(c.root / "reports" / "raw_validation.csv", dtype={"ticker": str}).set_index("ticker")
    rows = []
    for t in v.index:
        d = load_raw_ticker(c, t)
        bad = ((d["High"] < d[["Open", "Close"]].max(axis=1)) | (d["Low"] > d[["Open", "Close"]].min(axis=1))
               | (d["High"] < d["Low"]))
        close_out = (d["Close"] > d["High"]) | (d["Close"] < d["Low"])
        jumps = d.loc[d["Close"].pct_change().abs() > JUMP_SIMPLE, "Date"].dt.date.astype(str).tolist()
        mine = {"n_dup_date": int(d["Date"].duplicated().sum()), "n_ohlc_violation": int(bad.sum()),
                "n_zero_volume": int((d["Volume"] == 0).sum()), "n_gap_gt5d": int((d["Date"].diff().dt.days > 5).sum()),
                "n_jump_gt15pct": len(jumps)}
        for k, val in mine.items():
            rows.append({"ticker": t, "item": k, "existing": int(v.loc[t, k]), "a_recomputed": val})
        rows.append({"ticker": t, "item": "jump_dates", "existing": str(v.loc[t, "jump_dates"]) if pd.notna(v.loc[t, "jump_dates"]) else "",
                     "a_recomputed": ";".join(jumps)})
        rows.append({"ticker": t, "item": "a_only:n_close_outside_hl", "existing": "", "a_recomputed": int(close_out.sum())})
    out = pd.DataFrame(rows)
    out["match"] = [str(a) == str(b) if not i.startswith("a_only") else True
                    for i, a, b in zip(out["item"], out["existing"], out["a_recomputed"])]
    return out


def large_moves(c: Contract, tickers: list[str], threshold: float = JUMP_SIMPLE) -> pd.DataFrame:
    """|단순수익률| > 15% 인 관측치 — DQ flag 와 별개로, rolling 창 하나를 지배할 수 있는 '영향 큰 관측치'.
    삭제 대상이 아니라 A2 에서 창별 민감도를 확인할 목록이다."""
    r = returns(c, "raw", tickers=tickers)
    simple = np.expm1(r)
    flags = read_dq_flags(c)
    keyed = set(zip(flags["ticker"], flags["date"]))
    s = simple.stack()
    s = s[s.abs() > threshold]
    out = s.rename("simple_ret").reset_index().rename(columns={"level_1": "ticker"})
    out.columns = ["date", "ticker", "simple_ret"]
    out["logret"] = np.log1p(out["simple_ret"])
    out["has_dq_flag"] = [(t, d) in keyed for t, d in zip(out["ticker"], out["date"])]
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)
