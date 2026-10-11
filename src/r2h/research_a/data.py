"""Track A 데이터 로더 — raw(READ-ONLY) → Close 패널 → 일간 로그수익률 (variant 별).

정의는 config/research_a_contract.yaml 의 returns 절과 같다:
  r_t = ln(Close_t / Close_{t-1}), 공통 거래일, 첫 행 제외.
variant:
  raw               DQ 처리 없음 (= canonical data/interim/panel_logret.csv)
  dq_trim_canonical price_anomaly_unverified 셀만 NaN (행 유지, 보간 없음 — 기존 Rt/ANOM 과 동일)
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .contract import Contract, sha256_file

Variant = Literal["raw", "dq_trim_canonical"]


# ---------------------------------------------------------------- manifest / universe
def read_manifest(c: Contract) -> pd.DataFrame:
    return pd.read_csv(c.path("manifest"), dtype=str, keep_default_na=False)


def read_universe(c: Contract) -> pd.DataFrame:
    return pd.read_csv(c.path("universe"), dtype=str, keep_default_na=False)


def read_dq_flags(c: Contract) -> pd.DataFrame:
    d = pd.read_csv(c.path("dq_flags"), dtype=str, keep_default_na=False)
    d["date"] = pd.to_datetime(d["date"])
    return d


def verify_raw_sha(c: Contract) -> pd.DataFrame:
    """raw 파일 sha256 을 MANIFEST 와 대조. 불일치가 하나라도 있으면 이후 분석의 입력 버전이 달라진 것."""
    m = read_manifest(c)
    rows = []
    for t, expect in zip(m["ticker"], m["sha256"]):
        p = c.raw_dir / f"{t}.csv"
        got = sha256_file(p) if p.exists() else None
        rows.append({"ticker": t, "path": str(p), "exists": p.exists(), "sha256_manifest": expect,
                     "sha256_actual": got, "match": got == expect})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- raw → panel
def load_raw_ticker(c: Contract, ticker: str) -> pd.DataFrame:
    """정렬·중복제거 *하지 않은* raw 그대로. 검증 함수가 원 상태를 봐야 하므로."""
    d = pd.read_csv(c.raw_dir / f"{ticker}.csv")
    d["Date"] = pd.to_datetime(d["Date"])
    return d


def load_close_panel(c: Contract, tickers: list[str] | None = None, common_only: bool = True) -> pd.DataFrame:
    """Close wide panel (index=Date, columns=ticker, 단위=KRW).

    common_only=True 면 모든 ticker 에 Close 가 있는 날만 남긴다 — validate_raw.py 의 `close.dropna()` 와 같다.
    중복 날짜가 있으면 조용히 처리하지 않고 실패한다 (A1 QA 에서 0건 확인됨).
    """
    tickers = tickers or read_manifest(c)["ticker"].tolist()
    cols = {}
    for t in tickers:
        d = load_raw_ticker(c, t)
        if d["Date"].duplicated().any():
            raise ValueError(f"{t}: duplicate dates in raw — resolve explicitly before building panel")
        cols[t] = d.set_index("Date")["Close"].astype(float)
    panel = pd.DataFrame(cols).sort_index()
    panel.index.name = "Date"
    return panel.dropna() if common_only else panel


def log_returns(close: pd.DataFrame, formula: Literal["logdiff", "logratio"] = "logdiff") -> pd.DataFrame:
    """r_t = ln(C_t/C_{t-1}). 0 이하 가격은 로그가 정의되지 않으므로 실패시킨다(보정하지 않음).

    두 산식은 수학적으로 같지만 부동소수점에서 ≤2e-15 다르다:
      logdiff  = ln(C_t) - ln(C_{t-1})   — scripts/validate_raw.py → canonical panel_logret.csv (A 기본)
      logratio = ln(C_t / C_{t-1})       — analysis/scripts/universe_final.py, d0122
    Pearson 에는 무영향이지만, 같은 가격비에서 생기는 '동률'이 깨져 순위 통계(Spearman)가 1e-6 수준 달라진다.
    순위 기반 통계는 rank_safe() 로 반올림 후 계산한다.
    """
    if (close <= 0).any().any():
        raise ValueError("non-positive Close — log return undefined")
    if formula == "logdiff":
        return np.log(close).diff().iloc[1:]
    if formula == "logratio":
        return np.log(close / close.shift(1)).iloc[1:]
    raise ValueError(formula)


RANK_DECIMALS = 12


def rank_safe(r: pd.DataFrame | pd.Series, decimals: int = RANK_DECIMALS):
    """순위 통계 전용: 1e-12 로 반올림해 산식 차이(1e-15)로 갈라진 동률을 복원한다. 값 분석에는 쓰지 않는다."""
    return r.round(decimals)


def returns(c: Contract, variant: Variant = "raw", tickers: list[str] | None = None,
            formula: Literal["logdiff", "logratio"] = "logdiff") -> pd.DataFrame:
    """계약 정의대로 일간 로그수익률. 패널은 항상 20종 공통 날짜에서 만든 뒤 열을 고른다
    (부분 집합으로 공통 날짜를 다시 잡으면 canonical 패널과 index 가 달라질 수 있다)."""
    r = log_returns(load_close_panel(c), formula=formula)
    if variant == "dq_trim_canonical":
        r = r.copy()
        for t, days in c.dq_trim_cells.items():
            r.loc[pd.to_datetime(days), t] = np.nan
    elif variant != "raw":
        raise ValueError(variant)
    return r[tickers] if tickers else r


def load_interim(c: Contract) -> tuple[pd.DataFrame, pd.DataFrame]:
    """canonical interim 패널 (READ-ONLY). header 는 문자열로 읽히므로 069500 의 선행 0 이 보존된다."""
    close = pd.read_csv(c.interim_dir / "panel_close.csv", index_col=0, parse_dates=True)
    lr = pd.read_csv(c.interim_dir / "panel_logret.csv", index_col=0, parse_dates=True)
    for df in (close, lr):
        df.columns = df.columns.astype(str)
        df.index.name = "Date"
    return close, lr


def dq_flag_matrix(c: Contract, index: pd.DatetimeIndex, columns: list[str]) -> dict[str, pd.DataFrame]:
    """dq_class 별 bool 행렬 (index×ticker). 값을 바꾸지 않고 flag 를 '옆에 붙이는' 용도."""
    d = read_dq_flags(c)
    out = {}
    for cls, g in d.groupby("dq_class"):
        m = pd.DataFrame(False, index=index, columns=columns)
        for t, dt in zip(g["ticker"], g["date"]):
            if t in m.columns and dt in m.index:
                m.loc[dt, t] = True
        out[str(cls)] = m
    return out


# ---------------------------------------------------------------- ETF 명세
SPEC_COLS = ["slot", "ticker", "name", "asset_scope", "category", "benchmark", "benchmark_proxy_ticker",
             "leverage_inverse", "currency_hedge", "replication", "exposure", "trading_hours_note", "universe_status"]


def etf_manifest(c: Contract) -> pd.DataFrame:
    """20종 명세 + 분석 목적 필드. 원 유니버스 필드(universe.csv 그대로)와 A 가 정한 필드를 열 이름으로 분리한다."""
    u = read_universe(c)[SPEC_COLS].copy()
    m = read_manifest(c).set_index("ticker")
    u["in_original_universe"] = True
    u["a_analysis_role"] = u["ticker"].map(c.roles)
    u["a_in_base_panel"] = u["ticker"].isin(c.base_panel)
    u["a_comparison_group"] = u["ticker"].map(c.spec["analysis_roles"]["comparison_group_label"]).fillna("")
    u["a_exposure_group"] = u["ticker"].map(c.exposure_group_of)
    u["a_session_sync"] = u["ticker"].map(c.session_sync_of)
    u["raw_rows"] = u["ticker"].map(m["rows"]).astype(int)
    u["raw_first"] = u["ticker"].map(m["first"])
    u["raw_last"] = u["ticker"].map(m["last"])
    u["raw_sha256"] = u["ticker"].map(m["sha256"])
    u["source"] = u["ticker"].map(m["source"])
    u.loc[u["a_analysis_role"].eq("reference"), "a_comparison_group"] = "reference_kospi200"
    u.loc[u["a_analysis_role"].eq("control_not_independent"), "a_comparison_group"] = "control_kospi200_brand"
    return u
