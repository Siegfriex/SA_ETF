"""Phase A3 — 가격 사건(Price Events) 정의. 급등(+1)과 급락(−1)은 direction 으로 분리하고, 같은 절댓값 기준을 쓴
양방향 사건(direction=0)과 섞지 않는다. 정답(Ground Truth) 정의가 아니다 — 서로 다른 '가격 극단' 측정치의 후보 집합이다.

정의 (definition_id)
  P1_vol_scaled      r_t / σ_{t-1}(60) ≥ k (σ 는 shift(1), 당일 제외)                           → 실시간 관측 가능
  P2_quantile_exp    r_t ≥ q_{0.99}(r_{≤t-1}) (expanding, 최소 250 관측)                           → 실시간 관측 가능
  P2_quantile_full   r_t ≥ 전체표본 q_{0.99}                                                       → LOOK-AHEAD (서술용)
  P3_rz_D / P3_rz_A / P3_rz_eda   기존 robust-z 3 정의를 source 그대로 복원, z ≥ 3 / ≤ −3                → 실시간 관측 가능
  P4_cum_h{5,10,20}  Σ_{s=t}^{t+h-1} r_s / (σ_{t-1}(60)·√h) ≥ k — 미래 h 일 누적 → 사후 라벨 (identified_at = t+h-1)
  P5_legacy_{abs3,q99,rz3_D,rz3_A}   SHOCK_MAP 의 개별 ETF 극단 (abs3 는 양방향 |r|>3% 를 부호로 분리)
  P5_common_shock    SHOCK_MAP 공통충격일: rz3_D dedup 동반 ETF ≥ 5 (universe 수준, series_id='UNIVERSE')

시점 필드 (price)
  candidate_at = event_start, event_end, confirmed_at = identified_at (그 날 종가에 정의가 충족), available_at = identified_at
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .correlation_events import DEF_VERSION, EVENT_COLS

K200 = ["069500", "102110", "122630", "114800"]


# ---------------------------------------------------------------- robust-z 3 정의 (source 그대로)
def rz_D(r: pd.Series | pd.DataFrame):
    """analysis/scripts/universe_final.py rz_D (= robustness_final.py)."""
    med = r.rolling(60, min_periods=20).median().shift(1)
    mad = (r - med).abs().rolling(60, min_periods=20).median().shift(1) * 1.4826
    return (r - med) / mad


def rz_A(r: pd.Series | pd.DataFrame):
    """analysis/scripts/universe_final.py rz_A (= src/final_figs.robust_z, 창 내부 MAD, mp40)."""
    med = r.rolling(60, min_periods=40).median().shift(1)
    mad = r.rolling(60, min_periods=40).apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True).shift(1)
    return (r - med) / (1.4826 * mad)


def rz_eda(r: pd.Series):
    """src/eda_utils.extreme_definitions robust_z_shift1_w60_k3 — 원 코드는 |·| 만 반환. 부호 분리를 위해 부호 있는 z 로 복원."""
    base = r.shift(1)
    med = base.rolling(60, min_periods=30).median()
    mad = (base - med).abs().rolling(60, min_periods=30).median()
    return (r - med) / (1.4826 * mad.replace(0, np.nan))


RZ = {"P3_rz_D": rz_D, "P3_rz_A": rz_A, "P3_rz_eda": rz_eda}


def _eid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def _rows(mask: pd.Series, score: pd.Series, direction: int, did: str, params: dict, meta: dict,
          h: int = 1, idx: pd.DatetimeIndex | None = None) -> list[dict]:
    idx = idx if idx is not None else mask.index
    pos = {d: i for i, d in enumerate(idx)}
    out = []
    for d in mask.index[mask.fillna(False).astype(bool)]:
        i = pos[d]
        end = idx[min(i + h - 1, len(idx) - 1)]
        out.append({**meta, "definition_id": did, "definition_version": DEF_VERSION, "event_family": "price",
                    "parameter_set": json.dumps(params), "direction": direction, "candidate_at": d, "event_end": end,
                    "confirmed_at": end, "available_at": end, "score": float(score.get(d, np.nan)), "window": h, "method": None})
    return out


def _episodes(mask: pd.Series, score: pd.Series, direction: int, did: str, params: dict, meta: dict, h: int,
              idx: pd.DatetimeIndex) -> list[dict]:
    """겹치는 h 일 창은 연속된 시작일이 같은 상승을 중복 계산한다 → 연속 시작일을 1 episode 로 병합.
    event_start = 첫 시작일, event_end = 마지막 시작일 + h − 1, identified_at = 첫 시작일 + h − 1 (첫 창이 완성되는 날 — 사후 라벨),
    score = episode 안 최대 |누적수익률|."""
    m = mask.fillna(False).astype(bool).to_numpy()
    out, i, n = [], 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            seg = score.iloc[i:j + 1]
            out.append({**meta, "definition_id": did, "definition_version": DEF_VERSION, "event_family": "price",
                        "parameter_set": json.dumps(params), "direction": direction, "candidate_at": idx[i],
                        "event_end": idx[min(j + h - 1, n - 1)], "confirmed_at": idx[min(i + h - 1, n - 1)],
                        "available_at": idx[min(i + h - 1, n - 1)], "score": float(seg.abs().max()), "window": h, "method": None})
            i = j + 1
        else:
            i += 1
    return out


def _sided(z: pd.Series, k: float, did: str, params: dict, meta: dict, score: pd.Series | None = None) -> list[dict]:
    score = z if score is None else score
    return _rows(z >= k, score, +1, did, params, meta) + _rows(z <= -k, score, -1, did, params, meta)


def series_price_events(r: pd.Series, ticker: str, asset_class: str, snapshot_id: str = "", run_id: str = "",
                        k_vol: float = 3.0, q: float = 0.99, k_rz: float = 3.0, cum_h=(5, 10, 20), k_cum: float = 2.5,
                        quality: pd.Series | None = None) -> pd.DataFrame:
    meta = {"series_id": ticker, "reference_series_id": None, "asset_class": asset_class, "data_snapshot_id": snapshot_id, "run_id": run_id}
    rows = []
    sig = r.rolling(60, min_periods=40).std().shift(1)
    zv = r / sig
    rows += _sided(zv, k_vol, "P1_vol_scaled", {"k": k_vol, "sigma": "std60 shift1 mp40"}, meta)
    qe_hi = r.expanding(min_periods=250).quantile(q).shift(1)
    qe_lo = r.expanding(min_periods=250).quantile(1 - q).shift(1)
    rows += _rows(r >= qe_hi, r, +1, "P2_quantile_exp", {"q": q, "min_obs": 250, "lookahead": False}, meta)
    rows += _rows(r <= qe_lo, r, -1, "P2_quantile_exp", {"q": q, "min_obs": 250, "lookahead": False}, meta)
    rows += _rows(r >= r.quantile(q), r, +1, "P2_quantile_full", {"q": q, "lookahead": True}, meta)
    rows += _rows(r <= r.quantile(1 - q), r, -1, "P2_quantile_full", {"q": q, "lookahead": True}, meta)
    for did, f in RZ.items():
        rows += _sided(f(r), k_rz, did, {"k": k_rz, "source": f.__doc__.split("\n")[0]}, meta)
    for h in cum_h:
        fwd = r[::-1].rolling(h, min_periods=h).sum()[::-1]          # Σ_{s=t}^{t+h-1} r_s
        zc = fwd / (sig * np.sqrt(h))
        for sgn in (+1, -1):
            rows += _episodes(zc * sgn >= k_cum, fwd, sgn, f"P4_cum_h{h}",
                              {"h": h, "k": k_cum, "posthoc_label": True, "declustered": "consecutive start days merged"}, meta, h, r.index)
    # P5 legacy (SHOCK_MAP 개별 ETF)
    rows += _rows(r > 0.03, r, +1, "P5_legacy_abs3", {"thr": 0.03, "note": "|log r|>3% 를 부호로 분리"}, meta)
    rows += _rows(r < -0.03, r, -1, "P5_legacy_abs3", {"thr": 0.03, "note": "|log r|>3% 를 부호로 분리"}, meta)
    q99 = r.abs().quantile(.99)
    rows += _rows((r.abs() > q99) & (r > 0), r, +1, "P5_legacy_q99", {"lookahead": True}, meta)
    rows += _rows((r.abs() > q99) & (r < 0), r, -1, "P5_legacy_q99", {"lookahead": True}, meta)
    df = pd.DataFrame(rows)
    if quality is not None and len(df):
        df["quality_flag"] = np.where(quality.reindex(pd.to_datetime(df["candidate_at"])).fillna(False).to_numpy(), "dq_flag_day", None)
    return df


def common_shock_events(r: pd.DataFrame, snapshot_id: str = "", run_id: str = "", k: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """SHOCK_MAP 공통충격일 (rz3_D, KOSPI200 4종 dedup, rz3_A warm-up 이후) — 기존 표 재현 대상: k≥5 dedup = 40 일."""
    RZD, RZA = rz_D(r), rz_A(r)
    valid = RZA.notna().sum(axis=1) >= 15
    F = (RZD.abs() > 3) & valid.values[:, None]
    nd = F.drop(columns=K200).sum(axis=1) + F[K200].any(axis=1).astype(int)
    sgn = np.sign(r["069500"]).astype(int)
    meta = {"series_id": "UNIVERSE", "reference_series_id": None, "asset_class": "universe", "data_snapshot_id": snapshot_id, "run_id": run_id}
    rows = []
    for d in nd.index[nd >= k]:
        rows.append({**meta, "definition_id": "P5_common_shock", "definition_version": DEF_VERSION, "event_family": "price",
                     "parameter_set": json.dumps({"def": "rz3_D", "dedup": "K200", "k": k}), "direction": int(sgn[d]),
                     "candidate_at": d, "event_end": d, "confirmed_at": d, "available_at": d, "score": float(nd[d]), "window": 1, "method": None})
    return pd.DataFrame(rows), nd


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    if not len(df):
        return pd.DataFrame(columns=EVENT_COLS)
    for c in EVENT_COLS:
        if c not in df:
            df[c] = None
    df["event_id"] = [_eid(a, b, c, d) for a, b, c, d in zip(df.series_id, df.definition_id, df.parameter_set, df.candidate_at)]
    for c in ("candidate_at", "confirmed_at", "available_at", "event_end"):
        df[c] = pd.to_datetime(df[c])
    return df[EVENT_COLS].reset_index(drop=True)


def build_price_events(r: pd.DataFrame, tickers: list[str], asset_class: dict[str, str], snapshot_id: str = "", run_id: str = "",
                       quality: dict[str, pd.Series] | None = None, **kw) -> pd.DataFrame:
    parts = [series_price_events(r[t], t, asset_class.get(t, ""), snapshot_id, run_id,
                                 quality=(quality or {}).get(t), **kw) for t in tickers]
    cs, _ = common_shock_events(r, snapshot_id, run_id)
    return finalize(pd.concat(parts + [cs], ignore_index=True))
