"""Phase A3 — 상관 사건 C 와 가격 사건 P 의 정렬·매칭·우연 수준 비교 (결정적; 난수는 seed 고정).

시간차 lag = pos(P.candidate_at) − pos(C.<c_time>)  [거래일]
  lag = 0   co_occur      같은 거래일
  lag > 0   corr_leads    상관 사건이 먼저 (가격 사건이 뒤)
  lag < 0   price_leads   가격 사건이 먼저 (상관이 뒤에 반응)
  매칭 없음  corr_only / price_only
매칭은 |lag| ≤ max_lag 안에서 **일대일 greedy** (|lag| 오름차순, 동률이면 이른 C 우선). 한 P 가 여러 C 에 중복 계산되지 않는다.
품질: C 또는 P 의 quality_flag 가 있으면 dq_uncertain=True (유형은 유지하되 해석 불확실로 별도 집계).

**근접은 인과가 아니다.** 또한 Ground Truth 가 없으므로 hit rate 는 '탐지 정확도'가 아니라 두 사건 집합의 시간적 중첩 정도다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pos(idx: pd.DatetimeIndex, dates: pd.Series) -> np.ndarray:
    m = pd.Series(np.arange(len(idx)), index=idx)
    return m.reindex(pd.to_datetime(dates)).to_numpy()


def match_events(C: pd.DataFrame, P: pd.DataFrame, idx: pd.DatetimeIndex, max_lag: int,
                 c_time: str = "candidate_at", p_time: str = "candidate_at") -> pd.DataFrame:
    """반환: C·P 각 행에 대해 1 행 (매칭 쌍은 1 행으로 합침). 열: c_event_id, p_event_id, c_date, p_date, lag, match_type, dq_uncertain."""
    cp, pp = _pos(idx, C[c_time]), _pos(idx, P[p_time])
    pairs = []
    for i, a in enumerate(cp):
        if np.isnan(a):
            continue
        d = pp - a
        for j in np.where(np.abs(d) <= max_lag)[0]:
            pairs.append((abs(d[j]), a, i, j, int(d[j])))
    pairs.sort()
    used_c, used_p, rows = set(), set(), []
    for _, _, i, j, lag in pairs:
        if i in used_c or j in used_p:
            continue
        used_c.add(i); used_p.add(j)
        mt = "co_occur" if lag == 0 else ("corr_leads" if lag > 0 else "price_leads")
        rows.append((i, j, lag, mt))
    out = []
    Cq = C["quality_flag"].notna().to_numpy() if "quality_flag" in C else np.zeros(len(C), bool)
    Pq = P["quality_flag"].notna().to_numpy() if "quality_flag" in P else np.zeros(len(P), bool)
    for i, j, lag, mt in rows:
        out.append({"c_event_id": C["event_id"].iat[i], "p_event_id": P["event_id"].iat[j], "c_date": C[c_time].iat[i],
                    "p_date": P[p_time].iat[j], "lag": lag, "match_type": mt, "dq_uncertain": bool(Cq[i] or Pq[j])})
    for i in set(range(len(C))) - used_c:
        out.append({"c_event_id": C["event_id"].iat[i], "p_event_id": None, "c_date": C[c_time].iat[i], "p_date": pd.NaT,
                    "lag": np.nan, "match_type": "corr_only", "dq_uncertain": bool(Cq[i])})
    for j in set(range(len(P))) - used_p:
        out.append({"c_event_id": None, "p_event_id": P["event_id"].iat[j], "c_date": pd.NaT, "p_date": P[p_time].iat[j],
                    "lag": np.nan, "match_type": "price_only", "dq_uncertain": bool(Pq[j])})
    return pd.DataFrame(out, columns=["c_event_id", "p_event_id", "c_date", "p_date", "lag", "match_type", "dq_uncertain"])


def summarize_matches(m: pd.DataFrame, n_c: int, n_p: int) -> dict:
    vc = m["match_type"].value_counts()
    matched = int(vc.get("co_occur", 0) + vc.get("corr_leads", 0) + vc.get("price_leads", 0))
    lags = m["lag"].dropna()
    return {"n_C": n_c, "n_P": n_p, "n_matched": matched,
            "share_C_matched": matched / n_c if n_c else np.nan, "share_P_matched": matched / n_p if n_p else np.nan,
            "n_co_occur": int(vc.get("co_occur", 0)), "n_corr_leads": int(vc.get("corr_leads", 0)),
            "n_price_leads": int(vc.get("price_leads", 0)), "n_corr_only": int(vc.get("corr_only", 0)),
            "n_price_only": int(vc.get("price_only", 0)), "n_dq_uncertain": int(m["dq_uncertain"].sum()),
            "lag_median": float(lags.median()) if len(lags) else np.nan, "lag_mean": float(lags.mean()) if len(lags) else np.nan}


def proximity_count(cpos: np.ndarray, ppos: np.ndarray, max_lag: int) -> int:
    """C 중 ±max_lag 안에 P 가 하나라도 있는 개수 (다대일 허용 — 귀무비교용 단순 통계량)."""
    if len(cpos) == 0 or len(ppos) == 0:
        return 0
    ps = np.sort(ppos)
    lo = np.searchsorted(ps, cpos - max_lag, side="left")
    hi = np.searchsorted(ps, cpos + max_lag, side="right")
    return int((hi > lo).sum())


def circular_shift_test(C: pd.DataFrame, P: pd.DataFrame, idx: pd.DatetimeIndex, max_lag: int, B: int = 1000,
                        seed: int = 20261011, c_time: str = "candidate_at", min_shift: int | None = None) -> dict:
    """H0: C 와 P 의 시간 배치는 서로 무관하다 (각자의 군집 구조는 유지).
    P 의 위치 전체를 같은 offset u 만큼 원형 이동(mod n) → P 내부 간격·군집 보존. u ~ U{min_shift, …, n−min_shift}.
    통계량 T = proximity_count. p = (1 + #{T_null ≥ T_obs}) / (B + 1)  (단측: 우연보다 가까운가)."""
    n = len(idx)
    cpos = _pos(idx, C[c_time]); cpos = cpos[~np.isnan(cpos)].astype(int)
    ppos = _pos(idx, P["candidate_at"]); ppos = ppos[~np.isnan(ppos)].astype(int)
    t_obs = proximity_count(cpos, ppos, max_lag)
    ms = min_shift if min_shift is not None else 2 * max_lag + 1
    rng = np.random.default_rng(seed)
    us = rng.integers(ms, n - ms, size=B) if n - ms > ms else np.zeros(B, int)
    null = np.array([proximity_count(cpos, (ppos + u) % n, max_lag) for u in us])
    return {"T_obs": t_obs, "T_null_mean": float(null.mean()), "T_null_p95": float(np.quantile(null, .95)),
            "p_value": float((1 + (null >= t_obs).sum()) / (B + 1)), "B": B, "seed": seed, "n_C": len(cpos), "n_P": len(ppos),
            "ratio_obs_null": float(t_obs / null.mean()) if null.mean() > 0 else np.nan}


def circular_shift_test_fast(cpos: np.ndarray, ppos: np.ndarray, n: int, max_lag: int, B: int = 1000, seed: int = 20261011,
                             min_shift: int | None = None) -> dict:
    """circular_shift_test 와 같은 귀무가설·통계량을 B 개 shift 에 대해 한 번의 searchsorted 로 계산 (행마다 2n 오프셋)."""
    cpos = np.asarray(cpos, int); ppos = np.asarray(ppos, int)
    t_obs = proximity_count(cpos, ppos, max_lag)
    if len(cpos) == 0 or len(ppos) == 0:
        return {"T_obs": t_obs, "T_null_mean": 0.0, "T_null_p95": 0.0, "p_value": 1.0, "B": B, "seed": seed,
                "n_C": len(cpos), "n_P": len(ppos), "ratio_obs_null": np.nan}
    ms = min_shift if min_shift is not None else 2 * max_lag + 1
    rng = np.random.default_rng(seed)
    us = rng.integers(ms, n - ms, size=B)
    off = (np.arange(B) * 4 * n)[:, None]
    P = np.sort((ppos[None, :] + us[:, None]) % n, axis=1) + off
    flat = P.ravel()
    lo = np.searchsorted(flat, (cpos[None, :] - max_lag + off).ravel(), side="left").reshape(B, -1)
    hi = np.searchsorted(flat, (cpos[None, :] + max_lag + off).ravel(), side="right").reshape(B, -1)
    null = (hi > lo).sum(axis=1)
    return {"T_obs": t_obs, "T_null_mean": float(null.mean()), "T_null_p95": float(np.quantile(null, .95)),
            "p_value": float((1 + (null >= t_obs).sum()) / (B + 1)), "B": B, "seed": seed, "n_C": len(cpos), "n_P": len(ppos),
            "ratio_obs_null": float(t_obs / null.mean()) if null.mean() > 0 else np.nan}


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q 값 (탐색적 다중비교 보정)."""
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out
