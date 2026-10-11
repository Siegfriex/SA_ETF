"""Phase A3 — 상관 사건(Correlation Events) 정의. 모든 사건은 A Result Contract schema(EVENT_COLS)로 반환한다.

정의 (definition_id)
  C1_zero_cross       rho 가 0 선을 넘은 날 (raw). 잡음 교차 포함 — 단독으로 구조변화를 뜻하지 않는다.
  C2_large_change     비겹침 창 Fisher z 차이 |z| > z_thr 인 구간의 첫날 (방향 = Δrho 부호)
  C3_sign_regime      히스테리시스 부호 전환 (|rho|>band 가 persist 일 연속). candidate_at=새 run 첫날, confirmed_at=persist 일째
  C4_multi_window     C3 또는 C2 가 서로 다른 창 ≥2 개에서 ±tol 거래일 안에 같은 방향으로 발생 (창 간 동시 변화)

시점 필드
  candidate_at  사건 후보가 처음 관측 가능한 거래일 (그날 종가 기준)
  confirmed_at  persistence 등 추가 조건이 충족돼 확정된 거래일 (C1·C2 는 = candidate_at)
  available_at  실시간으로 '사건'이라고 말할 수 있는 최초 시점 = confirmed_at
  event_end     사건 구간의 끝 (C2: 초과 구간 마지막 날, C3: 다음 전환 직전 또는 표본 끝)
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import rolling_correlation as RC

EVENT_COLS = ["event_id", "series_id", "reference_series_id", "asset_class", "definition_id", "definition_version",
              "parameter_set", "event_family", "direction", "candidate_at", "confirmed_at", "available_at", "event_end",
              "score", "window", "method", "quality_flag", "data_snapshot_id", "run_id"]
DEF_VERSION = "1.0"


def _eid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def _finish(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    for k, v in meta.items():
        df[k] = v
    if not len(df):
        return pd.DataFrame(columns=EVENT_COLS)
    df["event_id"] = [_eid(r.series_id, r.reference_series_id, r.definition_id, r.parameter_set, r.window, r.method, r.candidate_at)
                      for r in df.itertuples()]
    for c in EVENT_COLS:
        if c not in df:
            df[c] = None
    return df[EVENT_COLS]


def c1_zero_cross(rho: pd.Series, **meta) -> pd.DataFrame:
    zc = RC.zero_crossings(rho)
    df = pd.DataFrame({"candidate_at": zc["candidate_at"], "confirmed_at": zc["candidate_at"], "available_at": zc["candidate_at"],
                       "event_end": zc["candidate_at"], "direction": zc["new_sign"], "score": zc["rho_at"].abs()})
    return _finish(df, {**meta, "definition_id": "C1_zero_cross", "definition_version": DEF_VERSION, "event_family": "correlation",
                        "parameter_set": json.dumps({"band": 0})})


def c2_large_change(rho: pd.Series, z_thr: float = 2.58, **meta) -> pd.DataFrame:
    """meta["window"] 가 비교 간격(비겹침 창)이다."""
    ev = RC.large_change_events(rho, int(meta["window"]), z_thr=z_thr)
    df = pd.DataFrame({"candidate_at": ev["candidate_at"], "confirmed_at": ev["candidate_at"], "available_at": ev["candidate_at"],
                       "event_end": ev["end_at"], "direction": ev["direction"], "score": ev["peak_z"].abs()})
    return _finish(df, {**meta, "definition_id": "C2_large_change", "definition_version": DEF_VERSION, "event_family": "correlation",
                        "parameter_set": json.dumps({"z_thr": z_thr, "compare": "rho_t vs rho_{t-w}"})})


def c3_sign_regime(rho: pd.Series, rule: RC.RegimeRule, **meta) -> pd.DataFrame:
    sr = RC.sign_regimes(rho, rule)
    sr = sr[~sr["is_initial"]].reset_index(drop=True) if len(sr) else sr
    if len(sr):
        nxt = list(sr["candidate_at"].iloc[1:]) + [rho.dropna().index[-1]]
        df = pd.DataFrame({"candidate_at": sr["candidate_at"], "confirmed_at": sr["confirmed_at"], "available_at": sr["confirmed_at"],
                           "event_end": nxt, "direction": sr["to_sign"], "score": sr["prev_peak_abs"]})
    else:
        df = pd.DataFrame(columns=["candidate_at", "confirmed_at", "available_at", "event_end", "direction", "score"])
    return _finish(df, {**meta, "definition_id": "C3_sign_regime", "definition_version": DEF_VERSION, "event_family": "correlation",
                        "parameter_set": json.dumps({"band": rule.band, "persist": rule.persist})})


def c4_multi_window(events: pd.DataFrame, tol: int, trading_index: pd.DatetimeIndex, min_windows: int = 2,
                    source_defs=("C3_sign_regime", "C2_large_change")) -> pd.DataFrame:
    """같은 series·method·정의 안에서 서로 다른 창의 사건이 ±tol 거래일 안에 같은 방향으로 모이면 1 건.
    candidate_at = 묶음의 가장 이른 candidate, confirmed_at = 묶음이 min_windows 개 창을 처음 채운 시점의 confirmed_at."""
    pos = {d: i for i, d in enumerate(trading_index)}
    out = []
    for (s, ref, m, d, ps), g in events[events.definition_id.isin(source_defs)].groupby(
            ["series_id", "reference_series_id", "method", "definition_id", "parameter_set"]):
        g = g.sort_values("candidate_at").reset_index(drop=True)
        used = set()
        for i, row in g.iterrows():
            if i in used:
                continue
            p0 = pos[row.candidate_at]
            members = g[(g.direction == row.direction) & g.candidate_at.map(lambda x: abs(pos[x] - p0) <= tol)]
            members = members[~members.index.isin(used)]
            if members.window.nunique() >= min_windows:
                used |= set(members.index)
                # 서로 다른 창이 min_windows 개 모이는 confirmed 시점
                firsts = members.sort_values("confirmed_at").drop_duplicates("window")
                conf = firsts["confirmed_at"].iloc[min_windows - 1]
                out.append({"series_id": s, "reference_series_id": ref, "method": m, "asset_class": row.asset_class,
                            "candidate_at": members.candidate_at.min(), "confirmed_at": conf, "available_at": conf,
                            "event_end": members.event_end.max(), "direction": row.direction,
                            "score": float(members.window.nunique()), "window": "+".join(map(str, sorted(members.window.unique()))),
                            "parameter_set": json.dumps({"source": d, "source_params": json.loads(ps), "tol": tol, "min_windows": min_windows}),
                            "quality_flag": ";".join(sorted({q for q in members.quality_flag if isinstance(q, str) and q})) or None,
                            "data_snapshot_id": row.data_snapshot_id, "run_id": row.run_id})
    df = pd.DataFrame(out)
    return _finish(df, {"definition_id": "C4_multi_window", "definition_version": DEF_VERSION, "event_family": "correlation"}) \
        if len(df) else pd.DataFrame(columns=EVENT_COLS)


def build_correlation_events(r: pd.DataFrame, ref: str, tickers: list[str], asset_class: dict[str, str],
                             windows=(30, 60, 120), methods=("pearson", "spearman"), min_periods_frac: float = 0.9,
                             rules: list[RC.RegimeRule] | None = None, z_thr: float = 2.58, c4_tol: int = 10,
                             snapshot_id: str = "", run_id: str = "", quality: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """quality: ticker -> bool Series(날짜) — True 인 날이 rolling 창 안에 있으면 quality_flag='dq_in_window'."""
    rules = rules or [RC.RegimeRule(0.1, 10)]
    evs = []
    for t in tickers:
        q = None
        for w in windows:
            if quality and t in quality:
                q = quality[t].reindex(r.index).fillna(False).astype(int).rolling(w, min_periods=1).max().astype(bool)
            for m in methods:
                rho = RC.rolling_corr(r[t], r[ref], w, m, min_periods=int(np.ceil(w * min_periods_frac)))["rho"]
                meta = dict(series_id=t, reference_series_id=ref, asset_class=asset_class.get(t, ""), window=w, method=m,
                            data_snapshot_id=snapshot_id, run_id=run_id)
                parts = [c1_zero_cross(rho, **meta), c2_large_change(rho, z_thr, **meta)]
                parts += [c3_sign_regime(rho, ru, **meta) for ru in rules]
                for p in parts:
                    if q is not None and len(p):
                        p["quality_flag"] = pd.Series(np.where(q.reindex(pd.to_datetime(p["candidate_at"])).to_numpy(), "dq_in_window", None), index=p.index, dtype=object)
                    evs.append(p)
    ev = pd.concat([e for e in evs if len(e)], ignore_index=True)
    ev = pd.concat([ev, c4_multi_window(ev, c4_tol, r.index)], ignore_index=True)
    for c in ("candidate_at", "confirmed_at", "available_at", "event_end"):
        ev[c] = pd.to_datetime(ev[c])
    return ev


def c5_method_concordant(events: pd.DataFrame, trading_index: pd.DatetimeIndex, tol: int = 5,
                         source_def: str = "C3_sign_regime") -> pd.DataFrame:
    """C5 — 추정량 일치 부호전환: 같은 series·window·규칙의 C3 가 Pearson 과 Spearman 양쪽에서 ±tol 거래일 안에 같은 방향으로 발생.
    Pearson 만의 전환은 소수의 큰 수익률(꼬리)에, Spearman 만의 전환은 순위·동률(0 수익률)에 좌우될 수 있다 → 둘 다 보이면
    '측정 방법에 덜 의존하는' 후보. candidate_at = 이른 쪽, confirmed_at = 늦은 쪽의 confirmed_at."""
    pos = {d: i for i, d in enumerate(trading_index)}
    src = events[events.definition_id == source_def]
    out = []
    for (s, ref, w, ps), g in src.groupby(["series_id", "reference_series_id", "window", "parameter_set"]):
        gp, gs = g[g.method == "pearson"], g[g.method == "spearman"]
        used = set()
        for _, a in gp.iterrows():
            cand = gs[(gs.direction == a.direction) & ~gs.index.isin(used)]
            if not len(cand):
                continue
            d = cand.candidate_at.map(lambda x: abs(pos[x] - pos[a.candidate_at]))
            if d.min() <= tol:
                b = cand.loc[d.idxmin()]; used.add(d.idxmin())
                out.append({"series_id": s, "reference_series_id": ref, "asset_class": a.asset_class, "window": w, "method": "pearson+spearman",
                            "candidate_at": min(a.candidate_at, b.candidate_at), "confirmed_at": max(a.confirmed_at, b.confirmed_at),
                            "available_at": max(a.confirmed_at, b.confirmed_at), "event_end": max(a.event_end, b.event_end),
                            "direction": a.direction, "score": float(np.nanmean([a.score, b.score])),
                            "parameter_set": json.dumps({"source": source_def, "source_params": json.loads(ps), "tol": tol}),
                            "quality_flag": a.quality_flag if isinstance(a.quality_flag, str) else b.quality_flag if isinstance(b.quality_flag, str) else None,
                            "data_snapshot_id": a.data_snapshot_id, "run_id": a.run_id})
    df = pd.DataFrame(out)
    return _finish(df, {"definition_id": "C5_method_concordant", "definition_version": DEF_VERSION, "event_family": "correlation"}) \
        if len(df) else pd.DataFrame(columns=EVENT_COLS)
