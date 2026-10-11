"""Track A 시각화 — 가격·수익률·상관을 *같은 시간축*에 놓는 그림들. 모든 축에 단위·대상 ETF 를 적는다."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import rolling_correlation as RC

WCOL = {30: "#e67e22", 60: "#2a6fdb", 120: "#2c3e50", 250: "#95a5a6"}
UP, DN = "#c0392b", "#1f5fa8"
PERIOD_SHADE = {"val_2025": "#f4d03f", "holdout_2026": "#e59866"}


def shade_periods(ax, alpha=.10):
    for k, col in PERIOD_SHADE.items():
        a, b = RC.PERIODS[k]
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=col, alpha=alpha, lw=0)


def rho_panel(ax, rhos: dict[int, pd.Series], title: str = "", band_window: int | None = None, events: pd.DataFrame | None = None,
              legend=True):
    """rhos: window -> rho Series. band_window 를 주면 그 창의 독립 가정 ±1.96/√w 띠를 회색으로."""
    if band_window:
        b = RC.null_band(band_window)
        ax.axhspan(-b, b, color="#bdc3c7", alpha=.35, lw=0, label=f"±1.96/√{band_window} (독립 가정 오차대)")
    for w, s in rhos.items():
        ax.plot(s.index, s, color=WCOL.get(w, None), lw=.9 if w == 30 else 1.3, label=f"{w}일")
    ax.axhline(0, color="k", lw=.6)
    if events is not None and len(events):
        for e in events.itertuples():
            ax.axvline(e.candidate_at, color=UP if e.direction > 0 else DN, lw=.8, ls=":", alpha=.9)
            if pd.notna(e.confirmed_at) and e.confirmed_at != e.candidate_at:
                ax.axvline(e.confirmed_at, color=UP if e.direction > 0 else DN, lw=.8, ls="-", alpha=.5)
    ax.set_ylim(-1, 1); ax.set_ylabel("rolling ρ")
    if title:
        ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(fontsize=7, ncol=4, loc="lower left")


def pair_overview(close: pd.DataFrame, r: pd.DataFrame, ref: str, t: str, rhos: dict[int, pd.Series], labels: dict[str, str],
                  events: pd.DataFrame | None = None, start=None, end=None, vol_window: int = 20, figsize=(14, 9)):
    """4 단: 정규화 가격(두 ETF) · 일간 로그수익률(두 ETF) · 20일 연환산 변동성 · rolling ρ(30/60/120). sharex."""
    sl = slice(start, end)
    fig, ax = plt.subplots(4, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1.2, 1, .8, 1.4]})
    c = close.loc[sl, [ref, t]]
    (c / c.iloc[0]).plot(ax=ax[0], color=["#7f8c8d", "#c0392b"], lw=1.1)
    ax[0].set_ylabel("정규화 가격 (시작=1)"); ax[0].legend([labels[ref], labels[t]], fontsize=8)
    rr = r.loc[sl, [ref, t]] * 100
    ax[1].plot(rr.index, rr[ref], color="#7f8c8d", lw=.6, label=labels[ref]); ax[1].plot(rr.index, rr[t], color="#c0392b", lw=.6, alpha=.8, label=labels[t])
    ax[1].set_ylabel("일간 로그수익률 (%)"); ax[1].legend(fontsize=8)
    v = r.loc[:, [ref, t]].rolling(vol_window).std().loc[sl] * np.sqrt(252) * 100
    ax[2].plot(v.index, v[ref], color="#7f8c8d", lw=.9); ax[2].plot(v.index, v[t], color="#c0392b", lw=.9)
    ax[2].set_ylabel(f"{vol_window}일 변동성\n(연환산 %)"); ax[2].set_yscale("log")
    rho_panel(ax[3], {w: s.loc[sl] for w, s in rhos.items()}, band_window=60,
              events=None if events is None else events[(events.candidate_at >= (start or "1900")) & (events.candidate_at <= (end or "2100"))])
    for a in ax:
        shade_periods(a)
    ax[3].xaxis.set_major_locator(mdates.YearLocator() if start is None else mdates.AutoDateLocator())
    fig.suptitle(f"{labels[t]} vs {labels[ref]} — 가격 · 수익률 · 변동성 · rolling ρ (같은 시간축; 노랑=2025 검증, 주황=2026 holdout 관찰)", y=1.0, fontsize=11)
    fig.tight_layout()
    return fig


def case_plot(close: pd.DataFrame, r: pd.DataFrame, ref: str, t: str, rhos: dict[int, pd.Series], labels: dict[str, str],
              center, half: int = 60, c_events: pd.DataFrame | None = None, p_events: pd.DataFrame | None = None, title: str = ""):
    """사례 확대: center ± half 거래일. 상관 사건(점선=candidate, 실선=confirmed)과 가격 사건(▲▼)을 같이 표시."""
    idx = r.index
    i = idx.get_indexer([pd.Timestamp(center)], method="nearest")[0]
    a, b = idx[max(0, i - half)], idx[min(len(idx) - 1, i + half)]
    fig, ax = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.3]})
    c = close.loc[a:b, [ref, t]]
    (c / c.iloc[0]).plot(ax=ax[0], color=["#7f8c8d", "#c0392b"], lw=1.2); ax[0].legend([labels[ref], labels[t]], fontsize=8)
    ax[0].set_ylabel("정규화 가격")
    rr = r.loc[a:b, [ref, t]] * 100
    wbar = 0.4
    ax[1].bar(rr.index - pd.Timedelta(days=wbar / 2), rr[ref], width=wbar, color="#7f8c8d", label=labels[ref])
    ax[1].bar(rr.index + pd.Timedelta(days=wbar / 2), rr[t], width=wbar, color="#c0392b", label=labels[t])
    ax[1].set_ylabel("일간 로그수익률 (%)"); ax[1].legend(fontsize=8); ax[1].axhline(0, c="k", lw=.5)
    if p_events is not None and len(p_events):
        pe = p_events[(p_events.candidate_at >= a) & (p_events.candidate_at <= b)]
        for e in pe.itertuples():
            y = r.at[e.candidate_at, e.series_id] * 100 if e.series_id in r.columns else 0
            ax[1].scatter([e.candidate_at], [y], marker="^" if e.direction > 0 else "v", s=70, color=UP if e.direction > 0 else DN,
                          edgecolor="k", zorder=5)
    ce = None if c_events is None else c_events[(c_events.candidate_at >= a) & (c_events.candidate_at <= b)]
    rho_panel(ax[2], {w: s.loc[a:b] for w, s in rhos.items()}, band_window=60, events=ce)
    ax[2].axvline(pd.Timestamp(center), color="k", lw=1.2, alpha=.3)
    fig.suptitle(title or f"{labels[t]} vs {labels[ref]} @ {pd.Timestamp(center).date()}", fontsize=11)
    fig.tight_layout()
    return fig
