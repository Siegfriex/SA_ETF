"""Phase 00.5 (C2-B1): 20 ETF Final Notebook generator. Deterministic.

수치는 build 시점에 src/final_figs.compute() 로 사전계산해 markdown 에 박는다 (GitHub 에서 출력 없이 읽힘).
같은 함수가 notebook 실행 시 figure 를 그리므로 텍스트와 그림의 정의가 일치한다.
산출: notebooks/final_eda/{slot:02d}_{ticker}_final_eda.ipynb, reports/tables/etf_final_stats.csv

usage: python scripts/build_final_notebooks.py [--only TICKER ...]
env: ETF_RAW_DIR (raw 경로 override)
"""
import argparse, hashlib, re, sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import final_figs as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="*")
args = ap.parse_args()


def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{d}f}%"


def f2(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def card_section(tk, slot, head):
    p = ROOT / "reports" / "ETF_CARDS" / f"{slot:02d}_{tk}.md"
    if not p.exists():
        return [], ""
    txt = p.read_text()
    m = re.search(rf"## {head}\n(.*?)(\n## |\Z)", txt, re.S)
    lines = [l for l in (m.group(1).splitlines() if m else []) if l.strip().startswith("-")]
    return lines, txt


def limitations(tk, row, S, card_txt, dq):
    """ETF별 '결론 불가' 근거 풀. key → 문장. 출처: dq_flags.csv · ETF 카드 · universe.csv."""
    L = {}
    d = dq[dq.ticker == tk]
    n_ohlc = int((d.flag == "ohlc_violation").sum())
    spikes = sorted(d[d.flag == "isolated_spike_reversal"].date.unique())
    if n_ohlc:
        L["ohlc"] = (f"dq_flags.csv 에 OHLC 위반(Close>High 등) {n_ohlc}일이 있다 (D-013: ≤4원, 원인 UNRESOLVED). "
                     "종가 수익률에는 영향이 미미하지만 High/Low 기반 range 해석은 이 notebook 에서 하지 않는다.")
    if spikes:
        L["spike"] = (f"dq_flags.csv 가 {', '.join(spikes)} 을 isolated_spike_reversal(다음날 되돌림) 후보로 표시한다. "
                      "이 날의 극단값이 기초자산 움직임인지 ETF 가격 괴리인지는 NAV 없이 구분할 수 없다.")
    if "low_liquidity" in card_txt:
        L["liq"] = (f"카드가 low_liquidity 로 표시한 ETF다 (median volume {S['vol_med']:,.0f}주). "
                    "큰 일간 변화와 거래량 급증이 NAV 괴리·호가 공백 때문인지 펀더멘털 뉴스 때문인지 OHLCV 만으로 가를 수 없다.")
    li = row["leverage_inverse"]
    if li in ("2x", "-1x"):
        L["gear"] = (f"{li} 일간 목표배율 상품이라 다기간 수익률은 기초지수 × {li} 가 아니다 (일일 리밸런싱·경로의존). "
                     "누적 수익·낙폭 차이를 '추적 실패'로 해석할 수 없다.")
    if row["asset_scope"] == "foreign_equity" or row["asset_scope"] in ("commodity", "fx"):
        L["async"] = (f"기초시장 거래시간이 KRX 와 다르다 ({row['trading_hours_note'] or '해외/선물 시장'}). "
                      "같은 날짜 수익률의 상관·β 는 비동기로 과소추정될 수 있으며 (CL-15: 2일 비중첩 상관도 낮음) lead-lag 원인 분해는 불가.")
    if row["currency_hedge"].startswith("hedged"):
        L["hedge"] = "환헤지(H) 상품이라 원화수익률 = 기초 + 헤지비용/basis. 헤지 비용·롤 효과를 이 데이터로 분리하지 못한다."
    elif row["currency_hedge"] == "unhedged":
        L["hedge"] = "환노출(비헤지) 상품이라 원화수익률에 USD/KRW 변동이 섞여 있다. 환율 데이터 없이 지수 효과와 환 효과를 분리할 수 없다."
    if "선물" in row["name"] or "선물" in row["benchmark"]:
        L["fut"] = "선물형 상품이라 만기 롤오버·basis 가 경로에 섞인다. 현물 가격 회복이 ETF 회복을 뜻하지 않는다."
    if "추정" in row["benchmark"]:
        L["bm"] = f"기초지수 명칭이 '{row['benchmark']}' 로 추정 상태다. 지수 데이터가 없어 추적오차를 측정할 수 없고 benchmark 는 ETF proxy 다."
    if tk == "153130":
        L["tick"] = (f"초저변동(연 {pct(S['ann_vol'],2)}) + 틱 이산성(0수익률 비율 {pct(S['zero_ret_ratio'])})으로 분모가 작아 "
                     "robust-z·첨도가 폭발한다 (CL-25: −0.47% 가 rz −34). 통계적 극단 ≠ 경제적 충격.")
    if tk == "261240":
        t = S["trim_anom"]
        L["anom"] = (f"2019-03-14/15 ±20% 는 USD/KRW(+0.21%)와 무관한 price_anomaly_unverified (CL-05). 이 2일 제거 시 "
                     f"ann_vol {pct(S['ann_vol'])}→{pct(t['ann_vol'])}, ex_kurt {f2(S['ex_kurt'],1)}→{f2(t['ex_kurt'],1)}, "
                     f"069500 상관 {f2(S['mkt_corr'],3)}→{f2(t['mkt_corr'],3)}, 2026/2019-25 vol 비율 {f2(S['vol_ratio_26'],2)}×→{f2(t['vol_ratio_26'],2)}× (기준기간 σ 를 2일이 부풀려 축소처럼 보이지만 제거 시 확대 — D 0120) (DOCX 의 '쌍 제외 kurt≈3.9' 와 trim3 {f2(S['ex_kurt_trim3'],1)} 은 제거 방식이 다르다).")
    if tk == "148070":
        L["proxyart"] = (f"universe.csv 의 benchmark proxy 가 153130(단기채)이라 proxy β={f2(S['proxy_beta'])}, ρ={f2(S['proxy_corr'],3)} 이 나온다. "
                         "DOCX 카드의 'corr 0.100 · beta 1.87' 은 이 proxy 산출물이다. β 는 분모(153130 분산, 연 0.32%)가 극히 작아 불안정한 artifact 다: "
                         f"153130 의 |r| 상위 2일(2024-01-30/31, 각 ≈0.5%)을 빼면 proxy β 가 {f2(S['proxy_beta'])} → {f2(S['proxy_beta_x2'])} 로 뛴다 (R² {f2(S['proxy_r2'])}). "
                         "즉 1.87 은 경제적 민감도가 아니라 두 날짜 레버리지 점의 산물이다. "
                         f"시장 기준(069500)으로는 β={f2(S['mkt_beta'],3)}, ρ={f2(S['mkt_corr'],3)}.")
    if row["structural_notes"]:
        L["struct"] = "universe.csv structural_notes: " + row["structural_notes"].replace("\n", " ")[:260]
    return L


def pick(L, keys, fallback):
    out = [L[k] for k in keys if k in L]
    return " ".join(out[:2]) if out else fallback


def build(tk, slot, row, S, dq):
    ident, card_txt = card_section(tk, slot, "정체성")
    L = limitations(tk, row, S, card_txt, dq)
    proxy_note = "" if row["benchmark_proxy_ticker"] else " (universe.csv proxy 비어 있음 → 069500 명시 사용)"
    vy = S["vol_by_year"]
    vmax_y = max(vy, key=vy.get); vmin_y = min(vy, key=vy.get)
    ep = S["episodes"]
    e1 = ep[0]
    X2 = X4 = ""
    if tk in ("069500", "102110", "122630", "114800"):
        X2 = (f" **2026-07-31 {pct(S['jul31'],1)}** (log, 단순수익률 {pct(np.expm1(S['jul31']),1)}) 은 raw 실제 값이다 — 같은 날 102110 도 {pct(S['jul31_102110'],1)} 로 동일 방향·크기라 "
              "개별 ETF 데이터 오류가 아니라 KOSPI200 공통 움직임이다 (D: ROBUSTNESS_MATRIX). 전후 07-29 / 08-03 대폭 하락과 묶인 A4 창이다.")
    if tk in ("132030", "261220"):
        X4 = (f" 2026/2019-25 변동성 비율 {f2(S['vol_ratio_26'])}× — 2026 변동성 확대는 국내주식만의 현상이 아니며 원자재에서도 나타난다 "
              "(reports/ROBUSTNESS_MATRIX.csv). 다만 국내 지수형(≈3×)과 크기·시점이 다르다.")
    if tk == "261240":
        X4 = (f" price_anomaly 2일(2019-03-14/15) 제거 시 ann_vol {pct(S['ann_vol'],2)} → {pct(S['trim_anom']['ann_vol'],2)} "
              f"— 전체기간 변동성의 상당 부분이 검증되지 않은 2일에서 온다. 2026/2019-25 비율도 {f2(S['vol_ratio_26'],2)}× → {f2(S['trim_anom']['vol_ratio_26'],2)}× 로 '축소'에서 '확대'로 뒤집힌다.")
    dq_row = dq_c1 = dag = c8 = ""
    if tk == "261240":
        t = S["trim_anom"]
        dq_row = (f"\n| **DQ 2일 제외** {pct(t['ann_vol'])} | — | **{f2(t['vol_ratio_26'])}×** | {f2(t['ex_kurt'],1)} | — | — | — | ρ {f2(t['mkt_corr'],3)} | — |"
                  "\n\n> 위 첫 행은 price_anomaly_unverified 2일(2019-03-14/15) 포함, 둘째 행은 제외. **해석은 둘째 행 기준** — 2026 은 축소가 아니라 확대다 (D 0120/0123).")
        dq_c1 = f" — **DQ 2일 제외 시 {pct(t['ann_vol'])}, {f2(t['vol_ratio_26'])}× (확대)**"
    if S.get("proxy") == "153130":
        dag = "†"
        dq_row += "\n\n† proxy β 는 153130(단기채, 연 0.32%) 분모가 극소해 불안정한 artifact (R² ≈ 0.01). 시장 관계는 market 열을 본다."
    if row["asset_scope"] != "domestic_equity":
        c8 = " — 비국내주식이라 KOSPI200 대비 누적차는 참고용 (결론 강등)"
    cells = []
    md = lambda s: cells.append(nbformat.v4.new_markdown_cell(s.strip()))
    code = lambda s: cells.append(nbformat.v4.new_code_cell(s.strip()))

    md(f"""
# {slot:02d} · {tk} {row['name']} — Final EDA (Phase 00.5)

> 기간 {S['start']} ~ {S['end']} · n={S['n_obs']} 일간 로그수익률 (FDR Close, 분배 소급조정 추정 · DEC-3) · UNIVERSE_STATUS=PROVISIONAL.
> 이 notebook 의 모든 수치는 `scripts/build_final_notebooks.py` 가 `src/final_figs.compute()` 로 사전계산해 적었고, 아래 코드 셀이 같은 함수로 그림을 다시 그린다.

## 왜 이런 ETF인가
| 항목 | 값 |
|---|---|
| asset_scope / category | {row['asset_scope']} / {row['category']} |
| benchmark (상품 기초) | {row['benchmark']} |
| 배율 / 환헤지 / 복제 | {row['leverage_inverse']} / {row['currency_hedge']} / {row['replication']} |
| peer_group | {row['peer_group']} |
| 비교 기준 ① proxy | {S['proxy']} {S['proxy_name']}{proxy_note} |
| 비교 기준 ② market reference | {S['market']} {S['market_name']} |

카드(reports/ETF_CARDS/{slot:02d}_{tk}.md) 정체성 요약:
{chr(10).join(ident) if ident else '- (카드 없음)'}

## 핵심 수치
| ann_vol | 2026 vol | 2026 / 2019-25 | ex_kurt (trim3) | Q01 / Q99 | Max DD | β·ρ vs proxy | β·ρ vs market | |rz60|>3 (+/−) |
|---|---|---|---|---|---|---|---|---|
| {pct(S['ann_vol'])} | {pct(S['vol_2026'])} | {f2(S['vol_ratio_26'])}× | {f2(S['ex_kurt'],1)} ({f2(S['ex_kurt_trim3'],1)}) | {pct(S['q01'],2)} / {pct(S['q99'],2)} | {pct(S['max_dd'])} ({S['max_dd_date']}) | {f2(S['proxy_beta'])}{dag} · {f2(S['proxy_corr'],3)} | {f2(S['mkt_beta'])} · {f2(S['mkt_corr'],3)} | {S['n_ext_pos']} / {S['n_ext_neg']} |{dq_row}

정의: ann_vol=std·√252 · ex_kurt=Fisher(편향보정) · trim3=|r| 상위 3일 제거 · rz60=(r−직전60일 median)/(1.4826·직전60일 MAD), shift(1) · β/ρ=동일일 OLS·Pearson.
""")
    code(f"""
import sys
from pathlib import Path
ROOT = next(q for q in [Path.cwd(), *Path.cwd().parents] if (q / "config" / "universe.csv").exists())
sys.path.insert(0, str(ROOT))
%matplotlib inline
import pandas as pd
from src import final_figs as F
TK = "{tk}"
c = F.Ctx(TK)
S = F.compute(TK, c.close, c.vol, c.R)
pd.Series({{k: v for k, v in S.items() if not isinstance(v, (dict, list))}}).to_frame("value")
""")

    def figblock(n, key, title, measures, see, why, cannot):
        md(f"## Fig {n}. {title}")
        code(f"F.{dict(F.FIGS)[key].__name__}(c)")
        md(f"""
**WHAT THIS FIGURE MEASURES** — {measures}

**WHAT WE SEE** — {see}

**WHY IT MATTERS FINANCIALLY** — {why}

**WHAT WE CANNOT CONCLUDE** — {cannot}
""")

    figblock(1, "01_price_drawdown", "가격(로그축)과 고점 대비 낙폭",
             "위: FDR Close 로그축 (같은 높이 = 같은 % 변화). 아래: 직전 최고가 대비 낙폭 %. 주황 띠 = config/event_anchors.csv 공통충격 anchor.",
             f"누적 로그수익률 {pct(S['cum_logret'])}. 최대낙폭 {pct(S['max_dd'])} (저점 {S['max_dd_date']}, 고점 {e1['peak']}, 회복 {e1['recovered']}). 마지막 날 낙폭 {pct(S['last_dd'])}.",
             "낙폭 깊이와 회복 시간은 변동성 한 숫자가 보여주지 못하는 '보유자가 실제로 겪는 손실 경로'다. 같은 변동성이라도 회복 기간이 길면 자금 회수 위험이 크다.",
             pick(L, ["gear", "fut", "anom", "tick"], "낙폭이 기초지수 하락인지 분배·추적 차이인지는 기초지수/NAV 없이 분리하지 못한다. 미래 낙폭 확률을 이 경로 하나로 추정할 수 없다."))
    figblock(2, "02_return_ts", "일간 수익률과 조건부 ±3σ 밴드",
             "막대 = 일간 로그수익률 (빨강 +, 파랑 −). 검은 선 = 직전 60일 표준편차 ×3 (shift(1), 당일 미포함). 회색 띠 = 짝수 연도.",
             f"최대 상승 {pct(S['max_up'],2)} ({S['max_up_date']}), 최대 하락 {pct(S['max_dn'],2)} ({S['max_dn_date']}). 변동성 최고 연도 {vmax_y} ({pct(vy[vmax_y])}), 최저 {vmin_y} ({pct(vy[vmin_y])}). 밴드 폭이 시기마다 달라 변동성 군집이 보인다." + X2,
             "고정 절대 threshold 대신 조건부(직전 변동성 대비) 기준을 써야 하는 근거다. 같은 −3% 도 저변동기에는 극단, 고변동기에는 평범하다.",
             pick(L, ["spike", "anom", "tick", "liq"], "밴드 밖 하루가 정보 충격인지 유동성 노이즈인지는 이 그림으로 판별하지 않는다. 밴드는 60일 창 길이 선택에 의존한다."))
    figblock(3, "03_distribution_tail", "수익률 분포 · Q-Q · 꼬리 생존함수",
             "왼쪽: 히스토그램(로그 밀도) vs 같은 평균·표준편차의 정규. 가운데: 정규 Q-Q. 오른쪽: |r| 생존함수 P(|R|>x) log-log, 빨강 = 정규 기대치.",
             f"skew {f2(S['skew'])}, excess kurtosis {f2(S['ex_kurt'],1)} → 상위 3일 제거 시 {f2(S['ex_kurt_trim3'],1)}. 일간 Q01 {pct(S['q01'],2)} / Q99 {pct(S['q99'],2)}. Q-Q 양 끝이 직선에서 벗어나면 정규보다 두꺼운 꼬리.",
             "첨도가 소수 날짜에 좌우되면 '이 자산은 원래 꼬리가 두껍다'가 아니라 '몇 날의 기억'이다. 리스크 지표(VaR 등)를 정규 가정으로 만들면 꼬리를 과소평가한다.",
             pick(L, ["anom", "tick", "spike"], f"trim3 후에도 kurtosis {f2(S['ex_kurt_trim3'],1)} 가 남지만, 이것이 구조적 꼬리인지 표본기간(2020·2026 국면) 산물인지는 이 그림만으로 가를 수 없다 (CL-07: 2026-07~08 창 기여)."))
    figblock(4, "04_rolling_vol", "rolling 변동성 (20/60일) 과 연도별 변동성",
             "왼쪽: 20일·60일 rolling std ×√252, 점선 = 전체기간. 오른쪽: 연도별 연환산 변동성 (빨강 = 2026).",
             f"전체 {pct(S['ann_vol'])}, 2019-25 {pct(S['vol_2019_25'])}, 2026 {pct(S['vol_2026'])} → 비율 {f2(S['vol_ratio_26'])}×. 연도별: " + ", ".join(f"{y} {pct(v,0)}" for y, v in vy.items()) + "." + X4,
             "변동성은 국면 변수다. 전체기간 평균으로 '평시'를 정의하면 2020·2026 같은 고변동 구간이 기준선을 끌어올린다. universe 비교에서는 2026 비율이 국내주식 국면을 가르는 축이다 (CL-30).",
             pick(L, ["async", "hedge", "tick", "gear"], "연도 경계는 임의 구획이다. 2026 은 10월까지 부분 연도라 연말까지 같은 수준일지 알 수 없다."))
    eps = "; ".join(f"#{i+1} {pct(e['depth'])} ({e['peak']}→{e['trough']}, 하락 {e['dd_days']}일, 회복 {e['rec_days'] if e['rec_days'] is not None else '미회복'})" for i, e in enumerate(ep[:3]))
    figblock(5, "05_drawdown_episodes", "상위 낙폭 에피소드 (깊이 · 하락기간 · 회복기간)",
             "회색 = 낙폭 곡선. 색 띠 = 깊이 상위 5개 에피소드의 고점→회복 구간 (미회복은 마지막 날까지). 주석 = 깊이·고점→저점 거래일·저점→회복 거래일.",
             f"{eps}.",
             "같은 깊이라도 회복기간이 다르면 위험의 성격이 다르다 (빠른 V자 vs 장기 레짐). 낙폭 에피소드는 이후 event window 와 regime 정의의 후보 구간을 준다.",
             pick(L, ["gear", "fut", "bm"], "회복 여부는 분배 소급조정 가격(DEC-3) 기준이다. 에피소드의 원인(거시·업종·상품 구조)은 가격만으로 식별하지 않는다."))
    vr_line = (f"극단일 거래량 배수 median: − 극단 {f2(S['vr_med_ext_neg'])}×, + 극단 {f2(S['vr_med_ext_pos'])}× (전체 일 median {f2(S['vr_med_all'])}×). "
               f"corr(|r|, log 거래량배수) = {f2(S['corr_absr_vr'],3)}. 거래량배수 Q99 = {f2(S['vr_q99'],1)}×.")
    figblock(6, "06_volume_joint", "거래량 배수와 수익률–거래량 joint",
             "왼쪽: 거래량 / 직전 20일 median (shift(1)) 로그축, 빨강 점 = 상위 1%. 오른쪽: x=일간 수익률, y=log2 거래량 배수, 빨강 = |rz60|>3.",
             vr_line,
             "가격 극단이 거래량 급증을 동반하는지(정보 충격/강제 매매)와 방향 비대칭(급락일 vs 급등일)은 CL-26 의 상품유형별 반대 패턴과 직접 연결된다.",
             pick(L, ["liq", "async", "gear"], "거래량은 주 단위이며 거래대금·호가 깊이가 아니다. LP 호가 공급 변화나 기관 설정·환매가 거래량에 섞여 투자자 수요로 해석할 수 없다."))
    figblock(7, "07_benchmark_scatter", "benchmark 대비 동일일 산점도와 OLS",
             f"x = 비교 ETF 일간 수익률, y = {tk} 일간 수익률, 빨강 = OLS 적합. 제목 = β, Pearson ρ, R², 잔차 표준편차(bp). 비교 ①proxy {S['proxy']} ②market {S['market']}.",
             f"proxy {S['proxy']}: β={f2(S['proxy_beta'])}, ρ={f2(S['proxy_corr'],3)}, Spearman={f2(S['proxy_spearman'],3)}, R²={f2(S['proxy_r2'])}, 잔차sd={f2(S['proxy_resid_sd_bp'],0)}bp. "
             f"market {S['market']}: β={f2(S['mkt_beta'])}, ρ={f2(S['mkt_corr'],3)}, Spearman={f2(S['mkt_spearman'],3)}, R²={f2(S['mkt_r2'])}, 잔차sd={f2(S['mkt_resid_sd_bp'],0)}bp.",
             "β 는 시장 1 움직임당 민감도, 잔차 sd 는 시장으로 설명되지 않는 고유 변동 크기다. 이상 탐지에서 '시장 공통분을 뺀 상대 움직임'을 쓸지 결정하는 기준이 된다.",
             pick(L, ["proxyart", "async", "anom", "bm"], "OLS β 는 전체기간 평균 관계이며 꼬리 구간의 관계(충격기 β)와 다를 수 있다. 상관은 같은 경제적 원인을 증명하지 않는다."))
    figblock(8, "08_rolling_corr_beta", "rolling 120일 상관과 β",
             f"120 거래일 창의 Pearson 상관(위)과 β(아래). 검정 = market {S['market']}, 주황 = proxy {S['proxy']} (같으면 하나만). 주황 띠 = anchor.",
             f"market 상관 범위 {f2(S['roll_corr_min'])} ({S['roll_corr_min_date']}) ~ {f2(S['roll_corr_max'])} ({S['roll_corr_max_date']}). 하위기간 상관: 2019-21 {f2(S['mkt_corr_p1'],3)} vs 2022-26 {f2(S['mkt_corr_p2'],3)}.",
             "관계 자체가 시간변화하는 상태 변수다. 충격기에 상관이 수렴하면 분산효과가 필요할 때 사라지고, 부호가 바뀌면 '헤지 자산' 라벨이 국면 한정이 된다.",
             pick(L, ["async", "proxyart", "anom", "struct"], "120일 창 상관은 표본오차가 크다 (n=120 에서 ρ 의 표준오차 ≈ 0.09). 창 길이를 바꾸면 전환 시점이 이동하므로 시점 자체를 확정하지 않는다."))
    figblock(9, "09_relative_return", f"market reference ({S['market']}) 대비 누적 상대수익",
             f"위: 본 ETF 와 {S['market']} 의 누적 로그수익률. 아래: 누적 차이 Σ(r − r_market) %p (빨강 = 앞섬, 파랑 = 뒤처짐). β 조정 없음.",
             f"본 ETF 누적 {pct(S['cum_logret'])}, 누적 차이 {S['rel_cum_end']*100:+.1f}%p (β 조정 없는 단순 차).",
             "상대수익이 장기 추세를 가지면 시장 factor 외 고유 factor(업종·자산군·통화)가 존재한다는 뜻이고, 이상 탐지 시 '시장 대비 초과' 정의의 기준선을 준다.",
             pick(L, ["gear", "hedge", "fut", "tick"], "β 조정을 하지 않았으므로 고β ETF 의 상대수익은 시장 방향을 그대로 반영한다. 상대 성과를 운용 능력이나 알파로 해석할 수 없다."))
    tops = "; ".join(f"{d} {r*100:+.2f}% (rz {z:+.1f})" for d, r, z in S["top_ext"])
    figblock(10, "10_extreme_timeline", "robust-z 극단일 타임라인 (+/− 분리)",
             "rz60 = (r − 직전60일 median)/(1.4826·직전60일 MAD), shift(1) 로 당일 미포함. 점선 = ±3. 빨강 = + 극단, 파랑 = − 극단, 주황 띠 = 공통충격 anchor. 표시는 ±15 에서 clip.",
             f"|rz60|>3: + {S['n_ext_pos']}일 / − {S['n_ext_neg']}일. 상위 |rz| 5일: {tops}.",
             "+/− 극단을 분리해야 급등·급락의 빈도·군집이 다르다는 점이 보인다. anchor 와 겹치는 극단은 공통충격, 겹치지 않는 극단은 고유충격 후보다.",
             pick(L, ["tick", "anom", "spike", "gear"], "MAD 식·창 길이·임계값을 바꾸면 극단일 수가 크게 변한다 (CL-08/20/32: 같은 'rz3' 도 MAD 식에 따라 공통일 52↔61). 고변동 국면 안의 연속 충격은 baseline 이 흡수해 과소 계수된다 (CL-21)."))
    anc = ", ".join(f"{k} {v*100:+.1f}%" for k, v in S["anchor_ret"].items())
    figblock(11, "11_event_window", "공통충격 anchor 전후 경로",
             f"각 anchor 시작일 t=0 기준 −5~+10 거래일 누적 로그수익률. 실선 = 본 ETF, 점선 = {S['market']}. anchor 정의는 config/event_anchors.csv.",
             f"anchor 구간(시작~끝) 누적 수익률: {anc}.",
             "공통충격에 대한 반응 방향·크기·회복 속도가 상품군 정체성(동반 하락, 반대 방향, 무관)을 가장 직접적으로 보여준다 (CL-22/27).",
             pick(L, ["async", "fut", "gear", "liq"], f"anchor {len(S['anchor_ret'])}개는 표본이 작아 평균 반응을 일반화할 수 없다. A4(2026-07-28~08-03)는 universe 공통이 아니라 국내 주식 국면 충격이다 (CL-24). "
             + ("A5(2025-04-07~10)는 FRAGILE anchor 다 — rz3·abs3 에서는 생존하나 ETF별 q99 정의에서 탈락 (CL-33). " if "A5" in S["anchor_ret"] else "")
             + "anchor 라벨은 날짜 식별용이며 원인(뉴스)을 귀속하지 않는다."))

    md(f"""
## 결론 ↔ 지지 figure
| # | 결론 (OBSERVATION) | 지지 figure | 근거 수치 |
|---|---|---|---|
| 1 | 평소 변동성 규모는 연 {pct(S['ann_vol'])}, 2026 은 2019-25 대비 {f2(S['vol_ratio_26'])}×{dq_c1} | Fig 4, Fig 2 | vol_2026 {pct(S['vol_2026'])} |
| 2 | 꼬리 두께 중 소수 날짜 기여 = kurt {f2(S['ex_kurt'],1)} → trim3 {f2(S['ex_kurt_trim3'],1)} | Fig 3 | Q01/Q99 {pct(S['q01'],2)}/{pct(S['q99'],2)} |
| 3 | 시장({S['market']}) 민감도 β={f2(S['mkt_beta'])}, ρ={f2(S['mkt_corr'],3)}; proxy({S['proxy']}) β={f2(S['proxy_beta'])}, ρ={f2(S['proxy_corr'],3)} | Fig 7 | 잔차sd {f2(S['mkt_resid_sd_bp'],0)}bp |
| 4 | 시장 관계는 시간변화: 120일 상관 {f2(S['roll_corr_min'])}~{f2(S['roll_corr_max'])}, 2019-21 {f2(S['mkt_corr_p1'],2)} vs 2022-26 {f2(S['mkt_corr_p2'],2)} | Fig 8 | — |
| 5 | 최대낙폭 {pct(S['max_dd'])} ({S['max_dd_date']}), 회복 {e1['recovered']} | Fig 1, Fig 5 | — |
| 6 | 극단일(|rz60|>3) + {S['n_ext_pos']} / − {S['n_ext_neg']}, 정의 민감 | Fig 10 | top: {S['top_ext'][0][0]} |
| 7 | 공통충격 반응: {anc} | Fig 11 | — |
| 8 | 시장 대비 누적 차이 {S['rel_cum_end']*100:+.1f}%p (β 미조정){c8} | Fig 9 | — |
| 9 | 극단일 거래량 배수 −{f2(S['vr_med_ext_neg'])}× / +{f2(S['vr_med_ext_pos'])}× vs 전체 {f2(S['vr_med_all'])}× | Fig 6 | — |

**이 notebook 이 주장하지 않는 것**: 인과(왜 움직였는가), 예측, 매매 신호. 모든 관계는 동일일 통계적 동행이다.

**ETF별 한계 요약**: {' '.join('- ' + v for v in L.values()) if L else '-'}
""")
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "hongik_26_2", "display_name": "Python (hongik 26-2 .venv CUDA)", "language": "python"}
    for i, c in enumerate(nb.cells):
        c.id = hashlib.sha1(f"final-{tk}-{i}".encode()).hexdigest()[:12]
    return nb


def main():
    U = F.universe()
    dq = pd.read_csv(ROOT / "reports" / "dq_flags.csv", dtype=str)
    close, vol, R = F.load_panel()
    out = ROOT / "notebooks" / "final_eda"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in U.iterrows():
        tk, slot = row["ticker"], int(row["slot"])
        if args.only and tk not in args.only:
            continue
        S = F.compute(tk, close, vol, R)
        nb = build(tk, slot, row, S, dq)
        p = out / f"{slot:02d}_{tk}_final_eda.ipynb"
        nbformat.write(nb, p)
        print("wrote", p.relative_to(ROOT))
        rows.append({k: S[k] for k in ["ticker", "name", "n_obs", "ann_vol", "vol_2026", "vol_2019_25", "vol_ratio_26",
                                       "ex_kurt", "ex_kurt_trim3", "q01", "q99", "max_dd", "max_dd_date",
                                       "proxy", "proxy_corr", "proxy_beta", "market", "mkt_corr", "mkt_beta",
                                       "mkt_corr_p1", "mkt_corr_p2", "n_ext_pos", "n_ext_neg"]})
    if not args.only:
        t = ROOT / "reports" / "tables"; t.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(t / "etf_final_stats.csv", index=False, float_format="%.10g")
        print("wrote reports/tables/etf_final_stats.csv")


main()
