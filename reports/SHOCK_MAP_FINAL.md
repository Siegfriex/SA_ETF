# SHOCK_MAP_FINAL — Phase 00.5 (A, C2-A1)

Supersedes `reports/SHOCK_MAP.md` (65b3f51, G3 ACCEPTED). 생성: `analysis/scripts/universe_final.py` → `reports/tables/universe_shock_*.csv`, `universe_breadth_daily.csv`, figures `10`–`13`. 해석: `notebooks/universe/universe_eda.ipynb` §4.
**동반은 동시성이며 전염·인과가 아니다.** 뉴스 매칭은 범위 밖.

## 1. 정의 (모두 기준선 shift(1))
| 이름 | 정의 | 비고 |
|---|---|---|
| rz3_D | \|r − med60\| / (1.4826·MAD60) > 3, MAD = \|r−trailing median\| 의 rolling median, min_periods 20 | D 식 |
| rz3_A | 창 내부 median/MAD, min_periods 40 | A 식 (CL-32) |
| q99 | ETF별 전체기간 \|r\| 99% 분위 초과 | look-ahead 포함, 서술용 |
| abs3 | \|log r\| > 3% | D 표는 단순수익률 ≥3% → 175 (본 표 177) |
| dedup | KOSPI200 4종(069500·102110·122630·114800) → 1 | CL-23 |
공정 비교를 위해 rz3_A warm-up(≥15종 유효) 이후 날만 집계.

## 2. 정의 민감도 (`10_shock_definition_sensitivity_dedup.png`, `universe_shock_definition_sensitivity.csv`)
| k | rz3_D raw/dedup | rz3_A | q99 | abs3 |
|---|---|---|---|---|
| ≥5 | **52 / 40** | 61 / 45 | 22 / 15 | 177 / 139 |
- 같은 'rz3' 라도 MAD 식만 바꾸면 52→61 (+17%). 정의 간 최대 8배 (D: 4정의 max/min 7.95).
- dedup 은 k≥5 일 수를 21–32% 줄인다.
- k≥5 dedup 집합 Jaccard: rz3_D–rz3_A 0.73 · rz3_D–q99 0.31 · rz3_D–abs3 0.25 · q99–abs3 **0.11** (`universe_shock_def_jaccard_k5dedup.csv`).
- 판정: shock count 는 **DEFINITION_DEPENDENT** (ROBUSTNESS_MATRIX DOCX-11).

## 3. Anchor (`12_shock_participation_sign.png`, `universe_shock_anchor_survival.csv`)
| ID | 날짜 | 069500 r | rz3_D (dedup) | rz3_A | q99 | abs3 | 같은부호/반대 | 상태 |
|---|---|---|---|---|---|---|---|---|
| A1 | 2020-03-19 | −7.6% | 15 (12) | 15 | 15 | 14 | 16/3 | CONFIRMED_DATA |
| A1 | 2020-03-20 | +7.4% | 14 (11) | 14 | 12 | 14 | 17/2 | CONFIRMED_DATA |
| A1 | 2020-03-23 | −6.1% | 12 (9) | 12 | 5 | 14 | 16/3 | CONFIRMED_DATA |
| A1 | 2020-03-24 | +8.9% | 16 (13) | 15 | 13 | 17 | 17/2 | CONFIRMED_DATA |
| A2 | 2024-08-05 | −9.6% | 15 (12) | 16 | 13 | 15 | 16/3 | CONFIRMED_DATA |
| A3 | 2026-03-04 | −13.3% | 13 (10) | 14 | 12 | 14 | 16/3 | CONFIRMED_DATA |
| F1 | 2025-04-07 | −5.9% | 17 (14) | 17 | **4** | 16 | 16/3 | **FRAGILE** — q99 에서 k≥5 탈락 (CL-33) |
| F1 | 2025-04-10 | +6.0% | 16 (13) | 16 | **4** | 16 | 17/2 | **FRAGILE** |
(같은부호/반대: \|r\|<0.1% 는 0 처리. 반대 고정 = 114800 기계적 + 261240 달러; 2024-08-05 는 148070 국고채 +0.9% 반대.)
- A1~A3 는 네 정의 모두 k≥5 생존 → `config/event_anchors.csv` 확정 anchor 유지. F1 은 FRAGILE 로 **추가 필요** (C 소유 config).
- 2026-07-28~08-03 은 universe 공통이 아니라 국내주식 국면 충격 (abs3 만 생존, CL-24) — anchor 아님. `11_breadth_timeline.png` 하단에서 2026 하반기 abs3 밀집, 상단 rz3_D 미반응(CL-21)으로 확인.

## 4. 참여·부호 구조 (`13_shock_participation_scatter.png`, `universe_shock_participation_by_etf.csv`; 공통일 = rz3_D dedup≥5, n=40)
| 그룹 | 참여율 | 069500 과 같은 부호 | 하락 공통일 평균 r |
|---|---|---|---|
| KOSPI200 지수형 (069500/102110/122630) | 0.73–0.78 | 1.00 | −4.9% / −4.9% / −10.0% |
| 114800 인버스 | 0.78 | 0.00 | +4.7% (기계적) |
| 국내 업종·테마·전략 | 0.43–0.68 | 0.98–1.00 | −4.2 ~ −6.1% |
| 해외주식 219480/133690/192090 | 0.58 / 0.43 / 0.15 | 1.00 / 0.88 / 0.75 | −3.2 / −2.7 / −1.8% |
| 148070 국고채10년 | 0.25 | 0.63 | −0.25% |
| 132030 금 / 261220 WTI | 0.28 / 0.40 | 0.73 / 0.68 | −1.3 / −3.1% |
| 261240 달러 | 0.38 | **0.075** | **+1.06%** |
- 공통 충격 구조 = 주식 동반 + 달러 반대 + 채권·금 혼재 (CL-22, CL-27). 데이터상 일관된 비기계적 반대축은 달러뿐.
- 달러 '폭락일 상승 비율' 은 정의 종속: 0.72 (≤−3%) ~ 0.91 (rz≤−3) (ROBUSTNESS_MATRIX DOCX-08). DOCX 의 '86%' 단일값은 재현되지 않음.

## 5. Idiosyncratic shock 민감도 (D 수행, ROBUSTNESS_MATRIX CL-idio)
universe median \|rz\| 기준 c = 0.75 / 1.0 / 1.25 → solo 사례 123 / 201 / 257, c=1.0 대비 생존율 0.61 / 1.0 / 1.0 → 기준 하향 시 39% 탈락, **DEFINITION_DEPENDENT**. 저변동 ETF(153130)의 rz 단독 극단은 경제적 크기 필터 필요 (CL-25).

## 6. 말할 수 없는 것
anchor 8일은 사례이지 통계적 표본이 아니다. F1 의 q99 탈락 원인(2026 고변동이 q99 상향)은 추정·미검증. 해외 ETF 의 KRX 종가 정렬은 기초시장 사건 날짜와 1일 어긋날 수 있다.
