# CHATGPT REVIEW PACKET — ETF Universe EDA Phase 00.5

이 문서는 검증용 지도다. 새 분석은 담지 않았다. 아래 수치는 모두 canonical commit 의 파일에 이미 있는 값을 옮긴 것이고, 경로 실재 여부는 `reports/review_reference_check.csv` 에서 자동으로 확인했다.

---

# 0. Canonical Snapshot

| 항목 | 값 |
|---|---|
| repository | https://github.com/Siegfriex/SA_ETF |
| branch | `main` (= `integration/eda-hour1` = `claude-c/eda-control-v1`) |
| commit SHA | `293e504` (closure manifest commit) · clean rerun 검증 SHA: `e9a6880` (외부 세션), `840f8e8` (C) |
| analysis period | 2019-01-02 ~ 2026-10-02 (일봉, FDR Close, 로그수익률) |
| ETF count | 20 (UNIVERSE_STATUS = PROVISIONAL) |
| raw rows / ETF | 1,903 행 (수익률 1,902) · raw 는 git 밖에 있고 `data/raw/MANIFEST.csv` 에 sha256 이 기록되어 있다 |
| notebook count | final ETF 20 + universe 1 = 21 (그 밖에 Phase 00 `notebooks/00_raw_eda/` 20개는 보존) |
| figure count | 243 (etf 220 = 20×11 · universe 18 = 00 smoke + 01–17 · robustness 5) |
| robustness test count | 79 (`ROBUSTNESS_MATRIX.csv` 74 + `ROBUSTNESS_MATRIX_C_ADDENDUM.csv` 5) |
| clean rerun result | PASS — tables 29/29 max\|Δ\| = 0, figure 243 바이트 동일, notebook 41 실행 오류 0 (`reports/CLOSURE_MANIFEST.json` → `gate.clean_rerun`, `gate.clean_rerun_final`) |
| 범위 밖 | 뉴스 · PELT · XGBoost · forecasting · trading agent · portfolio |

---

# 1. Where to Start

| 순서 | 파일 | 무엇을 보나 |
|---|---|---|
| 1 | `reports/EDA_EVIDENCE_CLOSURE.md` | 판정 요약 (§2 표), 새로 확인된 것 (§3), 주장하지 않는 것 (§4), 한계 (§5) |
| 2 | `reports/CLAIM_EVIDENCE_MATRIX.csv` | DOCX claim 16개 × (원 수치 · 최종 수치 · figure · robustness · final_status) |
| 3 | `notebooks/universe/universe_eda.ipynb` | §1 스케일 (cell 4–12) · §2 군집 (13–21) · §3 상관 상태 (22–28) · §4 충격 (29–37) · §4b peer·구조 (38–42) · §5 claim 재확인 표 (43–44) |
| 4 | `reports/ETF_RELATION_MAP_FINAL.md` | R1–R12 (묶임) · S1–S4 (동조) · B1–B5 (깨짐) · H1–H5 (5칸 블록) · §4 DOCX 변경 |
| 5 | `reports/SHOCK_MAP_FINAL.md` | §1 정의 · §2 정의 민감도 · §3 anchor · §4 참여/부호 · §5 idio 민감도 |
| 6 | `reports/ROBUSTNESS_MATRIX.csv` (+ `reports/ROBUSTNESS_MATRIX_C_ADDENDUM.csv`) | test 별 statistic · CI · verdict |
| 7 | `notebooks/final_eda/04_122630_final_eda.ipynb`, `10_143860`, `16_148070`, `20_261240`, `17_153130` | 구조 확인 (레버리지), peer 수정 (헬스케어), 정정 (148070), DQ 효과 (달러), 분모 문제 (단기채) |

ETF notebook 의 구조는 20개 모두 같다. cell 0 은 "왜 이런 ETF인가"와 핵심 수치, cell 2/5/8/…/32 는 Fig 1–Fig 11 (Fig N 은 cell 3N−1 의 heading, 그다음 셀이 코드, 그다음이 4섹션 해석), cell 35 는 "결론 ↔ 지지 figure" 표다.

---

# 2. Claim Ledger (16)

ID 대응: 이 문서와 `CLAIM_EVIDENCE_MATRIX.csv` 는 `DX-xx` 를 쓰고, `ROBUSTNESS_MATRIX.csv` 는 `DOCX-xx` 를 쓴다. 대응표는 각 행의 ROBUSTNESS 칸에 적었다. CI 는 모두 95% block bootstrap (block 10, B=1000, seed 고정)이고, 비율 CI 는 Wilson 이다. U-nb 는 `notebooks/universe/universe_eda.ipynb`, E-nb(tk) 는 notebooks/final_eda/ 아래 해당 ticker 의 notebook 이다 (경로는 §6).

### DX-01 — 변동성 scale
- **ORIGINAL**: 연변동성 0.32% (단기채) ~ 56.65% (레버리지), 약 175배
- **FINAL STATUS**: CONFIRMED
- **FINAL VALUE**: 153130 0.32% · 122630 56.6%. 261240 은 13.49% 이지만 DQ 2일을 빼면 8.9%
- **METHOD**: std(log r)·√252
- **CI**: —
- **SOURCE TABLE**: `reports/tables/universe_vol_rank.csv`, `reports/universe_profile.csv`
- **NOTEBOOK**: U-nb cell 5
- **FIGURE**: `figures/universe/01_vol_ranking.png`, `figures/universe/03_quantile_interval.png`
- **ROBUSTNESS**: DOCX-01 계열 (기술통계라 별도 검정 없음)
- **INTERPRETATION**: 절대 수익률 threshold 하나로는 ETF 간 비교가 금융적으로 의미가 없다
- **LIMITATION**: 261240 은 DQ 처리에 따라 값이 바뀐다

### DX-02 — KODEX200–TIGER200
- **ORIGINAL**: ρ ≈ 0.999, 일간차 sd ≈ 8bp
- **FINAL STATUS**: CONFIRMED
- **FINAL VALUE**: ρ 0.9990 · sd 8.05bp
- **METHOD**: Pearson, sd(r1−r2)
- **CI**: ρ [0.9987, 0.9992] · sd [7.15, 9.11] bp
- **SOURCE TABLE**: `reports/ROBUSTNESS_MATRIX.csv` (DOCX-01, DOCX-02), `reports/tables/universe_claim_checks.csv`
- **NOTEBOOK**: E-nb(102110) Fig 7 (cell 20–22)
- **FIGURE**: `figures/etf/102110/07_benchmark_scatter.png`, `figures/robustness/forest_claims.png`
- **ROBUSTNESS**: ROBUST
- **INTERPRETATION**: 같은 exposure 다. breadth·PCA 에서 dedup 해야 한다
- **LIMITATION**: NAV·추적오차 데이터가 없다

### DX-03 — 레버리지 β
- **ORIGINAL**: β ≈ 1.99
- **FINAL STATUS**: CONFIRMED
- **FINAL VALUE**: β 1.986. 기간별 1.974 / 1.988 / 1.992. rolling 60D β [1.887, 2.055]
- **METHOD**: 동일일 OLS (122630 ~ 069500)
- **CI**: [1.952, 2.015]
- **SOURCE TABLE**: `reports/ROBUSTNESS_MATRIX.csv` (DOCX-03 ×4), `reports/tables/universe_rolling_beta60.csv`
- **NOTEBOOK**: E-nb(122630) Fig 7–8 (cell 20–25) · U-nb cell 39
- **FIGURE**: `figures/etf/122630/07_benchmark_scatter.png`, `figures/etf/122630/08_rolling_corr_beta.png`, `figures/universe/16_rolling_beta_geared.png`
- **ROBUSTNESS**: ROBUST
- **INTERPRETATION**: 일간 2배 설계가 데이터에서 재현된다
- **LIMITATION**: 다기간 수익률은 2배가 아니다 (경로 의존). fig16 해석은 HOLD (§8)

### DX-04 — 인버스 β
- **ORIGINAL**: β ≈ −1.02, ρ ≈ −0.996
- **FINAL STATUS**: CONFIRMED
- **FINAL VALUE**: β −1.016 · ρ −0.996. rolling 60D β [−1.062, −0.960]
- **METHOD**: OLS
- **CI**: [−1.036, −0.997]
- **SOURCE TABLE**: `reports/ROBUSTNESS_MATRIX.csv` (DOCX-04 ×4)
- **NOTEBOOK**: E-nb(114800) Fig 7–8 · U-nb cell 39
- **FIGURE**: `figures/etf/114800/07_benchmark_scatter.png`, `figures/universe/16_rolling_beta_geared.png`
- **ROBUSTNESS**: ROBUST
- **INTERPRETATION**: KOSPI200 factor 의 부호를 바꾼 것이다
- **LIMITATION**: 기초는 선물이라 현물 지수와 basis 가 있다

### DX-05 — 헬스케어 peer
- **ORIGINAL**: 143860 은 KOSDAQ150 과 더 가깝다 (≈0.76)
- **FINAL STATUS**: CONFIRMED
- **FINAL VALUE**: Spearman 0.756 (229200) vs 0.481 (069500), Δρ +0.275. 60D Δ>0 비율 0.96
- **METHOD**: Spearman Δρ, block bootstrap
- **CI**: [0.234, 0.317]. Pearson Δρ 0.279 [0.220, 0.338]
- **SOURCE TABLE**: `reports/ROBUSTNESS_MATRIX.csv` (DOCX-05 ×8), `reports/tables/universe_peer_stability.csv`
- **NOTEBOOK**: U-nb cell 16, 41 · E-nb(143860)
- **FIGURE**: `figures/universe/14_empirical_peer_rolling.png`, `figures/universe/17_empirical_peer_scatter.png`
- **ROBUSTNESS**: ROBUST (3기간 모두 Δ>0)
- **INTERPRETATION**: 라벨(섹터)보다 성장·코스닥 factor 가 더 맞다
- **LIMITATION**: 2023-08 에 일시 역전이 있다. 저유동성 ETF 다

### DX-06 — 고배당–은행
- **ORIGINAL**: 161510 은 은행과 가깝다 (≈0.83)
- **FINAL STATUS**: CONFIRMED
- **FINAL VALUE**: Spearman 0.830 vs 0.603, Δρ +0.227. 60D/120D Δ>0 0.95/0.97
- **METHOD**: Spearman Δρ, block bootstrap
- **CI**: [0.185, 0.270]
- **SOURCE TABLE**: `reports/ROBUSTNESS_MATRIX.csv` (DOCX-06 ×8), `reports/tables/universe_peer_stability.csv`
- **NOTEBOOK**: U-nb cell 14, 16, 41 · E-nb(161510)
- **FIGURE**: `figures/universe/05_corr_spearman_clustered.png`, `figures/universe/17_empirical_peer_scatter.png`
- **ROBUSTNESS**: ROBUST
- **INTERPRETATION**: 금융·가치 factor 가 겹친다
- **LIMITATION**: 구성종목 데이터가 없어 겹침을 직접 확인하지 못했다

### DX-07 — 통계 cluster ≠ taxonomy
- **ORIGINAL**: 인버스와 달러가 묶인다. 통계 군집과 경제 분류는 다르다
- **FINAL STATUS**: CONFIRMED_WITH_LIMITATION
- **FINAL VALUE**: 114800+261240 동반은 부호 artifact 이고, 1−\|ρ\| 거리로 바꾸면 풀린다. average/complete k=5 는 ROBUST, ward·2026 단독은 FRAGILE. LOETF k=5 ARI ≥0.98, k=8 0.40
- **METHOD**: hierarchical clustering, ARI
- **CI**: — (min/median ARI)
- **SOURCE TABLE**: `reports/ROBUSTNESS_MATRIX.csv` (DOCX-12), `reports/ROBUSTNESS_MATRIX_C_ADDENDUM.csv`, `reports/tables/universe_clusters.csv`, `analysis/robustness/d0122/cluster_leave_one_etf_out.csv`
- **NOTEBOOK**: U-nb cell 18
- **FIGURE**: `figures/universe/06_dendrogram.png`, `figures/robustness/cluster_ari_heatmap.png`
- **ROBUSTNESS**: k=5 ROBUST / k=3·k=8 FRAGILE
- **INTERPRETATION**: 군집 경계는 linkage 와 k 에 따라 달라진다
- **LIMITATION**: 상관 기반 거리이므로 인과가 아니다

### DX-08 — 2026 변동성 확대
- **ORIGINAL**: 국내주식에 집중
- **FINAL STATUS**: REVISED
- **FINAL VALUE**: 국내주식 12/12 >1 (069500 3.30×, top3일 제외 2.90×). 해외 0.73 / 0.87 / 0.94. 금 2.21×, WTI 1.50×, 달러 1.26× (DQ 제외)
- **METHOD**: σ2026/σ2019-25, block CI, Mann–Whitney (국내 vs 해외)
- **CI**: 069500 [2.53, 4.21] · 금 [1.39, 3.10] · WTI [1.02, 2.15] · 달러 [1.04, 1.49] · 219480 [0.57, 0.90]
- **SOURCE TABLE**: `reports/ROBUSTNESS_MATRIX.csv` (DOCX-07 ×21), `reports/tables/universe_vol_ratio_2026.csv`
- **NOTEBOOK**: U-nb cell 7
- **FIGURE**: `figures/universe/02_vol_year_heatmap.png`, `figures/robustness/vol_ratio_2026.png`
- **ROBUSTNESS**: 국내 ROBUST (MW p=0.0022, 모든 leave-out 에서 유지) · 132030 FRAGILE
- **INTERPRETATION**: 국내주식 국면이고 해외주식은 아니다. 원자재·달러도 함께 확대됐다
- **LIMITATION**: 2026 은 n=184 이다. `etf_final_stats.csv` 의 261240 은 DQ 포함 값 0.80 이다 (§8 ISSUE)

### DX-09 — shock 정의 민감도
- **ORIGINAL**: k≥5 공통충격 52 / 22 / 175
- **FINAL STATUS**: CONFIRMED
- **FINAL VALUE**: rz3 (D식 MAD) 52 (dedup 40) · rz3 (A식 MAD) 61 · q99 22 · abs3 175 (단순) / 177 (log). max/min 7.95×
- **METHOD**: 정의 × k × dedup grid
- **CI**: —
- **SOURCE TABLE**: `reports/ROBUSTNESS_MATRIX.csv` (DOCX-11), `reports/tables/universe_shock_definition_sensitivity.csv`
- **NOTEBOOK**: U-nb cell 30
- **FIGURE**: `figures/universe/10_shock_definition_sensitivity_dedup.png`, `figures/robustness/shock_definition_grid.png`
- **ROBUSTNESS**: DEFINITION_DEPENDENT (정의 의존 자체가 claim 이다)
- **INTERPRETATION**: event 정의는 measurement problem 이다. robust-z 는 MAD 식까지 명시해야 재현된다
- **LIMITATION**: ETF notebook 의 rz60 은 A식이다

### DX-10 — 단기채·달러 kurtosis
- **ORIGINAL**: 153130 첨도 322.9, 261240 300, 소수 극단치 영향
- **FINAL STATUS**: CONFIRMED
- **FINAL VALUE**: 153130 323.8 → trim3 6.08 / winsor1% 0.98. 261240 301.1 → trim3 2.43 / anomaly 2일 제거 3.87 / winsor 1.14
- **METHOD**: excess kurtosis (unbiased), 처리 4종
- **CI**: —
- **SOURCE TABLE**: `reports/ROBUSTNESS_MATRIX.csv` (DOCX-09 = 261240, DOCX-10 = 153130)
- **NOTEBOOK**: E-nb(153130), E-nb(261240) Fig 3 (cell 8–10)
- **FIGURE**: `figures/robustness/kurtosis_variants.png`, `figures/etf/153130/03_distribution_tail.png`, `figures/etf/261240/03_distribution_tail.png`
- **ROBUSTNESS**: ROBUST
- **INTERPRETATION**: 통계적 극단과 경제적 크기는 다르다
- **LIMITATION**: DOCX 의 "3.9" 는 anomaly 2일 제거, "2.4" 는 trim3 로 방식이 다르다

### DX-11 — 상관은 time-varying state
- **ORIGINAL**: cross-ETF 상관 자체가 상태 정보다
- **FINAL STATUS**: CONFIRMED_WITH_LIMITATION
- **FINAL VALUE**: 국내 9종 120D 평균 ρ 0.27–0.80. 2026 vs 2019-25 Fisher \|z\|>3.4 쌍 44/136 (dedup). 수준차·vol 동조는 60/120/250 × Pearson/Spearman 에서 유지된다
- **METHOD**: rolling 평균 pairwise ρ, Fisher z
- **CI**: —
- **SOURCE TABLE**: `reports/tables/universe_rolling_state.csv`, `reports/tables/universe_corr_period_diff.csv`, `analysis/robustness/d0122/rolling_corr_window_estimator_summary.csv`
- **NOTEBOOK**: U-nb cell 23, 25
- **FIGURE**: `figures/universe/08_rolling_corr_dispersion.png`, `figures/universe/07_period_diff_corr.png`
- **ROBUSTNESS**: ROBUST (addendum)
- **INTERPRETATION**: 충격기에는 공통 factor 가 강해진다
- **LIMITATION**: Fisher z 는 독립을 가정하므로 낙관적이다. fig08 의 Pearson 계단은 단일 극단일 효과다

### DX-12 — 공통충격 anchor
- **ORIGINAL**: 2020-03, 2024-08-05, 2026-03-04 는 정의에 안정적
- **FINAL STATUS**: CONFIRMED
- **FINAL VALUE**: A1–A3 는 4정의 × dedup 에서 생존. A5 2025-04-07/10 은 q99 에서 4종뿐이라 FRAGILE. A4 2026-07-28~08-03 은 국내 국면
- **METHOD**: 정의별 참여 수
- **CI**: —
- **SOURCE TABLE**: `reports/tables/universe_shock_anchor_survival.csv`, `config/event_anchors.csv`
- **NOTEBOOK**: U-nb cell 32, 34
- **FIGURE**: `figures/universe/11_breadth_timeline.png`, `figures/universe/12_shock_participation_sign.png`
- **ROBUSTNESS**: CL-10/33
- **INTERPRETATION**: universe 공통 충격은 소수다
- **LIMITATION**: anchor label 은 날짜 식별용이고 원인을 귀속하지 않는다

### DX-13 — 달러 헤지
- **ORIGINAL**: 주식 폭락일에 86% 상승
- **FINAL STATUS**: REVISED
- **FINAL VALUE**: 정의에 따라 72–91% (§3-B)
- **METHOD**: 상승 비율
- **CI**: Wilson (§3-B)
- **SOURCE TABLE**: `reports/ROBUSTNESS_MATRIX.csv` (DOCX-08 ×5)
- **NOTEBOOK**: U-nb cell 34 · E-nb(261240) Fig 11
- **FIGURE**: `figures/universe/12_shock_participation_sign.png`, `figures/etf/261240/11_event_window.png`
- **ROBUSTNESS**: DEFINITION_DEPENDENT
- **INTERPRETATION**: 일관된 반대 방향 자산이지만 비율은 정의에 따라 달라진다
- **LIMITATION**: 86% 는 어떤 정의로도 재현되지 않았다

### DX-14 — 148070 β
- **ORIGINAL**: corr 0.100 · β 1.87
- **FINAL STATUS**: CORRECTED
- **FINAL VALUE**: proxy 153130 기준 β 1.87, ρ 0.100, R² 0.01 → artifact. 069500 기준 β 0.029, ρ 0.134
- **METHOD**: OLS. proxy 극단 2일을 빼면 5.57 로 불안정하다
- **CI**: —
- **SOURCE TABLE**: `reports/tables/etf_final_stats.csv`
- **NOTEBOOK**: E-nb(148070) cell 0 (†주석), Fig 7 (cell 20–22)
- **FIGURE**: `figures/etf/148070/07_benchmark_scatter.png`
- **ROBUSTNESS**: 재계산
- **INTERPRETATION**: proxy 의 분모 분산이 극소라서 생긴 artifact 다
- **LIMITATION**: KTB 10년 지수 데이터가 없다

### DX-15 — 2차전지–철강
- **ORIGINAL**: 일부 시기 철강과 더 가깝다
- **FINAL STATUS**: DOWNGRADED
- **FINAL VALUE**: 철강 우세 (Δρ>0) 비율 60D 0.53 / 120D 0.53. 2019-21 은 반도체 0.66 > 철강 0.54
- **METHOD**: rolling Spearman Δρ
- **CI**: —
- **SOURCE TABLE**: `reports/tables/universe_peer_stability.csv`
- **NOTEBOOK**: U-nb cell 16
- **FIGURE**: `figures/universe/14_empirical_peer_rolling.png`
- **ROBUSTNESS**: CL-17 HOLD
- **INTERPRETATION**: 안정적인 peer 가 없다 (REGIME_DEPENDENT)
- **LIMITATION**: 테마 지수 구성이 바뀌었는지 확인하지 못했다

### DX-16 — WTI 부호 반전
- **ORIGINAL**: WTI–주식 상관이 2026 에 반전
- **FINAL STATUS**: CONFIRMED
- **FINAL VALUE**: WTI–KOSPI +0.12~0.48 → −0.36. 261220–219480 +0.33 → −0.61 (z −13.5). WTI–달러 −0.30 → +0.40
- **METHOD**: 기간별 ρ, Fisher z
- **CI**: —
- **SOURCE TABLE**: `reports/tables/universe_corr_period_diff.csv`, `reports/tables/universe_fx_pairs_by_period.csv`
- **NOTEBOOK**: U-nb cell 27
- **FIGURE**: `figures/universe/15_wti_equity_sign_flip.png`
- **ROBUSTNESS**: CL-31
- **INTERPRETATION**: 2026 에 원유가 위험자산 축에서 달러 축으로 이동했다
- **LIMITATION**: 원인은 범위 밖이다

---

# 3. Most Important Corrections

## A. 2026 volatility expansion (DX-08)

- 기존: "국내주식 집중"
- 최종: 국내주식 확대는 강하다 (12/12 >1, 069500 3.30× [2.53, 4.21], top3일을 빼도 2.90×). 해외주식은 확대되지 않았다 (219480 0.73 [0.57, 0.90], 133690 0.87, 192090 0.94). 그러나 다음 자산도 확대됐다.

| 자산 | σ2026/σ2019-25 | 95% CI |
|---|---|---|
| 132030 금 | 2.21× | [1.39, 3.10] |
| 261220 WTI | 1.50× | [1.02, 2.15] |
| 261240 달러 (DQ 제외) | 1.26× | [1.04, 1.49] |

- 근거: `reports/tables/universe_vol_ratio_2026.csv`, `figures/robustness/vol_ratio_2026.png`, `figures/universe/02_vol_year_heatmap.png`

## B. USD hedge claim (DX-13)

- 기존: "폭락일 86% 상승"
- 최종: 폭락일 정의에 따라 72–91% 다.

| 폭락일 정의 (069500 기준) | n | 261240 상승 비율 | Wilson 95% CI |
|---|---|---|---|
| ≤ −3% | 57 | 0.719 | [0.592, 0.819] |
| ≤ q05 | 96 | 0.802 | [0.711, 0.870] |
| ≤ −2% | 115 | 0.809 | [0.727, 0.870] |
| rz ≤ −3 (D식 MAD) | 34 | 0.912 | [0.770, 0.970] |

- 근거: `reports/ROBUSTNESS_MATRIX.csv` (DOCX-08)

## C. 148070 beta (DX-14)

- 기존: β 1.87 / corr 0.100
- 최종: 분산이 매우 작은 proxy (153130 단기채, 연 0.32%) 를 분모로 써서 생긴 artifact 다. R² 0.01 이고, proxy 극단 2일을 빼면 β 가 5.57 로 뛴다. KOSPI200 (069500) 기준으로는 β ≈ 0.029, ρ 0.134 다.
- 근거: `figures/etf/148070/07_benchmark_scatter.png` (좌: proxy, 우: market), `reports/tables/etf_final_stats.csv`

## D. 2차전지–철강 peer (DX-15)

- 기존: 일부 시기 철강과 더 가깝다
- 최종: 철강이 반도체보다 가까운 날의 비율이 60D/120D 모두 약 0.53 이다. 안정적인 peer 라고 부르기 어렵다 → DOWNGRADED
- 근거: `reports/tables/universe_peer_stability.csv`, `figures/universe/14_empirical_peer_rolling.png`

---

# 4. High-Value Confirmed Results

| 결과 | 숫자 | 95% CI | figure |
|---|---|---|---|
| 069500–102110 상관 | ρ 0.9990 · 일간차 8.05bp | [0.9987, 0.9992] · [7.15, 9.11] bp | `figures/etf/102110/07_benchmark_scatter.png` |
| 122630 β | 1.986 | [1.952, 2.015] | `figures/universe/16_rolling_beta_geared.png` |
| 114800 β / ρ | −1.016 / −0.996 | [−1.036, −0.997] | `figures/etf/114800/07_benchmark_scatter.png` |
| 143860 Δρ (KOSDAQ − KOSPI) | +0.275 (Spearman) | [0.234, 0.317] | `figures/universe/17_empirical_peer_scatter.png` |
| 161510 Δρ (은행 − KOSPI) | +0.227 (Spearman) | [0.185, 0.270] | `figures/universe/17_empirical_peer_scatter.png` |
| 2026 국내주식 vol 확대 | 12/12 >1, 069500 3.30× (top3 제외 2.90×), MW p=0.0022 | 069500 [2.53, 4.21] | `figures/robustness/vol_ratio_2026.png` |
| shock 정의 민감도 | k≥5: 52 / 61 / 22 / 175 → 7.95× | — | `figures/robustness/shock_definition_grid.png` |
| same-exposure 중복 효과 | rz3 D식 k≥5 52 → dedup 40 · PC1 45.7% → dedup 39.3% | — | `figures/universe/10_shock_definition_sensitivity_dedup.png`, `figures/universe/09_pca_scree_loading.png` |
| cluster 안정/불안정 | LOETF k=5 ARI ≥0.98 (ROBUST) · k=8 0.40 · 24변형 k=3 min 0.15 (FRAGILE) | — | `figures/robustness/cluster_ari_heatmap.png` |
| 261240 DQ 효과 | ann_vol 13.49% → 8.9% · kurt 301 → 3.87 · 2026 비율 0.80 → 1.26× · KOSPI ρ −0.308 → −0.458 | 비율 [1.04, 1.49] | `figures/etf/261240/03_distribution_tail.png`, `figures/universe/01_vol_ranking.png` |

---

# 5. Figure Review Index (28)

| FIG_ID | PATH | QUESTION | WHAT TO CHECK | CLAIM |
|---|---|---|---|---|
| U01 | `figures/universe/01_vol_ranking.png` | 스케일 차이는 얼마인가 | 로그축, 261240 의 DQ 제외 8.9% 병기 | DX-01 |
| U02 | `figures/universe/02_vol_year_heatmap.png` | ETF × year 변동성 | 2026 열: 국내 붉음, 해외 1 미만, 금·WTI 확대 | DX-08 |
| U03 | `figures/universe/03_quantile_interval.png` | Q01–Q99 폭 | 레버리지·WTI 폭, 단기채 거의 0 | DX-01 |
| U05 | `figures/universe/05_corr_spearman_clustered.png` | 전체 상관 구조 | 군집 순서, 143860–229200, 161510–091170 | DX-05, DX-06 |
| U06 | `figures/universe/06_dendrogram.png` | 통계 군집 vs taxonomy | 114800·261240 의 위치 (signed vs 1−\|ρ\|) | DX-07 |
| U07 | `figures/universe/07_period_diff_corr.png` | 2026 상관 구조 변화 | dedup 44/136, 부호 | DX-11, DX-16 |
| U08 | `figures/universe/08_rolling_corr_dispersion.png` | 국내 주식 간 rolling 상관 | Pearson vs Spearman 선, anchor 근처 급등, 07-31 음영 | DX-11 |
| U09 | `figures/universe/09_pca_scree_loading.png` | PC1 비중과 dedup | 45.7% vs 39.3%, 상관행렬 기준 표기 | DX-07 |
| U10 | `figures/universe/10_shock_definition_sensitivity_dedup.png` | breadth 정의 민감도 | 정의 × k × dedup 차이 | DX-09 |
| U11 | `figures/universe/11_breadth_timeline.png` | 공통충격 시점 | anchor A1–A3, A5 FRAGILE, 2026-07 국내 음영 | DX-12 |
| U12 | `figures/universe/12_shock_participation_sign.png` | 참여·부호 행렬 | anchor 일 수익률, 달러 반대 부호, 이산 범례 | DX-12, DX-13 |
| U13 | `figures/universe/13_shock_participation_scatter.png` | ETF 별 참여율 vs 부호 | 국내 ≈1, 달러 음(−) | DX-12 |
| U14 | `figures/universe/14_empirical_peer_rolling.png` | peer 안정성 | 헬스케어·고배당 Δ>0 일관, 2차전지 진동 | DX-05, DX-06, DX-15 |
| U15 | `figures/universe/15_wti_equity_sign_flip.png` | WTI 부호 반전 | 2026 구간 음(−), WTI–달러 + | DX-16 |
| U16 | `figures/universe/16_rolling_beta_geared.png` | 레버리지·인버스 구조 | 2 / −1 기준선 ±5% 밴드 이내 | DX-03, DX-04 |
| U17 | `figures/universe/17_empirical_peer_scatter.png` | 헬스케어·고배당 peer 산점도 | 라벨 benchmark 보다 peer 쪽이 촘촘한가 | DX-05, DX-06 |
| R1 | `figures/robustness/forest_claims.png` | 핵심 claim CI | CI 가 0 또는 기준선을 넘지 않는가 | DX-02~06 |
| R2 | `figures/robustness/vol_ratio_2026.png` | 2026 비율 CI | 국내 CI >1, 해외 ≤1, 금·WTI·달러 >1 | DX-08 |
| R3 | `figures/robustness/shock_definition_grid.png` | 정의 grid | 52/61/22/175 | DX-09 |
| R4 | `figures/robustness/cluster_ari_heatmap.png` | cluster 민감도 | ward·2026 단독에서 ARI 하락 | DX-07 |
| R5 | `figures/robustness/kurtosis_variants.png` | 단기채·달러 tail | 처리별 kurtosis 붕괴 폭 | DX-10 |
| E1 | `figures/etf/102110/07_benchmark_scatter.png` | KODEX200 / TIGER200 | 거의 완전한 대각선 | DX-02 |
| E2 | `figures/etf/122630/08_rolling_corr_beta.png` | 레버리지 rolling β | ≈2 유지 | DX-03 |
| E3 | `figures/etf/114800/07_benchmark_scatter.png` | 인버스 구조 | 기울기 −1 | DX-04 |
| E4 | `figures/etf/148070/07_benchmark_scatter.png` | 148070 β artifact | 좌측 proxy 분포가 수직선 | DX-14 |
| E5 | `figures/etf/261240/03_distribution_tail.png` | 261240 DQ 전후 | ±20% 2일의 영향 | DX-10 |
| E6 | `figures/etf/153130/03_distribution_tail.png` | 단기채 tail / kurtosis | 틱 이산성, 분모 극소 | DX-10 |
| E7 | `figures/etf/069500/04_rolling_vol.png` | 2026 국내 vol 국면 | 2026-07-31 국면, 연도 band | DX-08 |

---

# 6. ETF Notebook Coverage (20/20)

모든 notebook 은 figure 11개를 갖는다 (`figures/etf/<ticker>/01–11`). benchmark 열은 `config/universe.csv` 의 상품 기초지수이고, 괄호 안은 비교에 쓴 proxy ETF 다.

| ticker | name | notebook | fig | benchmark (proxy) | primary domain insight | important limitation |
|---|---|---|---|---|---|---|
| 069500 | KODEX 200 | `notebooks/final_eda/01_069500_final_eda.ipynb` | 11 | KOSPI 200 (102110) | 시장 기준선. 2026 vol 3.30× | 2026-07-31 +21.7% 는 실데이터 (국면) |
| 102110 | TIGER 200 | `notebooks/final_eda/02_102110_final_eda.ipynb` | 11 | KOSPI 200 (069500) | 069500 과 ρ 0.999, 차이 8bp | breadth 에서 중복 집계 위험 |
| 229200 | KODEX 코스닥150 | `notebooks/final_eda/03_229200_final_eda.ipynb` | 11 | KOSDAQ 150 (069500) | KOSPI β 0.81, ρ 0.69 — 별도 위험요인 | 국내주식 단일 모집단 가정 금지 |
| 122630 | KODEX 레버리지 | `notebooks/final_eda/04_122630_final_eda.ipynb` | 11 | KOSPI200 2× (069500) | β 1.986, 기간별 안정 | 다기간 2배 아님, MDD −68% |
| 114800 | KODEX 인버스 | `notebooks/final_eda/05_114800_final_eda.ipynb` | 11 | KOSPI200 선물 −1× (069500) | β −1.016 | 선물 basis, MDD −91% |
| 091230 | TIGER 반도체 | `notebooks/final_eda/06_091230_final_eda.ipynb` | 11 | KRX 반도체 추정 (069500) | β 1.18, 2026 지수 결합 0.95 | 기초지수 명칭 추정 |
| 091170 | KODEX 은행 | `notebooks/final_eda/07_091170_final_eda.ipynb` | 11 | KRX 은행 추정 (069500) | 고배당과 0.83 | 금리 단일요인 해석 금지 |
| 091180 | KODEX 자동차 | `notebooks/final_eda/08_091180_final_eda.ipynb` | 11 | KRX 자동차 추정 (069500) | β 0.73, 2026 vol 2.12× | 환율 효과 분리 불가 |
| 305540 | TIGER 2차전지테마 | `notebooks/final_eda/09_305540_final_eda.ipynb` | 11 | FnGuide 2차전지 추정 (069500) | 연 40.8%, MDD −72.5% | 안정 peer 없음 (DX-15) |
| 143860 | TIGER 헬스케어 | `notebooks/final_eda/10_143860_final_eda.ipynb` | 11 | KRX 헬스케어 추정 (069500) | KOSDAQ factor 에 붙음 (Δρ +0.275) | 저유동성, OHLC 경미 위반 |
| 117680 | KODEX 철강 | `notebooks/final_eda/11_117680_final_eda.ipynb` | 11 | KRX 철강 추정 (069500) | β 0.68, 2026 1.74× | 업종 고유 이벤트를 분리하려면 시장효과 제거 필요 |
| 161510 | PLUS 고배당주 | `notebooks/final_eda/12_161510_final_eda.ipynb` | 11 | FnGuide 고배당 추정 (069500) | 은행 factor 에 붙음 (Δρ +0.227) | 구성종목 미확인 |
| 219480 | KODEX 미국S&P500선물(H) | `notebooks/final_eda/13_219480_final_eda.ipynb` | 11 | S&P500 선물 H (133690) | 2026 0.73× — 국내 국면과 분리 | 거래시간 비동기 |
| 133690 | TIGER 미국나스닥100 | `notebooks/final_eda/14_133690_final_eda.ipynb` | 11 | NASDAQ-100 (219480) | 비헤지, 2026 0.87× | 환율 효과 혼재 |
| 192090 | TIGER 차이나CSI300 | `notebooks/final_eda/15_192090_final_eda.ipynb` | 11 | CSI 300 (069500) | KOSPI ρ 0.28 | 중국 휴장 불일치 |
| 148070 | KIWOOM 국고채10년 | `notebooks/final_eda/16_148070_final_eda.ipynb` | 11 | KTB 10년 추정 (153130) | 주식과 거의 무상관 (β 0.029) | proxy β 1.87 은 artifact (DX-14) |
| 153130 | KODEX 단기채권 | `notebooks/final_eda/17_153130_final_eda.ipynb` | 11 | 단기채 추정 (148070) | 연 0.32%, 현금성 | 분모·틱 문제로 kurt 324 (DX-10) |
| 132030 | KODEX 골드선물(H) | `notebooks/final_eda/18_132030_final_eda.ipynb` | 11 | S&P GSCI Gold 추정 (069500) | 2026 2.21× — 국내만의 현상 아님 | 위기일 헤지 아님, 선물 롤 |
| 261220 | KODEX WTI원유선물(H) | `notebooks/final_eda/19_261220_final_eda.ipynb` | 11 | S&P GSCI Crude 추정 (069500) | 연 47.9%, 2026 주식 상관 반전 | 롤오버 경로 의존, MDD −86.5% |
| 261240 | KODEX 미국달러선물 | `notebooks/final_eda/20_261240_final_eda.ipynb` | 11 | USD/KRW 선물 (069500) | 가장 일관된 반(反)주식 축, 2026 1.26× (DQ 제외) | 2019-03-14/15 price_anomaly — headline 은 두 행으로 표기 |

---

# 7. Reproducibility

| 항목 | 값 | 기록 위치 |
|---|---|---|
| run command | `ETF_RAW_DIR=<path>/data/raw bash scripts/run_all_final.sh` (PY / JUPYTER / JOBS 는 env 로 override 가능) | `scripts/run_all_final.sh` |
| runtime | 약 83초 (C, `840f8e8`) / 1m23s (외부 세션, `e9a6880`) | `reports/CLOSURE_MANIFEST.json` → `gate.clean_rerun`, `gate.clean_rerun_final` |
| clean worktree SHA | `e9a6880` (외부), `840f8e8` (C, 최종) | 동일 |
| table parity | 29/29, max\|Δ\| = 0 (rtol 1e-9) | `gate.clean_rerun` |
| figure parity | 243 장 바이트 동일 | `gate.clean_rerun`, `gate.clean_rerun_final` (비-ipynb 변경 0) |
| notebook execution | 41/41 오류 0 · 한글 glyph 경고 0 · 미실행 셀 0 (final 20 + universe 1 + 00_raw_eda 20). 바뀌는 것은 실행 metadata 뿐 | `gate.clean_rerun`, `gate.korean_font` |
| fallback behavior | 병렬 (xargs −P 6) 실행이 실패하면 순차로 1회 재시도. 원인은 jupyter kernel 포트 race 이고 출력에는 영향이 없다 | `scripts/run_all_final.sh`, `reproducibility_notes` |
| 결정성 | bootstrap seed 고정. 스크립트를 두 번 실행했을 때 table md5 가 같다 | `analysis/robustness/robustness_final.py`, `analysis/scripts/universe_final.py` |

참고: 외부 세션의 상세 재현 로그는 local bus (gitignored) 에만 있다. 공개 근거는 manifest 의 수치 요약이다.

---

# 8. Unresolved but Non-blocking Items

## 8.1 Closure backlog (기존)

| 항목 | severity | affected artifact | 수치 영향 | 외부 검토자가 볼 필요 |
|---|---|---|---|---|
| 국내 ETF 7종의 2026-07-31 캡션 누락 | LOW | `notebooks/final_eda/` 국내 7종 해석 셀 | 없음 (설명 문구) | 낮음 — KOSPI200 4종 notebook 에는 명시되어 있다 |
| Universe notebook 생성 스크립트가 repo 에 없음 | MED | `notebooks/universe/universe_eda.ipynb` | 없음 (notebook 자체는 재실행되고 출력이 같다) | 중간 — notebook 본문 수정 이력은 추적할 수 없다 |
| fig16 rolling β 해석 HOLD | LOW | `figures/universe/16_rolling_beta_geared.png` | 없음 (β 범위 수치는 유효) | 낮음 — β 가 튀는 시점은 극단일이 창을 지배한 결과다 |
| 그림 여백·범례 사소한 문제 4건 | LOW | `figures/universe/11`, `12`, `17`, `RELATION_MAP_FINAL` H5 제목 | 없음 | 없음 |

## 8.2 ISSUE — 이번 검토 중 발견 (수정하지 않고 기록만 함)

| ISSUE | 내용 | severity | 수치 영향 | 외부 검토자가 볼 필요 |
|---|---|---|---|---|
| I-1 | `reports/ROBUSTNESS_MATRIX_C_ADDENDUM.csv` 의 rolling corr 행이 `claim_id = DOCX-09` 로 되어 있다. 본 matrix 에서 DOCX-09 는 261240 kurtosis 이고, 이 행이 지지하는 claim 은 DX-11 (time-varying corr) 이다 | LOW | 없음 (라벨 오류) | 예 — ID 로 join 하면 잘못 묶인다 |
| I-2 | `reports/tables/etf_final_stats.csv` 의 261240 `vol_ratio_26` 은 DQ 포함 값 0.796 이다. universe 표와 ROBUSTNESS 는 DQ 제외 값 1.26 이다. notebook headline 에는 두 행이 모두 있다 | MED | 정의 차이 (계산 오류 아님) | 예 — 이 표만 보면 "축소"로 읽힌다 |
| I-3 | `reports/ROBUSTNESS_MATRIX.csv` 의 ARI/PC1 행 (DOCX-12, CL-13) 은 `ci_low`/`ci_high` 열에 CI 가 아니라 다른 요약값 (median 등) 을 담고 있다 | LOW | 없음 | 예 — 이 행들을 CI 로 읽지 말 것 |
| I-4 | `ROBUSTNESS_MATRIX.csv` 의 132030 2026 비율은 CI [1.39, 3.10] 가 1 을 넘는데도 verdict 가 FRAGILE 이다. verdict 기준이 행에 적혀 있지 않다 | LOW | 없음 | 참고 |
| I-5 | 로컬 저장소에는 `main` 브랜치가 없다 (원격 `origin/main` 만 있음). canonical 은 `origin/main` = `293e504` 로 확인했다 | INFO | 없음 | 없음 |
