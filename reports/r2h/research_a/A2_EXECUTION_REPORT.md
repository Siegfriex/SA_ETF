# A2 Execution Report — Cross-Asset Rolling Correlation

- run_id `A2-20261011T145943-fd9713f7` · 실행 47 s · 최대 RSS 369 MB · commit `95b93ea` (코드) / `a76c7fd` (노트북)
- 노트북: A01 `01_채권_코스피상관시각화.ipynb` (7.6 s), A012 `012_기타_코스피_상관_시각화.ipynb` (≈6 s), 오류 0, 해석 수치 assert
- 파생: `data/processed/research_a/A2-…/rolling_correlations.parquet` 517,344 행 (ref 069500·102110 × raw·dq_trim × 30/60/120/250 × Pearson·Spearman)

| 표 (`reports/r2h/research_a/a2/`) | 내용 |
|---|---|
| etf_pair_manifest.csv | 쌍 · 기준 역할 · session_sync · 환헤지 · 노트북 |
| correlation_crossings.csv | C1 교차 + entering-day driven + 오차대 체류 |
| correlation_regime_candidates.csv · regime_counts_by_rule.csv | C3 규칙 grid (b 0/0.1/0.2/null, k 5/10/20) · candidate/confirmed |
| window_sensitivity.csv | 창 × 방법 요약 · 기간별 평균 |
| static_corr_by_period.csv | 전체·기간·연도별 정적 상관 |
| control_102110_comparison.csv | 069500 vs 102110 기준 차이 |
| data_quality_sensitivity.csv | 261240 raw/trim, spike 마스킹, 261220 >15%, 069500 2026-07-31 |
| zero_return_diagnostics_153130.csv | 0→NaN (index 유지, n_eff 기록), Pearson/Spearman |
| rank_policy_sensitivity.csv · min_periods_sensitivity.csv | 반올림·산식·min_periods 효과 (rolling 0) |
| lead_lag_corr.csv · async_aggregated_corr.csv | lag −3..3, 2/5일 합산, Dimson |
| legacy_parity_d0122.csv | 기존 평균 pairwise 60/120/250 재현 (NaN 창 밖 ≤5e-7) |

핵심 수치: 148070 60일 평균 ρ 0.037, 오차대 체류 51%, 교차 45 → C3 13 · 153130 30일 교차 137 (0→NaN 97, n_eff 25.0) ·
261240 DQ 영향 30/60/120일 ~04-25/06-11/09-04 · 주식·달러 부호 고정 · 원유 2026 −0.30.
