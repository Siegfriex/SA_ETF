# ETF_RELATION_MAP_FINAL — Phase 00.5 (A, C2-A1)

Supersedes `reports/ETF_RELATION_MAP.md` (b9fc6b4 draft, G3 ACCEPTED) as the canonical relation map. 기존 수치는 raw 에서 재계산해 일치 여부를 표기.
생성: `analysis/scripts/universe_final.py` (결정적, seed 20261004) → `reports/tables/universe_*.csv`, `figures/universe/*.png`. 해석 notebook: `notebooks/universe/universe_eda.ipynb`.
정의: r = log Close 수익률 2019-01-03~2026-10-02 (n=1902). 상관·군집은 261240 DQ 2일(2019-03-14/15, CL-05) 제외. Robustness 수치는 `reports/ROBUSTNESS_MATRIX.csv` (D) 인용.
**모든 관계는 동시성(co-movement)이며 인과가 아니다.**

## 1. 어떻게 묶이는가 — taxonomy vs statistical cluster

| # | 관계 | 수치 (재계산) | 근거 figure / table | Robustness | Claim |
|---|---|---|---|---|---|
| R1 | KOSPI200 동일 exposure: 069500≈102110 | Pearson 0.9990 (D CI [0.9987, 0.9992]) · 일간차 sd **8.05bp** [7.15, 9.11] | `05_corr_spearman_clustered.png` · `universe_claim_checks.csv` | ROBUST | CL-23 |
| R2 | 레버리지 = 2× 변환 | β(122630\|069500) **1.986** [1.952, 2.015]; 2019-21/22-25/2026 = 1.974/1.988/1.992 | `universe_claim_checks.csv` | ROBUST | — |
| R3 | 인버스 = −1× 변환 | β **−1.016** [−1.036, −0.997]; ρ −0.996 | 동상 | ROBUST | — |
| R4 | 헬스케어 143860 의 경험적 peer = 코스닥150 | Spearman 0.756 vs 069500 0.481, Δ **+0.28 [0.23, 0.32]**; 기간별 Δ 0.32/0.19/0.37 | `14_empirical_peer_rolling.png` · `universe_claim_bootstrap_ci.csv` | ROBUST (3기간 모두 >0) | DOCX §4.4 |
| R5 | 고배당 161510 의 경험적 peer = 은행 | Spearman 0.830 vs 0.603, Δ **+0.22 [0.18, 0.27]**; 기간별 0.16/0.25/0.24 | 동상 | ROBUST | DOCX §4.4 |
| R6 | 2차전지 305540 ↔ 철강 117680 | 2019-21 반도체 0.66 > 철강 0.54 · 2022-25 철강 0.57 > 반도체 0.42 · 2026 철강 0.82 > 0.54 | `14_empirical_peer_rolling.png` | **REGIME-DEPENDENT** (HOLD) | CL-17 |
| R7 | 114800 인버스 + 261240 달러 가 같은 군집 | signed 1−ρ average k=5 동일 군집 → 1−\|ρ\| 로 바꾸면 114800 은 069500 군집 | `06_dendrogram.png` · `universe_clusters.csv` | **ARTIFACT** (부호) | CL-14 |
| R8 | 군집 골격: 주식형(국내11+해외3)+WTI / 인버스·달러 / 채권·금 | avg k=3 [15, 2, 3]; 업종 간 분화는 이 거리에서 안 보임 | `06_dendrogram.png` | k=5 average/complete ROBUST (min ARI 0.53), ward·2026단독 FRAGILE (min 0.26), k=3 FRAGILE | CL-11, D DOCX-12 |
| R9 | PC1 = 국내주식 공통 factor, 비중은 구성 함수 | PC1 all20 **45.7%** → dedup17 39.3% → 비국내+069500 31.2%; D 변형 범위 0.43–0.52 | `09_pca_scree_loading.png` · `universe_pca_evr.csv` | DEFINITION_DEPENDENT | CL-13 |
| R10 | PC2 = 해외·원자재(+) vs 채권·금(−) | 133690 0.45 · 261220 0.42 · 192090 0.37 vs 148070 −0.45 · 132030 −0.29 | `09_pca_scree_loading.png` | 사후 라벨 | — |
| R11 | 달러 261240 = 국내주식의 비기계적 반대축 | Spearman ρ(069500) **−0.519**, Pearson −0.458 (DQ 제외) | `05_corr_spearman_clustered.png` | ROBUST (부호) | CL-22, CL-27 |
| R12 | 채권·금·단기채는 주식과 거의 무상관 | \|ρ\| < 0.2 전 쌍 | `05_corr_spearman_clustered.png` | — | CL-27 |

**결론 (how they group):** 20 ticker 는 대략 **국내주식 공통 factor 1개(KOSPI200 4종 중복 포함) + 해외주식/원자재 축 + 안전자산(채권·금) + 달러 반대축** 으로 압축된다. 경제적 taxonomy(asset_scope→category)는 해석의 축으로, 통계 군집은 중복·부호 artifact 를 보정한 뒤의 보조 증거로만 쓴다.

## 2. 언제 같이 움직이는가 — 상관은 상태변수

| # | 관찰 | 수치 | 근거 | 주의 |
|---|---|---|---|---|
| S1 | 국내주식 9종(K200 중복 제외) 120일 평균 pairwise ρ | 중앙값 0.51 · 최저 **0.27 (2023-09-11)** · 최고 **0.80 (2020-03-24)** | `08_rolling_corr_dispersion.png` · `universe_rolling_state.csv` | 120일 창 lag |
| S2 | 전체 17종 평균 ρ | 0.11 (2021-09) ~ 0.34 (2020-04) | 동상 | — |
| S3 | 상관 수준 ↔ 시장 변동성 | Spearman(120d 평균 ρ, 120d 평균 069500 vol) = **0.60** | 동상 | vol 상승 시 Pearson 상향 편향(Forbes–Rigobon) 미보정 |
| S4 | 공통 충격일 국내 지수형 참여율 0.73–0.78, 업종 0.43–0.68, CSI300 0.15 | rz3_D dedup≥5, n=40일 | `13_shock_participation_scatter.png` | 정의 종속 |

## 3. 언제 관계가 깨지는가

| # | 관찰 | 수치 | 근거 | 판정 |
|---|---|---|---|---|
| B1 | 2026 vs 2019-25 상관 구조 변화 | Fisher z \|z\|>3.4 (Bonferroni 근사) **60/190 쌍** | `07_period_diff_corr.png` · `universe_corr_period_diff.csv` | 독립 가정 낙관적 (D block bootstrap 방향 동일) |
| B2 | WTI–주식 부호 반전 | 261220–219480 **+0.33 → −0.61** (z −13.5) · 261220–069500 +0.29 → −0.36 | `15_wti_equity_sign_flip.png` | OBSERVED, 원인 범위 밖 · CL-31 |
| B3 | 반도체–지수 결합 강화 | 091230–069500 0.80 → **0.95** (z +9.1) | `07_period_diff_corr.png` | OBSERVED · CL-18 |
| B4 | 2차전지 peer 전환 | R6 참조 | `14_empirical_peer_rolling.png` | REGIME-DEPENDENT · CL-17 |
| B5 | 2026 변동성 국면 | σ2026/σ2019-25: K200 계열 **3.3배** [2.5, 4.2], 반도체 2.55, **금 2.21 [1.39, 3.10]**, WTI 1.50 [1.02, 2.15] vs 해외주식 0.73/0.87/0.94 | `02_vol_year_heatmap.png` · `universe_vol_ratio_2026.csv` · ROBUSTNESS_MATRIX DOCX-07 | DOCX "국내주식 집중" → **하향: domestic equity + commodity, 해외주식 아님** |

## 4. DOCX claim 대비 변경

- "2026 변동성 확대는 국내주식에 집중" → **domestic equity + commodity(금 2.21×, WTI 1.50×), 해외주식 아님** (D 반례).
- "달러 주식 폭락일 86% 상승" → 어떤 정의로도 86% 재현 안 됨: 069500 ≤q05 0.80, ≤−2% 0.81, rz≤−3(D식) 0.91, ≤−3% 0.72 (ROBUSTNESS_MATRIX DOCX-08). **정의와 함께 범위 0.72–0.91 로 인용**.
- "달러–KOSPI ≈ −0.52" → Spearman −0.519 / Pearson −0.458 (척도 명시).
- "통계 cluster ≠ taxonomy" → 유지, 단 군집 경계는 linkage/기간에 FRAGILE (ward·2026 단독).
- 148070 의 DOCX 'corr 0.100 · beta 1.87' 은 benchmark proxy=153130(단기채) 기준이라 시장관계 지표로 쓰면 안 됨 (C 지적; ETF notebook 에서 069500 기준 병기).

## 5. 이 지도로 말할 수 없는 것
높은 상관·같은 군집은 같은 경제적 원인이나 인과를 증명하지 않는다. 해외 ETF 는 KRX 종가와 기초시장 종가가 비동기라 동일일 상관이 과소추정될 수 있다(CL-15). holdings·NAV 없이 '헬스케어=코스닥 factor' 의 메커니즘은 검증되지 않았다.
