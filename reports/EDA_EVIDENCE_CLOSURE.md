# EDA EVIDENCE CLOSURE — Phase 00.5 (ETF Universe, 20 ETFs × 1,902 returns)

C (Research Director) · 2026-10-04 · branch `integration/eda-hour1` · reference: `ETF_Universe_Domain_EDA_Report.docx` (CLAIM MAP, 정답 아님)
범위: EDA evidence 만. 뉴스 · PELT · XGBoost · forecasting · trading agent · portfolio 는 다루지 않았다. 모든 관계는 동일일 통계적 동행이고 인과가 아니다.

**GATE 판정은 `reports/CLOSURE_MANIFEST.json` 의 `gate` 를 따른다. 이 문서는 내용을 요약한다.**

## 1. 어디서 무엇을 보는가

| 질문 | 열 곳 |
|---|---|
| ETF 하나: 왜 이런 ETF인가, 실제로 어떻게 움직였나, benchmark 와의 관계, 어느 그래프가 결론을 지지하나 | `notebooks/final_eda/NN_<ticker>_final_eda.ipynb` (20개, 각 11 figure + 4섹션 해석 + 맨 끝 "결론 ↔ 지지 figure" 표) |
| 20개가 어떻게 묶이고, 언제 같이 움직이고, 언제 관계가 깨지나 | `notebooks/universe/universe_eda.ipynb` + `reports/ETF_RELATION_MAP_FINAL.md` |
| 공통 충격은 언제였고, 정의에 따라 얼마나 달라지나 | `reports/SHOCK_MAP_FINAL.md` |
| DOCX 의 각 주장이 어디까지 맞나 | `reports/CLAIM_EVIDENCE_MATRIX.csv` (16 claim) · `reports/ROBUSTNESS_MATRIX.csv` (74 test) |
| 모든 그림 | `reports/VISUAL_INDEX.md` |
| 다시 만들기 | `bash scripts/run_all_final.sh` (`ETF_RAW_DIR` = raw 경로, 약 1분) |

## 2. DOCX claim 판정 요약

| # | DOCX claim | 판정 | 핵심 수치 (95% CI = block bootstrap, block 10) | 대표 figure |
|---|---|---|---|---|
| DX-01 | 변동성 scale 175배 | CONFIRMED | 0.32% (153130) ~ 56.65% (122630); 261240 은 DQ 2일 제외 시 13.49% → 8.9% | universe/01 |
| DX-02 | KODEX200–TIGER200 ρ≈0.999 | CONFIRMED | ρ 0.9990 [0.9987, 0.9992], 일간차 sd 8.05bp | etf/102110/07 |
| DX-03 | 레버리지 β≈1.99 | CONFIRMED | β 1.986 [1.95, 2.02] | etf/122630/07·08 |
| DX-04 | 인버스 β≈−1.02 | CONFIRMED | β −1.016 | etf/114800/07·08 |
| DX-05 | 헬스케어는 KOSDAQ150 과 더 가깝다 | CONFIRMED | Δρ +0.275 [0.23, 0.32] | universe/05·14 |
| DX-06 | 고배당은 은행과 가깝다 | CONFIRMED | Δρ +0.227 [0.19, 0.27] | universe/05·06 |
| DX-07 | 통계 cluster ≠ taxonomy | CONFIRMED_WITH_LIMITATION | 114800+261240 동반은 부호 artifact. cluster 경계는 ward·2026 단독에서 FRAGILE (min ARI 0.15) | universe/06 · robustness/cluster_ari |
| DX-08 | 2026 vol 확대는 국내주식 집중 | **REVISED** | 국내주식 12/12 >1, 069500 3.3× (top3일 제외 2.9×), 해외 0.73–0.94. 다만 금 2.21×, WTI 1.50×, 달러 1.26× (DQ 제외)도 확대 → "국내주식 집중, 해외주식 아님, 원자재·달러도 확대" | universe/02 · robustness/vol_ratio |
| DX-09 | shock 수는 정의에 민감 | CONFIRMED | k≥5: rz3(D식 MAD) 52 · rz3(A식 MAD) 61 · q99 22 · abs3 175 → 최대/최소 7.95× | universe/10 · robustness/shock_grid |
| DX-10 | 단기채·달러 kurtosis 는 극단치 영향 | CONFIRMED | 153130 323 → trim3 6.1; 261240 300 → trim3 2.4 / anomaly 2일 제거 3.9 | robustness/kurtosis |
| DX-11 | corr 자체가 time-varying state | CONFIRMED_WITH_LIMITATION | 국내 9종 120D 평균 ρ 0.27–0.80, anchor 에서 급등. 2026 vs 2019-25 유의 변화 44/136 쌍 (dedup, Fisher z 독립가정) | universe/07·08 |
| DX-12 | 공통충격 anchor | CONFIRMED | A1 2020-03, A2 2024-08-05, A3 2026-03-04 는 4정의×dedup 생존. A5 2025-04-07/10 FRAGILE (q99 4종) | universe/11·12 |
| DX-13 | 달러는 폭락일 86% 상승 | **REVISED** | 정의에 따라 72–91% (≤−3% 0.72 · ≤q05 0.80 · ≤−2% 0.81 · rz≤−3 0.91). 86% 는 재현되지 않음 | universe/12 |
| DX-14 | 148070 corr 0.100 / β 1.87 | **CORRECTED** | proxy 153130 의 분모 극소 artifact (R² 0.01). 069500 기준 β 0.029, ρ 0.134 | etf/148070/07 |
| DX-15 | 2차전지는 일부 기간 철강과 가깝다 | **DOWNGRADED** | 철강 vs 반도체 Δρ>0 인 날 54% → 안정적 peer 없음 (REGIME_DEPENDENT) | universe/14 |
| DX-16 | WTI–주식 2026 부호 반전 | CONFIRMED | +0.12~0.48 → −0.36 (KOSPI); WTI–달러 −0.30 → +0.40 | universe/15 |

반박(CONTRADICTED)된 claim 은 없다. 수정 3건(DX-08, DX-13, DX-14), 하향 1건(DX-15)이다.

## 3. 무엇이 새로 확인되었나 (DOCX 이후)
- **2026-07-31 +21.7% 는 데이터 오류가 아니다**: 102110 +22.1%, 122630 ≈2×, 114800 ≈−1×, 앞뒤도 극단일이다. 하루 단위 오류가 아니라 국면이다 (D 0120).
- **robust-z 이름만으로는 재현할 수 없다**: MAD 추정식(D식: 잔차 rolling median, mp20 / A식: 창 내 MAD, mp40)에 따라 k≥5 공통일이 52 와 61 로 갈린다. final notebook 의 rz60 은 A식이고, notebook 에 정의를 명시했다.
- **idiosyncratic shock 기준은 FRAGILE 하다**: universe median|rz| 기준을 1.0 에서 0.75 로 조이면 사례가 61% 만 남는다.
- **PC1 비중은 universe 구성에 따라 달라진다**: 전체 0.457, K200 중복 제거 0.393. universe 의 특성으로 보고하지 않는다.

## 4. 이 EDA 가 주장하지 않는 것
- 상관이나 cluster 가 같은 경제적 원인, 또는 인과를 뜻한다는 주장.
- OHLCV 만으로 저유동성 괴리(NAV)와 펀더멘털 충격을 분리할 수 있다는 주장.
- anchor label(COVID 등)은 날짜를 식별하려는 것이고 원인을 귀속하지 않는다.
- universe 가 대표성을 갖는다는 주장: PROVISIONAL 20종이고, 현재 상장·현재 규모로 골랐으므로 survivorship 이 있다.

## 5. 남은 한계 (blocker 아님)
- Fisher z 기간 비교는 독립성을 가정하므로 낙관적이다.
- 해외 ETF 는 KRX 와 거래시간이 달라 동일일 상관이 과소 추정될 수 있다 (CL-15).
- benchmark 기초지수 데이터가 없어 ETF proxy 를 쓴다. 추적오차는 측정하지 않았다.
- 00_raw_eda notebook(Phase 00)은 보존한다. C-0022 카드 재주입은 final notebook 이 카드를 직접 참조하므로 SUPERSEDED.
- `notebooks/universe/universe_eda.ipynb` 는 commit 된 notebook 자체가 원본이다. 재실행은 되지만 생성기 스크립트는 repo 에 없다. 20 final notebook 은 `scripts/build_final_notebooks.py` 로 다시 만들어진다.
- fig16 (rolling 60D β): β 가 크게 튀는 시점은 anchor 일이 창에 들어오고 나가는 날과 겹친다. 이를 해석하지 않는다 (단일 극단일이 창을 지배한 결과).
- fig08 Pearson 평균상관의 계단도 단일 극단일 효과다. Spearman 선을 병기했고, 수준 차이 결론은 60/120/250 × Pearson/Spearman 에서 유지된다 (D 0122).
- 2026-07-31 캡션은 KOSPI200 계열 4종과 일부 국내 ETF 에만 있다. 나머지 국내 7종은 backlog 로 넘긴다. 사소한 그림 여백·범례 4건(A 0131)도 backlog 로 넘긴다.
- PCA 는 상관행렬 기준이다 (cov 0.565 / rank 0.430). cluster 는 k=5 에서 leave-one-ETF-out ARI ≥0.98 로 ROBUST, k=8 에서 0.40 으로 FRAGILE 이다.

## 6. 검수 이력 (요약)
D 0103 (74 test) → D 0120 (07-31 SUPPORT, vol ratio 버그 발견) → A 0121 (P0 3 · P1 10 · P2 3) → 재작업 C2-A2 · B headline → A 0131 재검수 16/16 해소 → D 0122 (window/LOETF) · D 0123 (notebook 정적 검사) → freeze e9a6880 → clean 재현.
