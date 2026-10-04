# CANONICAL PATCH NOTE — Phase 00.5 (external audit)

- base: `origin/main` = `293e504`
- branch: `claude-c/postclosure-canonical-patch-v1`
- 외부 감사 판정: ANALYTICAL STATUS = CLOSED · CANONICAL PACKAGE STATUS = PASS_WITH_CANONICAL_PATCH

이 patch 는 evidence lineage, 표 의미, 통계 보고 방식만 정리한다. 수치 분석을 다시 하지 않았고 결론 수치도 바뀌지 않았다. notebook 21개와 figure 243장은 바이트 단위로 그대로다 (§5 QA).

## 1. 고친 것

| ISSUE | 문제 | 조치 | 바뀐 파일 |
|---|---|---|---|
| I-2 (P0) | `etf_final_stats.csv` 의 261240 `vol_ratio_26` = 0.7956 은 DQ 포함 값이다. claim 은 DQ 제외 값 1.2598 [1.0396, 1.4865] 를 쓰므로 같은 이름 아래 두 정의가 섞였다 | DQ 영향 지표 4개를 `*_raw` / `*_dq_excl` 로 나누고 `dq_policy` 열을 더했다. raw 값은 지우지 않았다 | `scripts/build_final_notebooks.py` (`stats_row`, `--stats-only`), `reports/tables/etf_final_stats.csv` |
| I-1 | addendum rolling-corr 행의 `claim_id=DOCX-09` 는 오기다. 본 matrix 에서 DOCX-09 는 261240 kurtosis 이다 | `claim_id=DX-11` 로 고치고 `dx_id` 열을 더했다 | `reports/ROBUSTNESS_MATRIX_C_ADDENDUM.csv` |
| I-6 | `CLAIM_EVIDENCE_MATRIX` 의 DX-01 이 KODEX200–TIGER200 상관 검정(DOCX-01)을 가리켰다. DX-02 는 sd 근거만 연결돼 있었다 | `robustness_rows` 열(`<파일>:<claim_id>`)을 더했다. 별도 검정이 없는 claim 은 `NONE` + "DESCRIPTIVE / NO SEPARATE ROBUSTNESS TEST" 로 적는다. DX-02 → DOCX-01 + DOCX-02 | `reports/CLAIM_EVIDENCE_MATRIX.csv` |
| I-3 | 7개 행의 `ci_low/ci_high` 에 CI 가 아닌 값(ARI Q1/median, PC1 min/max, dedup PC1, 정의 간 min/max)이 들어 있었다 | `interval_type`, `summary_stat`, `summary_low`, `summary_high` 열을 신설해 비-CI 값을 옮겼다. CI 열에는 `block_bootstrap_95` 와 `wilson_95` 만 남는다 | `analysis/robustness/robustness_final.py`, `reports/ROBUSTNESS_MATRIX.csv` |
| I-4 | 132030 금 (2.2055, CI [1.3898, 3.1004], BF p≪.001) 의 verdict 가 FRAGILE 이었다 | 추적 결과, 비국내 행에 `ci_low ≤ 1.2` 라는 임의 문턱을 적용해 생긴 라벨이었다. FRAGILE 을 만든 재현 가능한 민감도 검정은 없었다. 같은 규칙 때문에 WTI(lo 1.02)와 달러(lo 1.04)는 CI>1 인데도 ROBUST 였다. → 행 verdict 를 그 ETF 의 역할로 다시 정의했다 (아래). 기존 D 0120 의 ex-top3 leave-out 을 스크립트에 넣어 재현했다: 금 2.21→1.74, WTI 1.50→1.19, 달러 1.26→1.17. 셋 다 >1 을 유지한다 | 동상 |
| I-7 | Mann–Whitney p=0.0022 는 국내 12종에 KOSPI200 동일 exposure 4종이 반복된 표본이다 | verdict 를 SECONDARY 로 바꿨다. KOSPI200 4종을 069500 하나로 dedup 한 판을 추가했다 (n=9 vs 3, one-sided p=0.0045, exact 최소 p = 1/220). headline 근거는 within-series block CI 와 leave-out 이다 | 동상 |

## 2. Verdict taxonomy 변경 (ROBUSTNESS_MATRIX)

| 라벨 | 의미 | 적용 |
|---|---|---|
| ROBUST / FRAGILE / DEFINITION_DEPENDENT / CONTRADICTED | 기존과 같다 | — |
| **COUNTEREXAMPLE** (신규) | 행 자체는 통계적으로 확실하다(CI>1). 다만 DOCX 원 서술("국내주식 집중")의 반례다 | DOCX-07 132030 · 261220 · 261240 |
| **SECONDARY** (신규) | 독립성 가정이 약해 headline 근거로 쓰지 않는 보조 검정이다 | DOCX-07 Mann–Whitney 2행 |

DOCX-07 per-ETF 행의 규칙:
- 국내주식: CI>1 이면 ROBUST(지지), 아니면 FRAGILE
- 비국내: CI>1 이면 COUNTEREXAMPLE, 아니면 ROBUST(비확대)

| 집계 | 293e504 | patch |
|---|---|---|
| ROBUSTNESS_MATRIX 행 | 74 | 75 (+MW exposure-dedup) |
| ROBUST | 57 | 54 |
| FRAGILE | 5 | 4 |
| DEFINITION_DEPENDENT | 12 | 12 |
| COUNTEREXAMPLE | — | 3 |
| SECONDARY | — | 2 |
| CONTRADICTED | 0 | 0 |
| ADDENDUM | 5 (ROBUST 4 · FRAGILE 1) | 5 (같음) |

verdict 가 바뀐 행은 4개다:
- 132030: FRAGILE → COUNTEREXAMPLE
- 261220: ROBUST → COUNTEREXAMPLE
- 261240: ROBUST → COUNTEREXAMPLE
- MW n=12: ROBUST → SECONDARY

기존 74행의 `statistic` 과 실제 CI 값은 max\|Δ\| = 0 이다.

## 3. Claim ↔ robustness mapping

| DX | robustness_rows | 상태 |
|---|---|---|
| DX-01 | NONE | DESCRIPTIVE / NO SEPARATE ROBUSTNESS TEST |
| DX-02 | ROBUSTNESS_MATRIX:DOCX-01 (ρ) ; ROBUSTNESS_MATRIX:DOCX-02 (sd) | ROBUST |
| DX-03 | ROBUSTNESS_MATRIX:DOCX-03 | ROBUST |
| DX-04 | ROBUSTNESS_MATRIX:DOCX-04 | ROBUST |
| DX-05 | ROBUSTNESS_MATRIX:DOCX-05 | ROBUST |
| DX-06 | ROBUSTNESS_MATRIX:DOCX-06 | ROBUST |
| DX-07 | ROBUSTNESS_MATRIX:DOCX-12 ; ADDENDUM:DOCX-12 | k 의존 (ROBUST/FRAGILE) |
| DX-08 | ROBUSTNESS_MATRIX:DOCX-07 ; ADDENDUM:DOCX-07 | 국내 ROBUST · 비국내 COUNTEREXAMPLE · MW SECONDARY |
| DX-09 | ROBUSTNESS_MATRIX:DOCX-11 | DEFINITION_DEPENDENT |
| DX-10 | ROBUSTNESS_MATRIX:DOCX-09 (261240) ; ROBUSTNESS_MATRIX:DOCX-10 (153130) | ROBUST |
| DX-11 | ADDENDUM:DX-11 | ROBUST |
| DX-12 | NONE | DESCRIPTIVE (`universe_shock_anchor_survival.csv`) |
| DX-13 | ROBUSTNESS_MATRIX:DOCX-08 | DEFINITION_DEPENDENT |
| DX-14 | NONE | DESCRIPTIVE (`etf_final_stats.csv`) |
| DX-15 | NONE | DESCRIPTIVE (`universe_peer_stability.csv`) |
| DX-16 | NONE | DESCRIPTIVE (`universe_corr_period_diff.csv`) |

DOCX 밖의 보조 claim 인 CL-13 (PC1 구성 의존)과 CL-idio (idiosyncratic 기준 민감도)는 `dx_id` 를 비워 두었다. join audit 의 허용 목록이다.

## 4. 열 의미표 — DQ 정의 (261240 2019-03-14/15)

| 파일 | 열 | 정의 |
|---|---|---|
| `reports/tables/etf_final_stats.csv` | `ann_vol_raw`, `vol_ratio_26_raw`, `ex_kurt_raw`, `mkt_corr_raw` | 전체 일자 (DQ 포함) |
| 동상 | `*_dq_excl` | DQ 2일 제외 — **claim 기준**. DQ 일자가 없는 19종은 raw 와 같다 |
| 동상 | 나머지 열 (`vol_2026`, `vol_2019_25`, `ex_kurt_trim3`, `q01`, …) | raw (`dq_policy` 에 명시) |
| `reports/tables/universe_vol_rank.csv` | `ann_vol` / `ann_vol_trimdq` | raw / dq_excl |
| `reports/tables/universe_vol_ratio_2026.csv` | `vol_ratio_2026_vs_2019_25` | **dq_excl** (열 이름에는 없음 — I-8) |
| `reports/tables/universe_vol_by_year.csv` | 연도 열 | **dq_excl** (I-8) |
| `reports/ROBUSTNESS_MATRIX.csv` | DOCX-07 / 261240 | dq_excl (1.2598) |

I-8: universe notebook 이 이 열 이름을 직접 읽는다. notebook 불변 원칙에 따라 rename 하지 않고 이 표로 정의를 고정한다.

## 5. QA (patch 후)

| 검사 | 결과 |
|---|---|
| claim ID join audit (16 claim → robustness 행, `dx_id` 일치) | wrong 0 · orphan 0 |
| CI semantic audit (CI 행은 `interval_type` ∈ {block_bootstrap_95, wilson_95}, lo ≤ hi, Wilson 은 추정치 포함; 비-CI 행은 CI·interval_type 공란) | PASS |
| 261240 raw vs dq_excl | `etf_final_stats` 1.259816 = universe 1.25982 = ROBUSTNESS 1.2598. raw 0.795647 은 `_raw` 에만 있다. 비-DQ 19종은 raw = dq_excl |
| verdict count | manifest `robustness_verdicts` / `robustness_addendum_verdicts` = 표 집계 |
| Review Packet reference | `reports/review_reference_check.csv` missing 0 |
| 예상 밖 변경 | 추적 파일 md5 비교. 변경은 아래 목록뿐이고 notebook·figure·그 밖의 table 은 0건이다 |

변경 파일:
- `scripts/build_final_notebooks.py`
- `analysis/robustness/robustness_final.py`
- `reports/tables/etf_final_stats.csv`
- `reports/ROBUSTNESS_MATRIX.csv`
- `reports/ROBUSTNESS_MATRIX_C_ADDENDUM.csv`
- `reports/CLAIM_EVIDENCE_MATRIX.csv`
- `reports/EDA_EVIDENCE_CLOSURE.md`
- `reports/CLOSURE_MANIFEST.json`
- `reports/CHATGPT_REVIEW_PACKET.md`
- `reports/review_reference_check.csv`
- `reports/CANONICAL_PATCH_NOTE.md` (이 문서)

재현: `bash scripts/run_all_final.sh` 는 위 생성기를 그대로 호출하므로 patch 후 표를 다시 만든다. `robustness_final.py` 를 재실행했을 때 `figures/robustness/*.png` 와 `analysis/robustness/*.csv` 는 바이트 동일했다.

## 6. 바뀌지 않은 것
- 분석 결론: CONFIRMED 10 · CONFIRMED_WITH_LIMITATION 2 · REVISED 2 · CORRECTED 1 · DOWNGRADED 1
- ETF notebook 20, universe notebook 1, figure 243
- clean rerun 증거: `e9a6880` / `840f8e8` (manifest `gate.clean_rerun*`)
- non-blocking backlog 4건: 국내 7종 07-31 캡션, universe notebook 생성기 미편입, fig16 해석 HOLD, 그림 여백·범례
