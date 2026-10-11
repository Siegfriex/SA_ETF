# Track A → Track C 인계 명세 (비차단)

Track C 는 아직 시작하지 않는다. 이 문서는 A 결과를 C 에서 B(Hybrid SVR–GARCH CUSUM) 결과와 비교할 때 필요한 **인터페이스**만 정의한다. A 는 B 코드·결과를 쓰지 않았다.

## 1. 사건 테이블 schema (A Result Contract)
파일: `a3/correlation_event_candidates.csv`, `a3/price_event_candidates.csv` (같은 열; 적용 불가 필드는 null)

| 열 | 타입 | 의미 |
|---|---|---|
| event_id | str(12) | sha1(series, ref, definition, params, window, method, candidate_at) |
| series_id / reference_series_id | str(6) | ETF ticker (선행 0 보존) / 기준 ticker (가격 사건은 null) |
| asset_class | str | A comparison_group (bond_long, fx_usd …) |
| definition_id / definition_version | str / "1.0" | C1~C4 · P1~P5 (C5 는 `correlation_events.c5_method_concordant`) |
| parameter_set | JSON str | 정의 파라미터 (posthoc_label, lookahead 플래그 포함) |
| event_family / direction | correlation·price / +1·−1 | 부호 분리 |
| candidate_at / confirmed_at / available_at / event_end | date | 실시간 사용 시 **available_at 만** 사용 |
| score / window / method | float / int·str / pearson·spearman | C4 window 는 "30+60" 형식 |
| quality_flag | str·null | dq_in_window / dq_flag_day |
| data_snapshot_id / run_id | str | input_data_version[:16] / A3 run |

## 2. 반드시 보존할 정의 차이
1. 수익률 = logdiff, `dq_trim_canonical` (261240 2019-03-14·15 NaN). B 가 다른 산식·DQ 처리를 쓰면 같은 날짜 비교 전에 맞출 것.
2. `candidate_at` ≠ 실시간. CUSUM 경보 시점과 비교할 때는 `available_at` 를 쓸 것.
3. KOSPI200 exposure group(069500·102110·122630·114800)은 사건 1건.
4. robust-z 정의 ID(P3_rz_D / rz_A / rz_eda) — canonical 미정.
5. P4 · P2_full · P5_q99 는 사후·look-ahead.
6. 같은 날 큰 수익률의 기계적 중복 — CUSUM 경보·상관 사건·가격 사건 모두 같은 관측치에 반응할 수 있음 → leave-event-days-out 비교 권장 (`a3_pipeline.mechanical_removed_retest`).

## 3. 재사용 함수
`event_matching.match_events` · `circular_shift_test_fast` · `bh_qvalues` 는 임의 사건 테이블(위 schema)에 그대로 적용 가능.
