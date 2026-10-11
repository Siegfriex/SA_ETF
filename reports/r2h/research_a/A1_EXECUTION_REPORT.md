# A1 Execution Report — Data Contract

- run_id `A1-20261011T142600-fd9713f7` · input_data_version `fd9713f7f985e950…` · commit `2443f57`
- 산출: `config/research_a_contract.yaml` (1.1.0) · `src/r2h/research_a/{contract,data,qa,a1_pipeline}.py` · `reports/r2h/research_a/a1/` 13 표 · A00 노트북

| 검증 | 결과 |
|---|---|
| raw sha256 vs MANIFEST | 20/20 일치 · 각 1,903행 · 2019-01-02~2026-10-02 · 중복·역순·결측·0이하 가격 0 |
| canonical interim | panel_close 정확 일치, panel_logret index·열·NaN 0·값 ≤1e-16 |
| 기존 closure 표 재현 | ann_vol·Pearson·Spearman 상대 ≤3e-6, raw_validation.csv 5항목+점프일 20종 일치 |
| 수익률 구현 | logdiff(기본) vs logratio ≤1.84e-15 → 전체표본 Spearman 동률 함정(1.5e-6). rolling 창 안에서는 효과 0 (A2 확인) |
| 가격 조정 | ESTIMATED (Change 열은 Close 파생 → 독립 증거 아님) |
| DQ 220 | 212 OHLC(Close>High, 경로 없음) · 6 spike 후보 · 2 price_anomaly(261240) → ρ −0.308 → −0.458 |
| 비-DQ 대형 변동 | |r|>15% 10건 (261220 8 · 069500/102110 2026-07-31) |
| 환경 | 공용 .venv (python 3.12.3, pandas 3.0.3, numpy 2.4.6, scipy 1.18.0) · 설치 0 · FinanceDataReader 0.9.202 존재(감사 기록과 불일치, 미사용) |
