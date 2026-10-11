# Track A — Research Session Log

| 날짜 | 세션 | 한 일 | 갱신한 곳 |
|---|---|---|---|
| 2026-10-11 | Claude A (sa-4d) | A1 데이터 계약: codex-a worktree 생성(d5c2f79), raw SHA 20/20, interim parity ≤1e-16, 기존 표 재현(vol/Pearson/Spearman 상대 3e-6, raw_validation 20종 일치), logdiff vs logratio Spearman 동률 함정 발견, 261240 DQ 2일 ρ −0.308→−0.458, 153130 0수익률 16.8%. 테스트 20 통과, A00 노트북 오류 0. run_id A1-20261011T142600-fd9713f7 | config/research_a_contract.yaml · src/r2h/research_a/ · tests/research_a/ · notebooks/research_a/00 · reports/r2h/research_a/a1/ |
| 2026-10-11 | Claude A (sa-4d) | A2: rolling pair 상관 51.7만 행(30/60/120/250 × P/S × raw/trim × ref 069500/102110), d0122 재현(5e-7; Spearman NaN창만 9e-5 정의차), 261240 DQ 영향기간, 153130 0수익률 진단. A3: C1~C5·P1~P5 사건, 46,000 검정(B=1000; p≤0.01 1,034개 B=20000 재추정 → BH 해상도 착시 정정), 기계적 동시성·사건일 제거 재검정, 쌍별 사례 51행(발견 38/부재 13). 테스트 34 통과 | src/r2h/research_a/{rolling_correlation,correlation_events,price_events,event_matching,robustness,visualization,a2_pipeline,a3_pipeline}.py · reports/r2h/research_a/a2,a3 · tests/research_a/test_a2_a3.py |
