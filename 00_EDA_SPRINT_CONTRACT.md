# 00 EDA Sprint Contract — ETF Universe Reconnaissance (Phase 00)

frozen_by: C · frozen_at: 2026-10-03T20:58+09:00 · sprint_t0: 2026-10-03T20:51+09:00 · deadline: 21:51

## 목적
한국 상장 ETF 20종을 동일 데이터 계약으로 확보하고, 개별(Layer I) · 상호(Layer II) · 전체/시간(Layer III) 구조를 탐색해
이후 preprocessing / normalization / surge definition / detector design 의 경험적 기반을 만든다. **객체는 급등이 아니라 Universe 자체.**

비목적: 뉴스 · PELT · XGBoost · forecasting · final surge label/threshold · 매매전략/포트폴리오 · agent trading.

## Universe
- 공식 수업 ticker list: 탐색 결과 **없음** (B 0001, C 확인: Digital_Twin/, ~/projects-wsl, SA/10.pdf).
- **UNIVERSE_STATUS=PROVISIONAL** — `fdr.StockListing("ETF/KR")` 모집단에서 선정. 공식 list 입수 시 교체.
- 선정 기준: 2020-01-01 이전 상장 · 2019-01-01 이후 가격 이력 충분 · 소형/비유동 제외 · 브랜드 편향 회피 · asset_scope/category 다양성.
  listing 실제 column 을 runtime 에서 확인 후 구현(필드명 추측 금지). 부족하면 pykrx 를 metadata 보조로만.
- 레버리지/인버스는 최대 2개(구조 대조용), 나머지는 1x.

## Taxonomy (config/universe.csv 필수 column)
ticker, name, provider_brand, listing_date, asset_scope, category, benchmark, peer_group, active_passive, leverage_inverse,
raw_file, universe_status, selection_reason. 주 grouping = asset_scope → category → benchmark → peer_group (브랜드 아님).

## Data contract
- primary: `fdr.DataReader(ticker, "2019-01-01")` → 최신 가용일. pykrx 는 NAV/거래대금/메타 보조만.
- 저장: `<ROOT>/data/raw/{ticker}.csv` (ROOT=/home/sieg/projects-wsl/hongik_univ_26_2/SA/ETF_EDA_SCAFFOLD, 모든 worktree 가 이 절대경로를 읽는다). raw 는 Git 금지.
  Git 에는 `data/raw/MANIFEST.csv`(ticker, rows, first, last, sha256, source, fetched_at)만.
- 가격 조정: FDR Close 를 그대로 쓰고 **adjusted 여부를 MANIFEST/notebook 에 명시**. 분배금·분할 의심 점프는 flag 만, 수정 금지.
- 기준 계산: rolling baseline 은 **shift(1)** (당일 제외). 양(+)/음(−)/절대 후보는 컬럼을 분리. volume 결측/0 은 명시적 처리.

## Notebook contract
- SSOT: `notebooks/_template/` 1개 + 생성 스크립트 → `notebooks/00_raw_eda/00_{slot:02d}_{ticker}_raw_eda.ipynb` ≤ 20.
- 모든 주요 plot/statistic 뒤 Markdown: **WHAT WE SEE / WHY IT MATTERS / WHAT WE DO NOT KNOW / NEXT QUESTION** (실제 수치 인용). 그래프만 = 미완료.
- 실행: kernel `hongik_26_2`, nbconvert --execute, 오류 0.

## Hypothesis policy
SEED / OBSERVED(근거 필수) / TESTABLE(RQ·H0·H1) / HOLD. ACCEPTED/REJECTED 금지. H01~H10 사전 배정 금지.

## Required final artifacts
config/universe.csv · config/event_anchors.csv · scripts/(collect, validate, build_notebooks, profile) · notebooks/_template · notebooks/00_raw_eda/*.ipynb ·
reports/ETF_CARDS/*.md · reports/universe_profile.csv · reports/ETF_RELATION_MAP.md · reports/SHOCK_MAP.md · reports/HYPOTHESIS_BACKLOG.csv ·
reports/EDA_INTELLIGENCE_BRIEF.md · reports/RUN_MANIFEST.json

## Evidence
T1 raw/runtime > T2 reproducible computation > T3 schema > T4 C decision > T5 docs > T6 narrative.
주장 type 구분: OBSERVATION / ANALYSIS / DECISION / PROJECTION.

## Git
remote = https://github.com/Siegfriex/SA_ETF.git 만. main 은 C 가 final gate 후에만 갱신.
branches: claude-a/eda-analysis-v1 · claude-b/eda-production-v1 · claude-c/eda-control-v1 · claude-d/eda-sandbox-v1 · integration/eda-hour1.
worktrees: SA/.worktrees/SA_ETF/{a,b,c,d}. 각 agent 는 자기 branch 에만 commit/push. 경로 소유권:
- A: reports/ETF_CARDS, reports/ETF_RELATION_MAP.md, reports/SHOCK_MAP.md, reports/HYPOTHESIS_BACKLOG.csv(A rows), analysis/
- B: config/, scripts/, src/, notebooks/, data/raw/MANIFEST.csv, reports/universe_profile.csv, reports/RUN_MANIFEST.json
- D: sandbox/ (non-canonical) · findings 는 bus 로만
- C: 00_EDA_SPRINT_CONTRACT.md, reports/EDA_INTELLIGENCE_BRIEF.md, 통합

## Bus
local-only `<ROOT>/.agent_bus/etf_eda_v1/{tickets,acks,completions,findings,heartbeats,locks}/` + event_log.jsonl.
파일명 `{seq:04d}_{FROM}_{TYPE}_{slug}.md`, immutable. 정정은 FACT_CORRECTION/SUPERSEDE. heartbeat 는 `heartbeats/{agent}.md` 덮어쓰기(≤180s).
