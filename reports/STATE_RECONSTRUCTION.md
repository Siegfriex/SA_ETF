# STATE RECONSTRUCTION — Phase 00.5 (EDA Evidence Closure) · C · 2026-10-04 12:28 KST

검증 근거: `git branch -a -vv`, `git worktree list`, `.agent_bus/etf_eda_v1/` (59 파일), 실제 산출물, reference DOCX 본문 추출. 기억/handoff 는 bootstrap hint 로만 사용.

## Git (실측)
| branch | HEAD | 원격 일치 | 내용 |
|---|---|---|---|
| integration/eda-hour1 | cd09224 → (C 작업) | origin 동일 시작 | A SHOCK_MAP + C-0022 cards + RELATION_MAP draft + B run_all |
| claude-a/eda-analysis-v1 | b9fc6b4 | 일치 | RELATION_MAP "wip" (G3 에서 내용상 ACCEPTED) · 미추적 `analysis/shock/rz60.csv` |
| claude-b/eda-production-v1 | 011979e | 일치 | run_all.sh, env override. clean clone 재현성 미완 |
| claude-c/eda-control-v1 | 09934b9 | 일치 | integration + EDA_INTELLIGENCE_BRIEF.md v0 → integration 으로 merge 완료 |
| claude-d/eda-sandbox-v1 | 03e940a | 일치 | sandbox 26 산출물 (regime, shock sens, counter) |
| main (origin) | 042cf2d | — | Initial commit 만. promotion 미실시 |

## 산출물 상태
| 항목 | 상태 | 근거 |
|---|---|---|
| raw 20/20 (1,903행, 2019-01-02~2026-10-02) | ACCEPTED | data/raw/MANIFEST.csv, CL-02 (B+D 독립 2중) |
| config/universe.csv taxonomy | ACCEPTED_WITH_LIMITATION | PROVISIONAL universe (CL-01), benchmark 추정 표기 |
| 00_raw_eda notebook 20 | ACCEPTED_WITH_LIMITATION | PASS 19 / PASS_WITH_LIMITATION 1. 1개 재실행 12s 정상. figure 밀도·해석 4섹션 기준 미달 → final 로 대체 |
| ETF_CARDS 20 | ACCEPTED | efada2e + C-0022 970032e. **notebook 재주입 미완** (B C-0021) |
| HYPOTHESIS_BACKLOG 47 | ACCEPTED_WITH_LIMITATION | curation 미완 |
| ETF_RELATION_MAP.md | ACCEPTED (G3) → FINAL 필요 | 시각 증거·period-diff·dedup 부족 |
| SHOCK_MAP.md | ACCEPTED (G3) → FINAL 필요 | 2025-04 anchor 누락 → event_anchors A5 FRAGILE 로 추가 (본 세션) |
| D sensitivity | INCOMPLETE | idio 0.75/1.25 미수행 (v1 0033) |
| universe_profile.csv | ACCEPTED | DEC-7 kurt 정의 일치 (diff 4.4e-16) |
| clean-clone reproducibility | INCOMPLETE | B C-0021 미완 |
| EDA_INTELLIGENCE_BRIEF v0 | SUPERSEDED (예정) | → EDA_EVIDENCE_CLOSURE.md |
| Universe notebook / Visualization atlas / Robustness matrix | 없음 | 본 Sprint 생산 대상 |
| 한글 폰트 | ACCEPTED | src/kfont.py, Noto Sans CJK KR, missing glyph 0 (NanumGothic 은 U+2212 결손 → 2순위) |

## DOCX 와 데이터의 이미 확인된 불일치 (CLAIM_EVIDENCE_MATRIX 반영)
1. 148070 "corr 0.100 · beta 1.87" 는 proxy=153130(단기채) 대비 값 → benchmark artifact. 069500 기준 병기 필요.
2. k≥5 공통충격 "52/22/175": 52 는 D 식 MAD, A 식은 61 (CL-32). MAD 식 명시 필요.
3. 달러 "폭락일 86% 상승" vs D-020 "34일 0.91": 폭락일 정의 상이.
4. 달러 "이상치 제외 kurt 3.9" vs trim3 2.4: 제거 방식 상이 — 병기.
5. 153130 "corr 0.100" 은 148070 과의 값 (DOCX 에 benchmark 명시 없음).

## 이번 Sprint 운영
- v2 bus: `.agent_bus/etf_eda_v2/` (local only). 경로 소유권 분리 · commit 은 C 만 (integration 에서).
- A = universe notebook + RELATION/SHOCK FINAL · B = 20 final notebook + run_all_final · D = ROBUSTNESS_MATRIX + 미완 민감도.
