# 공개 브랜치에서 뺀 산출물

SA_ETF 공개 브랜치에는 코드·테스트·보고서·요약표·그림·노트북만 담았다. 아래 디렉터리는 크기(반복별 원시 결과) 또는
재배포 권리(Yahoo Finance 시계열) 때문에 제외했다. digest = 정렬된 `경로 sha256` 줄들의 sha256 (로컬 원본 기준, 2026-10-12).
MC·ablation 결과는 seed 파생이 결정적이므로 README 의 재현 명령으로 다시 만들 수 있다 (재생성 일치 확인: R2H `research/REPRODUCTION_STATUS.md`).
원시 시계열은 `data/raw/data_manifest.json` 의 URL·기간·sha256 으로 다시 받아 대조한다. 원문 PDF 는 DOI 10.3390/e22050578.

| 경로 | 파일 수 | 바이트 | digest |
|---|---|---|---|
| `results/mc/` | 1486 | 18,563,795 | `aa7f19518b7a45e0013498eeb44a9ca233eec28b93cb84a82f90cfa47fe6bbc2` |
| `results/ablation/` | 5402 | 63,261,226 | `9a1a400df285b064d33016a790e8f2069f8f67a6685d41e5560eb00e69097928` |
| `data/raw/` | 3 | 119,127 | `614c6323bc1215a7c78dddca0acd96a68bd77799d8a3b66f89611050786e8c23` |
