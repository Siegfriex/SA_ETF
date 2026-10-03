
# ETF Raw EDA Research Scaffold

## 목표
20개 ETF를 **각 1개 notebook**으로 원자료 수준에서 탐색한다.
현재 단계는 final preprocessing / normalization / labeling / modeling 단계가 아니다.

## Numbering
기존 stage-numbering 습관을 살리되, `stage_slot`으로 분리했다.

- `00_01` ~ `00_20`: raw EDA, ETF 20종
- 추후 확장 예시
  - `10_xx`: preprocessing / normalization
  - `20_xx`: feature construction
  - `30_xx`: detector / modeling
  - `90_xx`: auxiliary / audit / release

현재 notebook은 **정확히 20개**다.

## Tree
```text
ETF_EDA_SCAFFOLD/
├─ config/
│  ├─ etf_universe.csv
│  └─ hypothesis_catalog.csv
├─ data/
│  ├─ raw/                 # 원본 수정 금지
│  ├─ interim/
│  │  └─ 00_profiles/
│  └─ processed/           # 현재 단계 미사용
├─ notebooks/
│  └─ 00_raw_eda/
│     ├─ 00_01_ETF01_raw_eda.ipynb
│     ├─ ...
│     └─ 00_20_ETF20_raw_eda.ipynb
└─ src/
   ├─ style.py
   └─ eda_utils.py
```

## 처음 해야 할 일
1. `config/etf_universe.csv`의 20개 placeholder를 실제 ETF로 교체
2. 각 raw file 경로 입력
3. 가능하면 `category`, `benchmark_ticker`, `peer_group` 입력
4. `primary_hypothesis`를 H01~H10 중 ETF 성격에 맞게 1개 지정
5. `data/raw/`는 immutable로 유지

## 모든 notebook의 공통 프레임
1. 가설 계약
2. provenance
3. schema/data quality
4. raw OHLCV
5. return/tail
6. volatility
7. trading activity
8. candidate event map
9. persistence/reversal
10. benchmark/peer
11. optional PCA
12. Observation → Implication → Decision
13. 종료조건

시각화는 별도 notebook으로 분리하지 않는다.

## 패키지
```bash
pip install pandas numpy matplotlib scipy statsmodels scikit-learn jupyter
```
Parquet: `pyarrow`, Excel: `openpyxl`
