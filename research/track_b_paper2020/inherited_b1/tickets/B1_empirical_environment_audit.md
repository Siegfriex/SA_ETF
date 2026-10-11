# B1 empirical data and environment source audit

> 범위: Lee, Kim & Lee (2020)의 영어 원문 §5/Figures 1–4와 원 저자 실행 환경, 현재 로컬 실행 환경 및 원 실증 데이터의 로컬 존재 여부만 감사한다. 외부 다운로드·패키지 설치·실험 실행은 하지 않았다.

## 1. 감사 기준과 소스 식별

| 항목 | 확인값 | 근거 |
|---|---|---|
| 원문 PDF | `/home/sieg/projects-wsl/hongik_univ_26_2/SA/1011/Hybrid_SVR_GARCH_CUSUM_.pdf` | 로컬 파일 직접 확인 |
| PDF SHA-256 | `97c95e308af8dd95e8d283f01674932f52349ab8b4d0ea2a93d4dbe45fcf3cd6` | `sha256sum` |
| PDF 페이지 수 | 45 | `pdfinfo` |
| 영어 원문 범위 | PDF 30–45쪽, 논문 표기 1–16쪽 | PDF 본문 머리말/쪽번호 직접 확인 |
| 이 감사의 핵심 원문 페이지 | 실행 환경: PDF 36쪽(논문 7쪽); §5와 Figure 1: PDF 40쪽(논문 11쪽); Figures 2–4: PDF 41쪽(논문 12쪽); 결과 서술: PDF 41–42쪽(논문 12–13쪽) | 텍스트 추출과 150 dpi 원본 렌더링을 함께 대조 |

## 2. §5 실증 데이터 계약

영어 원문의 명시 사항만 전사하면 다음과 같다.

| 데이터 | 원문 명칭/대상 | 전체 기간 | 입력/분석량 | 출처 | 훈련 구간 계약 | Figure |
|---|---|---|---|---|---|---|
| S&P 500 | daily index of the S&P 500 | 2012-01-03–2016-09-30 | 원 지수와 그 일별 로그수익률 | `investing.com` | 전체 시계열의 앞쪽 절반. 다만 본문 예시는 `2012-04-02–2014-04-16`이라고 명시 | Figure 1(a), Figure 2 |
| KOSPI | Korea Composite Stock Price Index (KOSPI) | 2012-10-02–2019-06-28 | 원 지수와 그 일별 로그수익률 | `investing.com` | 전체 시계열의 앞쪽 절반; 정확한 시작/종료일은 원문에 없음 | Figure 1(b), Figure 3 |
| KRW/USD | exchange rate between USD and KRW, denoted KRW/USD | 2012-01-02–2016-09-30 | 원 환율지수와 그 일별 로그수익률 | `investing.com` | 전체 시계열의 앞쪽 절반; 정확한 시작/종료일은 원문에 없음 | Figure 1(c), Figure 4 |

근거는 PDF 40쪽(논문 11쪽) §5 첫 문단이다. 여기서 원문은 세 시계열을 `investing.com`에서 얻었다고만 적고, URL·다운로드 일시·심볼·열 이름·snapshot hash는 제공하지 않는다.

### 훈련/검정 분할과 Figure 1

- 각 전체 금융시계열의 **첫 절반(first half)** 이 training time series이다(PDF 40쪽/논문 11쪽).
- Figure 1은 각 training 시계열의 **로그수익률**에 대해 ACF와 PACF를 **lag 25까지** 표시한다.
- 원문은 이 ACF/PACF가 세 training 데이터의 정상성을 대체로 지지한다고 해석한다. 이는 시각적 서술이며, 별도의 단위근/정상성 검정통계량은 제시하지 않는다.
- Figures 2–4에서 training 구간은 노란 음영, 후반 구간은 연두 음영, 검출 변화점은 빨간 수직선으로 나타난다(PDF 40–41쪽/논문 11–12쪽).
- Figure 2–4 각각은 패널 (a) 원 가격/지수, 패널 (b) 로그수익률을 제시한다.

### 원문 목표값

| 데이터 | 변화점 위치 | CUSUM statistic `T` | 5% 판정 | 근거 |
|---|---:|---:|---|---|
| S&P 500 | 2015-08-18 | 2.4751 | 기각 | PDF 41쪽 하단–42쪽 상단(논문 12–13쪽) |
| KOSPI | 2018-01-22 | 1.5081 | 기각 | PDF 42쪽(논문 13쪽) |
| KRW/USD | 2014-09-26 | 4.2754 | 기각 | PDF 42쪽(논문 13쪽) |

원문 §3의 5% 임계값은 `1.3397`이므로 세 값 모두 이를 넘는다. **변화점 날짜와 통계량은 회귀 테스트의 목표값이지, 이를 맞추기 위한 사후 파라미터 튜닝의 허가가 아니다.**

## 3. 원문 내부 재현 불명확성

| ID | 불명확성/충돌 | 영향 | Phase B2/B3에서 필요한 결정 또는 증거 |
|---|---|---|---|
| EMP-01 | S&P 500 전체 기간은 `2012-01-03` 시작인데, 같은 문단의 training 예시는 `2012-04-02–2014-04-16`이다. Figure 2의 x축과 노란 음영은 시각상 2012-01-03부터 시작하는 것처럼 보인다. | 첫 절반의 관측치 수와 SVR 학습 표본이 달라져 목표 통계량/날짜가 바뀔 수 있음 | 원 저자 데이터 또는 코드 확보 전에는 자동 보정 금지. 두 해석을 별도 계약으로 보존하고 우선 원문 문장 그대로 구현 후보로 둔다. |
| EMP-02 | `first half`가 달력기간 절반인지, 정제 후 관측행 수의 앞 절반인지 명시되지 않는다. KOSPI/KRW/USD의 정확한 midpoint 날짜도 없다. | train/test 경계와 lag/proxy 경계 변경 | 원 snapshot의 행 수·거래일을 복원한 뒤 인덱스 기준 후보를 명시적으로 비교해야 함 |
| EMP-03 | 로그수익률 식, 자연로그 여부, `×100` 스케일, 첫 결측행 처리 방식이 없다. Figure y축은 `%`로 보이지만 식은 없다. | SVR `C`, `epsilon`, 분산 proxy의 수치 스케일과 CUSUM 잔차가 달라짐 | raw index에서 가능한 변환을 선등록하고 목표값을 보지 않은 상태에서 원문 관례/그림 축과 일치하는 정책 확정 |
| EMP-04 | Investing.com의 정확한 symbol/market, 가격 열(Open/Close/Adjusted), 거래일·휴장일, 결측/중복/정렬, 소수점/지역 locale, 수정 이력과 다운로드 시점이 없다. | 논문 당시 snapshot의 정확 재현 불가 가능성 | B3 전에 원본 snapshot 및 checksum 확보; 불가능하면 method-equivalent와 numerically exact를 분리 보고 |
| EMP-05 | 실증 데이터에 사용한 `(C, gamma^2, epsilon)` grid의 간격, tie-break, feature scaling 여부, 선택 결과가 보고되지 않는다. | 동일 데이터라도 모델/잔차/변화점 불일치 | 원 저자 코드 확보 또는 모든 미명시 선택을 gap register에 고정하고 민감도 결과를 원문 복제와 분리 |
| EMP-06 | 난수 seed는 보고되지 않는다. SVR solver의 tolerance/cache/shrinking 설정도 없다. | 최적화 세부 차이가 수치 재현에 영향 | deterministic 설정·solver metadata 기록; R parity run이 가능하면 별도 수행 |
| EMP-07 | Figure 1은 정상성을 시각적으로 지지한다고만 하며 confidence-band 산식과 PACF 추정 method를 명시하지 않는다. | Figure의 미세한 모양/경계 불일치 | R 3.5.1 기본 `acf`/`pacf` 후보를 R parity 사양으로 별도 검증 |

## 4. 원 저자 환경 계약

PDF 36쪽(논문 7쪽) §4는 다음만 명시한다.

- OS: **Windows 10**
- R: **3.5.1**
- packages: **`e1071`**, **`fGarch`**

패키지 버전, BLAS, compiler, R RNG kind/seed, locale/time zone, `e1071::svm`/`fGarch` 호출 인자 및 기본값 변경 여부는 원문에 없다. 따라서 `R 3.5.1 + 패키지명`은 환경 완전 고정이 아니며, 현재 Python 구현을 수치적으로 동일하다고 선언할 근거로 충분하지 않다.

## 5. 현재 로컬 환경 스냅샷

감사 시각: 2026-10-11 14:08 KST. 패키지 설치나 환경 변경 없이 실행경로와 metadata만 조회했다.

| 항목 | 확인값 | 상태/해석 |
|---|---|---|
| 실행 계층 | WSL2, Linux kernel `6.18.33.2-microsoft-standard-WSL2`, Ubuntu 24.04.4 LTS, x86_64 | 원문의 Windows 10/R 환경과 다름 |
| locale / timezone | `C.UTF-8`; shell 시각 `KST +0900` | 명시 정책은 별도 계약 필요 |
| Python | 3.12.3; `/home/sieg/projects-wsl/hongik_univ_26_2/.venv/bin/python`; `sys.prefix=/home/sieg/projects-wsl/hongik_univ_26_2/.venv` | 프로젝트 공용 venv, paper2020 전용 환경 아님 |
| Jupyter kernel | `hongik_26_2` 및 `python3`가 모두 위 공용 venv를 가리킴 | B00 fresh-kernel 실행 가능 후보이나 isolation은 없음 |
| NumPy / pandas / SciPy | 2.4.6 / 3.0.3 / 1.18.0 | 설치됨 |
| matplotlib / seaborn / plotly | 3.11.0 / 0.13.2 / 6.8.0 | 설치됨 |
| scikit-learn / statsmodels | 1.9.0 / 0.14.6 | 설치됨 |
| PyYAML | 6.0.3 | B1 YAML 계약 로드 가능 |
| JupyterLab / notebook / ipykernel / nbconvert / nbformat | 4.6.0 / 7.6.0 / 7.3.0 / 7.17.1 / 5.10.4 | 설치됨 |
| `arch` | 없음 | B2/B3에서 설치 정책 승인 전 사용 불가; 원문 DGP를 이 API로 대체해서도 안 됨 |
| R / Rscript | 실행 파일 없음 | 현재 R parity 검증은 차단 |
| `finance-datareader` / `FinanceDataReader` | 0.9.202 설치됨 | 원문 출처 Investing.com과 동일 snapshot을 보장하지 않음 |
| `yfinance` | 없음 | 원문 출처도 아니며 필수 아님 |
| 환경 관리자 | `uv` 실행파일은 존재, `conda`/`poetry` 없음 | worktree 자체에는 lockfile/pyproject/requirements가 없음 |

공유 상위 프로젝트의 `pyproject.toml`과 `requirements.txt`는 Python `>=3.12`와 주요 패키지의 **하한만** 선언한다. `uv.lock`은 상위 루트와 이 worktree에서 발견되지 않았다. `finance-datareader`도 해당 선언 파일에 고정되어 있지 않다. 그러므로 현재 설치 버전은 관측된 snapshot이지 재생 가능한 lock 계약이 아니다.

## 6. Python–R method parity 위험

1. **RBF `gamma` 의미:** 논문 커널은 `exp(-||x-y||^2 / (2 gamma^2))`; scikit-learn/libsvm 형식은 `exp(-gamma_sklearn ||x-y||^2)`이다. 따라서 `gamma_sklearn = 1 / (2 gamma_paper^2)`를 사용해야 한다. 논문의 `gamma^2` 숫자를 scikit-learn `gamma`에 그대로 넣으면 다른 모델이다.
2. **feature scaling:** 로컬 scikit-learn `SVR` 기본 `gamma='scale'`이고, 원문은 grid의 `gamma^2`를 지정한다. 자동 `scale` 사용은 금지해야 한다. R `e1071`의 입력 scaling 여부와 동일한 전처리도 명시적으로 재현해야 한다.
3. **solver defaults:** scikit-learn 1.9.0의 관측 기본은 `tol=0.001`, `shrinking=True`, `max_iter=-1`, `C=1.0`, `epsilon=0.1`이다. 원 저자 호출 인자가 없으므로 optimizer tolerance와 stopping 차이는 numerical parity risk다.
4. **데이터 변환 스케일:** 로그수익률을 비율 단위로 둘지 `%`로 둘지는 SVR의 `C/epsilon` 의미를 바꾼다. 이 선택은 목표값을 본 뒤 조정하면 안 된다.
5. **GARCH 초기값/DGP:** 원문은 `fGarch`를 사용했다고만 하므로 초기 conditional variance, burn-in, RNG, 정규분포 생성기의 R/Python 차이가 있다. method-equivalent 결과와 R parity 결과를 분리해야 한다.
6. **날짜/locale:** Investing.com CSV의 날짜 표현과 소수점/천단위 표시는 locale 의존적일 수 있다. 현재 `C.UTF-8` parser 동작을 원 snapshot 해석과 동일하다고 가정하면 안 된다.

## 7. 로컬 원 실증 데이터 존재 여부

확인한 범위:

- worktree의 tracked `csv/parquet/feather/pickle/json/xls/xlsx/RData/RDS` 경로 인벤토리
- worktree와 `/home/sieg/projects-wsl/hongik_univ_26_2/SA/1011`에서 `S&P 500`, `SP500`, `KRW/USD`, 목표 변화점 날짜, `Investing.com` 문자열 검색
- 전체 `SA` 아래에서 `*sp500*`, `*kospi*`, `*krw*usd*`, `*investing*`, `*garch*`, `*cusum*` 파일명 검색(`.git`/venv 제외)

결과:

- **논문 §5의 Investing.com 원본 3종 snapshot은 발견되지 않았다.** 해당 파일명 검색에서는 PDF만 발견됐다.
- worktree에는 Track A용 ETF CSV/보고서가 다수 있으나, 예컨대 `219480 KODEX 미국S&P500선물(H)`는 한국 상장 ETF 선물 proxy이며 논문의 S&P 500 현물 daily index가 아니다. 이를 원 논문 데이터로 대체할 수 없다.
- KOSPI 관련 ETF와 보고서도 원문의 KOSPI 지수 snapshot이 아니며, KRW/USD 원 환율 snapshot도 확인되지 않았다.
- 따라서 Figures 1–4의 exact data replication은 현재 **BLOCKED**다. B1에서는 다운로드하지 않고 gap으로 유지한다.

이 판정은 파일명·tracked inventory·관련 문자열 기반이다. 일반적인 이름의 미추적 바이너리 내부까지 역공학한 것은 아니지만, 현재 repository 구조에서 논문 원 데이터로 식별 가능한 manifest/snapshot/checksum은 없다.

## 8. B1 결론과 후속 조건

- **Source contract:** §5의 대상, 전체 기간, 출처, 앞쪽 절반 훈련, lag-25 ACF/PACF, Figure 구성, 세 변화점/T 목표값은 원문에서 확정 가능하다.
- **Exact empirical replication readiness:** 원 데이터 snapshot과 데이터 정제/로그수익률/훈련 경계/선택 hyperparameter가 없어 `BLOCKED`다.
- **Method-equivalent Python readiness:** 현재 Python/Jupyter 핵심 스택으로 B2 설계는 가능하지만 RBF 매핑, scaling, solver/seed 정책을 계약으로 잠그기 전 구현 결과를 원문 수치 복제로 부르면 안 된다.
- **R parity readiness:** R/Rscript가 없고 패키지 버전도 원문에 없어 `BLOCKED`; 설치는 사용자 승인 이후 별도 격리 환경에서만 수행해야 한다.
- **B3 데이터 수집 승인 전 필수 결정:** Investing.com 당시 snapshot 확보 경로와 라이선스/수집 정책, S&P 500 training 시작일 충돌 처리, 로그수익률 단위, 관측행 기준 midpoint 정의, exact replication과 method-equivalent의 판정 기준.

