# B1 Mathematical Source Audit — Lee, Kim & Lee (2020)

## 1. 감사 범위와 원문 권위

- 감사 대상: Sangyeol Lee, Chang Kyeom Kim, Sangjo Lee, *Hybrid CUSUM Change Point Test for Time Series with Time-Varying Volatilities Based on Support Vector Regression*, **Entropy** 22(5), 578 (2020), DOI `10.3390/e22050578`.
- 로컬 원본: `/home/sieg/projects-wsl/hongik_univ_26_2/SA/1011/Hybrid_SVR_GARCH_CUSUM_.pdf`
- 파일 SHA-256: `97c95e308af8dd95e8d283f01674932f52349ab8b4d0ea2a93d4dbe45fcf3cd6`
- 이 PDF의 물리 페이지 30~45가 영어 원문 16쪽이다. 아래의 `원문 p.`는 논문 자체 페이지 번호, `PDF p.`는 결합 PDF의 물리 페이지 번호를 뜻한다.
- 검증 방법: 영어 원문 범위를 `pdftotext -layout`으로 추출한 뒤, 수식이 있는 PDF p.32~36(원문 p.3~7)을 180 dpi로 렌더링하여 첨자, 제곱, 절댓값, 분모, 초기값을 직접 대조했다.
- 이 문서는 원문의 전사·추적 계약이다. 아래에서 명시적으로 `원문 미명시` 또는 `원문 표기 의문`으로 표시한 사항을 구현 가정으로 확정하지 않는다.

## 2. Eq. (1)~(9) 원문 전사와 의미

### Eq. (1): 비선형 GARCH 표현

출처: 원문 p.3 / PDF p.32, §2.

$$
y_t = \sigma_t e_t,
$$

$$
\sigma_t^2 = g\!\left(y_{t-1}^2,\ldots,y_{t-p}^2,
\sigma_{t-1}^2,\ldots,\sigma_{t-q}^2\right). \tag{1}
$$

원문 조건: $g$는 미지 함수, $p,q$는 음이 아닌 정수, $\{e_t\}$는 평균 0과 분산 1을 갖는 i.i.d. 확률변수열이다.

### Eq. (2): $\epsilon$-insensitive loss

출처: 원문 p.3 / PDF p.32, §2.

원문은 함수 표기에 로마자 `e`와 매우 유사한 기호를 사용하지만, SVR 관례와 뒤의 설명에서는 $\epsilon$-band를 뜻한다.

$$
\ell_{\epsilon}(y,f(x))=
\begin{cases}
|y-f(x)|-\epsilon, & |y-f(x)|\ge \epsilon,\\
0, & \text{otherwise}.
\end{cases} \tag{2}
$$

경계 $|y-f(x)|=\epsilon$에서는 첫 식도 0이므로 두 분기 간 수치 충돌은 없다.

### Eq. (3): SVR primal

출처: 원문 p.3 / PDF p.32, §2.

먼저 원문은

$$
f(x)=w^T\phi(x)+b
$$

를 둔다. 입력 $x_i$, 출력 $y_i$, $i=1,\ldots,n$, $C>0$에 대해:

$$
\underset{w,b,\tilde\xi_{1,i},\tilde\xi_{2,i}}{\operatorname{minimize}}
\quad \frac12\lVert w\rVert^2
+C\sum_{i=1}^{n}(\tilde\xi_{1,i}+\tilde\xi_{2,i}) \tag{3}
$$

subject to

$$
\begin{aligned}
y_i-w^T\phi(x_i)-b &\le \epsilon+\tilde\xi_{2,i},\\
w^T\phi(x_i)+b-y_i &\le \epsilon+\tilde\xi_{1,i},\\
\tilde\xi_{1,i}&\ge0,\qquad \tilde\xi_{2,i}\ge0.
\end{aligned}
$$

원문 본문은 직후 slack variables를 `> 0`이라고 서술하지만, 표시 수식의 제약은 `>= 0`이다. 구현 계약은 최적화 문제의 표시 수식과 표준 SVR에 맞춰 비음수 제약을 사용해야 하며, 본문 불일치는 gap으로 남긴다.

### Eq. (4): Lagrangian

출처: 원문 p.4 / PDF p.33, §2.

$$
\begin{aligned}
\underset{}{\operatorname{minimize}}\quad L := {}&
\frac12\lVert w\rVert^2
+C\sum_{i=1}^{n}(\tilde\xi_{1,i}+\tilde\xi_{2,i})\\
&-\sum_{i=1}^{n}(\beta_{1,i}\tilde\xi_{1,i}
+\beta_{2,i}\tilde\xi_{2,i})\\
&-\sum_{i=1}^{n}\alpha_{1,i}
(\epsilon+\tilde\xi_{1,i}-y_i+w^T\phi(x_i)+b)\\
&-\sum_{i=1}^{n}\alpha_{2,i}
(\epsilon+\tilde\xi_{2,i}+y_i-w^T\phi(x_i)-b).
\end{aligned} \tag{4}
$$

### Eq. (5): KKT stationarity/feasibility conditions

출처: 원문 p.4 / PDF p.33, §2.

$$
\begin{aligned}
w-\sum_{i=1}^{n}(\alpha_{1,i}-\alpha_{2,i})\phi(x_i)&=0,\\
\sum_{i=1}^{n}(\alpha_{1,i}-\alpha_{2,i})&=0,\\
C-\alpha_{1,i}-\beta_{1,i}&=0, &&i=1,\ldots,n,\\
C-\alpha_{2,i}-\beta_{2,i}&=0, &&i=1,\ldots,n,\\
\alpha_{1,i},\alpha_{2,i},\tilde\xi_{1,i},\tilde\xi_{2,i}&\ge0,
&&i=1,\ldots,n.
\end{aligned} \tag{5}
$$

### Eq. (6): KKT complementarity conditions

출처: 원문 p.4 / PDF p.33, §2.

$$
\begin{aligned}
\alpha_{1,i}(\epsilon+\tilde\xi_{1,i}-y_i+w^T\phi(x_i)+b)&=0,
&&i=1,\ldots,n,\\
\alpha_{2,i}(\epsilon+\tilde\xi_{2,i}+y_i-w^T\phi(x_i)-b)&=0,
&&i=1,\ldots,n,\\
(C-\alpha_{1,i})\tilde\xi_{1,i}&=0,&&i=1,\ldots,n,\\
(C-\alpha_{2,i})\tilde\xi_{2,i}&=0,&&i=1,\ldots,n.
\end{aligned} \tag{6}
$$

### Eq. (7): dual

출처: 원문 p.4 / PDF p.33, §2.

$$
\begin{aligned}
\underset{\alpha_1,\alpha_2}{\operatorname{maximize}}\quad
&-\frac12\sum_{i=1}^{n}\sum_{j=1}^{n}
(\alpha_{1,i}-\alpha_{2,i})(\alpha_{1,j}-\alpha_{2,j})
\phi(x_i)^T\phi(x_j)\\
&-\epsilon\sum_{i=1}^{n}(\alpha_{1,i}+\alpha_{2,i})
+\sum_{i=1}^{n}(\alpha_{1,i}-\alpha_{2,i})y_i
\end{aligned} \tag{7}
$$

subject to

$$
\sum_{i=1}^{n}(\alpha_{1,i}-\alpha_{2,i})=0,
\qquad 0\le\alpha_{1,i}\le C,
\qquad 0\le\alpha_{2,i}\le C.
$$

이로부터 원문은 다음을 제시한다.

$$
\hat w=\sum_{i=1}^{n}(\alpha_{1,i}-\alpha_{2,i})\phi(x_i),
$$

$$
\hat b=
\begin{cases}
y_i-\hat w^T\phi(x_i)-\epsilon,&0<\alpha_{1,i}<C,\\
y_i-\hat w^T\phi(x_i)+\epsilon,&0<\alpha_{2,i}<C.
\end{cases}
$$

원문은 두 $\hat b$ 경우가 드물게 일치하지 않을 수 있으며 그 두 값의 평균을 한 선택으로 언급한다.

### Eq. (8): kernel representation

출처: 원문 p.5 / PDF p.34, §2.

$$
\begin{aligned}
\hat f(x)
&=\sum_{i=1}^{n}(\alpha_{1,i}-\alpha_{2,i})
\phi(x_i)^T\phi(x)+\hat b\\
&=\sum_{i=1}^{n}(\alpha_{1,i}-\alpha_{2,i})K(x_i,x)+\hat b,
\end{aligned} \tag{8}
$$

where $K(x,y)=\phi(x)^T\phi(y)$.

### Gaussian kernel 및 라이브러리 파라미터 대응

출처: 원문 p.5 / PDF p.34, Eq. (8) 직후.

원문 커널:

$$
K_{\text{paper}}(x,y)=
\exp\!\left(-\frac{\lVert x-y\rVert^2}{2\gamma_{\text{paper}}^2}\right).
$$

원문 grid-search 영역:

$$
1\le C\le100,\qquad
0.1\le\gamma_{\text{paper}}^2\le1,\qquad
0.1\le\epsilon\le1.
$$

원문은 이 **연속 구간 내 실제 grid 좌표/간격/점 개수는 명시하지 않는다**.

scikit-learn `SVR(kernel="rbf")`의 커널이

$$
K_{\text{sklearn}}(x,y)=\exp(-\gamma_{\text{sklearn}}\lVert x-y\rVert^2)
$$

이므로 지수 계수를 같게 두면

$$
\boxed{\gamma_{\text{sklearn}}=
\frac{1}{2\gamma_{\text{paper}}^2}}.
$$

따라서 논문의 $\gamma^2$ 구간 $[0.1,1]$은 sklearn의 `gamma` 구간 $[0.5,5]$에 역순 대응한다. 예: 논문 $\gamma^2=0.1\mapsto5$, $0.5\mapsto1$, $1\mapsto0.5$. Phase B2에서는 임의의 서로 다른 두 벡터와 최소 세 grid 점에서 두 커널 값을 기계 정밀도 내 비교하는 수치 테스트가 필요하다. 논문의 $\gamma^2$ 값을 sklearn `gamma`에 그대로 전달하면 원문 커널을 재현하지 못한다.

### Eq. (9): SVR-GARCH feature vector

출처: 원문 p.5 / PDF p.34, §2.

$$
x_s=(\breve y_s^T,\breve\sigma_s^T)^T,
$$

$$
\breve y_s=(y_{s-p}^2,y_{s-p+1}^2,\ldots,y_{s-1}^2)^T,
$$

$$
\breve\sigma_s=(\sigma_{s-q}^2,\sigma_{s-q+1}^2,
\ldots,\sigma_{s-1}^2)^T. \tag{9}
$$

즉 행 $x_s$는 시간 $s$ 이전의 $p$개 제곱 관측치와 $q$개 조건부분산으로 구성된다. 뒤 §3과 §4의 실험은 $p=q=1$만 고려한다고 원문 p.7 / PDF p.36에서 명시한다.

## 3. 원문 SVR–GARCH Procedure Steps 1~5

출처: 원문 p.5~6 / PDF p.34~35, §2.

길이 $n+n'$의 시계열과 tuning-parameter space가 주어진다.

1. **시간순 분할과 평가점 지정.** Parameter space 안에서 평가할 점들을 정하고, 주어진 시계열을 크기 $n$의 training 시계열과 크기 $n'$의 validation 시계열로 나눈다. 목적은 fitted SVR-GARCH를 validation하여 최적 tuning parameter set을 정하는 것이다.
2. **조건부분산 proxy.** 알 수 없는 $\sigma_t^2$를 training 시계열의 moving average proxy $\tilde\sigma_t^2$로 대체한다:

   $$
   \tilde\sigma_t^2=\frac1m\sum_{j=1}^{m}y_{t-j+1}^2.
   $$

   $t<m$이면

   $$
   \tilde\sigma_t^2=\frac1t\sum_{j=1}^{t}y_{t-j+1}^2.
   $$

   이 정의는 시간 $t$의 proxy에 $y_t^2$를 포함한다. 통상적인 lagged rolling variance로 바꾸면 안 된다.
3. **SVR 적합과 분산 예측.** 주어진 tuning-parameter set에서, Eq. (1)의 $g$를 SVR의 $\hat g$로 추정하되 $\sigma_t^2$를 $\tilde\sigma_t^2$로 바꾼다. 이후:

   $$
   \hat\sigma_t^2=\hat g(y_{t-1}^2,\ldots,y_{t-p}^2,
   \tilde\sigma_{t-1}^2,\ldots,\tilde\sigma_{t-q}^2).
   $$

4. **Validation MAE.** 추정된 모델과 Step 2와 같은 validation proxy 공식을 사용해:

   $$
   \operatorname{MAE}=\frac1{n'}\sum_{t=1}^{n'}
   |\hat\sigma_t^2-\tilde\sigma_t^2|.
   $$

5. **선택과 재적합.** Step 1에서 정한 모든 tuning set에 대해 Steps 2~4를 반복하고 MAE가 최소인 조합을 선택한다. 이후 training과 validation 시계열을 합쳐 Steps 2와 3을 수행하여 residual 계산에 사용할 final model을 정한다.

### 경계·인덱스 관련 원문 미명시 사항

- validation proxy의 local $t=1$에서 이동평균을 validation 구간만으로 다시 시작하는지, 직전 training 관측을 이어 쓰는지 명시하지 않는다.
- validation feature의 첫 $p,q$개 lag가 training 끝부분을 사용하는지, validation 내부에서만 완전한 lag가 생긴 뒤 시작하는지 명시하지 않는다.
- training에서 Eq. (9)의 lag 때문에 소실되는 최초 관측 수와 MAE 분모 $n'$ 간 조정이 명시되지 않는다.
- SVR의 직접 target이 $\tilde\sigma_t^2$라는 해석은 Step 2~4의 절차상 강하게 시사되지만, “target vector는 이것이다”라는 독립된 표시식은 없다.
- 선택 후 test 구간에서 $\hat\sigma_t^2$를 만들 때 lagged variance 입력에 proxy를 계속 쓰는지, 예측값을 재귀 입력하는지, 관측 가능한 test $y_t^2$로 proxy를 갱신하는지 명시하지 않는다.
- variance prediction의 음수 가능성에 대한 positivity transform, floor, clipping, 또는 실패 정책은 없다.
- feature scaling/standardization, solver tolerance, tie-breaking, SVR intercept 처리, support-vector tolerance가 명시되지 않는다.

## 4. Remarks 1~3 계약

출처: 원문 p.6 / PDF p.35, §2.

### Remark 1

- random search 같은 다른 tuning selection 방식이 가능하지만, 저자들은 simulation study에서 grid search가 random search보다 우수하다고 보아 grid search를 사용한다.
- proxy volatility의 $m$ 선택은 안정적인 검정 구성에 중요하다.
- 원문은 $m=5$가 가장 일관되게 합리적으로 만족스러운 결과를 제공하여 이 경우를 보고한다고 한다.
- 따라서 exact-protocol 경로의 $m$은 5로 고정하고, 다른 $m$은 원문 결과와 분리된 sensitivity 경로에서만 허용해야 한다.

### Remark 2

- $\hat\sigma_t^2$를 수렴 조건까지 반복 갱신하는 방법을 고려할 수 있으나, 저자들은 이때 $\hat g$가 반복마다 “flatten out”되어 결국 상수함수가 되거나 초기 $\tilde\sigma_t^2$의 peak outlier를 극단적으로 따라가는 현상을 보고한다.
- 그래서 원문 연구는 iterative update를 부적절한 것으로 보고 사용하지 않는다. exact-protocol 구현에 반복 재적합을 삽입하면 안 된다.

### Remark 3

- MAE 대신 MSE, RMSE, MAPE를 고려할 수 있으나 저자들은 suboptimal 결과 가능성을 설명한다.
- MSE/RMSE는 손실값 크기에 따라 손실을 확대 또는 축소해 결과가 악화된다고 설명한다.
- 원문 문장은 MAPE가 “unobservable $\tilde\sigma_t^2$ rather than $\hat\sigma_t^2$”를 사용하기 때문에 불일치한 결과를 낸다고 적는다. 그러나 같은 절차에서 $\tilde\sigma_t^2$는 관측치로 만든 proxy이므로 이 `unobservable` 표현은 의미가 불명확하다.
- exact-protocol tuning criterion은 MAE이며, 이 문구의 의미를 임의 교정하지 않는다.

## 5. §3 CUSUM of Squares 계약

### 5.1 원문이 먼저 제시하는 parametric GARCH(1,1) 기준

출처: 원문 p.6 / PDF p.35, §3.

$$
y_t=\sigma_t e_t,
$$

$$
\sigma_t^2=\omega+\alpha y_{t-1}^2+\beta\sigma_{t-1}^2,
\qquad t=1,\ldots,n,
$$

with $\omega>0$, $\alpha\ge0$, $\beta\ge0$, $\alpha+\beta<1$, and i.i.d. $e_t$ with mean 0 and variance 1.

가설은 다음과 같다.

$$
H_0:\ \omega,\alpha,\beta\text{ remain the same over }t=1,\ldots,n,
$$

$$
H_1:\ \text{not }H_0.
$$

QMLE 같은 일치추정량 $\hat\omega,\hat\alpha,\hat\beta$를 사용한 residual은:

$$
\hat e_t=\frac{y_t}{\hat\sigma_t},
$$

$$
\hat\sigma_t^2=\hat\omega+\hat\alpha y_{t-1}^2
+\hat\beta\sigma_{t-1}^2,\qquad t=1,\ldots,n.
$$

원문에 인쇄된 초기값은 정확히

$$
y_0=0,\qquad \sigma_0^2=y_1
$$

이다.

주의: 위 recursion 마지막 항은 PDF상 $\hat\beta\sigma_{t-1}^2$로, 이전 분산에 hat이 없다. 또한 $\sigma_0^2=y_1$은 분산과 관측값의 차원이 맞지 않고 $y_1$이 음수일 수도 있다. 두 표기는 렌더링 원문에서 재확인했으나 명백한 교정 지침이 없다. B2 구현에서 조용히 $\hat\sigma_{t-1}^2$ 또는 $y_1^2$로 바꾸지 말고, 원문-문자적 경로와 추정 typo 교정 경로를 분리하거나 저자 구현/보충자료로 확인해야 한다.

### 5.2 중심화 제곱누적합, 분산 정규화, supremum 통계량

출처: 원문 p.6~7 / PDF p.35~36, §3.

각 $k=1,\ldots,n$에 대해 중심화한 제곱 residual 누적합을

$$
D_n(k)=\sum_{t=1}^{k}\hat e_t^2
-\frac{k}{n}\sum_{t=1}^{n}\hat e_t^2
$$

로 두면 원문 검정통계량은

$$
\boxed{
\hat T_n=
\max_{1\le k\le n}
\frac{1}{\sqrt n\,\hat\tau_n}
\left|D_n(k)\right|
}.
$$

분산 정규화 항은

$$
\boxed{
\hat\tau_n^2=
\frac1n\sum_{t=1}^{n}\hat e_t^4
-\left(\frac1n\sum_{t=1}^{n}\hat e_t^2\right)^2
}.
$$

즉 $\hat\tau_n^2$는 residual square의 표본 2차 중심모멘트를 분모 $n$으로 계산한 형태이고, $\hat T_n$의 분모에는 그 비음수 제곱근 $\hat\tau_n$이 들어간다. `ddof=1` 표본분산으로 바꾸면 원문과 달라진다.

참오차 $e_t$를 사용한 이론 통계량은:

$$
T_n=
\max_{1\le k\le n}
\frac{1}{\sqrt n\,\tau}
\left|
\sum_{t=1}^{k}e_t^2
-\frac{k}{n}\sum_{t=1}^{n}e_t^2
\right|,
$$

$$
\tau^2=\operatorname{Var}(e_1^2).
$$

### 5.3 Brownian Bridge 기준과 기각 판정

출처: 원문 p.7 / PDF p.36, §3.

원문은 regularity conditions 아래 $H_0$에서 $n\to\infty$일 때 $\hat T_n$이 $T_n$과 점근적으로 동일하게 행동하며,

$$
\hat T_n\ \Rightarrow\ \sup_{0\le s\le1}|B^\circ(s)|,
$$

여기서 $B^\circ$는 단위구간 Brownian bridge라고 설명한다. 유의수준 0.05의 기각 계약은:

$$
\boxed{\text{reject }H_0\quad\text{if}\quad \hat T_n\ge1.3397.}
$$

원문은 이 임계값이 Monte Carlo simulation으로 얻어질 수 있다고 적는다. p-value 계산법은 제시하지 않는다.

### 5.4 변화점 위치

출처: 원문 p.7 / PDF p.36, §3.

변화가 존재한다는 조건 아래 위치는:

$$
\boxed{
\hat k_n=
\operatorname*{arg\,max}_{1\le k\le n}
\left|
\sum_{t=1}^{k}\hat e_t^2
-\frac{k}{n}\sum_{t=1}^{n}\hat e_t^2
\right|.
}
$$

기각 여부와 $\hat k_n$은 서로 다른 출력이어야 한다. $\hat k_n$ 계산 가능성이 곧 $H_0$ 기각을 뜻하지 않는다. 원문은 동률 argmax의 처리 규칙을 정하지 않는다.

### 5.5 SVR–GARCH residual로의 전이 범위

원문은 위 표시식을 parametric GARCH(1,1) residual에 대해 먼저 설명한 다음, SVR-GARCH에서 얻은 residual이 참오차 $e_t$처럼 행동할 가능성이 높으므로 같은 CUSUM procedure를 적용할 수 있다고 서술한다. 하지만 §3에는 SVR final model의 test residual을 만드는 별도의 완결된 recursion/표시식이 없다. 따라서 다음 연결은 원문에 명시된 것과 그렇지 않은 것을 구분해야 한다.

- 명시됨: residual의 개념은 관측값을 추정 volatility로 나눈 $y_t/\hat\sigma_t$이다.
- 명시됨: CUSUM 입력은 그 standardized residual의 제곱이다.
- 명시됨: §4 실험에서 training/validation 뒤 남은 $n$개가 testing set이다.
- 미명시: SVR 예측이 $\hat\sigma_t^2<0$을 내면 $\hat\sigma_t$를 어떻게 정의하는가.
- 미명시: test feature의 lagged variance를 proxy, 이전 prediction, 또는 다른 초기화 중 무엇으로 채우는가.
- 미명시: $\hat\tau_n=0$ 또는 부동소수점상 $\hat\tau_n^2<0$일 때의 정책.
- 미명시: 여러 $k$가 같은 최대 절댓값을 가질 때 선택할 위치.

원문은 Inclán and Tiao [24] 방식을 따르면 multiple change points도 검출할 수 있다고 언급하지만, 본 논문의 표시식과 보고 결과는 위 단일 argmax 계약이다. multiple-change segmentation을 exact-protocol 핵심에 임의 삽입하지 않는다. 또한 이는 전체 test 구간을 본 뒤 계산하는 retrospective test이며 rolling online alarm이 아니다.

## 6. B2용 수학 golden-test 요구사항

이 문서는 구현을 포함하지 않지만, 원문 전사의 오역을 막기 위해 B2에서 최소한 다음 독립 검산이 필요하다.

1. 작은 고정 residual 배열 $e=(e_1,\ldots,e_n)$을 사용한다.
2. $e_t^2$, $e_t^4$, 전체 제곱합, 각 prefix 제곱합을 손계산 fixture로 고정한다.
3. 각 $D_n(k)$, $\hat\tau_n^2$, $\hat\tau_n$, $\hat T_n$, 첫/명시된 tie-rule의 $\hat k_n$을 라이브러리 없는 독립 계산과 비교한다.
4. 분산 계산이 반드시 분모 $n$인지 확인하는 반례를 둔다.
5. 절댓값을 빼면 argmax가 달라지는 배열을 포함한다.
6. `reject = (T >= 1.3397)`와 `change_index = argmax(...)`를 별도 assertion으로 둔다.
7. sklearn RBF mapping은 $\gamma_{\text{paper}}^2\in\{0.1,0.5,1.0\}$과 서로 다른 벡터 쌍에서 직접 지수값을 비교한다.
8. $\hat\tau_n=0$, 음의 variance prediction, argmax tie는 원문 미명시 예외로서 fail-closed 또는 명시적 policy branch를 요구하고, 조용한 보정은 금지한다.

## 7. 원문 충실성 gap 요약

| ID | 원문 위치 | gap / ambiguity | B2 처리 계약 |
|---|---|---|---|
| MATH-01 | p.3 Eq. (3) | 표시식 slack `>=0`, 본문 `>0` | 표시식/표준 SVR의 `>=0`; 차이 기록 |
| MATH-02 | p.3 본문 | $b$를 $w$와 함께 “vectors”라 하고 $w,b$가 $x$와 같은 차원이라고 기술 | $b$는 표시식의 scalar intercept로 취급하되 원문 오류 기록 |
| MATH-03 | p.5 kernel | grid 범위만 있고 grid 좌표/간격 없음 | exact grid 확정 전 `UNKNOWN`; 임의 기본 grid 금지 |
| MATH-04 | p.5 Eq. (9), Steps 2~4 | 최초 lag와 validation 경계 정의 없음 | 후보 정책 분리·민감도 기록; 조용한 선택 금지 |
| MATH-05 | p.5 Step 3 | SVR target vector를 독립적으로 명시하지 않음 | proxy target 해석을 `inferred`로 표시 |
| MATH-06 | p.6 Step 5 / §3 | final SVR의 test-time variance recursion 없음 | 원문-근거 추가 확인 전 확정 금지 |
| MATH-07 | p.6 Remark 3 | MAPE 설명의 `unobservable tilde sigma` 의미 불명확 | 문구 보존; MAE만 exact protocol에 사용 |
| MATH-08 | p.6 §3 | $\hat\beta\sigma_{t-1}^2$에서 이전 분산에 hat 없음 | 문자적/추정 typo 경로 분리 |
| MATH-09 | p.6 §3 | 초기값 $\sigma_0^2=y_1$은 차원·양수성 문제 | $y_1^2$로 조용히 교정 금지 |
| MATH-10 | p.5~7 | 음의 SVR variance prediction 처리 없음 | positivity/clipping은 추가 정책으로 명시·분리 |
| MATH-11 | p.7 §3 | $\hat\tau_n=0$, argmax tie 처리 없음 | 명시적 예외/tie 정책과 sensitivity 필요 |
| MATH-12 | p.7 §3 | SVR residual CUSUM의 점근 정당화는 서술 수준 | 원문 주장 범위를 넘는 보장 금지 |

## 8. 추적 결론

- Eq. (1)~(9), Gaussian kernel, tuning 범위, five-step procedure, Remarks 1~3, §3의 residual/CUSUM/normalization/supremum/Brownian-bridge/critical-value/change-location 식을 영어 원문 렌더링과 대조했다.
- 원문 충실 구현에서 가장 중요한 수치 대응은 `sklearn gamma = 1 / (2 * paper gamma_squared)`이다.
- CUSUM 통계량과 변화점 위치는 모두 중심화 제곱 residual 누적합의 **절댓값**을 사용하며, 정규화 분산은 분모 $n$이다.
- 5% 기각은 $\hat T_n\ge1.3397$이고, 기각 여부와 argmax 위치는 분리해야 한다.
- test-time SVR recursion, validation boundary, positivity, exact grid, parametric recursion의 hat/초기값 문제는 원문만으로 확정할 수 없다. 이 항목들은 unsupported assumption 없이 gap으로 유지한다.
