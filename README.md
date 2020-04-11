# covid19-modelling

### A Outline of Project for MATH371 Wi2020}

by Natasha Ting
University of Alberta 2020

#### SIRD Model
The fraction of individuals susceptible to contract COVID-19 in a population , $\frac{S}{N} = s$, fraction of infected individuals, $\frac{I}{N} = i$, fraction of removed individuals, $\frac{R}{N} = r$ and fraction of dead, $\frac{D}{N} = d$ change in dynamic described in the following planar system: 
\begin{equation}
       \begin{array}{ll}
      \dot{s} = - \beta s i  & \quad s(0) = s_0 \\
       \dot{i} = \beta s i - (\alpha + \mu)i & \quad i(0) = i_0 \\
       \dot{r} = \alpha i  & \quad r(0) = r_0 \\ 
       \dot{d} = \mu i & \quad d(0) = 0 
        \end{array}
\end{equation}
where:
\begin{conditions}
\beta & infection rate; \quad \quad $\frac{1}{\beta}$ = average days to be infected \\
\alpha & recovery rate; \quad \quad $\frac{1}{\alpha}$ = average days to recover \\
\mu & death rate \\
R_0 & basic reproduction number, the avg number infected by 1 person \\
\end{conditions}
\end{frame}

#### Stability Analysis of the SIRD
Taking out $d = 1 - s - i - d $, the planar system in (1) can be rewritten as $$\mathbf{\dot{x}} = \mathbf{F}(\mathbf{x})$$ 
where $\mathbf{x}^{T} = [s \;\;i\;\; r]$ \\
	at steady state (ss), $\mathbf(\dot{x}) = 0 \implies $ $$\mathbf{x} = \hat{\mathbf{x}} = \begin{pmatrix} s_{ss} \\ 0 \\ r \end{pmatrix}$$ 
	with Jacobian matrix $$J{\mathbf{\hat{x}}} = \begin{pmatrix} 0 & \beta s_{ss} & 0\\ 0 & \beta s_{ss} - (\alpha + \mu) & 0 \\ 0 & \alpha & 0  \end{pmatrix} $$ and eigenvalues $\lambda_1, \lambda_2 = 0, \lambda_3 = bs - (\alpha + \mu) <0 $ since $s_{ss} < s_{I_{max}}$
	
The phase plane plot reveals that $i = 0$, $s = 0$ is a stable point. 

#### Assumptions
\begin{itemize}
	\item instantaneous infection, recovery of individuals
	\item no delay in any effect, including death
\end{itemize}
which is not captured in the publicly available data collected by the health authority. \\

\bigskip 
In addition, while the solutions to the SIR model is unique, the optimal parameters found by equation (3) takes different values given different initial guess fed to the solver. 


#### Methods of Estimation}
Least-square (Plateaued)

To estimate parameters from data from regions that \textbf{have reached} a plateau-ing trend in the number of confirmed cases (infected), we solve the following problem: 

\begin{equation}
\begin{aligned}
& \underset{\alpha, \beta=1, \mu, N}{\text{minimize}} & &  \left(\frac{I_{max} - \widehat{I}_{max}}{I_{max}}\right)^2 + \frac{1}{T} \sum_{t=1}^{T} \left(\frac{R_t - \widehat{R_t}}{R_{max}}\right)^2 + \left(\frac{D_t - \widehat{D_t}}{D_{max}}\right)^2\\
& \text{subject to}
& & 0 \leq \frac{\alpha + \mu}{\beta} \leq \left(\frac{N-( I + R + D )}{N} \right)\bigg\rvert_{I=I_{max}}
 \end{aligned}
\end{equation}
\end{frame}

\begin{frame}{For Non-Plateaued Data}
For data from regions that have not reached the plateauing stage of the data, we solve the following minimisation problem for each compartment $i, r, d$. \\ 
I.e. assuming $s = 1$, for $i$: \\
\begin{equation}
\begin{aligned}
& \underset{i_0, k_0}{\text{minimize}} & &  E = \sum_{t=1}^{T} (i_t - \widehat{i_t})^2 \\
& \text{subject to}
& & \hat{i} = i_0 e^{k_0 i} \\
& \text{where } k = \beta -\alpha - \mu &
 \end{aligned}
\end{equation}
\end{frame}


\begin{frame}{For Non-Plateaued Data}
From Eq(4) we get
\begin{equation}
\begin{aligned}
& \hat{i} = i_0 e^{k_0 t} \\
& \hat{r} = r_0 e^{k_0 t} \\
& \hat{d} = d_0 e^{k_0 t} 
 \end{aligned}
\end{equation}
In addition, $$r_0 = \frac{i_0 \alpha}{k_0}, \quad d_0 = \frac{i_0 \mu}{k_0} $$
$$\implies \alpha = \frac{r_0 k_0}{ i_0}, \quad \mu = \frac{d_0 k_0}{ i_0} $$
$$\implies \beta = k_0 + \frac{k_0}{i_0} (d_0 + r_0)$$

\end{frame}


\subsection{Time Scale Matching}
\begin{frame}{Time Scale of Estimated Data}
Using Euler’s method, a differential form $\dot{s}=-\beta si$ can be expressed as
	$$s_{t+1} = s_t + dt (-\beta s_t i_t)$$
Which we consider as 
	$$s_{t+1} = s_t + \beta dt (- s_t i_t)$$
Thus, a change in $\beta$ leads to a change in the timescale. In Eq(3) we assumed $\beta$ = 1. \\
\end{frame}

\begin{frame}{The Time-scale constant}
The estimated data has time scale $t'$. It is scaled to match the timescale used by the data in $t$ time scale. We obtain 
\begin{equation}k = \frac{t_m}{t'_m} \end{equation} 
where $t_m $ is the time at which $i(t)$ is maximum and  $t'_m$ is the time when $\hat{i}(t')$ is maximum. 
\bigskip
\\
Eq (6) implies $t' = \frac{t}{k}$. Thus, to match timescales, $\hat{s}(t') = \hat{s}(\frac{t}{k}) $. \\ 
In addition, since $\beta = 1$ in Eq(3), the estimated parameters are scaled i.e. 
$$\alpha = \frac{\hat{\alpha}}{k}, \beta = \frac{1}{k}, \mu = \frac{\hat{\mu}}{k}, N = \hat{N}$$.
\end{frame}

