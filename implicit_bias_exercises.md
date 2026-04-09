# Exercises
We use the same notations as in the presentation
## Problem 1: loss landscape geometry
- Consider a two layers diagonal linear network learning a teacher matrix $\text{diag}(s_1,...,s_d)$. Show that the student $\text{diag}(s_1,...,s_r,0,...,0)$ with $r<d$ is a saddle point.
- Show that the multiplication map $\mu(\theta)=W_L...W_1$ is invariant under action of product of $GL_{d_l}$ What does this implies about the loss landscape geometry?
## Problem 2: Gradient flow and NTK
- Derive the gradient flow equation of a two layer Deep Linear Network for each layer.
- Show that $$G_l = W_{l+1}^{\top}W_{l+1} - W_{l}W_{l}^{\top}$$ is conserved through the gradient flow
- Using the balanced conditions, derive the Gradient flow equation in function space (i.e. in terms of the student $W = \mu(\theta)$)
  - Show that we have a self consistent equation in function space
- Define the NTK operator as $$K[f(W)] := \sum_k (WW^{\top})^{\frac{L-k}{L}}f(W)(W^{\top}W)^{\frac{k-1}{L}}$$. Take $L=2$, assume that the weights are balanced. express gradient flow in terms of the NTK operator in function space.
## Problem 3: Lazy and rich regimes
- For small initialization show that for a DLN of depth $L$ with initialization $\theta(0)\sim N(0,\sigma)$ and $\sigma^2 = d_L^{-\gamma}, \ \gamma < 1$:
$$
\begin{align*}
    \dot{W} & \approx K_0[M - W]
\end{align*}
$$
- In the rich regime $\gamma>1$. Assume that the weights are balanced. Furthermore, assume that the NTK is already aligned to the task i.e. the singular vectors of $K$ align with the singular vectors of the teacher matrix. Show that we have the following system of 1D ODE: 
$$\begin{equation*}
    \dot{w}_{\alpha} = 2\left(s_{\alpha} - w_{\alpha}\right)w_{\alpha}.
\end{equation*}$$
- Show that the time scale of learning a particular feature is:
$$
t=\frac{1}{2s}\,\ln\!\left(\frac{w_f\,(s-w_0)}{w_0\,(s-w_f)}\right), \qquad s\neq 0.
$$
- In general we have a mixture of lazy and rich regimes. An ansatz for the self consistent equation is the following ($\sigma$ is initialization variance and $w$ is width)
$$\dot{W} = -\sqrt{WW^{\top} + \sigma^4w^2I}\nabla L(W) - \eta \nabla L(W)\sqrt{W^{\top}W + \sigma^4w^2I} $$ 
- By comparing the singular values of $W$ and $\sigma^4w^2$ discuss when will the model be in a lazy and rich regime
- Some directions might transition from lazy to rich regime, how can this happen?

## Problem 4 (bonus): Stochasticity
Consider the Langevin model of SGD:
$$
d\theta_t=-\nabla L(\theta_t)\,dt+\sqrt{\eta\,\Sigma(\theta_t)}\,dB_t,
$$
Where $\Sigma(\theta_t)$ is the covariance matrix of gradient noise coming from batching. 
Consider the associated Fokker-Planck equation:
$$\begin{align*}
    \partial_t p & := - \nabla\cdot j \\
    j &:=- \nabla L(\theta)p(\theta) - \frac{\eta}{2}\nabla^\top\left(\Sigma(\theta)p(\theta)\right) 
 \end{align*}$$
Assume:
- Stationary distribution: $\partial_tp^{\ast}(x,t) = 0$ (white noise)
- Thermal equilibrium: $j=0$ 
- White noise: $\Sigma=\sigma^2 I$ 

Show that the equilibrium distribution satisfies:
$$ p^{\ast}(\theta) \propto \exp\left(-\frac{2}{\eta\sigma^2}L_N(\theta)\right)$$

