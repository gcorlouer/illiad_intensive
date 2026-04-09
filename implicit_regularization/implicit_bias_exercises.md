# Implicit Regularization in Deep Linear Networks

## Exercise Session — 1h30, pen and paper

**References**:
- Saxe, McClelland & Ganguli, *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks*, ICLR 2014 / PNAS 2019
- Achour, Malgouyres & Gerchinovitz, *The loss landscape of deep linear neural networks: a second-order analysis*, JMLR 2024
- Tu, Aranguri & Jacot, *Mixed Dynamics In Linear Networks: Unifying the Lazy and Active Regimes*, 2024

---

## Setup and notation

We study deep linear networks (DLNs) with $L$ weight matrices $W_1 \in \mathbb{R}^{d_1 \times d_0}, W_2 \in \mathbb{R}^{d_2 \times d_1}, \ldots, W_L \in \mathbb{R}^{d_L \times d_{L-1}}$. The network computes:

$$f(x) = W_L W_{L-1} \cdots W_1 \, x =: W x$$

where $W = W_L \cdots W_1 \in \mathbb{R}^{d_L \times d_0}$ is the end-to-end (or "student") matrix. We train on a dataset $\{(x_\mu, y_\mu)\}_{\mu=1}^P$ with the squared loss:

$$\mathcal{L}(\theta) = \frac{1}{2P} \sum_{\mu=1}^{P} \|y_\mu - W_L \cdots W_1 \, x_\mu\|^2$$

In the population limit with whitened inputs $\Sigma_X = I$, this becomes:

$$\mathcal{L}(W) = \frac{1}{2}\|M - W\|_F^2$$

where $M = \Sigma_{YX} \Sigma_X^{-1}$ is the "teacher" matrix (the OLS solution) with SVD $M = U \,\text{diag}(s_1, \ldots, s_r, 0, \ldots, 0)\, V^\top$, with $s_1 \geq s_2 \geq \cdots \geq s_r > 0$.

---

## Problem 1 — Loss Landscape Geometry (15 min)

This problem explores the critical point structure of deep linear networks, following Achour et al. (2024).

**(a)** Consider the **diagonal** case: $d_0 = d_1 = \cdots = d_L = d$, and the teacher is diagonal $M = \text{diag}(s_1, \ldots, s_d)$ with $s_1 > s_2 > \cdots > s_d > 0$. Restrict attention to diagonal weight matrices $W_l = \text{diag}(w_l^{(1)}, \ldots, w_l^{(d)})$. Show that the loss decomposes into $d$ independent scalar problems:

$$\mathcal{L} = \frac{1}{2}\sum_{\alpha=1}^d \left(s_\alpha - \prod_{l=1}^L w_l^{(\alpha)}\right)^2$$

**(b)** For a single scalar mode with target $s > 0$, find all first-order critical points of $\ell(w_1, \ldots, w_L) = \frac{1}{2}(s - \prod_l w_l)^2$, for depth $L = 2$. Show that the critical points are:
- The global minimum: $w_1 w_2 = s$
- The origin: $w_1 = w_2 = 0$

Classify the origin as a saddle point by computing the Hessian $H$ of $\ell$ at $(0,0)$ and showing it has both positive and negative eigenvalues.

*Hint*: The gradient equations are $\frac{\partial \ell}{\partial w_1} = -(s - w_1 w_2) w_2 = 0$ and $\frac{\partial \ell}{\partial w_2} = -(s - w_1 w_2) w_1 = 0$.

**(c)** Now consider the full (non-diagonal) DLN with $d_l \geq d_L$ for all $l$. Achour et al. show that every first-order critical point $\theta = (W_1, \ldots, W_L)$ satisfies: there exists a subset $S \subseteq \{1, \ldots, r\}$ such that:

$$W = W_L \cdots W_1 = P_S M$$

where $P_S = U_S U_S^\top$ is the orthogonal projector onto the span of the left singular vectors $\{u_\alpha\}_{\alpha \in S}$ of $M$. Argue (without proof) that these critical points with $|S| < r$ are saddle points, not local minima. Why does this mean there are no spurious local minima?

**(d)** The group $GL_h := GL_{d_1} \times \cdots \times GL_{d_{L-1}}$ acts on the weights by:

$$(W_1, \ldots, W_L) \mapsto (g_1 W_1, g_2 W_2 g_1^{-1}, \ldots, W_L g_{L-1}^{-1})$$

Verify that the student map $\mu(\theta) = W_L \cdots W_1$ is invariant under this action: $\mu(g \cdot \theta) = \mu(\theta)$. What does this imply about the dimension of the set of global minima in parameter space?

---

## Problem 2 — Gradient Flow and Conserved Quantities (20 min)

We now study gradient flow: $\dot{\theta}(t) = -\nabla_\theta \mathcal{L}(\theta(t))$, the continuous-time limit of gradient descent with infinitesimal learning rate.

**(a)** For a two-layer DLN ($L=2$) with loss $\mathcal{L} = \frac{1}{2}\|M - W_2 W_1\|_F^2$ and whitened inputs, derive the gradient flow equations for each layer. Show that:

$$\dot{W}_1 = W_2^\top(M - W_2 W_1), \qquad \dot{W}_2 = (M - W_2 W_1) W_1^\top$$

*Hint*: Use the matrix identity $\frac{\partial}{\partial A}\text{Tr}(B A C) = B^\top C^\top$ and the chain rule. Recall $\|X\|_F^2 = \text{Tr}(X^\top X)$.

**(b)** Define the **balancedness matrix**:

$$G := W_2^\top W_2 - W_1 W_1^\top$$

Show that $G$ is conserved under the gradient flow, i.e. $\dot{G} = 0$.

*Hint*: Compute $\dot{G} = \dot{W}_2^\top W_2 + W_2^\top \dot{W}_2 - \dot{W}_1 W_1^\top - W_1 \dot{W}_1^\top$, substitute the gradient flow equations from (a), and simplify. Several terms will cancel.

**(c)** From now on, assume **balanced initialization**: $G(0) = 0$, which by part (b) means $W_2^\top W_2 = W_1 W_1^\top$ for all time. We want to derive the gradient flow in function space (the ODE for the student $W = W_2 W_1$). This requires several steps.

**(c.i)** First, compute $\dot{W} = \dot{W}_2 W_1 + W_2 \dot{W}_1$ using the equations from (a). Show that:

$$\dot{W} = (M - W) W_1^\top W_1 + W_2 W_2^\top (M - W)$$

**(c.ii)** We now need to express $W_1^\top W_1$ and $W_2 W_2^\top$ in terms of $W = W_2 W_1$. Recall the **polar decomposition**: any matrix $A$ can be written as $A = Q P$ where $Q$ is orthogonal and $P = (A^\top A)^{1/2}$ is positive semi-definite. Apply this to $W_1$: write $W_1 = Q_1 (W_1^\top W_1)^{1/2}$. Deduce that:

$$W_1^\top W_1 = \left[(W_1^\top W_1)^{1/2}\right]^2$$

and that $W = W_2 W_1 = (W_2 Q_1)(W_1^\top W_1)^{1/2}$. Using this, show that:

$$W^\top W = (W_1^\top W_1)^{1/2} \underbrace{Q_1^\top W_2^\top W_2 Q_1}_{= Q_1^\top (W_1 W_1^\top) Q_1 \text{ by balance}} (W_1^\top W_1)^{1/2}$$

Now use the fact that $Q_1^\top W_1 W_1^\top Q_1 = W_1^\top W_1$ (verify this from $W_1 = Q_1(W_1^\top W_1)^{1/2}$) to conclude:

$$W^\top W = (W_1^\top W_1)^2 \qquad \Longrightarrow \qquad W_1^\top W_1 = (W^\top W)^{1/2}$$

**(c.iii)** By an analogous polar decomposition argument on $W_2^\top$ (or by using balancedness directly), show that $W_2 W_2^\top = (W W^\top)^{1/2}$.

**(c.iv)** Substitute the results of (c.ii) and (c.iii) into (c.i) to obtain:

$$\dot{W} = (W W^\top)^{1/2}(M - W) + (M - W)(W^\top W)^{1/2}$$

**(d)** Define the NTK operator for $L = 2$ as:

$$K[F] := (WW^\top)^{1/2} F + F (W^\top W)^{1/2}$$

Show that the gradient flow from (c) can be written as $\dot{W} = K[M - W]$. Generalize to depth $L$ on the balanced manifold (you may state without proof):

$$\dot{W} = \sum_{k=1}^{L} (WW^\top)^{\frac{L-k}{L}} (M - W) (W^\top W)^{\frac{k-1}{L}}$$

---

## Problem 3 — Rich Regime: Incremental Learning (25 min)

This is the core problem. We derive the exact solution of the rich regime dynamics directly from the self-consistent equation of Problem 2, emphasizing the NTK perspective.

**(a)** **Alignment assumption.** Start from the balanced gradient flow equation derived in Problem 2(c):

$$\dot{W} = (WW^\top)^{1/2}(M - W) + (M - W)(W^\top W)^{1/2}$$

We work in the **rich regime** (small initialization). Assume that the NTK is **aligned to the task**: the singular vectors of $W(t)$ coincide with those of the teacher $M$ at all times. Concretely, write:

$$M = U\,\text{diag}(s_1, \ldots, s_d)\,V^\top, \qquad W(t) = U\,\text{diag}(w_1(t), \ldots, w_d(t))\,V^\top$$

where $U, V$ are the left/right singular vectors of $M$, and $w_\alpha(t) \geq 0$ are the evolving singular values of $W$.

**(a.i)** Under this alignment assumption, show that:

$$(WW^\top)^{1/2} = U\,\text{diag}(w_1, \ldots, w_d)\,U^\top, \qquad (W^\top W)^{1/2} = V\,\text{diag}(w_1, \ldots, w_d)\,V^\top$$

and that the residual is $M - W = U\,\text{diag}(s_1 - w_1, \ldots, s_d - w_d)\,V^\top$.

*Hint*: Recall that for a matrix $A = U\,\text{diag}(\sigma_i)\,V^\top$, we have $AA^\top = U\,\text{diag}(\sigma_i^2)\,U^\top$, and therefore $(AA^\top)^{1/2} = U\,\text{diag}(|\sigma_i|)\,U^\top$.

**(a.ii)** Substitute into the self-consistent equation. Show that the matrix equation decouples into $d$ independent scalar ODEs:

$$\dot{w}_\alpha = 2\,w_\alpha\,(s_\alpha - w_\alpha), \qquad \alpha = 1, \ldots, d$$

*Hint*: Compute the product $(WW^\top)^{1/2}(M - W)$ in the SVD basis. It is diagonal with entries $w_\alpha(s_\alpha - w_\alpha)$. The second term contributes the same, giving the factor of 2.

**(b)** **Exact solution.** The ODE $\dot{w} = 2w(s - w)$ is a logistic equation. Solve it by separation of variables. Show that:

$$w(t) = \frac{s}{1 + \left(\frac{s}{w_0} - 1\right)e^{-2st}}$$

where $w_0 = w(0)$.

*Hint*: Use partial fractions: $\frac{1}{w(s-w)} = \frac{1}{s}\left(\frac{1}{w} + \frac{1}{s-w}\right)$. Integrate both sides and solve for $w(t)$.

**(c)** **Timescale of learning and contrast with the lazy regime.** From the solution in (b), compute the time $t_\alpha$ it takes for mode $\alpha$ to go from initial strength $w_0$ to a final strength $w_f$. Show that:

$$t_\alpha = \frac{1}{2 s_\alpha} \ln\left(\frac{w_f(s_\alpha - w_0)}{w_0(s_\alpha - w_f)}\right)$$

Deduce that modes with larger singular values $s_\alpha$ are learned faster. This is the **incremental learning** phenomenon: the network learns features in decreasing order of their singular value strength.

Now contrast with the lazy regime from Problem 4 below, where $\dot{W} \approx K_0[M - W]$ with $K_0$ approximately constant. In that case, all modes are learned at the same rate set by $K_0$. Explain in one or two sentences why the rich regime has a **separation of timescales** (learning time $\propto 1/s_\alpha$) while the lazy regime does not.

**(d)** **Implicit bias toward simplicity.** Consider a teacher with $r$ nonzero singular values and a small uniform initialization $w_\alpha(0) = w_0 \ll s_r$ for all $\alpha$.

- Argue that at early times, $W(t)$ is approximately rank 1 (only the top mode $s_1$ has grown significantly).
- At intermediate times, successively more modes switch on and the rank increases incrementally.
- In the limit $t \to \infty$, $W(t) \to M$ (full recovery).

Why does this constitute an implicit bias toward low-rank (i.e. simple) solutions? Connect this to the critical point structure from Problem 1(c): explain how the gradient flow trajectory passes near the saddle points $P_S M$ as modes are recruited one by one.

---

## Problem 4 — Lazy Regime (10 min)

**(a)** Consider a DLN initialized with $W_l(0) \sim \mathcal{N}(0, \sigma^2)$ with $\sigma$ large (specifically $\sigma^2 = d_L^{-\gamma}$ with $\gamma < 1$). In this regime, the initial NTK $K_0$ is large and approximately constant. Argue from the gradient flow equation $\dot{W} = K[M - W]$ that if $K$ doesn't change much during training, the solution is approximately:

$$W(t) \approx M - e^{-K_0 t}(M - W(0))$$

What is the learning timescale in this regime? Does it depend on the singular values of $M$?

**(b)** In the lazy regime, all modes are learned at comparable timescales (set by $K_0$), rather than sequentially. Explain why this means there is **no implicit low-rank bias** in the lazy regime: the network converges to the full OLS solution without passing through a hierarchy of increasingly complex approximations.

**(c)** Contrast the generalization properties: if the teacher $M$ is low-rank but the data is noisy, which regime (lazy or rich) is more likely to recover the low-rank structure? Why?

---

## Problem 5 — Mixed Dynamics: Unifying Lazy and Rich (15 min)

In practice, networks are neither purely lazy nor purely rich. Tu, Aranguri & Jacot (2024) provide a unified description. This problem develops the key ideas.

**(a)** Consider a depth-$L$ DLN with balanced initialization at scale $\sigma$ and width $w$. In the SVD basis of the teacher, an ansatz for the gradient flow of the student is:

$$\dot{W} = -\sqrt{WW^\top + \sigma^{2L} w^2 I}\;\nabla_W \mathcal{L}(W) \;-\; \nabla_W \mathcal{L}(W)\;\sqrt{W^\top W + \sigma^{2L} w^2 I}$$

where $\nabla_W \mathcal{L}(W) = -(M - W)$. Compare this to the balanced gradient flow from Problem 2(c). What is the role of the term $\sigma^{2L} w^2 I$?

*Hint*: In the rich regime ($\sigma \to 0$), this reduces to the equation from Problem 2(c). In the lazy regime ($\sigma$ large), the square root is dominated by the $\sigma^{2L}w^2 I$ term and the dynamics become approximately linear.

**(b)** Working in the SVD basis where $W = \text{diag}(w_1, \ldots, w_d)$ and $M = \text{diag}(s_1, \ldots, s_d)$, show that the dynamics decouple into scalar ODEs:

$$\dot{w}_\alpha = 2(s_\alpha - w_\alpha)\sqrt{w_\alpha^2 + \sigma^{2L} w^2}$$

*Hint*: Substitute diagonal matrices into the ansatz. Each mode evolves independently.

**(c)** Define the **threshold** $\tau := \sigma^{L} w$. A mode $\alpha$ is in the lazy regime when $|w_\alpha| \ll \tau$ and in the rich regime when $|w_\alpha| \gg \tau$. Verify this by examining the two limits of the ODE from (b):
- When $|w_\alpha| \ll \tau$: show $\dot{w}_\alpha \approx 2\tau(s_\alpha - w_\alpha)$, which is linear (lazy).
- When $|w_\alpha| \gg \tau$: show $\dot{w}_\alpha \approx 2|w_\alpha|(s_\alpha - w_\alpha)$, which is the logistic/rich ODE.

**(d)** At initialization, all $w_\alpha$ start small ($\sim \sigma^L$, comparable to $\tau$). Describe the two-phase dynamics:
1. Initially all modes are lazy: the NTK is approximately fixed, the network aligns with the task (the singular vectors of $W$ align with those of $M$).
2. As some $w_\alpha$ grow past $\tau$, they enter the rich regime and accelerate, exhibiting the sigmoidal, incremental learning from Problem 3.

Explain why this transition from lazy to rich can happen **mode-by-mode**: a large singular value mode can already be in the rich regime while smaller modes are still lazy. How does this connect to the grokking phenomenon discussed in the other exercise session?

---

## Problem 6 (Bonus) — Stochastic Implicit Bias (10 min)

SGD introduces noise from mini-batching. A continuous model is the Langevin SDE:

$$d\theta_t = -\nabla \mathcal{L}(\theta_t)\,dt + \sqrt{\eta\,\Sigma(\theta_t)}\,dB_t$$

where $\Sigma(\theta)$ is the covariance of the stochastic gradient noise and $\eta$ is the learning rate.

**(a)** The probability density $p(\theta, t)$ of the parameter evolves according to the Fokker–Planck equation:

$$\partial_t p = -\nabla \cdot \mathbf{j}, \qquad \mathbf{j} = -\nabla \mathcal{L}(\theta)\,p(\theta) - \frac{\eta}{2}\nabla \cdot \left(\Sigma(\theta)\,p(\theta)\right)$$

where $\mathbf{j}$ is the probability current. Assume: (i) stationarity $\partial_t p^* = 0$, (ii) thermal equilibrium $\mathbf{j} = 0$, and (iii) isotropic noise $\Sigma = \sigma^2 I$. Show that the equilibrium distribution is the Boltzmann distribution:

$$p^*(\theta) \propto \exp\left(-\frac{2}{\eta \sigma^2}\mathcal{L}(\theta)\right)$$

*Hint*: Setting $\mathbf{j} = 0$ with $\Sigma = \sigma^2 I$ gives $\nabla \mathcal{L} \cdot p + \frac{\eta\sigma^2}{2}\nabla p = 0$. This is a first-order ODE for $p$ in terms of $\mathcal{L}$. Try the ansatz $p \propto e^{-\beta \mathcal{L}}$ and solve for $\beta$.

**(b)** The ratio $\frac{2}{\eta \sigma^2}$ plays the role of inverse temperature $\beta$. Interpret what happens to the equilibrium distribution when:
- $\eta$ is very small (low temperature)
- $\eta$ is very large (high temperature)

Which regime favors flatter minima, and why might this be beneficial for generalization?

**(c)** In practice, SGD noise is **not** isotropic: $\Sigma(\theta)$ depends on both the loss landscape and the data. Without solving anything, explain qualitatively why anisotropic noise can introduce an implicit bias that goes beyond what the loss function $\mathcal{L}$ alone would select. Specifically, why might SGD preferentially escape sharp directions of the loss while remaining in flat directions?