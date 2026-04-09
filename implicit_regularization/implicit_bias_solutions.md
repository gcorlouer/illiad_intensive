# Implicit Regularization in Deep Linear Networks

## Exercise Session — 1h30, pen and paper — Solutions

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

**(a)** Consider the **diagonal** case: $d_0 = d_1 = \cdots = d_L = d$, and the teacher is diagonal $M = \text{diag}(s_1, \ldots, s_d)$ with $s_1 > s_2 > \cdots > s_d > 0$. Restrict attention to diagonal weight matrices $W_l = \text{diag}(w_l^{(1)}, \ldots, w_l^{(d)})$. Show that the loss decomposes into $d$ independent scalar problems:

$$\mathcal{L} = \frac{1}{2}\sum_{\alpha=1}^d \left(s_\alpha - \prod_{l=1}^L w_l^{(\alpha)}\right)^2$$

**(b)** For a single scalar mode with target $s > 0$, find all first-order critical points of $\ell(w_1, \ldots, w_L) = \frac{1}{2}(s - \prod_l w_l)^2$, for depth $L = 2$. Show that the critical points are:
- The global minimum: $w_1 w_2 = s$
- The origin: $w_1 = w_2 = 0$

Classify the origin as a saddle point by computing the Hessian $H$ of $\ell$ at $(0,0)$ and showing it has both positive and negative eigenvalues.

**(c)** Now consider the full (non-diagonal) DLN with $d_l \geq d_L$ for all $l$. Achour et al. show that every first-order critical point $\theta = (W_1, \ldots, W_L)$ satisfies: there exists a subset $S \subseteq \{1, \ldots, r\}$ such that:

$$W = W_L \cdots W_1 = P_S M$$

where $P_S = U_S U_S^\top$ is the orthogonal projector onto the span of the left singular vectors $\{u_\alpha\}_{\alpha \in S}$ of $M$. Argue (without proof) that these critical points with $|S| < r$ are saddle points, not local minima. Why does this mean there are no spurious local minima?

**(d)** The group $GL_h := GL_{d_1} \times \cdots \times GL_{d_{L-1}}$ acts on the weights by:

$$(W_1, \ldots, W_L) \mapsto (g_1 W_1, g_2 W_2 g_1^{-1}, \ldots, W_L g_{L-1}^{-1})$$

Verify that the student map $\mu(\theta) = W_L \cdots W_1$ is invariant under this action: $\mu(g \cdot \theta) = \mu(\theta)$. What does this imply about the dimension of the set of global minima in parameter space?

---

### Solution 1

**(a)** When all $W_l$ are diagonal, the product $W = W_L \cdots W_1$ is also diagonal with entries $(W)_{\alpha\alpha} = \prod_{l=1}^L w_l^{(\alpha)}$. The teacher $M$ is diagonal with entries $s_\alpha$. Then:

$$\mathcal{L} = \frac{1}{2}\|M - W\|_F^2 = \frac{1}{2}\sum_{\alpha=1}^d (s_\alpha - (W)_{\alpha\alpha})^2 = \frac{1}{2}\sum_{\alpha=1}^d \left(s_\alpha - \prod_{l=1}^L w_l^{(\alpha)}\right)^2$$

Since the different $\alpha$-modes share no parameters, the loss decomposes into $d$ independent scalar problems.

**(b)** For $L = 2$, the gradient equations are:

$$\frac{\partial \ell}{\partial w_1} = -(s - w_1 w_2)w_2 = 0, \qquad \frac{\partial \ell}{\partial w_2} = -(s - w_1 w_2)w_1 = 0$$

**Case 1**: $s - w_1 w_2 = 0$, i.e. $w_1 w_2 = s$. This is the global minimum manifold (a hyperbola $w_2 = s/w_1$) with $\ell = 0$.

**Case 2**: Both $w_1 = 0$ and $w_2 = 0$ (otherwise, e.g. if $w_2 \neq 0$ and $w_1 = 0$, the first equation gives $0 = -s w_2 \neq 0$, a contradiction since $s > 0$).

The Hessian at $(0,0)$:

$$H = \begin{pmatrix} w_2^2 & -s + 2w_1 w_2 \\ -s + 2w_1 w_2 & w_1^2 \end{pmatrix}\bigg|_{(0,0)} = \begin{pmatrix} 0 & -s \\ -s & 0 \end{pmatrix}$$

The eigenvalues are $\pm s$. Since $s > 0$, $H$ has one positive and one negative eigenvalue. The origin is a **strict saddle point**.

**(c)** At a critical point with $W = P_S M$, the loss value is:

$$\mathcal{L} = \frac{1}{2}\|M - P_S M\|_F^2 = \frac{1}{2}\sum_{\alpha \notin S} s_\alpha^2$$

If $|S| < r$, then some $s_\alpha > 0$ is excluded, and the loss is strictly positive. Meanwhile, adding any missing mode $\alpha$ to $S$ would decrease the loss. The analysis of the Hessian (Achour et al., extending the diagonal case above) shows that the direction of adding a missing mode is a descent direction, making these critical points saddle points. All critical points with $|S| = r$ have $W = M$ and $\mathcal{L} = 0$: these are the global minima. Therefore there are no spurious local minima — every local minimum is global.

**(d)** Under the group action:

$$\mu(g \cdot \theta) = W_L g_{L-1}^{-1} \cdot g_{L-1} W_{L-1} g_{L-2}^{-1} \cdot \ldots \cdot g_1 W_1 = W_L W_{L-1} \cdots W_1 = \mu(\theta)$$

All the $g_l$ and $g_l^{-1}$ factors cancel telescopically. This means that every parameter point $\theta$ in the orbit $\mathcal{O}_\theta = GL_h \cdot \theta$ maps to the same student $W$. The global minima form the fiber $\mu^{-1}(M)$, which contains the entire orbit $GL_h \cdot \theta^*$ for any global minimizer $\theta^*$. The orbit has dimension $\sum_{l=1}^{L-1} d_l^2$ (the dimension of $GL_h$), so the set of global minima is a **continuous manifold** of very high dimension — far from being isolated points.

---

## Problem 2 — Gradient Flow and Conserved Quantities (20 min)

**(a)** For a two-layer DLN ($L=2$) with loss $\mathcal{L} = \frac{1}{2}\|M - W_2 W_1\|_F^2$ and whitened inputs, derive the gradient flow equations for each layer. Show that:

$$\dot{W}_1 = W_2^\top(M - W_2 W_1), \qquad \dot{W}_2 = (M - W_2 W_1) W_1^\top$$

**(b)** Define the **balancedness matrix**:

$$G := W_2^\top W_2 - W_1 W_1^\top$$

Show that $G$ is conserved under the gradient flow, i.e. $\dot{G} = 0$.

**(c)** From now on, assume **balanced initialization**: $G(0) = 0$, which by part (b) means $W_2^\top W_2 = W_1 W_1^\top$ for all time. We want to derive the gradient flow in function space (the ODE for the student $W = W_2 W_1$). This requires several steps.

**(c.i)** First, compute $\dot{W} = \dot{W}_2 W_1 + W_2 \dot{W}_1$ using the equations from (a). Show that:

$$\dot{W} = (M - W) W_1^\top W_1 + W_2 W_2^\top (M - W)$$

**(c.ii)** We now need to express $W_1^\top W_1$ and $W_2 W_2^\top$ in terms of $W = W_2 W_1$. Recall the **polar decomposition**: any matrix $A$ can be written as $A = Q P$ where $Q$ is orthogonal and $P = (A^\top A)^{1/2}$ is positive semi-definite. Apply this to $W_1$: write $W_1 = Q_1 (W_1^\top W_1)^{1/2}$. Deduce that:

$$W^\top W = (W_1^\top W_1)^2 \qquad \Longrightarrow \qquad W_1^\top W_1 = (W^\top W)^{1/2}$$

**(c.iii)** By an analogous polar decomposition argument on $W_2^\top$, show that $W_2 W_2^\top = (W W^\top)^{1/2}$.

**(c.iv)** Substitute into (c.i) to obtain:

$$\dot{W} = (W W^\top)^{1/2}(M - W) + (M - W)(W^\top W)^{1/2}$$

**(d)** Define the NTK operator for $L = 2$ as:

$$K[F] := (WW^\top)^{1/2} F + F (W^\top W)^{1/2}$$

Show that the gradient flow from (c) can be written as $\dot{W} = K[M - W]$. Generalize to depth $L$ on the balanced manifold (you may state without proof):

$$\dot{W} = \sum_{k=1}^{L} (WW^\top)^{\frac{L-k}{L}} (M - W) (W^\top W)^{\frac{k-1}{L}}$$

---

### Solution 2

**(a)** We have $\mathcal{L} = \frac{1}{2}\text{Tr}\left[(M - W_2 W_1)^\top(M - W_2 W_1)\right]$. Taking the gradient with respect to $W_1$:

$$\nabla_{W_1}\mathcal{L} = -W_2^\top(M - W_2 W_1)$$

since $\frac{\partial}{\partial W_1}\text{Tr}(W_2 W_1 A) = W_2^\top A^\top$ and here $A = -(M - W_2 W_1)^\top$. The gradient flow $\dot{W}_1 = -\nabla_{W_1}\mathcal{L}$ gives:

$$\dot{W}_1 = W_2^\top(M - W_2 W_1)$$

Similarly, $\nabla_{W_2}\mathcal{L} = -(M - W_2 W_1) W_1^\top$, so:

$$\dot{W}_2 = (M - W_2 W_1) W_1^\top$$

**(b)** Compute:

$$\dot{G} = \dot{W}_2^\top W_2 + W_2^\top \dot{W}_2 - \dot{W}_1 W_1^\top - W_1 \dot{W}_1^\top$$

Let $E := M - W_2 W_1$ (the residual). Substituting:

$$\dot{W}_2^\top W_2 = (W_1 E^\top)(W_2) = W_1 E^\top W_2$$

$$W_2^\top \dot{W}_2 = W_2^\top E W_1^\top$$

$$\dot{W}_1 W_1^\top = W_2^\top E \cdot W_1^\top$$

$$W_1 \dot{W}_1^\top = W_1 (W_2^\top E)^\top = W_1 E^\top W_2$$

Therefore:

$$\dot{G} = W_1 E^\top W_2 + W_2^\top E W_1^\top - W_2^\top E W_1^\top - W_1 E^\top W_2 = 0$$

$G$ is conserved.

**(c)** We proceed step by step.

**(c.i)** Compute $\dot{W} = \dot{W}_2 W_1 + W_2 \dot{W}_1$:

$$\dot{W} = (M - W_2 W_1)W_1^\top W_1 + W_2 W_2^\top(M - W_2 W_1) = (M - W) W_1^\top W_1 + W_2 W_2^\top (M - W)$$

**(c.ii)** Apply the polar decomposition to $W_1$: write $W_1 = Q_1 P_1$ where $Q_1$ is orthogonal and $P_1 = (W_1^\top W_1)^{1/2}$. Then:

$$W = W_2 W_1 = (W_2 Q_1)(W_1^\top W_1)^{1/2}$$

Computing $W^\top W$:

$$W^\top W = (W_1^\top W_1)^{1/2} \cdot Q_1^\top W_2^\top W_2 Q_1 \cdot (W_1^\top W_1)^{1/2}$$

Now use balancedness: $W_2^\top W_2 = W_1 W_1^\top$. And from $W_1 = Q_1(W_1^\top W_1)^{1/2}$:

$$W_1 W_1^\top = Q_1 (W_1^\top W_1)^{1/2} \cdot (W_1^\top W_1)^{1/2} Q_1^\top = Q_1 (W_1^\top W_1) Q_1^\top$$

So $Q_1^\top W_2^\top W_2 Q_1 = Q_1^\top (W_1 W_1^\top) Q_1 = Q_1^\top Q_1 (W_1^\top W_1) Q_1^\top Q_1 = W_1^\top W_1$.

Substituting back:

$$W^\top W = (W_1^\top W_1)^{1/2} \cdot W_1^\top W_1 \cdot (W_1^\top W_1)^{1/2} = (W_1^\top W_1)^2$$

Taking square roots: $\boxed{W_1^\top W_1 = (W^\top W)^{1/2}}$.

**(c.iii)** By an analogous argument using the polar decomposition of $W_2^\top = Q_2 (W_2 W_2^\top)^{1/2}$ and balancedness, one obtains $\boxed{W_2 W_2^\top = (WW^\top)^{1/2}}$.

Alternatively: from balancedness, $W_1$ and $W_2$ have the same singular values $\{\sigma_i\}$. Since $W = W_2 W_1$ has singular values $\{\sigma_i^2\}$ (by the decoupled structure on the balanced manifold), we get $(WW^\top)^{1/2}$ has singular values $\{\sigma_i^2\}$ while $W_2 W_2^\top$ has singular values $\{\sigma_i^2\}$ with the same left singular vectors — giving the result.

**(c.iv)** Substituting into (c.i):

$$\dot{W} = (WW^\top)^{1/2}(M - W) + (M - W)(W^\top W)^{1/2}$$

**(d)** Identifying $F = M - W$, the equation from (c) is:

$$\dot{W} = (WW^\top)^{1/2}F + F(W^\top W)^{1/2} = K[F]$$

For general depth $L$ on the balanced manifold, the same approach (using $L$-fold balanced conditions $W_{l+1}^\top W_{l+1} = W_l W_l^\top$ for all $l$) yields:

$$\dot{W} = \sum_{k=1}^{L}(WW^\top)^{\frac{L-k}{L}}(M-W)(W^\top W)^{\frac{k-1}{L}}$$

The key point is that each term in the sum involves a different weighting of the left and right singular structure of $W$, but they all act on the residual $M - W$. The NTK operator encodes how the current learned representation shapes the learning dynamics.

---

## Problem 3 — Rich Regime: Incremental Learning (25 min)

This is the core problem. We derive the exact solution of the rich regime dynamics directly from the self-consistent equation of Problem 2, emphasizing the NTK perspective.

**(a)** **Alignment assumption.** Start from the balanced gradient flow equation derived in Problem 2(c). Assume NTK alignment: $W(t) = U\,\text{diag}(w_1(t), \ldots, w_d(t))\,V^\top$ shares the singular vectors of $M$.

**(a.i)** Compute $(WW^\top)^{1/2}$, $(W^\top W)^{1/2}$, and $M - W$ in the SVD basis.

**(a.ii)** Substitute into the self-consistent equation and derive the scalar ODEs.

**(b)** **Exact solution.** Solve $\dot{w} = 2w(s - w)$.

**(c)** **Timescale of learning and contrast with lazy regime.**

**(d)** **Implicit bias toward simplicity.**

---

### Solution 3

**(a.i)** Under the alignment assumption $W = U\,\text{diag}(w_\alpha)\,V^\top$:

$$WW^\top = U\,\text{diag}(w_1^2, \ldots, w_d^2)\,U^\top$$

Since $w_\alpha \geq 0$, taking the matrix square root:

$$(WW^\top)^{1/2} = U\,\text{diag}(w_1, \ldots, w_d)\,U^\top$$

Similarly:

$$W^\top W = V\,\text{diag}(w_1^2, \ldots, w_d^2)\,V^\top \quad \Longrightarrow \quad (W^\top W)^{1/2} = V\,\text{diag}(w_1, \ldots, w_d)\,V^\top$$

The residual is:

$$M - W = U\,\text{diag}(s_1 - w_1, \ldots, s_d - w_d)\,V^\top$$

**(a.ii)** Now substitute into $\dot{W} = (WW^\top)^{1/2}(M-W) + (M-W)(W^\top W)^{1/2}$:

**First term:**
$$\underbrace{U\,\text{diag}(w_\alpha)\,U^\top}_{(WW^\top)^{1/2}} \cdot \underbrace{U\,\text{diag}(s_\alpha - w_\alpha)\,V^\top}_{M - W} = U\,\text{diag}\big(w_\alpha(s_\alpha - w_\alpha)\big)\,V^\top$$

where we used $U^\top U = I$ to collapse the middle.

**Second term:**
$$\underbrace{U\,\text{diag}(s_\alpha - w_\alpha)\,V^\top}_{M-W} \cdot \underbrace{V\,\text{diag}(w_\alpha)\,V^\top}_{(W^\top W)^{1/2}} = U\,\text{diag}\big((s_\alpha - w_\alpha)w_\alpha\big)\,V^\top$$

using $V^\top V = I$.

Summing:

$$\dot{W} = U\,\text{diag}\big(2\,w_\alpha(s_\alpha - w_\alpha)\big)\,V^\top$$

Since $W = U\,\text{diag}(w_\alpha)\,V^\top$ and the singular vectors are constant (by the alignment assumption), we can read off the scalar ODEs:

$$\boxed{\dot{w}_\alpha = 2\,w_\alpha\,(s_\alpha - w_\alpha)}$$

This is the key result: the self-consistent equation from Problem 2, combined with NTK-task alignment, directly yields the Saxe et al. logistic dynamics — without needing to go back to the layer-wise equations. The NTK perspective makes the structure transparent: the factor $2w_\alpha$ comes from the NTK operator $K$, which amplifies learning in directions that are already strong, while $(s_\alpha - w_\alpha)$ is the residual in that mode.

**(b)** Separate variables:

$$\int_{w_0}^{w(t)} \frac{dw}{w(s - w)} = 2t$$

Using partial fractions $\frac{1}{w(s-w)} = \frac{1}{s}\left(\frac{1}{w} + \frac{1}{s-w}\right)$:

$$\frac{1}{s}\Big[\ln w - \ln(s-w)\Big]_{w_0}^{w(t)} = 2t$$

$$\frac{1}{s}\ln\frac{w(t)(s - w_0)}{w_0(s - w(t))} = 2t$$

Exponentiating and solving for $w(t)$:

$$\frac{w(t)}{s - w(t)} = \frac{w_0}{s - w_0}\,e^{2st}$$

$$w(t)\left(s - w_0 + w_0\,e^{2st}\right) = s\,w_0\,e^{2st}$$

$$w(t) = \frac{s\,w_0\,e^{2st}}{s - w_0 + w_0\,e^{2st}} = \frac{s}{1 + \left(\frac{s}{w_0} - 1\right)e^{-2st}}$$

This is a **sigmoid** (logistic) curve. For small $w_0 \ll s$, the solution plateaus near $w_0$ for a long time, then transitions rapidly to $w \approx s$ over a timescale $\sim 1/(2s)$.

**(c)** From the integrated form:

$$t_\alpha = \frac{1}{2s_\alpha}\ln\left(\frac{w_f(s_\alpha - w_0)}{w_0(s_\alpha - w_f)}\right)$$

For fixed $w_0$ and $w_f/s_\alpha$, the timescale $t_\alpha \propto 1/s_\alpha$. **Modes with larger singular values are learned faster.** The network successively builds up complexity: mode $s_1$ is learned first, then $s_2$, etc.

**Contrast with the lazy regime:** In the lazy regime, the NTK is frozen at $K_0$ and all modes satisfy $\dot{w}_\alpha \approx -\lambda_0(w_\alpha - s_\alpha)$ where $\lambda_0$ is an eigenvalue of $K_0$ that depends on the initialization but **not** on $s_\alpha$. All modes converge at the same exponential rate $e^{-\lambda_0 t}$. There is no separation of timescales.

In the rich regime, the NTK is state-dependent ($K$ depends on $W$ through $w_\alpha$), and the effective learning rate for mode $\alpha$ is $2w_\alpha$, which is proportional to how much that mode has already been learned. This creates a positive feedback loop (rich get richer) that amplifies the timescale differences encoded in $s_\alpha$, producing a strong separation of timescales: $t_1 \ll t_2 \ll \cdots \ll t_r$.

**(d)** With uniform initialization $w_\alpha(0) = w_0$ for all $\alpha$, the sigmoid solution shows that mode $\alpha$ remains near $w_0$ until time $t \approx \frac{1}{2s_\alpha}\ln(s_\alpha/w_0)$, then rapidly transitions to $w_\alpha \approx s_\alpha$.

- **Early times** ($t \sim \frac{1}{2s_1}\ln(s_1/w_0)$): Only $w_1$ has grown significantly. $W(t) \approx w_1(t)\,u_1 v_1^\top$, which is **rank 1**.

- **Intermediate times**: As $t$ increases, $w_2$ switches on, then $w_3$, etc. The student matrix is approximately $W(t) \approx \sum_{\alpha=1}^{k(t)} w_\alpha(t)\,u_\alpha v_\alpha^\top$ where $k(t)$ increases stepwise. The **rank increases incrementally**.

- **Long times** ($t \to \infty$): All modes have converged, $w_\alpha \to s_\alpha$, and $W \to M$.

This is an **implicit bias toward low-rank solutions**: at any finite time, the network represents the best rank-$k$ approximation to the teacher, where $k$ is the number of modes that have switched on. The gradient flow effectively performs a greedy SVD.

The connection to the saddle structure (Problem 1c) is direct: the saddle points $P_S M$ with $|S| = k$ represent the best rank-$k$ approximation to $M$, with loss $\frac{1}{2}\sum_{\alpha \notin S} s_\alpha^2$. The gradient flow trajectory passes **near** these saddles as it incrementally recruits modes. The **plateaus** in the loss curve correspond to time spent near these saddle points (only $k$ modes active, the rest still dormant), and the **sharp transitions** correspond to escaping along the unstable direction that activates the next mode.

---

## Problem 4 — Lazy Regime (10 min)

**(a)** Consider a DLN initialized with $W_l(0) \sim \mathcal{N}(0, \sigma^2)$ with $\sigma$ large. Argue that the dynamics is approximately linear.

**(b)** Explain the absence of low-rank bias.

**(c)** Contrast generalization in lazy vs. rich.

---

### Solution 4

**(a)** When $\sigma$ is large, the initial student $W(0) = W_L(0) \cdots W_1(0)$ has large singular values, and hence the NTK operator $K$ has large eigenvalues. The key observation is that the NTK depends on $W$ through terms like $(WW^\top)^{(L-k)/L}$, and when $W$ is already large, small changes to $W$ during training produce small *relative* changes to $K$. Formally, $\delta K / K \sim \delta W / W \sim (M - W(0))/W(0) \ll 1$ when $\|W(0)\| \gg \|M\|$.

So we can freeze $K \approx K_0$ and the equation $\dot{W} = K_0[M - W]$ becomes a linear ODE. In the SVD basis of $K_0$, each mode $w_\alpha$ satisfies $\dot{w}_\alpha = -\lambda_\alpha(w_\alpha - s_\alpha)$ where $\lambda_\alpha$ are eigenvalues of $K_0$. The solution is:

$$w_\alpha(t) = s_\alpha - (s_\alpha - w_\alpha(0))e^{-\lambda_\alpha t}$$

or in matrix form $W(t) \approx M - e^{-K_0 t}(M - W(0))$. The learning timescale is $t \sim 1/\lambda_{\min}(K_0)$, determined by the smallest eigenvalue of the initial NTK. It does **not** depend on the singular values of $M$ the way the rich regime does.

**(b)** In the lazy regime, all modes are learned at timescales set by $K_0$, not by $s_\alpha$. If $K_0$ is approximately isotropic (as it is for large random initialization), all modes converge at comparable rates. There is no sequential, rank-by-rank buildup. The network goes directly from $W(0)$ to $M$ without passing through low-rank intermediate solutions. The implicit bias of the lazy regime is toward the **minimum-norm** perturbation of the initial weights, not toward low-rank structure.

**(c)** Suppose the true teacher is $M = M_{\text{signal}} + M_{\text{noise}}$ where $M_{\text{signal}}$ is low-rank and $M_{\text{noise}}$ is a small full-rank perturbation from noise. In the **rich regime**, the incremental dynamics learns $M_{\text{signal}}$ first (its singular values are large) and only later starts fitting $M_{\text{noise}}$. With early stopping, the network recovers the low-rank signal while ignoring the noise — good generalization.

In the **lazy regime**, all components of $M$ are learned simultaneously, including the noise. The network converges to the full OLS solution $M$ and fits the noise along with the signal. This is analogous to kernel ridge regression with the initial NTK, which has no mechanism to separate signal from noise based on singular value structure. Generalization is worse.

---

## Problem 5 — Mixed Dynamics: Unifying Lazy and Rich (15 min)

**(a)** The ansatz and the role of $\sigma^{2L} w^2 I$.

**(b)** Scalar ODE in the SVD basis.

**(c)** Lazy and rich limits.

**(d)** Mode-by-mode transition.

---

### Solution 5

**(a)** The term $\sigma^{2L} w^2 I$ acts as a **regularization of the square root**: it prevents the NTK operator from degenerating when $W$ has small or zero singular values. In the rich regime ($\sigma \to 0$), this term vanishes and we recover:

$$\dot{W} = \sqrt{WW^\top}(M-W) + (M-W)\sqrt{W^\top W}$$

which is the balanced gradient flow from Problem 2(c). In the lazy regime ($\sigma$ large), $\sigma^{2L} w^2 I$ dominates the square root, giving $\sqrt{WW^\top + \sigma^{2L}w^2 I} \approx \sigma^L w \cdot I$, and the dynamics reduce to:

$$\dot{W} \approx 2\sigma^L w (M - W)$$

which is the linear (lazy) ODE with rate $2\sigma^L w = 2\tau$. The $\sigma^{2L}w^2 I$ term interpolates between these two regimes by contributing a constant "floor" to the effective NTK eigenvalues.

**(b)** In the SVD basis, $W = \text{diag}(w_1, \ldots, w_d)$, $M = \text{diag}(s_1, \ldots, s_d)$, and:

$$\sqrt{WW^\top + \sigma^{2L}w^2 I} = \text{diag}\left(\sqrt{w_\alpha^2 + \tau^2}\right)$$

where $\tau = \sigma^L w$. The ansatz becomes:

$$\dot{w}_\alpha = (s_\alpha - w_\alpha)\sqrt{w_\alpha^2 + \tau^2} + (s_\alpha - w_\alpha)\sqrt{w_\alpha^2 + \tau^2} = 2(s_\alpha - w_\alpha)\sqrt{w_\alpha^2 + \tau^2}$$

Each mode decouples into a scalar ODE.

**(c)** **Lazy limit** ($|w_\alpha| \ll \tau$): $\sqrt{w_\alpha^2 + \tau^2} \approx \tau$, so:

$$\dot{w}_\alpha \approx 2\tau(s_\alpha - w_\alpha)$$

This is a linear ODE with solution $w_\alpha(t) = s_\alpha(1 - e^{-2\tau t})$ (assuming $w_\alpha(0) \approx 0$). All modes evolve at the same rate $2\tau$, independent of $s_\alpha$. This is the lazy regime.

**Rich limit** ($|w_\alpha| \gg \tau$): $\sqrt{w_\alpha^2 + \tau^2} \approx |w_\alpha|$, so:

$$\dot{w}_\alpha \approx 2|w_\alpha|(s_\alpha - w_\alpha) = 2w_\alpha(s_\alpha - w_\alpha)$$

(assuming $w_\alpha > 0$). This is exactly the logistic ODE from Problem 3(a), the rich regime with incremental learning.

**(d)** At initialization, $w_\alpha \sim \sigma^L \sim \tau$, so all modes start at the boundary between lazy and rich. In the early phase:

1. **Lazy phase**: All modes satisfy $|w_\alpha| \lesssim \tau$ and evolve at rate $\sim 2\tau$. The dynamics are approximately linear, and the NTK is approximately constant. During this phase, the network **aligns** with the task: the singular vectors of $W$ rotate to match those of $M$. (This alignment is crucial because the rich regime dynamics from Problem 3 assumed pre-alignment.)

2. **Lazy-to-rich transition**: Modes grow, and the first to cross $\tau$ is the one with the largest $s_\alpha$ (since it grows fastest in the lazy phase). Once $|w_\alpha| > \tau$, mode $\alpha$ transitions to the rich regime and begins its sigmoidal acceleration. Meanwhile, smaller modes may still be lazy.

3. **Rich phase**: Eventually all modes cross the threshold and undergo sigmoidal growth, but they do so sequentially — large modes first. This recovers the incremental learning phenomenon from Problem 3.

**Connection to grokking**: The grokking phenomenon (memorization followed by delayed generalization) can be understood as a lazy-to-rich transition. In the lazy phase, the network memorizes the training data via kernel regression with the initial NTK, which may be misaligned with the task. The test loss plateaus. Later, as modes cross into the rich regime, feature learning kicks in, the NTK rotates to align with the task structure, and the test loss drops. The timescale of the grokking delay is controlled by how long it takes the relevant modes to cross the threshold $\tau$ — which depends on the initialization scale $\sigma$ and width $w$. Larger $\sigma$ or $w$ increases $\tau$, extending the lazy phase and widening the grokking gap. This is exactly analogous to the role of $\alpha$ in the Kumar et al. model from the grokking exercise session.

---

## Problem 6 (Bonus) — Stochastic Implicit Bias (10 min)

**(a)** Derive the Boltzmann distribution.

**(b)** Interpret the temperature.

**(c)** Anisotropic noise.

---

### Solution 6

**(a)** Setting $\mathbf{j} = 0$ (thermal equilibrium) with isotropic noise $\Sigma = \sigma^2 I$:

$$0 = -\nabla \mathcal{L}(\theta)\,p^*(\theta) - \frac{\eta \sigma^2}{2}\nabla p^*(\theta)$$

Rearranging:

$$\frac{\nabla p^*}{p^*} = -\frac{2}{\eta \sigma^2}\nabla \mathcal{L}$$

The left side is $\nabla \ln p^*$, so:

$$\nabla \ln p^*(\theta) = -\frac{2}{\eta \sigma^2}\nabla \mathcal{L}(\theta)$$

Integrating:

$$\ln p^*(\theta) = -\frac{2}{\eta \sigma^2}\mathcal{L}(\theta) + \text{const}$$

$$\boxed{p^*(\theta) \propto \exp\left(-\frac{2}{\eta \sigma^2}\mathcal{L}(\theta)\right)}$$

This is the Boltzmann distribution with inverse temperature $\beta = \frac{2}{\eta \sigma^2}$.

**(b)** The effective temperature is $T = \frac{\eta \sigma^2}{2}$.

- **Small $\eta$ (low temperature)**: $\beta \to \infty$, the distribution concentrates sharply on the global minima of $\mathcal{L}$. SGD converges to the minimizer with the lowest loss, without exploring.

- **Large $\eta$ (high temperature)**: $\beta \to 0$, the distribution becomes nearly uniform over parameter space. SGD explores widely and does not settle into any particular minimum.

**Flat vs. sharp minima**: At moderate temperature, the Boltzmann distribution assigns more probability mass to **broad basins** (flat minima) than to narrow ones. This is because a flat minimum occupies a larger volume of parameter space at any given loss level. The width of the basin acts as an effective entropic contribution. Flat minima tend to generalize better because small perturbations to the parameters (or slight distribution shift in the data) don't dramatically change the loss, so the implicit bias toward flat minima from SGD noise is beneficial.

**(c)** When $\Sigma(\theta)$ is not isotropic, the noise pushes harder in some directions than others. The Fokker-Planck equation with anisotropic noise has a probability current:

$$\mathbf{j} = -\nabla \mathcal{L} \cdot p - \frac{\eta}{2}\nabla \cdot(\Sigma(\theta) p)$$

The term $\nabla \cdot(\Sigma(\theta) p)$ introduces an **Itô drift** that depends on the spatial variation of the noise covariance. In directions where $\Sigma$ has large eigenvalues (high noise), the effective diffusion is strong — the system escapes easily from sharp regions of the loss landscape. In directions where $\Sigma$ has small eigenvalues (low noise), the system is more stable and tends to remain.

This means that SGD noise can act as a direction-dependent regularizer: it preferentially pushes the parameters out of directions that have high gradient noise variance (which often correspond to sharp, overfitting directions) while preserving the parameters along low-noise, flat directions. This goes beyond what a simple Boltzmann distribution on $\mathcal{L}$ would predict — the equilibrium, if it exists, depends on $\Sigma(\theta)$ in a non-trivial way (and in general, detailed balance $\mathbf{j}=0$ may not hold, leading to persistent probability currents and non-equilibrium steady states).