# Implicit Regularization in Deep Linear Networks

## Exercise Session — Solutions

**References**:
- Saxe, McClelland & Ganguli, *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks*, ICLR 2014 / PNAS 2019
- Achour, Malgouyres & Gerchinovitz, *The loss landscape of deep linear neural networks: a second-order analysis*, JMLR 2024
- Tu, Aranguri & Jacot, *Mixed Dynamics In Linear Networks: Unifying the Lazy and Active Regimes*, 2024

---

## Problem 1 — Loss Landscape Geometry (15 min)

### Solution 1

**(a)** When all $W_l$ are diagonal, the product $W = W_L \cdots W_1$ is also diagonal with entries $(W)_{\alpha\alpha} = \prod_{l=1}^L w_l^{(\alpha)}$. The teacher $M$ is diagonal with entries $s_\alpha$. Then:

$$\mathcal{L} = \frac{1}{2}\|M - W\|_F^2 = \frac{1}{2}\sum_{\alpha=1}^d (s_\alpha - (W)_{\alpha\alpha})^2 = \frac{1}{2}\sum_{\alpha=1}^d \left(s_\alpha - \prod_{l=1}^L w_l^{(\alpha)}\right)^2$$

Since the different modes $\alpha$ share no parameters, the loss decomposes into $d$ independent scalar problems. $\square$

**(b)** For $L = 2$, the gradient equations are:

$$\frac{\partial \ell}{\partial w_1} = -(s - w_1 w_2)\,w_2 = 0, \qquad \frac{\partial \ell}{\partial w_2} = -(s - w_1 w_2)\,w_1 = 0$$

**Case 1**: $s - w_1 w_2 = 0$, i.e. $w_1 w_2 = s$. This is the global minimum manifold (a hyperbola $w_2 = s/w_1$) with $\ell = 0$.

**Case 2**: If $s - w_1 w_2 \neq 0$, then the first equation requires $w_2 = 0$ and the second requires $w_1 = 0$. (For instance, if $w_2 \neq 0$ and $w_1 = 0$, the first equation gives $-s\,w_2 = 0$, which contradicts $s > 0$ and $w_2 \neq 0$.) So the only other critical point is $w_1 = w_2 = 0$.

The Hessian at $(0,0)$. We need the second derivatives of $\ell = \frac{1}{2}(s - w_1 w_2)^2$:

$$\frac{\partial^2 \ell}{\partial w_1^2} = w_2^2, \qquad \frac{\partial^2 \ell}{\partial w_2^2} = w_1^2, \qquad \frac{\partial^2 \ell}{\partial w_1 \partial w_2} = -(s - w_1 w_2) + w_1 w_2 = 2w_1 w_2 - s$$

At $(0,0)$:

$$H = \begin{pmatrix} 0 & -s \\ -s & 0 \end{pmatrix}$$

The eigenvalues are $\pm s$. Since $s > 0$, $H$ has one positive and one negative eigenvalue. The origin is a **strict saddle point**. $\square$

**(c)** At the critical point $W = P_S M$ with $S = \{1, \ldots, r\}$, the student retains only the first $r$ modes: $W = \text{diag}(s_1, \ldots, s_r, 0, \ldots, 0)$. The loss is:

$$\mathcal{L} = \frac{1}{2}\|M - P_S M\|_F^2 = \frac{1}{2}\sum_{\alpha \notin S} s_\alpha^2 = \frac{1}{2}\sum_{\alpha=r+1}^d s_\alpha^2$$

This is strictly positive whenever $r < d$ (since all $s_\alpha > 0$), so these are not global minima. By the Hessian analysis (extending part (b)), the direction corresponding to "switching on" a missing mode $\alpha \notin S$ is a descent direction, making these critical points saddle points. All critical points with $|S| = d$ have $W = M$ and $\mathcal{L} = 0$: these are the global minima. Therefore there are **no spurious local minima** — every local minimum is global. $\square$

**(d)** Under the group action:

$$\mu(g \cdot \theta) = W_L g_{L-1}^{-1} \cdot g_{L-1} W_{L-1} g_{L-2}^{-1} \cdot \ldots \cdot g_1 W_1 = W_L W_{L-1} \cdots W_1 = \mu(\theta)$$

All $g_l$ and $g_l^{-1}$ factors cancel telescopically. This means every parameter point in the orbit $\mathcal{O}_\theta = GL_h \cdot \theta$ maps to the same student $W$. The global minima form the fiber $\mu^{-1}(M)$, which contains the entire orbit $GL_h \cdot \theta^*$ for any global minimizer $\theta^*$. The orbit has dimension $\sum_{l=1}^{L-1} d_l^2$ (the dimension of $GL_h$), so the set of global minima is a **continuous manifold of very high dimension** — far from being isolated points. $\square$

---

## Problem 2 — Gradient Flow and Conserved Quantities (20 min)

### Solution 2

**(a)** We have $\mathcal{L} = \frac{1}{2}\|M - W_2 W_1\|_F^2 = \frac{1}{2}\text{Tr}\left[(M - W_2 W_1)^\top(M - W_2 W_1)\right]$.

**General matrix derivation.** Expand:

$$\mathcal{L} = \frac{1}{2}\text{Tr}(M^\top M) - \text{Tr}(M^\top W_2 W_1) + \frac{1}{2}\text{Tr}(W_1^\top W_2^\top W_2 W_1)$$

Differentiating with respect to $W_1$, using $\frac{\partial}{\partial A}\text{Tr}(B^\top A) = B$ and $\frac{\partial}{\partial A}\text{Tr}(A^\top C A) = (C + C^\top)A$:

$$\nabla_{W_1}\mathcal{L} = -W_2^\top M + W_2^\top W_2 W_1 = -W_2^\top(M - W_2 W_1)$$

So $\dot{W}_1 = -\nabla_{W_1}\mathcal{L} = W_2^\top(M - W_2 W_1)$.

Similarly, $\nabla_{W_2}\mathcal{L} = -(M - W_2 W_1)W_1^\top$, giving $\dot{W}_2 = (M - W_2 W_1)W_1^\top$.

**Diagonal shortcut.** For diagonal matrices $W_1 = \text{diag}(a_\alpha)$, $W_2 = \text{diag}(b_\alpha)$, $M = \text{diag}(s_\alpha)$: the loss decouples as $\mathcal{L} = \frac{1}{2}\sum_\alpha(s_\alpha - b_\alpha a_\alpha)^2$. Then $\dot{a}_\alpha = -\partial\mathcal{L}/\partial a_\alpha = b_\alpha(s_\alpha - b_\alpha a_\alpha)$ and $\dot{b}_\alpha = a_\alpha(s_\alpha - b_\alpha a_\alpha)$, which is the diagonal version of the matrix equations above. $\square$

**(b)** Let $E := M - W_2 W_1$ denote the residual. Compute:

$$\dot{G} = \dot{W}_2^\top W_2 + W_2^\top \dot{W}_2 - \dot{W}_1 W_1^\top - W_1 \dot{W}_1^\top$$

Substitute the gradient flow equations. Using $\dot{W}_2 = E W_1^\top$ and $\dot{W}_1 = W_2^\top E$:

$$\dot{W}_2^\top W_2 = (E W_1^\top)^\top W_2 = W_1 E^\top W_2$$

$$W_2^\top \dot{W}_2 = W_2^\top E W_1^\top$$

$$\dot{W}_1 W_1^\top = W_2^\top E W_1^\top$$

$$W_1 \dot{W}_1^\top = W_1(W_2^\top E)^\top = W_1 E^\top W_2$$

Therefore:

$$\dot{G} = W_1 E^\top W_2 + W_2^\top E W_1^\top - W_2^\top E W_1^\top - W_1 E^\top W_2 = 0$$

The terms cancel pairwise. $G$ is conserved. $\square$

**(c.i)** Apply the product rule:

$$\dot{W} = \dot{W}_2 W_1 + W_2 \dot{W}_1 = (M - W_2 W_1)W_1^\top \cdot W_1 + W_2 \cdot W_2^\top(M - W_2 W_1)$$

Writing $W = W_2 W_1$:

$$\dot{W} = (M - W)\,W_1^\top W_1 + W_2 W_2^\top\,(M - W) \qquad \square$$

**(c.ii)** For diagonal matrices, $W_1 = \text{diag}(a_\alpha)$ and $W_2 = \text{diag}(b_\alpha)$ with $W = \text{diag}(a_\alpha b_\alpha)$. Balancedness gives $b_\alpha^2 = a_\alpha^2$ for each $\alpha$, i.e. $|b_\alpha| = |a_\alpha|$ (with matching signs for positive entries).

Now $W_1^\top W_1 = \text{diag}(a_\alpha^2)$ and $W^\top W = \text{diag}(a_\alpha^2 b_\alpha^2)$. Using $b_\alpha^2 = a_\alpha^2$:

$$W^\top W = \text{diag}(a_\alpha^2 \cdot a_\alpha^2) = \text{diag}(a_\alpha^4)$$

Taking the positive square root:

$$(W^\top W)^{1/2} = \text{diag}(a_\alpha^2) = W_1^\top W_1 \qquad \square$$

**(c.iii)** By the same argument, $W_2 W_2^\top = \text{diag}(b_\alpha^2) = \text{diag}(a_\alpha^2)$ (using $b_\alpha^2 = a_\alpha^2$). And $WW^\top = \text{diag}(a_\alpha^2 b_\alpha^2) = \text{diag}(a_\alpha^4)$, so $(WW^\top)^{1/2} = \text{diag}(a_\alpha^2) = W_2 W_2^\top$. $\square$

**(c.iv)** Substituting into (c.i):

$$\dot{W} = (M - W)(W^\top W)^{1/2} + (WW^\top)^{1/2}(M - W) \qquad \square$$

**(d)** With $F = M - W$, the equation from (c.iv) reads:

$$\dot{W} = (WW^\top)^{1/2}\,F + F\,(W^\top W)^{1/2} = K[F] = K[M - W] \qquad \square$$

The general depth-$L$ result $\dot{W} = \sum_{k=1}^{L}(WW^\top)^{(L-k)/L}(M-W)(W^\top W)^{(k-1)/L}$ follows from the same approach applied to the $L$-fold balanced conditions $W_{l+1}^\top W_{l+1} = W_l W_l^\top$ for all $l$, and extends to non-diagonal matrices via the polar decomposition.

---

## Problem 3 — Rich Regime: Incremental Learning (25 min)

### Solution 3

**(a.i)** Under the alignment assumption, $W = U\,\text{diag}(w_\alpha)\,V^\top$. Then:

$$WW^\top = U\,\text{diag}(w_\alpha^2)\,U^\top$$

Since $w_\alpha \geq 0$, the positive matrix square root is:

$$(WW^\top)^{1/2} = U\,\text{diag}(w_\alpha)\,U^\top$$

Similarly:

$$W^\top W = V\,\text{diag}(w_\alpha^2)\,V^\top \quad \Longrightarrow \quad (W^\top W)^{1/2} = V\,\text{diag}(w_\alpha)\,V^\top$$

The residual is immediate:

$$M - W = U\,\text{diag}(s_\alpha - w_\alpha)\,V^\top$$

The NTK operator acts on any matrix $F = U\,\text{diag}(f_\alpha)\,V^\top$ as:

$$K[F] = U\,\text{diag}(w_\alpha)\,U^\top \cdot U\,\text{diag}(f_\alpha)\,V^\top + U\,\text{diag}(f_\alpha)\,V^\top \cdot V\,\text{diag}(w_\alpha)\,V^\top = U\,\text{diag}(2 w_\alpha f_\alpha)\,V^\top$$

So $K[F]$ stays in the SVD basis of $M$ — the NTK is aligned to the task. $\square$

**(a.ii)** Substitute into $\dot{W} = (WW^\top)^{1/2}(M-W) + (M-W)(W^\top W)^{1/2}$:

**First term:**

$$U\,\text{diag}(w_\alpha)\,\underbrace{U^\top U}_{I}\,\text{diag}(s_\alpha - w_\alpha)\,V^\top = U\,\text{diag}\big(w_\alpha(s_\alpha - w_\alpha)\big)\,V^\top$$

**Second term:**

$$U\,\text{diag}(s_\alpha - w_\alpha)\,\underbrace{V^\top V}_{I}\,\text{diag}(w_\alpha)\,V^\top = U\,\text{diag}\big((s_\alpha - w_\alpha)\,w_\alpha\big)\,V^\top$$

Summing:

$$\dot{W} = U\,\text{diag}\big(2\,w_\alpha(s_\alpha - w_\alpha)\big)\,V^\top$$

Since $W = U\,\text{diag}(w_\alpha)\,V^\top$ and the singular vectors are constant by assumption, we read off:

$$\boxed{\dot{w}_\alpha = 2\,w_\alpha\,(s_\alpha - w_\alpha)}$$

The NTK perspective makes the structure transparent: the factor $2w_\alpha$ is the NTK eigenvalue for mode $\alpha$, which amplifies learning in directions that are already strong, while $(s_\alpha - w_\alpha)$ is the residual. $\square$

**(b)** From the logistic solution $w(t) = \frac{s}{1 + (s/w_0 - 1)e^{-2st}}$, we invert to find $t$ as a function of $w$. The derivation is equivalent to integrating $\dot{w} = 2w(s-w)$ by separation of variables:

$$\int_{w_0}^{w_f} \frac{dw}{w(s-w)} = 2\,t_\alpha$$

Using partial fractions $\frac{1}{w(s-w)} = \frac{1}{s}\!\left(\frac{1}{w} + \frac{1}{s-w}\right)$:

$$\frac{1}{s_\alpha}\Big[\ln w - \ln(s_\alpha - w)\Big]_{w_0}^{w_f} = 2\,t_\alpha$$

$$\frac{1}{s_\alpha}\ln\frac{w_f\,(s_\alpha - w_0)}{w_0\,(s_\alpha - w_f)} = 2\,t_\alpha$$

Therefore:

$$\boxed{t_\alpha = \frac{1}{2s_\alpha}\ln\left(\frac{w_f\,(s_\alpha - w_0)}{w_0\,(s_\alpha - w_f)}\right)}$$

For a fixed ratio $w_f/s_\alpha$ and fixed $w_0$, the learning time scales as $t_\alpha \propto 1/s_\alpha$: **modes with larger singular values are learned faster**. The network learns features in decreasing order of their strength — a strong separation of timescales.

**Contrast with the lazy regime** (anticipating Problem 4): in the lazy regime the NTK is frozen at $K_0 = 2w_0 \gg 1$, so $\dot{w}_\alpha \approx 2w_0(s_\alpha - w_\alpha)$. All modes converge at the same exponential rate $2w_0$, independent of $s_\alpha$. The rich regime has state-dependent NTK ($2w_\alpha$), creating a positive feedback loop — modes that are already large learn even faster — which amplifies the differences between $s_\alpha$ into a hierarchy of timescales $t_1 \ll t_2 \ll \cdots \ll t_r$. $\square$

**(c)** With uniform initialization $w_\alpha(0) = w_0$ for all $\alpha$, the sigmoid solution shows that mode $\alpha$ remains near $w_0$ until time $t \approx \frac{1}{2s_\alpha}\ln(s_\alpha/w_0)$, then transitions rapidly to $w_\alpha \approx s_\alpha$.

**Early times** ($t \sim \frac{1}{2s_1}\ln(s_1/w_0)$): Only mode 1 has grown significantly. The student is approximately $W(t) \approx w_1(t)\,u_1 v_1^\top$, which is **rank 1**.

**Intermediate times**: As $t$ increases, $w_2$ switches on, then $w_3$, etc. The student is approximately $W(t) \approx \sum_{\alpha=1}^{k(t)} w_\alpha(t)\,u_\alpha v_\alpha^\top$ where $k(t)$ increases stepwise. The **rank increases incrementally**.

**Long times** ($t \to \infty$): All modes converge, $w_\alpha \to s_\alpha$, and $W \to M$.

This is an **implicit bias toward low-rank (simple) solutions**: at any finite time, the network represents the best rank-$k$ approximation to the teacher. The gradient flow effectively performs a greedy SVD.

**Connection to the saddle structure (Problem 1c):** The saddle points $P_S M$ with $|S| = k$ are exactly the best rank-$k$ approximations to $M$, with loss $\frac{1}{2}\sum_{\alpha > k} s_\alpha^2$. The gradient flow trajectory passes **near** these saddles as it incrementally recruits modes. The **plateaus** in the loss curve correspond to time spent near saddle points (only $k$ modes active), and the **sharp transitions** correspond to escaping along the unstable direction that activates the next mode. $\square$

---

## Problem 4 — Lazy Regime (15 min)

### Solution 4

**(a)** Starting from the same ODE $\dot{w} = 2w(s - w)$, with $w_0 \gg s > 0$, the weight $w$ needs to decrease from $w_0$ to $s$. As long as $w$ has not changed much from $w_0$, we can write $w = w_0 + \delta w$ with $|\delta w| \ll w_0$, so:

$$2w = 2(w_0 + \delta w) \approx 2w_0$$

The ODE becomes:

$$\dot{w} \approx 2w_0(s - w) \qquad \square$$

This is a **linear** ODE — the NTK factor $2w$ has been frozen at its initial value $2w_0$. Note that only the NTK prefactor is linearized; the residual $(s - w)$ is kept exact since it drives learning.

**(b)** The linearized ODE $\dot{w} = 2w_0(s - w)$ is first-order linear. Set $\tilde{w} := w - s$, then $\dot{\tilde{w}} = -2w_0\,\tilde{w}$, with solution $\tilde{w}(t) = (w_0 - s)\,e^{-2w_0 t}$. Therefore:

$$\boxed{w(t) = s + (w_0 - s)\,e^{-2w_0 t}}$$

This is **exponential** convergence to $s$ (contrast with the sigmoid of Problem 3). The learning timescale is:

$$t_{\text{lazy}} \sim \frac{1}{2w_0}$$

It depends on the **initialization scale** $w_0$ but **not on the target** $s$. This is the defining feature of the lazy regime: the learning speed is set by the NTK at initialization, not by the structure of the task.

**Self-consistency:** During the learning phase ($t \lesssim 1/(2w_0)$), the displacement is $|w(t) - w_0| \leq |s - w_0| \approx w_0$ (since $w_0 \gg s$). The relative change in the NTK is $\Delta(2w)/(2w_0) = (w_0 - s)/w_0 = 1 - s/w_0 \approx 1$, so the NTK does change significantly in absolute terms. However, the key point is that the dynamics remain well approximated by the linear ODE because the convergence rate is dominated by $2w_0$, which is large and approximately constant throughout training. In the rich regime, by contrast, the NTK changes from $2w_0 \approx 0$ to $2s$ — a change of order $s/w_0 \to \infty$ relative to the initial value — making the linearization completely invalid. $\square$

**(c)** Restoring the mode index, each mode satisfies $\dot{w}_\alpha \approx 2w_0(s_\alpha - w_\alpha)$ with solution:

$$w_\alpha(t) = s_\alpha + (w_0 - s_\alpha)\,e^{-2w_0 t}$$

All modes converge at the **same exponential rate** $2w_0$, regardless of $s_\alpha$. The time for mode $\alpha$ to go from $w_0$ to within $(1-\epsilon)$ of $s_\alpha$ is:

$$t_\alpha^{\text{lazy}} = \frac{1}{2w_0}\ln\!\left(\frac{w_0 - s_\alpha}{\epsilon\,(w_0 - s_\alpha)}\right) = \frac{1}{2w_0}\ln\frac{1}{\epsilon}$$

which is **independent of $s_\alpha$**. All modes reach their targets simultaneously.

**Contrast with the rich regime:**

| | Rich regime | Lazy regime |
|---|---|---|
| NTK | State-dependent: $2w_\alpha$ | Frozen: $2w_0$ |
| Learning time for mode $\alpha$ | $t_\alpha \propto 1/s_\alpha$ | $t_\alpha \approx \text{const}$ |
| Timescale separation | Strong: $t_1 \ll t_2 \ll \cdots$ | None |
| Intermediate solutions | Low-rank (incremental) | Full-rank from the start |
| Implicit rank bias | Yes (low-rank → high-rank) | No |

In the lazy regime, all components of $M$ are learned simultaneously: signal and noise alike. The network converges directly to the full OLS solution without passing through low-rank intermediates. There is no mechanism to separate signal from noise based on singular value structure. With early stopping, the rich regime recovers a low-rank signal while ignoring noise; the lazy regime cannot. $\square$

---

## Problem 5 — Mixed Dynamics: Unifying Lazy and Rich (15 min)

### Solution 5

**(a)** The interpolating ODE is $\dot{w}_\alpha = 2\sqrt{w_\alpha^2 + \tau^2}\,(s_\alpha - w_\alpha)$.

**Rich limit ($\tau \to 0$):** The square root reduces to $\sqrt{w_\alpha^2} = |w_\alpha|$, giving:

$$\dot{w}_\alpha = 2|w_\alpha|(s_\alpha - w_\alpha)$$

For $w_\alpha > 0$ this is $2w_\alpha(s_\alpha - w_\alpha)$, exactly the logistic ODE from Problem 3. $\square$

**Lazy limit ($\tau \to \infty$, $w_\alpha$ bounded):** The square root is dominated by $\tau$: $\sqrt{w_\alpha^2 + \tau^2} \approx \tau$, giving:

$$\dot{w}_\alpha \approx 2\tau(s_\alpha - w_\alpha)$$

This is a linear ODE with rate $2\tau$, independent of $s_\alpha$ — the frozen-NTK regime of Problem 4 (with $\tau$ playing the role of $w_0$). $\square$

**(b)** At initialization, $w_\alpha(0) \sim \sigma^2 \ll \tau = \sigma^2\sqrt{n}$ (since $\sqrt{n} \gg 1$). So initially all modes satisfy $|w_\alpha| \ll \tau$.

**Early phase (lazy):** $\sqrt{w_\alpha^2 + \tau^2} \approx \tau$ for all $\alpha$. Every mode evolves at the same linear rate $2\tau$:

$$\dot{w}_\alpha \approx 2\tau(s_\alpha - w_\alpha) \qquad \Longrightarrow \qquad w_\alpha(t) \approx s_\alpha(1 - e^{-2\tau t})$$

(assuming $w_\alpha(0) \approx 0$). The dynamics are approximately linear and the NTK is approximately constant. During this phase, the network **aligns** with the task: each $w_\alpha$ grows toward $s_\alpha$ at the same rate. There is no timescale separation.

**Late phase (rich):** Once some $w_\alpha$ grow past $\tau$, the square root transitions to $\sqrt{w_\alpha^2 + \tau^2} \approx |w_\alpha|$, and those modes enter the rich regime with the sigmoidal, self-accelerating dynamics $\dot{w}_\alpha \approx 2w_\alpha(s_\alpha - w_\alpha)$. The state-dependent NTK creates timescale separation, and the modes that have crossed the threshold converge rapidly to their targets.

In summary: the network starts in a lazy phase where all modes grow uniformly (alignment), then transitions to a rich phase where modes accelerate individually (incremental learning). $\square$

**(c)** In the lazy phase, each mode grows as $w_\alpha(t) \approx s_\alpha(1 - e^{-2\tau t})$. At any given time, $w_\alpha(t) \propto s_\alpha$: **modes with larger $s_\alpha$ are larger**. Since the lazy-to-rich transition occurs when $|w_\alpha| \sim \tau$, the time for mode $\alpha$ to cross the threshold satisfies:

$$s_\alpha(1 - e^{-2\tau t_\alpha^*}) = \tau \qquad \Longrightarrow \qquad t_\alpha^* = \frac{1}{2\tau}\ln\frac{s_\alpha}{s_\alpha - \tau}$$

For $s_\alpha > \tau$, this crossing time exists and is smaller for larger $s_\alpha$. For $s_\alpha < \tau$, the mode never reaches the threshold and remains permanently lazy.

This means that **different modes can be in different regimes simultaneously**: large-$s_\alpha$ modes have crossed into the rich regime and are converging rapidly, while small-$s_\alpha$ modes are still in the lazy phase (or may never leave it). The network interpolates between the two regimes mode by mode.

**Connection to grokking:** This lazy-to-rich transition provides a mechanism for grokking. During the lazy phase, the network fits the training data via kernel regression with the initial (misaligned) NTK — it **memorizes**. The test loss plateaus because the frozen kernel cannot capture the task structure. Later, as modes cross the threshold into the rich regime, feature learning kicks in: the NTK rotates to align with the task, and the test loss drops — **generalization** is achieved. The grokking delay is controlled by the time it takes the relevant modes to cross $\tau$. Increasing the initialization scale $\sigma$ or the width $n$ increases $\tau$, which extends the lazy phase and widens the grokking gap. This is the same mechanism identified by Kumar et al. in the grokking exercise session. $\square$

---

## Problem 6 (Bonus) — Stochastic Implicit Bias (10 min)

### Solution 6

**(a)** Setting $\mathbf{j} = 0$ (thermal equilibrium) with isotropic noise $\Sigma = \sigma^2 I$:

$$0 = -\nabla\mathcal{L}(\theta)\,p^*(\theta) - \frac{\eta\sigma^2}{2}\nabla p^*(\theta)$$

Rearranging:

$$\frac{\nabla p^*}{p^*} = -\frac{2}{\eta\sigma^2}\nabla\mathcal{L}$$

The left-hand side is $\nabla\ln p^*$, so:

$$\nabla\ln p^*(\theta) = -\frac{2}{\eta\sigma^2}\nabla\mathcal{L}(\theta)$$

Integrating both sides:

$$\ln p^*(\theta) = -\frac{2}{\eta\sigma^2}\mathcal{L}(\theta) + \text{const}$$

$$\boxed{p^*(\theta) \propto \exp\!\left(-\frac{2}{\eta\sigma^2}\mathcal{L}(\theta)\right)}$$

This is the Boltzmann distribution with inverse temperature $\beta = 2/(\eta\sigma^2)$. $\square$

**(b)** The effective temperature is $T = \eta\sigma^2/2$.

**Small $\eta$ (low temperature, $\beta \to \infty$):** The distribution concentrates sharply on the global minima of $\mathcal{L}$. SGD converges to the lowest-loss minimizer without exploring.

**Large $\eta$ (high temperature, $\beta \to 0$):** The distribution becomes nearly uniform over parameter space. SGD explores broadly and does not settle into any particular minimum.

**Flat vs. sharp minima:** At moderate temperature, the Boltzmann distribution assigns more probability mass to **broad basins** (flat minima) than to narrow ones. This is because a flat minimum occupies a larger volume of parameter space at any given loss level — the width of the basin acts as an entropic contribution. Flat minima tend to generalize better because small perturbations to the parameters (or slight distribution shift in the data) do not dramatically change the loss. Thus the implicit bias of SGD noise toward flat minima is beneficial for generalization. $\square$

**(c)** When $\Sigma(\theta)$ is anisotropic, the noise strength varies by direction. The Fokker-Planck current becomes:

$$\mathbf{j} = -\nabla\mathcal{L}\,p - \frac{\eta}{2}\nabla\cdot(\Sigma(\theta)\,p)$$

The second term introduces an **Itô drift** $\frac{\eta}{2}\nabla\cdot\Sigma(\theta)$ that depends on the spatial variation of the noise covariance. In directions where $\Sigma$ has large eigenvalues (high noise), the effective diffusion is strong and the system escapes easily from sharp regions of the loss landscape. In directions where $\Sigma$ has small eigenvalues (low noise), the system is more stable and tends to remain there.

This creates a **direction-dependent regularizer**: SGD preferentially pushes parameters out of sharp directions (high gradient variance → large noise eigenvalue → fast escape) while preserving parameters along flat directions (low gradient variance → small noise eigenvalue → stability). This goes beyond what the Boltzmann distribution on $\mathcal{L}$ alone would predict. In general, detailed balance $\mathbf{j} = 0$ may not hold for anisotropic, state-dependent noise, leading to persistent probability currents and non-equilibrium steady states that further modify the implicit bias. $\square$
