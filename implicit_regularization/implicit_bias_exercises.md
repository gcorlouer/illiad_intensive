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

where $W = W_L \cdots W_1 \in \mathbb{R}^{d_L \times d_0}$ is the end-to-end (or "student") matrix. We train on a dataset $\{(x_\mu, y_\mu)\}_{\mu=1}^N$ with the squared loss:

$$\mathcal{L}(\theta) = \frac{1}{2N} \sum_{\mu=1}^{N} \|y_\mu - W_L \cdots W_1 \, x_\mu\|^2$$

In the population limit with whitened inputs $\Sigma_X = I$, this becomes:

$$\mathcal{L}(W) = \frac{1}{2}\|M - W\|_F^2$$

where $M = \Sigma_{YX} \Sigma_X^{-1}$ is the "teacher" matrix (the OLS solution) with SVD $M = U \,\text{diag}(s_1, \ldots, s_r, 0, \ldots, 0)\, V^\top$, and $s_1 \geq s_2 \geq \cdots \geq s_r > 0$.

---

## Problem 1 — Loss Landscape Geometry (15 min)

This problem explores the critical point structure of deep linear networks, following Achour et al. (2024).

**(a)** Consider the **diagonal** case: $d_0 = d_1 = \cdots = d_L = d$, and the teacher is diagonal, $M = \text{diag}(s_1, \ldots, s_d)$, with $s_1 > s_2 > \cdots > s_d > 0$. Restrict attention to diagonal weight matrices $W_l = \text{diag}(w_l^{(1)}, \ldots, w_l^{(d)})$. Show that the loss decomposes into $d$ independent scalar problems:

$$\mathcal{L} = \frac{1}{2}\sum_{\alpha=1}^d \left(s_\alpha - \prod_{l=1}^L w_l^{(\alpha)}\right)^2$$

**(b)** For a single scalar mode with target $s > 0$, find all first-order critical points of $\ell(w_1, w_2) = \frac{1}{2}(s - w_1 w_2)^2$ at depth $L = 2$. Show that the critical points are:
- The global minimum manifold: $w_1 w_2 = s$
- The origin: $w_1 = w_2 = 0$

Classify the origin as a saddle point by computing the Hessian $H$ of $\ell$ at $(0,0)$ and showing it has both positive and negative eigenvalues.

*Hint*: The gradient equations are $\frac{\partial \ell}{\partial w_1} = -(s - w_1 w_2)\, w_2 = 0$ and $\frac{\partial \ell}{\partial w_2} = -(s - w_1 w_2)\, w_1 = 0$.

**(c)** Now consider the full (non-diagonal) DLN with $d_l \geq d_L$ for all $l$. Achour et al. show that every first-order critical point $\theta = (W_1, \ldots, W_L)$ satisfies: there exists a subset $S \subseteq \{1, \ldots, d\}$ such that

$$W = W_L \cdots W_1 = P_S\, M$$

where $P_S = U_S U_S^\top$ is the orthogonal projector onto the span of the left singular vectors $\{u_\alpha\}_{\alpha \in S}$ of $M$. For the diagonal case $M = \text{diag}(s_1, \ldots, s_d)$, write the value of the loss at the critical point corresponding to $S = \{1, \ldots, r\}$ with $r < d$.

**(d)** The group $GL_h := GL_{d_1} \times \cdots \times GL_{d_{L-1}}$ acts on the weights by:

$$(W_1, \ldots, W_L) \mapsto (g_1 W_1,\; g_2 W_2 g_1^{-1},\; \ldots,\; W_L g_{L-1}^{-1})$$

Verify that the student map $\mu(\theta) = W_L \cdots W_1$ is invariant under this action: $\mu(g \cdot \theta) = \mu(\theta)$. What does this imply about the dimension of the set of global minima in parameter space?

---

## Problem 2 — Gradient Flow and Conserved Quantities (20 min)

We now study gradient flow: $\dot{\theta}(t) = -\nabla_\theta \mathcal{L}(\theta(t))$, the continuous-time limit of gradient descent with infinitesimal learning rate.

**(a)** For a two-layer DLN ($L=2$) with loss $\mathcal{L} = \frac{1}{2}\|M - W_2 W_1\|_F^2$ and whitened inputs, derive the gradient flow equations for each layer. Show that:

$$\dot{W}_1 = W_2^\top(M - W_2 W_1), \qquad \dot{W}_2 = (M - W_2 W_1) W_1^\top$$

*Hint*: Use the matrix identity $\frac{\partial}{\partial A}\text{Tr}(B A C) = B^\top C^\top$ and the chain rule. Recall $\|X\|_F^2 = \text{Tr}(X^\top X)$. If the matrix calculus is too involved, you may first derive the result for diagonal weight matrices and then state it for the general case.

**(b)** Define the **balancedness matrix**:

$$G := W_2^\top W_2 - W_1 W_1^\top$$

Show that $G$ is conserved under the gradient flow, i.e. $\dot{G} = 0$.

*Hint*: Compute $\dot{G} = \dot{W}_2^\top W_2 + W_2^\top \dot{W}_2 - \dot{W}_1 W_1^\top - W_1 \dot{W}_1^\top$, substitute the gradient flow equations from (a), and verify that the terms cancel pairwise.

**(c)** From now on, assume **balanced initialization**: $G(0) = 0$, which by part (b) means $W_2^\top W_2 = W_1 W_1^\top$ for all time. We want to derive the gradient flow in function space (the ODE for the student $W = W_2 W_1$). This requires several steps.

**(c.i)** Compute $\dot{W} := \frac{d}{dt}(W_2 W_1)$ using the gradient flow equations from (a). Show that:

$$\dot{W} = (M - W)\, W_1^\top W_1 + W_2 W_2^\top\, (M - W)$$

**(c.ii)** We now need to express $W_1^\top W_1$ and $W_2 W_2^\top$ in terms of $W = W_2 W_1$. For simplicity, assume that the weight matrices $W_l$ are diagonal. Show that:

$$W_1^\top W_1 = (W^\top W)^{1/2}$$

**(c.iii)** Similarly, show that $W_2 W_2^\top = (W W^\top)^{1/2}$.

**(c.iv)** Substitute the results of (c.ii) and (c.iii) into (c.i) to obtain:

$$\dot{W} = (W W^\top)^{1/2}(M - W) + (M - W)(W^\top W)^{1/2}$$

**(d)** Define the NTK operator for $L = 2$ as:

$$K[F] := (WW^\top)^{1/2}\, F + F\, (W^\top W)^{1/2}$$

Show that the gradient flow from (c) can be written as $\dot{W} = K[M - W]$.

It turns out that this result generalizes to depth $L$ on the balanced manifold and to non-diagonal weight matrices:

$$\dot{W} = \sum_{k=1}^{L} (WW^\top)^{\frac{L-k}{L}} (M - W) (W^\top W)^{\frac{k-1}{L}}$$

---

## Problem 3 — Rich Regime: Incremental Learning (25 min)

This is the core problem. We derive the exact solution of the rich regime dynamics directly from the self-consistent equation of Problem 2, emphasizing the NTK perspective.

**(a)** **Alignment assumption.** Start from the balanced gradient flow equation derived in Problem 2(c):

$$\dot{W} = (WW^\top)^{1/2}(M - W) + (M - W)(W^\top W)^{1/2}$$

We work in the **rich regime** (small initialization) with balanced weights. Assume that the student is **aligned** to the teacher: the singular vectors of $W(t)$ coincide with those of $M$ at all times. Concretely, write:

$$M = U\,\text{diag}(s_1, \ldots, s_d)\,V^\top, \qquad W(t) = U\,\text{diag}(w_1(t), \ldots, w_d(t))\,V^\top$$

where $U, V$ are the left/right singular vectors of $M$, and $w_\alpha(t) \geq 0$ are the evolving singular values of $W$.

**(a.i)** Under this alignment assumption, show that the NTK operator is aligned to the task, i.e. $K[F]$ preserves the SVD basis. In particular, show that:

$$(WW^\top)^{1/2} = U\,\text{diag}(w_1, \ldots, w_d)\,U^\top, \qquad (W^\top W)^{1/2} = V\,\text{diag}(w_1, \ldots, w_d)\,V^\top$$

and that the residual is $M - W = U\,\text{diag}(s_1 - w_1, \ldots, s_d - w_d)\,V^\top$.

*Hint*: Recall that for $A = U\,\text{diag}(\sigma_i)\,V^\top$, we have $AA^\top = U\,\text{diag}(\sigma_i^2)\,U^\top$, and therefore $(AA^\top)^{1/2} = U\,\text{diag}(|\sigma_i|)\,U^\top$.

**(a.ii)** Substitute into the self-consistent equation. Show that the matrix equation decouples into $d$ independent scalar ODEs:

$$\dot{w}_\alpha = 2\,w_\alpha\,(s_\alpha - w_\alpha), \qquad \alpha = 1, \ldots, d$$

*Hint*: Compute the product $(WW^\top)^{1/2}(M - W)$ in the SVD basis. It is diagonal with entries $w_\alpha(s_\alpha - w_\alpha)$. The second term contributes identically, giving the factor of 2.

**(b)** **Timescale of learning.** The ODE $\dot{w} = 2w(s - w)$ is a logistic equation whose solution is:

$$w(t) = \frac{s}{1 + \left(\frac{s}{w_0} - 1\right)e^{-2st}}$$

where $w_0 = w(0)$. Compute the time $t_\alpha$ it takes for mode $\alpha$ to travel from initial strength $w_0$ to a final strength $w_f$. Show that:

$$t_\alpha = \frac{1}{2 s_\alpha} \ln\left(\frac{w_f\,(s_\alpha - w_0)}{w_0\,(s_\alpha - w_f)}\right)$$

Deduce that modes with larger singular values $s_\alpha$ are learned faster. This is a **separation of timescales**: the network learns features in decreasing order of their singular value strength.

**(c)** **Incremental learning.** Consider a teacher with $r$ nonzero singular values and a small uniform initialization $w_\alpha(0) = w_0 \ll s_r$ for all $\alpha$. Discuss:

- What is the approximate rank of $W(t)$ at early times?
- How does the rank evolve at intermediate times?
- What happens in the limit $t \to \infty$?

What does this tell us about the sequence of critical points visited by the gradient flow?

---

## Problem 4 — Lazy Regime (15 min)

We now show that large initialization freezes the NTK and eliminates the timescale separation found in Problem 3. For clarity, we work with the **diagonal scalar** model from Problem 1(a), so each mode $\alpha$ is independent. This avoids matrix algebra and isolates the essential mechanism.

**(a)** **Setup.** Consider a single mode: a depth-2 diagonal DLN with scalar weights $a, b$ learning a target $s > 0$. From Problems 2–3, the balanced gradient flow for $w = ab$ is:

$$\dot{w} = 2w(s - w)$$

The factor $2w$ is the (scalar) NTK — it is **state-dependent**: the effective learning rate depends on the current value of $w$.

Now initialize at a **large** value $w_0 \gg s > 0$. Show that in the early phase of training, while $w$ has not changed much from $w_0$, the ODE becomes approximately:

$$\dot{w} \approx 2w_0(s - w)$$

**(b)** **Frozen-NTK solution.** Solve the linearized ODE from (a). Show that:

$$w(t) = s + (w_0 - s)\,e^{-2w_0 t}$$

What is the learning timescale?

**(c)** **No timescale separation.** Now restore the mode index $\alpha$. In the lazy regime, each mode satisfies $\dot{w}_\alpha \approx 2w_0(s_\alpha - w_\alpha)$ with the **same** rate $2w_0$ for all $\alpha$. Compare the lazy learning time $t_\alpha^{\text{lazy}}$ with the rich-regime timescale from Problem 3(b): $t_\alpha^{\text{rich}} \propto 1/s_\alpha$.

Discuss the difference in **rank bias** between the lazy and rich regimes.

---

## Problem 5 — Mixed Dynamics: Unifying Lazy and Rich (15 min)

In practice, networks are neither purely lazy nor purely rich. Tu, Aranguri & Jacot (2024) provide a unified description. We develop the key ideas using the diagonal scalar model.

**(a)** **The interpolating ODE.** In Problems 3 and 4, we studied the same ODE $\dot{w}_\alpha = 2w_\alpha(s_\alpha - w_\alpha)$ in two limits: small $w_0$ (rich) and large $w_0$ (lazy). A more general model for a depth-2 DLN with balanced initialization at scale $\sigma$ and width $n$ replaces the scalar NTK $2w_\alpha$ by:

$$\dot{w}_\alpha = 2\sqrt{w_\alpha^2 + \tau^2}\;(s_\alpha - w_\alpha)$$

where $\tau := \sigma^2 \sqrt{n}$ is a **threshold** parameter that depends on the initialization scale and width.

Verify that this ODE reproduces the two known regimes:
- **Rich limit** ($\tau \to 0$): recover $\dot{w}_\alpha = 2|w_\alpha|(s_\alpha - w_\alpha)$.
- **Lazy limit** ($\tau \to \infty$ with $w_\alpha$ bounded): recover $\dot{w}_\alpha \approx 2\tau(s_\alpha - w_\alpha)$.

**(b)** **Two-phase dynamics.** At initialization, all modes start at $w_\alpha(0) \sim \sigma^2 \ll \tau$ (since $\sqrt{n} \gg 1$). Observe that:
- When $|w_\alpha| \ll \tau$, the effective learning rate is $\approx 2\tau$, the same for all modes (lazy behavior).
- When $|w_\alpha| \gg \tau$, the effective learning rate is $\approx 2|w_\alpha|$, which is mode-dependent (rich behavior).

Describe the resulting two-phase dynamics in words: what happens first, and what happens later?

**(c)** **Mode-by-mode transition.** Not all modes cross the threshold $\tau$ at the same time which modes cross it first? Explain why this means that a single network can simultaneously have some modes in the rich regime and others still in the lazy regime.

---

## Problem 6 (Bonus) — Stochastic Implicit Bias (10 min)

SGD introduces noise from mini-batching. A continuous model is the Langevin SDE:

$$d\theta_t = -\nabla \mathcal{L}(\theta_t)\,dt + \sqrt{\eta\,\Sigma(\theta_t)}\,dB_t$$

where $\Sigma(\theta)$ is the covariance of the stochastic gradient noise and $\eta$ is the learning rate.

**(a)** The probability density $p(\theta, t)$ of the parameters evolves according to the Fokker–Planck equation:

$$\partial_t p = -\nabla \cdot \mathbf{j}, \qquad \mathbf{j} = -\nabla \mathcal{L}(\theta)\,p(\theta) - \frac{\eta}{2}\nabla \cdot \left(\Sigma(\theta)\,p(\theta)\right)$$

where $\mathbf{j}$ is the probability current. Assume: (i) stationarity $\partial_t p^* = 0$, (ii) thermal equilibrium $\mathbf{j} = 0$, and (iii) isotropic noise $\Sigma = \sigma^2 I$. Show that the equilibrium distribution is the Boltzmann distribution:

$$p^*(\theta) \propto \exp\left(-\frac{2}{\eta \sigma^2}\mathcal{L}(\theta)\right)$$

*Hint*: Setting $\mathbf{j} = 0$ with $\Sigma = \sigma^2 I$ gives $\nabla \mathcal{L}\, p + \frac{\eta\sigma^2}{2}\nabla p = 0$. This is a first-order ODE for $p$ in terms of $\mathcal{L}$. Try the ansatz $p \propto e^{-\beta \mathcal{L}}$ and solve for $\beta$.

**(b)** The ratio $\frac{2}{\eta \sigma^2}$ plays the role of an inverse temperature $\beta$. Interpret what happens to the equilibrium distribution when:
- $\eta$ is very small (low temperature)
- $\eta$ is very large (high temperature)

Which regime favors flatter minima, and why might this be beneficial for generalization?

**(c)** In practice, SGD noise is **not** isotropic: $\Sigma(\theta)$ depends on both the loss landscape and the data. Without solving anything, explain qualitatively why anisotropic noise can introduce an implicit bias that goes beyond what the loss function $\mathcal{L}$ alone would select. Specifically, why might SGD preferentially escape sharp directions of the loss while remaining stable along flat directions?