# Grokking as the Transition from Lazy to Rich Training Dynamics

## Exercise Session - 1h, pen and paper

**Reference**: Kumar, Bordelon, Gershman & Pehlevan, ICLR 2024


## Problem 1 - The Model and its Summary Statistics (10 min)

Consider a two-layer committee machine with $N$ hidden neurons, input $\boldsymbol{x} \in \mathbb{R}^D$, and fixed readout weights all equal to 1:

$$f(\boldsymbol{w}, \boldsymbol{x}) = \frac{\alpha}{N} \sum_{i=1}^{N} \varphi(w_i \cdot \boldsymbol{x}), \qquad \varphi(h) = h + \frac{\epsilon}{2} h^2$$

The target function is $y(\boldsymbol{x}) = \tfrac{1}{2}(\boldsymbol{\beta}_\star \cdot \boldsymbol{x})^2$, with $\boldsymbol{x} \sim \mathcal{N}(0, \tfrac{1}{D}\boldsymbol{I})$.

**(a)** Define the two summary statistics:
$$\bar{\boldsymbol{w}} = \frac{1}{N}\sum_{i=1}^N \boldsymbol{w}_i, \qquad \boldsymbol{M} = \frac{1}{N}\sum_{i=1}^N \boldsymbol{w}_i \boldsymbol{w}_i^\top$$

Show that the network output can be written as:
$$f(\boldsymbol{x}) = \alpha\, \bar{\boldsymbol{w}} \cdot \boldsymbol{x} \;+\; \frac{\alpha \epsilon}{2}\, \boldsymbol{x}^\top \boldsymbol{M}\, \boldsymbol{x}$$

**(b)** What are the values of $\bar{\boldsymbol{w}}$ and $\boldsymbol{M}$ at random initialisation (large $N$ limit)?

**(c)** For the network to perfectly fit the target on the test distribution, what must $\boldsymbol{M}$ and $\bar{\boldsymbol{w}}$ equal? (*Hint*: match the quadratic and linear parts of $f$ to $y$.)

---

## Problem 3 - The NTK of the Toy Model (10 min)

**(a)** Compute the NTK $K(\boldsymbol{x}, \boldsymbol{x}') = \sum_{i=1}^N \nabla_{w_i} f \cdot \nabla_{w_i} f$ for this model. Show that it can be expressed in terms of $\bar{\boldsymbol{w}}$ and $\boldsymbol{M}$ as:

$$K(\boldsymbol{x}, \boldsymbol{x}') = (\boldsymbol{x}\cdot\boldsymbol{x}') + \epsilon\,(\boldsymbol{x}\cdot\boldsymbol{x}')\,\bar{\boldsymbol{w}}\cdot(\boldsymbol{x}+\boldsymbol{x}') + \epsilon^2\,(\boldsymbol{x}\cdot\boldsymbol{x}')\,\boldsymbol{x}^\top \boldsymbol{M}\,\boldsymbol{x}'$$

(*Hint*: compute $\nabla_{w_i} f$ first, then form the dot product.)

**(b)** At initialisation ($\bar{\boldsymbol{w}}=0$, $\boldsymbol{M}=\boldsymbol{I}$), the kernel simplifies to $K_0(\boldsymbol{x},\boldsymbol{x}') = (\boldsymbol{x}\cdot\boldsymbol{x}') + \epsilon^2(\boldsymbol{x}\cdot\boldsymbol{x}')^2$. This kernel is a sum of two terms. The first term $(\boldsymbol{x}\cdot\boldsymbol{x}')$ is sensitive to *linear* structure in the data, and the second $\epsilon^2(\boldsymbol{x}\cdot\boldsymbol{x}')^2$ to *quadratic* structure. Compute the ratio of the typical magnitude of the quadratic term to the linear term when $x,x'$ are independent draws from $\mathcal{N}(0,\frac{1}{D}I)$. What does this tell you about how well-suited the initial kernel is for learning the (purely quadratic) target, especially when $\epsilon$ is small?

(*Hint*: use $\mathbb{E}[(\boldsymbol{x}\cdot\boldsymbol{x}')^2] = 1/D$ and $\mathbb{E}[(\boldsymbol{x}\cdot\boldsymbol{x}')^4] \sim 1/D^2$ for independent draws.)

**(c)** The target $y(x) = \frac{1}{2}(\beta_\star \cdot x)^2$ is purely quadratic. After feature learning, the network can align $\boldsymbol{M}$ with $\beta_\star\beta_\star^\top$, effectively reducing the problem to learning a single direction in $\mathbb{R}^D$. Argue intuitively that this requires $P \sim D$ samples (think about how many equations you need to identify a vector in $\mathbb{R}^D$). By contrast, a fixed kernel that treats all $\sim D^2$ quadratic directions equally would need far more samples. Why does this gap between $D$ and $D^2$ create an opportunity for grokking?

---

## Problem 4 - Loss Decomposition (15 min)

The test MSE of the model is $\mathcal{L} = \langle (y - f)^2 \rangle$ where $\langle\cdot\rangle$ is the expectation over $x \sim \mathcal{N}(0,\frac{1}{D}I)$.

**Useful identity (Isserlis' theorem for Gaussians).** For $x \sim \mathcal{N}(0, \frac{1}{D}I)$ and any symmetric matrices $A, B$:

$$\langle (x^\top A\, x)(x^\top B\, x) \rangle = \frac{1}{D^2}\left[(\text{Tr}\,A)(\text{Tr}\,B) + 2\,\text{Tr}(AB)\right]$$

In particular, $\langle (x^\top A\, x)^2 \rangle = \frac{1}{D^2}[(\text{Tr}\,A)^2 + 2|A|_F^2]$ where $|A|_F^2 = \text{Tr}(A^\top A)$.

**(a)** Decompose the network output as $f = f_{\text{lin}} + f_{\text{quad}}$ with $f_{\text{lin}} = \alpha\bar{\boldsymbol{w}}\cdot\boldsymbol{x}$ and $f_{\text{quad}} = \frac{\alpha\epsilon}{2}\boldsymbol{x}^\top\boldsymbol{M}\,\boldsymbol{x}$. Note that the target $y = \frac{1}{2}\boldsymbol{x}^\top\boldsymbol{\beta}_\star\boldsymbol{\beta}_\star^\top\boldsymbol{x}$ is purely quadratic. Use the fact that odd moments of isotropic Gaussians vanish (i.e. $\langle x_i\, g(x^\top A x)\rangle = 0$) to argue that the MSE splits as:

$$\mathcal{L} = \langle (y - f_{\text{quad}})^2 \rangle + \langle f_{\text{lin}}^2 \rangle$$

Then compute $\langle f_{\text{lin}}^2 \rangle$ explicitly. This is **Term C** (linear power).

**(b)** Now focus on the quadratic part. Define $A = \boldsymbol{\beta}_\star\boldsymbol{\beta}_\star^\top - \alpha\epsilon\,\boldsymbol{M}$, so that $y - f_{\text{quad}} = \frac{1}{2}\boldsymbol{x}^\top A\,\boldsymbol{x}$. Apply the Isserlis identity above to compute $\langle(y - f_{\text{quad}})^2\rangle = \frac{1}{4}\langle(\boldsymbol{x}^\top A\,\boldsymbol{x})^2\rangle$. Show that the result can be written as the sum of two terms:

$$\underbrace{\frac{1}{4D^2}(\text{Tr}\,A)^2}_{\text{Term A: variance error}} + \underbrace{\frac{1}{2D^2}|A|_F^2}_{\text{Term B: misalignment error}}$$

Write out Term A and Term B explicitly in terms of $|\boldsymbol{\beta}_\star|^2$, $\text{Tr}\,\boldsymbol{M}$, $\alpha$, $\epsilon$, and $|\alpha\epsilon\boldsymbol{M} - \boldsymbol{\beta}_\star\boldsymbol{\beta}_\star^\top|_F$.

**(c)** Interpret each of the three terms (A, B, C) in one sentence.

**(d)** At initialisation ($\bar{\boldsymbol{w}}=0$, $\boldsymbol{M}=\boldsymbol{I}$), which terms are large and which are small? What happens in early training: recalling from Problem 3(b) that the initial kernel is much more sensitive to linear structure than quadratic structure, which component of $f$ does the network use first to reduce train loss, and what does this do to the three terms?

---

## Problem 5 - The Role of $\alpha$: Controlling Laziness (10 min)

**(a)** Consider training with gradient descent on MSE loss $\mathcal{L} = \frac{1}{2}\sum_\mu (y_\mu - f(\boldsymbol{w}, \boldsymbol{x}_\mu))^2$. In the lazy regime, $f$ is linear in $\boldsymbol{w}$, so the gradient descent dynamics become $\dot{\boldsymbol{w}} = -\nabla_w \mathcal{L}$. For a linear model $f_\mu \approx f_\mu^0 + \boldsymbol{J}_\mu \cdot (\boldsymbol{w} - \boldsymbol{w}_0)$ where $\boldsymbol{J}_\mu = \nabla_w f|_{w_0}(\boldsymbol{x}_\mu)$, the converged solution satisfies $f_\mu = y_\mu$ for all training points. Show that the required parameter displacement is:

$$\boldsymbol{w}^* - \boldsymbol{w}_0 = \boldsymbol{J}^\top (K_0)^{-1} (\boldsymbol{y} - \boldsymbol{f}^0)$$

where $(K_0)_{\mu\nu} = \boldsymbol{J}_\mu \cdot \boldsymbol{J}_\nu$ is the NTK Gram matrix. Deduce that $|\boldsymbol{w}^* - \boldsymbol{w}_0|^2 = (\boldsymbol{y}-\boldsymbol{f}^0)^\top K_0^{-1} (\boldsymbol{y}-\boldsymbol{f}^0)$.

(*Hint*: the converged condition is $\boldsymbol{J}_\mu \cdot (\boldsymbol{w}^* - \boldsymbol{w}_0) = y_\mu - f_\mu^0$ for all $\mu$. This is a linear system.)

**(b)** Now recall that the network output is $f = \frac{\alpha}{N}\sum_i \varphi(w_i \cdot x)$. A small parameter change $\delta w$ produces a change in the output $\delta f \sim \alpha\,|\delta w|$ (because the Jacobian $\nabla_w f$ scales as $\alpha$). Since the target is $O(1)$, the network needs $\delta f \sim O(1)$ to fit the data. Use this to argue that $|\delta w| \sim 1/\alpha$. Why does this mean that large $\alpha$ keeps the network in the lazy regime?

**(c)** Feature learning corresponds to the *nonlinear* part of the dynamics - the deviation of $f(\boldsymbol{w})$ from its linearisation. One can show that the rate at which the NTK itself changes (i.e. the rate at which features evolve) scales as $\dot{K}/K \sim |\delta w|^2 \sim 1/\alpha^2$. Meanwhile, the kernel regression solution is reached on a timescale that does not grow with $\alpha$. This creates a separation of timescales.

On the diagram below, fill in the three labels (i), (ii), (iii) and mark the grokking gap. Then sketch a second curve (dashed) showing what happens when $\alpha$ is *increased*.

```
   Loss
    |
    |  ___________
    | /            \___________
    |/                         \
    |                           \_____________
    |
    +------+----------+-----------+-----------> log(time)
          (i)        (ii)       (iii)

    (i)   = ...................
    (ii)  = ...................
    (iii) = ...................

    Grokking gap = from ......... to .........
```

(*Hint*: one curve is train loss, one is test loss. They start together, then diverge, then reconverge.)

**(d)** What happens in the limits $\alpha \to 0$ and $\alpha \to \infty$?

---

## Problem 6 - The Role of $\epsilon$: Initial Kernel Alignment (5 min)

**(a)** From Problem 3, the initial kernel eigenvalue on quadratic functions is $\lambda_{\text{quad}} = 2\epsilon^2/D^2$. Define the *NTK alignment* $\varepsilon = \boldsymbol{y}^\top K_0 \boldsymbol{y} / (\|K_0\|_F \|\boldsymbol{y}\|^2)$. How does this alignment scale with $\epsilon$ in the toy model?

**(b)** Large $\epsilon$ means the initial kernel already places significant power on quadratic functions. Explain why this *reduces* or *eliminates* grokking. Conversely, why does small $\epsilon$ *increase* the grokking delay but lead to *lower* final test loss?

---

## Problem 7 - Synthesis: Why Grokking is a Lazy-to-Rich Transition (5 min)

**(a)** State the three conditions identified by Kumar et al. that are jointly sufficient for grokking.

**(b)** Using the loss decomposition from Problem 4, describe the two-phase dynamics during grokking:
  - Phase 1 (early training): What happens to Terms A, B, C?
  - Phase 2 (late training): What happens to Terms A, B, C?

**(c)** Explain in 2-3 sentences why weight decay is *sufficient but not necessary* for grokking, and how the lazy-to-rich framework subsumes weight-decay-based explanations.
