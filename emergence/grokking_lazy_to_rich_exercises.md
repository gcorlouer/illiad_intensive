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

**(b)** What are the values of $\bar{\boldsymbol{w}}$ and $\boldsymbol{M}$ at random initialisation in the large $N$ limit?

**(c)** For the network to perfectly fit the target on the test distribution, what must $\boldsymbol{M}$ and $\bar{\boldsymbol{w}}$ equal? (*Hint*: match the quadratic and linear parts of $f$ to $y$.)

---

## Problem 2 - The NTK of the Toy Model (10 min)

**(a)** Compute the NTK $K(\boldsymbol{x}, \boldsymbol{x}') = \sum_{i=1}^N \nabla_{w_i} f(\boldsymbol{x}) \cdot \nabla_{w_i} f(\boldsymbol{x}')$ for this model. Show that, up to the overall prefactor $\alpha^2/N$ coming from this parametrisation, it can be expressed in terms of $\bar{\boldsymbol{w}}$ and $\boldsymbol{M}$ as:

$$K(\boldsymbol{x}, \boldsymbol{x}') \propto (\boldsymbol{x}\cdot\boldsymbol{x}') + \epsilon\,(\boldsymbol{x}\cdot\boldsymbol{x}')\,\bar{\boldsymbol{w}}\cdot(\boldsymbol{x}+\boldsymbol{x}') + \epsilon^2\,(\boldsymbol{x}\cdot\boldsymbol{x}')\,\boldsymbol{x}^\top \boldsymbol{M}\,\boldsymbol{x}'$$

**(b)** At initialisation ($\bar{\boldsymbol{w}}=0$, $\boldsymbol{M}=\boldsymbol{I}$), the kernel simplifies to
$$K_0(\boldsymbol{x},\boldsymbol{x}') \propto (\boldsymbol{x}\cdot\boldsymbol{x}') + \epsilon^2(\boldsymbol{x}\cdot\boldsymbol{x}')^2.$$
This kernel is a sum of two terms. The first term $(\boldsymbol{x}\cdot\boldsymbol{x}')$ is sensitive to *linear* structure in the data, and the second $\epsilon^2(\boldsymbol{x}\cdot\boldsymbol{x}')^2$ to *quadratic* structure. Compute the ratio of the typical magnitude of the quadratic term to the linear term when $x,x'$ are independent draws from $\mathcal{N}(0,\frac{1}{D}I)$. What does this tell you about how well-suited the initial kernel is for learning the (purely quadratic) target, especially when $\epsilon$ is small?

(*Hint*: use $\mathbb{E}[(\boldsymbol{x}\cdot\boldsymbol{x}')^2]$ for typical magnitude)

**(c)** The target $y(x) = \frac{1}{2}(\beta_\star \cdot x)^2$ is purely quadratic. After feature learning, the network can align $\boldsymbol{M}$ with $\beta_\star\beta_\star^\top$, effectively reducing the problem to learning a single direction in $\mathbb{R}^D$. Argue intuitively that this requires $P \sim D$ samples. By contrast, a fixed kernel that treats all quadratic directions equally would need far more samples. Why does this create an opportunity for grokking?

---

## Problem 3 - Loss Decomposition (15 min)

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

**(d)** At initialisation ($\bar{\boldsymbol{w}}=0$, $\boldsymbol{M}=\boldsymbol{I}$), which terms are large and which are small? What happens in early training: recalling from Problem 2(b) that the initial kernel is much more sensitive to linear structure than quadratic structure, which component of $f$ does the network use first to reduce train loss, and what does this do to the three terms?

---

## Problem 4 — The Role of $\alpha$: Controlling Laziness (10 min)

Recall the model $f(\boldsymbol{w}, \boldsymbol{x}) = \frac{\alpha}{N}\sum_{i=1}^N \varphi(\boldsymbol{w}_i \cdot \boldsymbol{x})$ with target $y(\boldsymbol{x}) = \frac{1}{2}(\boldsymbol{\beta}_\star \cdot \boldsymbol{x})^2$. The parameter $\alpha$ multiplies the network output but does **not** appear in the target. We train with gradient flow on the MSE loss $\mathcal{L} = \frac{1}{2}\sum_\mu(y_\mu - f_\mu)^2$.

**Important note.** In the paper, large-$\alpha$ experiments are implemented using the centered predictor
$$\tilde f(\boldsymbol{w},\boldsymbol{x}) = f(\boldsymbol{w},\boldsymbol{x}) - f(\boldsymbol{w}_0,\boldsymbol{x})$$
and the learning rate is rescaled as $\eta \propto \alpha^{-2}$. Here we give the basic scaling intuition in raw gradient-flow time.

**(a)** **Scaling of the NTK.** Compute the Jacobian $\nabla_{\boldsymbol{w}_i} f$ and show that it scales as $\alpha$. Deduce that the NTK Gram matrix $(K_0)_{\mu\nu} = \sum_i \nabla_{\boldsymbol{w}_i}f(\boldsymbol{x}_\mu) \cdot \nabla_{\boldsymbol{w}_i}f(\boldsymbol{x}_\nu)$ scales as $\alpha^2$.

**(b)** **Kernel regression timescale.** In the linearized (lazy) regime, the prediction dynamics are $\dot{f}_\mu = -\sum_\nu K_{\mu\nu}(f_\nu - y_\nu)$. Argue that the timescale for the training predictions to converge to the targets is:

$$t_{\text{kernel}} \sim \frac{1}{\alpha^2}$$

Note that the targets $y_\mu = \frac{1}{2}(\boldsymbol{\beta}_\star \cdot \boldsymbol{x}_\mu)^2$ are $O(1)$ — they are independent of $\alpha$.

**(c)** **Parameter displacement.** The linearized model is $f_\mu \approx f_\mu^0 + \sum_i \nabla_{\boldsymbol{w}_i}f|_{\boldsymbol{w}_0} \cdot \delta\boldsymbol{w}_i$, where $\delta\boldsymbol{w}_i = \boldsymbol{w}_i - \boldsymbol{w}_i(0)$. To fit the training data, the network needs $\delta f_\mu \sim O(1)$. Using the fact that $\nabla_{\boldsymbol{w}_i}f \sim \alpha$, show that the required parameter displacement satisfies:

$$|\delta \boldsymbol{w}| \sim \frac{1}{\alpha}$$

Explain why this means large $\alpha$ keeps the network in the lazy regime.

**(d)** **Feature learning timescale.** The NTK depends on the parameters through $\varphi'(\boldsymbol{w}_i \cdot \boldsymbol{x})$. Show that the fractional change in the NTK during kernel regression is:

$$\frac{\Delta K}{K_0} \sim \frac{|\delta \boldsymbol{w}|}{|\boldsymbol{w}_0|} \sim \frac{1}{\alpha}$$

(using $|\boldsymbol{w}_0| \sim O(1)$ at standard initialization). Deduce that for large $\alpha$, the NTK has barely rotated by the time train loss reaches zero: memorization happens **before** any significant feature learning.

For the NTK to rotate by $O(1)$ (which is what generalization requires), the parameters must change by $O(1)$. This happens on a much longer timescale than the initial kernel fit. A standard scaling summary is:

$$t_{\text{feature}} \sim \alpha^2$$

**(e)** **The grokking gap.** Compute the ratio $t_{\text{feature}}/t_{\text{kernel}}$ as a function of $\alpha$. Describe what happens in the two limits $\alpha \to 0$ and $\alpha \to \infty$.

---

## Problem 5 - The Role of $\epsilon$: Initial Kernel Alignment (5 min)

Large $\epsilon$ means the initial kernel already places significant power on quadratic functions. Explain why this *reduces* or *eliminates* grokking. Conversely, why does small $\epsilon$ *increase* the grokking delay but lead to *lower* final test loss?

---

## Problem 6 - Synthesis: Why Grokking is a Lazy-to-Rich Transition (5 min)

**(a)** State the three conditions identified by Kumar et al. that are jointly sufficient for grokking.

**(b)** Using the loss decomposition from Problem 3, describe the two-phase dynamics during grokking:
  - Phase 1 (early training): What happens to Terms A, B, C?
  - Phase 2 (late training): What happens to Terms A, B, C?

**(c)** Explain in 2-3 sentences why weight decay can help produce grokking but is not necessary, and how the lazy-to-rich framework subsumes weight-decay-based explanations.