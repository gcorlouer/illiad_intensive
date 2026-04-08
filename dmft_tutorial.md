# Plan:
## Applications
###  DMFT in DLNs (30mn)
- Write the DMFT equations for DLNs
- Write systems of differential equations for Kernel H and G 
- Set $\Phi^0 = I$ and assume ansatz (74): derive equations (75)
- Using conservation low derive self consistent equation (79)
- Discuss how the values of $\gamma$ affect the lazy and rich regime

### DMFT application to grokking (30mn)
- Consider the quadratic model in Kumar et al. Write the DMFT equations for that model
- Discuss how grokking can be seen as a transition from the lazy to the rich regime using the DMFT equations
- Draw or discuss a qualitative phase diagram of grokking. 
- Discuss the role of explicit regularization (weight decay)

# Tutorial content:
## DMFT application to deep linear networks (30 mn)

### Goal
Reproduce the DLN section of the lecture notes, then use it to explain the lazy-to-rich transition in linear networks.

### 1. Start from the two-layer linear DMFT
Specialize the two-layer DMFT to the linear activation case and MSE loss.

### 2. Pass to the deterministic kernel system
Introduce vector notation and set MSE loss, so

$$
\Delta = y-f, \qquad \dot \Delta = -\dot f.
$$

Then derive the closed ODE system for the kernels \(H\), \(G\), and prediction \(f\):

### 3. Specialize to whitened inputs
Set

$$
\Phi^{(0)}=I.
$$

Simplify the system 

### 4. Introduce the rank-one ansatz
Assume the dynamics stay in the span of the target:

$$
H(t)=\bigl(H_y(t)-1\bigr)\frac{yy^\top}{|y|^2}+I,
\qquad
f(t)=f_y(t)\frac{y}{|y|}.
$$

Substitute the ansatz

### 5. Use the symmetry \(H_y=G\)
Since

$$
\dot H_y(t)=\dot G(t)
\qquad \text{and} \qquad
H_y(0)=G(0)=1,
$$

we get

$$
H_y(t)=G(t).
$$

So the system reduces to

$$
\frac{dH_y(t)}{dt}
=
\frac{2\eta_0\gamma_0^2}{P}(y-f_y)f_y,
$$

$$
\frac{df_y(t)}{dt}
=
\frac{2\eta_0}{P}H_y(t)(y-f_y).
$$

This is Eq. (76). :contentReference[oaicite:4]{index=4}

### 6. Derive the conservation law
Define

$$
L = H_y(t)^2-\gamma_0^2 f_y(t)^2.
$$

Show that it is a conservation law

### 7. Obtain the self-consistent scalar equation
Substitute the conservation law into the \(f_y\) equation to get the self consistent equations


### 8. Interpret the role of \(\gamma_0\)
Now explain lazy versus rich directly from the scalar equation.

### Bonus silent alignment
Read the following [paper](https://arxiv.org/abs/2111.00034) and explain the silent alignment effect with DMFT.

### 9. Bonus Connect to modern lazy/active analyses of linear networks
A useful [work](https://arxiv.org/abs/2405.17580) show that the active/rich regime in linear networks corresponds to dynamics where large singular directions accelerate relative to small ones. Recent work makes this explicit through the approximate self-consistent matrix dynamics

$$
\partial_t A_\theta(t)
\approx
-\eta\sqrt{A_\theta A_\theta^\top+\sigma^4w^2I}\,\nabla C(A_\theta)
-\eta\nabla C(A_\theta)\sqrt{A_\theta^\top A_\theta+\sigma^4w^2I},
$$


### 10. Final tutorial takeaway
The DLN story is:

1. DMFT closes exactly for linear networks.
2. Under $\Phi^{(0)}=I$ and the rank-one ansatz, the dynamics reduce to one scalar equation.
3. A conserved quantity turns the system into a self-consistent ODE.
4. The parameter $\gamma_0$ controls the transition:
   - small $\gamma_0$: lazy / NTK regime,
   - non-small $\gamma_0$: rich / active regime with acceleration.


## DMFT application to grokking (30mn)
We will consider mostly the lecture notes [here](https://pehlevan.seas.harvard.edu/sites/g/files/omnuum6471/files/pehlevan/files/princeton_lecture_notes.pdf) and the [paper](https://arxiv.org/abs/2310.06110) on grokking as lazy to rich transition from Kumar et. al 

- Write the DMFT for the quadratic model considered in Kumar et. al
  - Identify $\gamma_0$ with $\alpha$
- Find the lazy limit
- Introduce the macroscopic variable $\bar{w}$ and $M$ and interpret them
- Understand each term in the loss decomposition
$$
\mathcal L =
\underbrace{\frac14\left(\frac{|\beta_\star|^2}{D}-\frac{\alpha\varepsilon}{D}\operatorname{Tr}M\right)^2}_{\text{variance error}}
+
\underbrace{\frac{1}{2D^2}\left\|\alpha\varepsilon M-\beta_\star\beta_\star^\top\right\|_F^2}_{\text{alignment error}}
+
\underbrace{\frac{\alpha^2}{D}|\bar w|^2}_{\text{linear-mode error}}.
$$
- Discuss why early training is bad for generalization (lazy)
- Discuss the rich phase
- Discuss grokking as a lazy to rich transition: what are the control parameters and how to they affect the transition

## DMFT theory (30mn)
- From the definition of deep net equations write the gradient flow equations
- Write NTK layer recursion in terms of the kernels $\Phi$ and $G$ 
- Write the field $h$ and $z$
- Using the mean field assumption write the DMFT equations (i will introduce the partition function)

# Tutorial solution
## DLN
### DMFT equations
In the lecture notes this gives the closed stochastic system

$$
\{\chi_\mu\}_{\mu\in[P]} \sim \mathcal N(0,\Phi^{(0)}), \qquad \xi \sim \mathcal N(0,1),
$$

$$
h_\mu(t)=\chi_\mu+\frac{\eta_0\gamma_0}{P}\int_0^t ds\, g(s)\sum_\alpha \Phi^{(0)}_{\mu\alpha}\Delta_\alpha(s),
$$

$$
g(t)=\xi+\frac{\eta_0\gamma_0}{P}\int_0^t ds \sum_\alpha h_\alpha(s)\Delta_\alpha(s),
$$

$$
H_{\mu\alpha}(t,s)=\langle h_\mu(t)h_\alpha(s)\rangle, \qquad
G(t,s)=\langle g(t)g(s)\rangle,
$$

$$
\frac{df_\mu}{dt}
=
\frac{\eta_0}{P}\sum_\alpha
\bigl[H_{\mu\alpha}(t,t)+G(t,t)\Phi^{(0)}_{\mu\alpha}\bigr]\Delta_\alpha.
$$

This is Eq. (66) in the notes. :contentReference[oaicite:0]{index=0}

### Kernels


$$
\frac{dH(t)}{dt}
=
\frac{\eta_0\gamma_0}{P}\,\Phi^{(0)}(y-f)f^\top
+
\frac{\eta_0\gamma_0}{P}\,f(y-f)^\top\Phi^{(0)},
$$

$$
\frac{dG(t)}{dt}
=
\frac{2\eta_0\gamma_0}{P}\,f\cdot (y-f),
$$

$$
\frac{df(t)}{dt}
=
\frac{\eta_0}{P}\bigl[H(t)+G(t)\Phi^{(0)}\bigr](y-f),
$$

with

$$
H(0)=\Phi^{(0)}, \qquad G(0)=1, \qquad f(0)=0.
$$

This is Eq. (72) in the notes after using the identity
$$
\langle g(t)h(t)\rangle=\gamma_0 f(t).
$$
:contentReference[oaicite:1]{index=1}
### Set initial conditions
to

$$
\frac{dH(t)}{dt}
=
\frac{\eta_0\gamma_0^2}{P}(y-f)f^\top
+
\frac{\eta_0\gamma_0^2}{P}f(y-f)^\top,
$$

$$
\frac{dG(t)}{dt}
=
\frac{2\eta_0\gamma_0^2}{P}f\cdot (y-f),
$$

$$
\frac{df(t)}{dt}
=
\frac{\eta_0}{P}[H(t)+G(t)I](y-f),
$$

with

$$
H(0)=I, \qquad G(0)=1, \qquad f(0)=0.
$$

This is Eq. (73). :contentReference[oaicite:2]{index=2}
### Ansatz substitution


$$
\frac{dH_y(t)}{dt}
=
\frac{2\eta_0\gamma_0^2}{P}(y-f_y)f_y,
$$

$$
\frac{dG(t)}{dt}
=
\frac{2\eta_0\gamma_0^2}{P}(y-f_y)f_y,
$$

$$
\frac{df_y(t)}{dt}
=
\frac{\eta_0}{P}\bigl(H_y(t)+G(t)\bigr)(y-f_y),
$$

with

$$
H_y(0)=1,\qquad G(0)=1,\qquad f_y(0)=0.
$$

This is Eq. (75). :contentReference[oaicite:3]{index=3}
### Self consistent equation

$$
\frac{df_y(t)}{dt}
=
\frac{2\eta_0}{P}\sqrt{1+\gamma_0^2 f_y(t)^2}\,(y-f_y),
\qquad
f_y(0)=0.
$$

This is Eq. (79). :contentReference[oaicite:6]{index=6}

### Lazy and rich regimes

#### Lazy regime: \(\gamma_0 \to 0\)
If \(\gamma_0\) is small, then

$$
\sqrt{1+\gamma_0^2 f_y(t)^2}\approx 1,
$$

so

$$
\frac{df_y(t)}{dt}
\approx
\frac{2\eta_0}{P}(y-f_y),
$$

which is exactly the NTK/lazy linearized dynamics. This is Eq. (80) in the notes. :contentReference[oaicite:7]{index=7}

#### Rich regime: \(\gamma_0 = O(1)\) or larger
If \(\gamma_0\) is not small, then the prefactor

$$
\sqrt{1+\gamma_0^2 f_y(t)^2}
$$

grows with \(f_y\). So the effective learning rate is state-dependent and increases as learning progresses. This is the nonlinear feature-learning effect captured by the DMFT description. It is absent in the lazy limit and is the linear-network analogue of active/rich training. :contentReference[oaicite:8]{index=8}

## DMFT application to grokking

### Goal
Show that, in the Kumar toy model, grokking can be interpreted as a **transition from lazy to rich learning** within the two-layer DMFT framework.

### Step 1: Write the two-layer DMFT equations
For a two-layer network trained with MSE, write the representative-neuron DMFT:

$$
\{\chi_\mu\}_{\mu=1}^P \sim \mathcal N(0,\Phi^{(0)}), \qquad \xi \sim \mathcal N(0,1),
$$

$$
h_\mu(t)=\chi_\mu+\frac{\eta_0\gamma_0}{P}\int_0^t ds \sum_{\alpha=1}^P z(s)\,\phi'(h_\alpha(s))\,\Phi^{(0)}_{\mu\alpha}\,\Delta_\alpha(s),
$$

$$
z(t)=\xi+\frac{\eta_0\gamma_0}{P}\int_0^t ds \sum_{\alpha=1}^P \phi(h_\alpha(s))\,\Delta_\alpha(s),
$$

$$
\Phi_{\mu\alpha}(t,s)=\Big\langle \phi(h_\mu(t))\phi(h_\alpha(s))\Big\rangle,
$$

$$
G_{\mu\alpha}(t,s)=\Big\langle z(t)z(s)\phi'(h_\mu(t))\phi'(h_\alpha(s))\Big\rangle,
$$

$$
\partial_t f_\mu(t)=\frac{\eta_0}{P}\sum_{\alpha=1}^P
\Big[\Phi_{\mu\alpha}(t,t)+G_{\mu\alpha}(t,t)\Phi^{(0)}_{\mu\alpha}\Big]\Delta_\alpha(t),
$$

with

$$
\Delta_\alpha(t)=y_\alpha-f_\alpha(t).
$$

### Step 2: Specialize to the Kumar toy model
Use the Kumar activation and target:

$$
\phi(h)=h+\frac{\varepsilon}{2}h^2, \qquad \phi'(h)=1+\varepsilon h,
$$

$$
y(x)=\frac12(\beta_\star\cdot x)^2.
$$

Then the DMFT becomes

$$
h_\mu(t)=\chi_\mu+\frac{\eta_0\gamma_0}{P}\int_0^t ds \sum_\alpha
z(s)\bigl(1+\varepsilon h_\alpha(s)\bigr)\Phi^{(0)}_{\mu\alpha}\Delta_\alpha(s),
$$

$$
z(t)=\xi+\frac{\eta_0\gamma_0}{P}\int_0^t ds \sum_\alpha
\left(h_\alpha(s)+\frac{\varepsilon}{2}h_\alpha(s)^2\right)\Delta_\alpha(s).
$$

At this point identify the Kumar laziness parameter \(\alpha\) with the inverse DMFT feature-learning parameter:

$$
\gamma_0=\frac1\alpha.
$$

So:
- large \(\alpha\) = small \(\gamma_0\) = lazy regime,
- small \(\alpha\) = larger \(\gamma_0\) = richer feature learning.

### Step 3: Explain the lazy limit
In the limit

$$
\gamma_0 \to 0 \qquad (\text{equivalently } \alpha \to \infty),
$$

the memory terms in the DMFT are suppressed, so

$$
h_\mu(t)\approx \chi_\mu, \qquad z(t)\approx \xi,
$$

and therefore the kernels

$$
\Phi,\; G
$$

remain essentially frozen at their initial values.

Then the prediction dynamics reduce to static-kernel learning:

$$
\partial_t f(t)\approx \frac{\eta_0}{P}K_0\,(y-f(t)).
$$

Interpretation:
- the network behaves like its linearization,
- it first fits the training set using the **initial NTK**,
- no substantial feature learning has happened yet.

### Step 4: Introduce the macroscopic observables
For the Kumar committee machine define

$$
\bar w=\frac1N\sum_{i=1}^N w_i,
\qquad
M=\frac1N\sum_{i=1}^N w_i w_i^\top.
$$

Then the network function is

$$
f(x)=\alpha\,\bar w\cdot x+\frac{\alpha\varepsilon}{2}x^\top M x.
$$

Interpretation:
- \(\bar w\) controls the unwanted linear component,
- \(M\) controls the learned quadratic feature,
- feature learning means evolving \(M\), not just fitting coefficients in a fixed kernel.

### Step 5: Write the loss decomposition
Present the Kumar decomposition of the test loss:

$$
\mathcal L =
\underbrace{\frac14\left(\frac{|\beta_\star|^2}{D}-\frac{\alpha\varepsilon}{D}\operatorname{Tr}M\right)^2}_{\text{variance error}}
+
\underbrace{\frac{1}{2D^2}\left\|\alpha\varepsilon M-\beta_\star\beta_\star^\top\right\|_F^2}_{\text{alignment error}}
+
\underbrace{\frac{\alpha^2}{D}|\bar w|^2}_{\text{linear-mode error}}.
$$

Explain each term:
- **variance error**: overall scale of the learned quadratic component,
- **alignment error**: whether \(M\) points in the teacher direction \(\beta_\star\beta_\star^\top\),
- **linear-mode error**: penalty from spurious linear features.

### Step 6: Explain why early training is bad for generalization
In the lazy phase, the model follows the initial kernel bias.

For this task, the initial NTK is poorly aligned with the quadratic target when \(\varepsilon\) is small:
- the kernel is biased toward linear modes,
- the target is quadratic,
- train loss can decrease before the right quadratic feature is learned.

Interpretation:
- the network first behaves like a bad kernel regressor,
- it can partially fit the data without yet learning the teacher feature,
- this creates the train/test delay characteristic of grokking.

### Step 7: Explain the rich phase
At later time, the self-consistent kernels \(\Phi\) and \(G\) evolve.

In macroscopic terms:
- \(M(t)\) starts aligning with

$$
\beta_\star\beta_\star^\top,
$$

- \(\bar w(t)\to 0\).

This means:
- the linear-mode error decreases,
- the alignment error decreases,
- the network learns the correct feature representation,
- test loss finally drops.

Interpretation:
the network leaves the lazy regime and enters a **rich feature-learning regime**.

### Step 8: State the grokking mechanism
Grokking is the delayed drop in test loss caused by a **lazy-to-rich crossover**:

1. **Early time (lazy):**
   - kernels are approximately frozen,
   - dynamics are close to NTK regression,
   - train loss falls first,
   - generalization remains poor.

2. **Late time (rich):**
   - kernels evolve self-consistently,
   - \(M\) aligns with the teacher feature,
   - \(\bar w\) is suppressed,
   - test loss falls later.

### Step 9: Discuss the control parameters
Use the DMFT view to explain the qualitative phase diagram.

#### Effect of \(\alpha\)

$$
\gamma_0=\frac1\alpha
$$

so:
- larger \(\alpha\) makes training more lazy,
- feature learning starts later,
- grokking delay becomes stronger.

#### Effect of \(\varepsilon\)
- smaller \(\varepsilon\) means worse initial kernel-target alignment,
- the initial NTK does worse,
- more feature learning is needed,
- grokking becomes stronger.

#### Effect of dataset size \(P\)
There is a **goldilocks regime**:
- too little data: no eventual generalization,
- too much data: train and test track each other,
- intermediate data: delayed generalization is visible.

### Final takeaway
The Kumar toy model can be read as a concrete two-layer DMFT example in which grokking is not mysterious: it is the visible consequence of a transition from

- **lazy training** with nearly frozen kernels

to

- **rich training** with evolving kernels and learned features.

So the conceptual message is:

> grokking = delayed generalization produced by late-time feature learning after an initial NTK-like phase.


