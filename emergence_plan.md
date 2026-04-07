# Content

- What is emergence (1 slide)
  - More is different
  - Emergent of skills
- Phase transitions from memorization to generalization as a theoretical approach to predict emergence (1 slide)
  - Intuitively: rapid change in qualitative behaviour of something
  - Loss is blind to change
  - The "mirage" debate: metric choice vs real discontinuities
- Specific Examples
  -  DLNs
     - Alignment, plus amplification phase
  - Grokking: main topic with worked example 
     -  Memorization → generalization transition
     -  Lazy to rich transition
  -  Mention Induction heads as a training phase transition (Olsson et al.)
        - In-context learning as emergent capability: Refer to inference/interp session
-  Scaling laws
  - Here we are interested in emergence in training dynamics: train longer new capabilities arise that are not seen by the training loss
  - Emergence in scaling: scaling laws
  - Emergent misalignment
- Theoretical perspectives:
  - Toy models
    - NTK, DLNs, lazy to rich
  - DMFT
  - Tensor Programs
    - Statistical field theory
  - SLT
- Implications for alignment
  -  Difficulty of Predicting capabilities from smooth loss
  -  Controlling what is learned from data (see influence functions lecture)
  -  Open problems

# Plan
Learning objectives: 
* Understand that new skills emerge during training
* Main training phases of frontier models
* Smooth loss can hide discrete jumps
* Understand grokking as an example of phase transition and emergence
* Relevance to alignment with emergent misalignment
* Main applications of training dynamics for safety
  * Early detection of emergent unsafe behaviour
  * Robust unlearning
  * Controlling training 

Teaching phase: 35 mn + 15 mn Q&A
Learning phase: 
- Worked out example: Grokking (50mn)
- Reading and discussing the Emergent Misalignment paper (30mn reading)
- Proper lecture: 50mn + 10mn break

# Slide deck

## Emergence in training dynamics

Can we predict and control emergent phenomena during training?
## The mystery of emergence (5-10mn)


### Training LLMs
- pretraining cross entropy loss (sum of condiitonal log probs)
-  RLHF ranking loss  (note mention that it is used for training reward model which then is used for learning policy)
- Mention reasoning models: produce CoT and reward on correctness of the answer


### Emergence of capabilities in LLMs
image from Wei et al: figure 2 https://arxiv.org/pdf/2206.07682

Note that we will be interested in dynamic emergence (training time) not scale

### Key puzzle: hidden progress measures
New capabilities:

- Appear suddenly (not gradually)
- Are not predicted by the training signal (loss)
- Involve qualitative reorganisation of internal representations

### Physics analogy: phase transition
- More is different (Andersen): train for longer, train bigger get novel capabilities
- Rapid change in macroscopic behaviour driven by continuous change in control parameter 
- Ex: 1st oder (solid liquid gas), 2nd order (magnetisation in ferromagnet)
- Mathematically phase transition is a singularity in the Gibbs free energy

### Emergence in scaling and training
- Scaling: bigger models lead to novel capabilities 
-  Write Scaling laws (Kaplan, Chinchila)
-  Control parameter: Data, model size
- Train for longer: novel capabilities
- Control parameter: training time
- Can train bigger or longer

### Mirage debate
-  Put Figure 2 from [https://arxiv.org/pdf/2304.15004]
-  Mirage: one metric's emergence is another metric continuous phenomena
-  But we do have evidence of rapid skill acquisition and qualitive change in models

## Empirical examples of emergence (15-20mn)
### Silent alignment in DLNs
- Loss plateau: alignment of student toward teacher direction 
- Can be seen with NTK
- Figure 1 in. [https://arxiv.org/pdf/2111.00034]

### Sparse parity learning
- Task: SGD learns parity of a substring
  - (n,k)-sparse parity string: get random n-bits string
  - $ y= \Pi_{j\in k} x_j $
  - Learners see (x,y) must figure out k
- Hidden progress: figure 3 in [https://arxiv.org/pdf/2207.08799]
- If SGD random: $2^{O(n)}$ steps
- SGD not random: $n^{\Omega(k)}$ steps, polynomial (close to optimal)

### Grokking

- Setup: transformer learns modular arithmetic (addition of mult)
- Observation: 
  - Delayed generalization
  - Train loss is 0 
  - test loss is high 
  - Accuracy becomes 100%
- Figure 1 in [https://arxiv.org/pdf/2201.02177]


### Why is it an example of emergence
- Memorization phase then generalization phase
- Training loss gives no signal
- Can detect grokking internally 
- See phase diagram in tutorial

### Hidden progress measures:
Nanda mechanistic progress figure 7 in [https://arxiv.org/pdf/2301.05217]

Note for me:
- The top-left panel (excluded loss) shows that during circuit formation, the memorisation component is being replaced — excluded loss rises while train and test loss remain flat. This is the smoking gun: something is changing internally even though external performance looks unchanged.
- The top-right panel (restricted loss) shows the generalising circuit forming before grokking occurs — restricted loss starts declining well before test loss drops. This directly demonstrates hidden progress: the Fourier multiplication algorithm is already working if you isolate it, but the memorisation noise masks it.
- The bottom-left panel (Gini coefficients) shows the weights becoming sparse in the Fourier basis — the sharp increase during cleanup corresponds to the network committing to the structured solution.
- The bottom-right panel (sum of squared weights) links the whole process to weight decay as the driving force.

### Transition from memorization to generalization

Empirical LLC detect it (not predict) include figure 3 in [https://arxiv.org/pdf/2603.01192]
Interpretation: low loss basin that generalize better have lower LLC and are preferred

Note: grokking phase diagram will be discussed during the tutorial session

### Various explanations
- Circuit competitions: memorising circuit vs. generalising circuit (Varma et al., 2023) [https://arxiv.org/abs/2309.02390]
- Regularisation effect: weight decay favours efficient (generalising) solutions
- Lazy to rich transition nuance weight decay [https://arxiv.org/abs/2310.06110]: see tutorial 

### Induction Heads

During transformer training, a specific circuit forms: induction heads

Pattern: [A][B] ... [A] → predict [B]
Enables in-context learning

Plot image of induction heads

Note: can be seen as bump in the loss this time

### Emergent misalignment

Figure 1 in [https://arxiv.org/pdf/2602.07852]
Figure 1 in [https://arxiv.org/pdf/2502.17424]
Include original EM paper

### Emergent misalignment as phase transition

Figure 7, 9, 10 from [https://arxiv.org/pdf/2506.11613v1]

### EM as a generalization issue

Figure 5 from [https://arxiv.org/pdf/2602.07852]

he general solution is an attractor in the loss landscape, the narrow solution is unstable without explicit constraint, and diverse starting points all converge to the same place. That convergence is what makes it a generalisation story — the model isn't just memorising the dataset, it's finding a broader representation that happens to be misaligned.

See tutorial for more

## Stat mech + DMFT: a theoretical approach (30mn)

- From statistical physics: self-consistent field theory for wide networks
- Tracks **order parameters** (kernel matrices) at pairs of time points

### Stat mech crash course (10mn)

- Ising model
- Partition function
- Mean field: spins are ind. and interacting with mean from the other spins
- Effective free energy
- Equilibrium yield self consistent equation
- phase transition: 1st and 2nd order
- Bridge to neural networks: stochastic and dynamical system with large number of interactions, large width, large depth: quite natural to want to apply statistical physics.

### DMFT (20mn)
- Motivate DMFT
  - A wide neural network is a high-dimensional interacting dynamical system.
- Spins are weights (micro), energy is loss, macro are kernels
- Micro dynamics: DNN equation, gradient flow and NTK
  - Goal: average over initialization to isolate randomness
- Order parameters: Kernels
- Mean field statements for DNN: micro interact with macro fields
- Path integral is over weight config.
- self consistent equations for kernels (conjugate field?)
- Two layer DMFT equations
- Interpretation
- Extensions
- Limitations





## Tutorial on DMFT
- DMFT for linear networks
- Application to grokking


### Alternative appraoches
Not dynamical but still important
- Tensor program and other statistical field theory
- SLT
- Michaud quanta hypotehsis
- Arora: emergence of skills

## Open Problems (~3 min)

### — Implications for Alignment
- **Predicting capabilities from loss is unreliable**
  - Smooth loss ≠ smooth capability development
  - Emergent misalignment shows this extends to safety-relevant properties
- **Early detection of emergent unsafe behaviour**
  - LLC, progress measures, gradient norms as monitoring tools
  - But: we don't yet know what to monitor for in general
- **Controlling what is learned**
  - Can we steer training away from dangerous phase transitions?
  - Connection to influence functions (→ next lecture)

### Open Problems
1. **Predicting transition timing**: no framework can predict *when* a transition will occur for a new model on new data
2. **Scalability**: most theoretical tools work at <100M parameters; frontier models are 100×–1000× larger
3. **Bridging Bayesian-SGD gap**: SLT proves results about Bayesian posteriors; connection to Adam/SGD dynamics is empirical, not proven
4. **Automated monitoring**: can we build architecture-agnostic early warning systems for capability jumps?
5. **The emergence of emergence**: why do neural networks have phase transitions at all? Is it a necessary feature of gradient-based learning, or contingent on architecture?


## Deprecated
### NTK
h fixed kernel
  - "Lazy" regime: features don't change, only readout weights move
  - Predicts no phase transitions, no grokking
- **Rich regime**: features adapt to the task (finite width, μP)
  - Enables qualitative reorganisation → phase transitions possible
- The **lazy → rich transition** itself can cause grokking (Kumar et al., 2024)
  - Control parameter α: laziness ↔ feature learning rate
  - Grokking = network starts lazy, eventually escapes to rich regime

### Slide 20 — Dynamical Mean Field Theory (Bordelon & Pehlevan)
- From statistical physics: self-consistent field theory for wide networks
- Tracks **order parameters** (kernel matrices) at pairs of time points
- Key results:
  - NTK eigenstructure **rotates toward task-relevant directions** during training ("silent alignment")
  - Derives neural scaling laws with separate exponents for time and model size
  - Non-perturbative in feature learning strength (unlike NTK corrections)
- Provides the analytical engine behind the lazy-to-rich grokking framework
- Limitation: expensive solver, partial extension to discrete SGD

### Alternative appraoches
Not dynamical but still important
- Tensor program and other statistical field theory
- SLT
- Michaud quanta hypotehsis
- Arora: emergence of skills




### Grokking
* setup, model, task, loss, training
* key observation delayed generalization show train and test loss
  *  Power, Burda, Edwards, Babuschkin, and Misra (2022)
* Phase diagram 
  * Liu et al. (2023), "Omnigrok
* Role of Regularization
  * Is weight decay important? 
  * Kumar vs Kaifeng Lyu
* Hidden measures
* Algorithm
  * Competing circuits
* Lazy to rich transition 

# Emergent misalignment
    - Original paper 
    - Phase transition https://arxiv.org/pdf/2506.11613
    - EMERGENT MISALIGNMENT IS EASY,
NARROW MISALIGNMENT IS HARD



# Claude literature on grokking

Theoretical approaches to grokking: a literature review

**Grokking — the dramatic delayed generalization that occurs long after a neural network has memorized its training data — has spawned a rich theoretical ecosystem since Power et al. first documented it in 2022.** The field has converged toward a core narrative: grokking arises when networks transition from a kernel/lazy regime (memorization) to a feature-learning/rich regime (generalization), with regularization accelerating but not always causing this shift. Yet this consensus masks deep disagreements about mechanism, necessity of weight decay, and generality. This review catalogs the major theoretical frameworks, their mathematical tools, and their interconnections, with an eye toward identifying results tractable enough for graduate-level exercises.

The foundational observation comes from **Power, Burda, Edwards, Babuschkin, and Misra (2022)**, "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" (arXiv:2201.02177, ICLR 2022 Workshop). Small transformers trained on binary operations (e.g., division mod 97) achieve near-perfect training accuracy within ~10³ steps but require ~10⁶ steps before validation accuracy suddenly jumps to near-perfect. The phenomenon is robust across operations (modular addition, subtraction, permutation composition), optimizers, and model sizes. Smaller training fractions require exponentially more steps to generalize. This paper is purely empirical — it posed the questions that the subsequent theoretical literature has tried to answer.

## Implicit regularization and the kernel-to-rich transition

The most mathematically rigorous line of work explains grokking through the implicit bias of gradient descent and the role of regularization in driving a transition between training regimes.

**Lyu, Jin, Li, Du, Lee, and Hu (2024)**, "Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking" (ICLR 2024, arXiv:2311.18817), is the cornerstone theoretical paper. For *L*-homogeneous neural networks with large initialization scale α and small weight decay λ, they prove a sharp dichotomy: before a critical time *t*\* ≈ log(α)/λ, the network implements a kernel SVM using the initial NTK, while just after *t*\*, it attains first-order KKT conditions for margin maximization (min ‖θ‖² subject to correct classification with margin). The transition is provably sharp — the network jumps from kernel predictor to max-margin predictor. Concrete grokking examples are constructed for sparse linear classification and low-rank matrix completion. The framework is highly tractable: the proofs rely on gradient flow dynamics, norm tracking, and KKT conditions — all standard tools in advanced optimization courses.

**Kumar, Bordelon, Gershman, and Pehlevan (2024)**, "Grokking as the Transition from Lazy to Rich Training Dynamics" (ICLR 2024, arXiv:2310.06110), provides the crucial counterpoint by demonstrating grokking **without weight decay or regularization**. In two-layer MLPs on polynomial regression, they identify sufficient statistics for test loss and show three conditions produce grokking: NTK-task misalignment at initialization, a "Goldilocks" dataset size, and starting in the lazy regime. The key control parameter is the output scale α, which governs the feature-learning rate. This paper demonstrates that weight decay is sufficient but not necessary for grokking, and that the fundamental mechanism is the lazy-to-rich transition itself. The polynomial regression setting is analytically clean, making it excellent for exercises involving kernel methods.

**Mohamadi, Li, Wu, and Sutherland (2024)**, "Why Do You Grok?" (ICML 2024, arXiv:2407.12332), provides the strongest sample-complexity results specific to modular arithmetic. They prove a lower bound: any permutation-equivariant kernel method requires **Ω(p²) samples** for modular addition mod *p* — essentially all the data. But two-layer quadratic networks with bounded ℓ∞ norm generalize with only **Õ(p^{5/3}) samples**, the first non-trivial bound for this task. This formally establishes why memorization is unavoidable in the kernel regime and why escape to the rich regime is necessary.

**Boursier, Pesme, and Dragomir (2025)**, "A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation" (NeurIPS 2025, arXiv:2505.20172), offers the most general optimization-theoretic framework. They decompose gradient flow with small weight decay λ into two phases: a fast phase that follows unregularized gradient flow to the interpolation manifold, and a slow phase (timescale ~1/λ) where Riemannian gradient flow minimizes the ℓ₂ norm on that manifold. This result requires no task-specific assumptions, unifying the lazy-to-rich and weight-norm-reduction perspectives under one umbrella.

## Fourier analysis and mechanistic interpretability of learned algorithms

A parallel line of work reverse-engineers the actual algorithms that grokked networks implement, revealing a deep connection to Fourier analysis and representation theory.

**Nanda, Chan, Lieberum, Smith, and Steinhardt (2023)**, "Progress Measures for Grokking via Mechanistic Interpretability" (ICLR 2023 Oral, arXiv:2301.05217), is the foundational mechanistic paper. They fully reverse-engineer one-layer transformers trained on modular addition (a + b mod 113), showing the network implements a **Fourier Multiplication Algorithm**: embeddings map inputs to sin(ωₖa) and cos(ωₖa) at sparse key frequencies ωₖ, and the MLP combines these via the identity cos(ωa)cos(ωb) − sin(ωa)sin(ωb) = cos(ω(a+b)). Two progress measures — *restricted loss* (computed using only key frequencies, which declines steadily) and *excluded loss* (on non-key frequencies, which rises) — reveal that the generalizing circuit is being built continuously during the apparent memorization plateau. Training decomposes into three phases: memorization, circuit formation, and cleanup. This is among the most tractable papers for exercises: the trigonometric identities are elementary, the Fourier analysis of weights is concrete, and a Colab notebook is publicly available.

**Gromov (2023)**, "Grokking Modular Arithmetic" (arXiv:2301.02679, later ICLR 2024), provides **explicit analytic weight constructions** for two-layer networks that solve modular addition. The first-layer weights encode inputs as cos(2πka/p) and sin(2πka/p), and the second layer combines them via trigonometric identities. Crucially, Gromov shows grokking occurs even with vanilla SGD and MSE loss (no weight decay), with the Inverse Participation Ratio of weights in Fourier space serving as a progress measure. This paper is extremely tractable — the weight formulas are explicit and involve only basic trigonometry.

**Morwani, Edelman, Oncescu, Zhao, and Kakade (2024)**, "Feature Emergence via Margin Maximization" (ICLR 2024, arXiv:2311.07568), answers *why* these Fourier features emerge. They prove that the principle of **margin maximization** alone fully specifies the learned features: for modular addition, the max-margin solution necessarily uses Fourier features; for general finite group composition, it uses irreducible group-theoretic representations. This bridges Nanda et al.'s "what" with the optimization-theoretic "why," using results connecting gradient descent on homogeneous networks to max-margin classification (Soudry et al. 2018, Lyu and Li 2020).

**Chughtai, Chan, and Nanda (2023)**, "A Toy Model of Universality" (ICML 2023, arXiv:2302.03025), generalizes the Fourier picture to arbitrary finite groups using representation theory. They propose the **Group Composition via Representations** algorithm, where embeddings map to matrix representations ρ(g), the hidden layer computes the matrix product ρ(a)ρ(b), and the output computes traces tr(ρ(ab)ρ(c⁻¹)). For non-abelian groups like S₅, networks learn higher-dimensional irreducible representations. **Stander, Yu, Fan, and Biderman (2024)** (ICML 2024, arXiv:2312.06581) later showed that for S₅ and S₆, the more faithful description involves **coset circuits** rather than character-based algorithms, revealing richer algebraic structure.

**Zhong, Liu, Tegmark, and Andreas (2023)**, "The Clock and the Pizza" (NeurIPS 2023, arXiv:2306.17844), complicates the clean Fourier picture by showing networks can learn qualitatively distinct algorithms: the "Clock" (standard Fourier rotation) and the "Pizza" (vector mean followed by piecewise-linear slicing). Small hyperparameter changes determine which algorithm emerges. This result is pedagogically valuable for illustrating algorithmic diversity in neural networks.

**Mallinar, Beaglehole, Zhu, Radhakrishnan, Pandit, and Belkin (2025)** (ICML 2025, arXiv:2407.20199) extend the story beyond neural networks entirely, showing grokking occurs in Recursive Feature Machines via the Average Gradient Outer Product, with both neural and non-neural models learning **block-circulant feature matrices** diagonalized by the DFT. This demonstrates that the Fourier structure is universal across model classes.

## Phase transitions and statistical physics frameworks

Several papers frame grokking using the language and tools of statistical physics, treating it as a genuine phase transition.

**Rubin, Seroussi, and Ringel (2024)**, "Grokking as a First Order Phase Transition in Two Layer Networks" (ICLR 2024, arXiv:2310.03789), applies the adaptive kernel/mean-field approach to show that grokking corresponds to a transition from Gaussian Feature Learning (GFL) to Gaussian Mixture Feature Learning (GMFL), analogous to a first-order phase transition. They derive the transition location analytically for modular addition and cubic-polynomial teachers, reducing the problem to solving a low-dimensional nonlinear equation for order parameters. A counter-perspective appeared at NeurIPS 2025, "Is Grokking a Computational Glass Relaxation?", which maps grokking to slow relaxation in a glassy landscape rather than a sharp first-order transition, finding no entropy barrier in the memorization-to-generalization path.

**Žunkovič and Ilievski (2024)**, "Grokking Phase Transitions in Learning Local Rules with Gradient Descent" (JMLR 25(199):1–52, arXiv:2210.15435), provides two fully **analytically solvable** grokking models. They derive exact critical exponents, grokking probability, and grokking time distributions, showing grokking is a consequence of teacher locality. L₁ regularization yields higher grokking probability and shorter grokking times than L₂. The tensor network map to standard perceptron theory makes this ideal for graduate exercises.

**Liu, Kitouni, Nolte, Michaud, Tegmark, and Williams (2022)**, "Towards Understanding Grokking: An Effective Theory of Representation Learning" (NeurIPS 2022, arXiv:2205.10343), pioneered the physics-inspired approach. Using effective field theory, they model embedding dynamics as interacting particles, deriving an effective loss whose third eigenvalue λ₃ governs the grokking timescale: t_grok ~ 1/λ₃. Below a critical training data fraction (~0.4), λ₃ ≤ 0 and the structured representation is unstable. They map out four learning phases — comprehension, grokking, memorization, confusion — in a phase diagram over hyperparameters.

## Loss landscape geometry and the Omnigrok framework

**Liu, Michaud, and Tegmark (2023)**, "Omnigrok: Grokking Beyond Algorithmic Data" (ICLR 2023, arXiv:2210.01117), introduced the **LU mechanism**: when training loss versus weight norm ‖w‖ forms an "L" shape and test loss forms a "U" shape, their mismatch creates grokking. Generalizing solutions concentrate around a critical norm *w*_c at the U's minimum. Networks initialized with large norms first overfit in the high-norm region, then weight decay drives ‖w‖ toward *w*_c where generalization finally occurs, with grokking time *t* ∝ 1/λ. The key empirical contribution is inducing grokking on **MNIST, IMDb, and QM9** — far beyond algorithmic tasks — simply through large initialization plus weight decay. The constrained optimization technique (rescaling weights to fixed norm and evaluating loss) is a clean, teachable method.

**Prieto, Barsbey, Mediano, and Birdal (2025)**, "Grokking at the Edge of Numerical Stability" (ICLR 2025, arXiv:2501.04697), identifies a complementary numerical mechanism: without regularization, post-overfitting gradients align with a "Naïve Loss Minimization" direction that merely scales logits without improving predictions, driving weight growth until **Softmax Collapse** (floating-point errors in softmax) halts learning entirely. Weight decay prevents this scaling, explaining regularization's role. They introduce StableMax (a numerically stable activation enabling grokking without regularization) and ⊥Grad (which projects out the NLM direction). This paper's gradient decomposition analysis is mathematically accessible and practically illuminating.

## Circuit competition and efficiency

**Varma, Shah, Kenton, Kramár, and Kumar (2023)**, "Explaining Grokking Through Circuit Efficiency" (arXiv:2309.02390, Google DeepMind), formalizes grokking as competition between a memorizing circuit and a generalizing circuit with different "efficiencies" (logit magnitude per unit parameter norm). The memorizing circuit's cost scales with dataset size while the generalizing circuit's cost is fixed, yielding a **critical dataset size D_crit** where they cross. Weight decay favors the more efficient circuit. This theory uniquely predicts two novel phenomena, both experimentally confirmed: **ungrokking** (a grokked network regresses when retrained on smaller data) and **semi-grokking** (partial generalization at D ≈ D_crit). The two-circuit toy model is analytically simple — it involves parameterized product weights w_G = w_{G1}·w_{G2} with gradient descent — making it excellent for problem sets.

**Merrill, Tsilivis, and Shukla (2023)**, "A Tale of Two Circuits" (arXiv:2303.11873, ICLR 2023 Workshop), provided the empirical foundation by demonstrating that grokking on sparse parity corresponds to a sparse subnetwork dominating over a dense one, connected to the lottery ticket hypothesis. **Huang, Hu, Han, Liu, and Sun (2024)** (arXiv:2402.15175) extended the circuit competition framework to construct a two-dimensional phase diagram unifying grokking, double descent, and emergent abilities in LLMs.

## Scaling laws and discrete skill acquisition

**Michaud, Liu, Girit, and Tegmark (2023)**, "The Quantization Model of Neural Scaling Laws" (NeurIPS 2023, arXiv:2303.13506), proposes that neural network knowledge consists of **discrete quanta** (skills) that are either fully learned or not. Smooth power-law scaling laws emerge as statistical averages over many individual sharp phase transitions. The model posits: tasks decompose into enumerable quanta with learning threshold τ > 1, and quanta use-frequencies follow a Zipfian distribution. When quanta are learned in frequency order, the average loss follows L ∝ N^{−α}. Grokking is visible when tasks involve few quanta, making each individual phase transition observable. The mathematical core — summing over power-law-distributed quanta — is quite tractable and connects grokking to the broader neural scaling laws literature.

**Barak, Edelman, Goel, Kakade, Malach, and Zhang (2022)**, "Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit" (NeurIPS 2022, arXiv:2207.08799), studies grokking-like transitions in sparse parity learning. They show SGD makes continuous "hidden progress" through gradual amplification of sparse Fourier features driven by the population gradient's Fourier gap, achieving convergence in ~n^{O(k)} iterations near SQ lower bounds. The hidden progress measure ρ(w) — alignment of weights with the parity support — increases smoothly during the apparent loss plateau. This paper introduced the concept of "progress measures" that Nanda et al. then operationalized through mechanistic interpretability.

## The slingshot mechanism and training dynamics

**Thilak, Littwin, Zhai, Saremi, Paiss, and Susskind (2022)**, "The Slingshot Mechanism" (arXiv:2206.04817, NeurIPS 2022 Workshop, later TMLR), identifies cyclic phase transitions in adaptive optimizers (Adam, AdaGrad) at late training stages: weight norms grow rapidly, loss spikes, features undergo rapid evolution, then the cycle resets. Without explicit regularization, grokking almost exclusively occurs at slingshot onset. The slingshot acts as an **implicit regularizer** specific to adaptive optimizers and is absent with vanilla SGD. While primarily empirical, the phenomenology connects to the **catapult mechanism** of **Lewkowycz, Bahri, Dyer, Sohl-Dickstein, and Gur-Ari (2020)** (arXiv:2003.02218), which provides a theoretical foundation: at learning rates above η_crit = 2/λ₀ (top NTK eigenvalue), the network enters a phase where loss initially increases, curvature decreases, then training converges to flatter minima. The slingshot is effectively the cyclic version of the catapult under adaptive optimization.

**Notsawo, Zhou, Pezeshki, Rish, and Dumas (2023)** (arXiv:2306.13253, ICML 2023 Workshop) bridge these dynamics with loss landscape geometry, showing that low-frequency oscillatory signatures in early training loss reliably predict whether grokking will eventually occur. Fourier spectral analysis of the loss curve provides an early diagnostic tool.

## Complexity, compression, and simplicity bias

A cluster of papers explains grokking through the lens of complexity reduction and compression during training.

**Humayun, Balestriero, and Baraniuk (2024)**, "Deep Networks Always Grok and Here is Why" (ICML 2024, arXiv:2402.15555), uses spline theory (viewing ReLU networks as continuous piecewise affine operators) to show that grokking corresponds to a phase transition in **local complexity** — the density of linear regions near data points. During training, linear regions migrate from training samples toward decision boundaries, smoothing the mapping around data. They demonstrate grokking on CNNs/CIFAR-10 and ResNets/Imagenette, extending the phenomenon well beyond algorithmic tasks, and introduce "delayed robustness" (grokking adversarial examples).

**DeMoss et al. (2024)**, "The Complexity Dynamics of Grokking" (Physica D, 2025; arXiv:2412.09810), provides the most principled information-theoretic framework using **rate-distortion theory** and algorithmic complexity. Their lossy compression scheme (quantization plus low-rank approximation) yields a complexity measure that rises during memorization and falls sharply during generalization. They connect this to explicit **MDL-based generalization bounds**: min_M [H(D|M) + C(M)]. The rate-distortion framework is standard information theory, making it quite accessible.

**Liu, Zhong, and Tegmark (2023)**, "Grokking as Compression" (arXiv:2310.05918), define the **Linear Mapping Number** (LMN) — a generalization of the linear region count — as a neural network complexity proxy. LMN decreases steadily during the compression phase with a strong linear relationship to test loss, serving as a candidate for neural network Kolmogorov complexity. **Miller, O'Neill, and Bui (2024)** (TMLR, arXiv:2310.17247) extend the story further by showing grokking occurs even in Gaussian Processes, demonstrating it is not neural-network-specific and supporting the view that grokking fundamentally involves tension between complexity and fit.

## Singular learning theory approaches

A growing community applies Watanabe's Singular Learning Theory to grokking, using the Real Log Canonical Threshold (RLCT) and its practical estimator, the Local Learning Coefficient (LLC).

**Lau, Wei, and Murfet (2023)**, "The Local Learning Coefficient" (arXiv:2308.12108), introduced the LLC as a singularity-aware complexity measure estimated via SGLD, scaling to 100M-parameter networks. This is the foundational tool paper: all subsequent SLT grokking analyses rely on it. **Chen, Lau, Mendel, Wei, and Murfet (2023)** (arXiv:2310.06301) demonstrated the framework on Anthropic's Toy Model of Superposition, deriving closed-form losses and showing that LLC changes predict phase transitions in both Bayesian posteriors and SGD training.

**Cullen et al. (2025)**, "Grokking as a Phase Transition between Competing Basins" (arXiv:2603.01192), is the most direct SLT treatment of grokking. They interpret it as a transition between near-zero-loss basins with different LLCs: the memorizing basin has higher LLC (lower posterior concentration), and the generalizing basin has lower LLC. For **quadratic networks on modular arithmetic**, they derive **closed-form LLC expressions**, making the algebraic geometry concrete. The Bayesian free energy expansion F_n ≈ nL_n(w\*) + λ·log(n) + (m−1)·log(log(n)) connects the LLC λ to generalization, with LLC trajectories tracking validation loss during grokking.

**Lakkapragada (2025)** (arXiv:2512.00686) tests an **Arrhenius-style rate hypothesis**: grokking time scales as t_grok ~ exp(β_eff · |ΔF|), where ΔF is the free energy decrease between basins and β_eff depends on learning rate and batch size. **Hoogland, Wang, Farrugia-Roberts, Carroll, Wei, and Murfet (2024)** (arXiv:2402.02364) extend LLC-based analysis to transformer training dynamics, detecting discrete developmental stages as LLC plateaus, providing a framework that applies to grokking as one instance of stagewise development.

## Exactly solvable models as exercise foundations

Several papers provide complete analytical solutions ideal for building graduate exercises around.

**Levi, Beck, and Bar-Sinai (2024)**, "Grokking in Linear Estimators" (ICLR 2024, arXiv:2310.16441), show grokking in **linear networks on linear tasks** with Gaussian inputs. Full training dynamics are derived via training and generalization covariance matrices, yielding closed-form grokking time as a function of dimensionality, sample size, regularization, and initialization. The provocative finding: the sharp accuracy increase may be an artifact of the threshold-based accuracy measure rather than a true memorization-to-understanding transition. Grokking time diverges logarithmically near the interpolation threshold, analogous to critical phenomena. This is the simplest possible solvable model — pure linear algebra.

**Xu, Vardi, and Bar-Sinai (2026)**, "To Grok Grokking: Provable Grokking in Ridge Regression" (arXiv:2601.19791), provides the first **end-to-end provable** grokking result, with rigorous quantitative bounds: T_grok ∝ 1/λ and ∝ ln(ν²), where λ is the weight decay and ν is the initialization scale. **Beck, Levi, and Bar-Sinai (2024)** (arXiv:2410.04489) study grokking at the edge of linear separability in random feature models, proving grokking occurs precisely when the training set's convex hull contains the origin. Both are highly tractable.

**Xu, Wang, Frei, Vardi, and Hu (2024)**, "Benign Overfitting and Grokking in ReLU Networks for XOR Cluster Data" (ICLR 2024, arXiv:2310.02541), proves that both benign overfitting and grokking occur in two-layer ReLU networks on XOR clusters, bridging the benign overfitting and grokking literatures.

## Recent unifying frameworks and scaling laws

**Tian (2025)**, "Provable Scaling Laws of Feature Emergence from Learning Dynamics of Grokking" (NeurIPS 2025, arXiv:2509.21519), provides the most comprehensive dynamics framework. Their Li₂ framework identifies three stages: (I) lazy learning/memorization, (II) independent feature learning from backpropagated gradients, and (III) interactive feature learning where neurons diversify via repulsion. They prove emerging features are local maxima of an energy function and derive a scaling law: **O(M log M) samples** suffice for generalization of group arithmetic of order M.

**Truong et al. (2026)** (arXiv:2603.13331) derive a tight **scaling law for grokking delay**: T_grok − T_mem = Θ((1/γ_eff) · log(‖θ_mem‖²/‖θ_post‖²)), where γ_eff = ηλ for SGD. This is confirmed across 293 runs with R² > 0.97, providing the most precise quantitative prediction of grokking timing. **Notsawo, Zhou, Pezeshki, Rish, and Dumas (2025)** (arXiv:2506.05718) demonstrate that grokking can be induced by *any* form of regularization (ℓ₁, nuclear norm, implicit depth-based), generalizing the mechanism beyond weight decay.

## The landscape of competing explanations

The theoretical approaches form a layered narrative rather than truly competing alternatives:

- **What is learned**: Fourier features for abelian groups, irreducible representations for general groups, with alternative algorithms (Pizza) possible (Nanda, Gromov, Chughtai, Zhong, Morwani)
- **Why these features**: Margin maximization under gradient descent's implicit bias selects them (Morwani, Mohamadi)
- **Why the delay**: Initial kernel regime cannot generalize; escape to the rich regime requires time proportional to 1/λ · log(initialization scale) (Lyu, Kumar, Boursier)
- **How to measure progress**: LLC trajectories, Fourier progress measures, LMN, local spline complexity all detect hidden progress during the apparent plateau (Nanda, Cullen, Liu, Humayun)
- **How general is it**: Grokking occurs in linear models, GPs, RFMs, and practical vision tasks — it is not neural-network-specific (Levi, Miller, Mallinar, Humayun)

Key tensions remain. Weight decay's role is debated: Lyu et al. and Varma et al. treat it as essential, while Kumar et al. and Gromov demonstrate grokking without it (though Prieto et al. show that without regularization, Softmax Collapse can prevent grokking with cross-entropy loss, partially reconciling the disagreement). The phase transition order is contested: Rubin et al. call it first-order, while the NeurIPS 2025 glass relaxation paper finds no entropy barrier. And while circuit competition and complexity compression both describe grokking well, Miller et al.'s demonstration of grokking in GPs challenges any explanation requiring neural-network-specific circuit structure.

## Conclusion

For designing graduate exercises on training dynamics, several papers stand out for mathematical tractability:

- **Levi et al. (2024)** for an exactly solvable linear model (pure linear algebra)
- **Varma et al. (2023)** for a two-circuit toy model with gradient descent dynamics and the elegant ungrokking/semi-grokking predictions
- **Gromov (2023)** for explicit Fourier weight constructions requiring only trigonometry
- **Nanda et al. (2023)** for hands-on mechanistic interpretability with DFT analysis
- **Žunkovič and Ilievski (2024)** for solvable perceptron models with exact critical exponents
- **Lyu et al. (2024)** for the sharp kernel-to-margin transition in homogeneous networks
- **Xu et al. (2026)** for end-to-end provable grokking bounds in ridge regression
- **Michaud et al. (2023)** for deriving scaling laws from quantized skill distributions

The field's deepest insight is that grokking is not an anomaly but a window into the fundamental structure of neural network learning: the tension between memorization and generalization, manifest as a sharp transition that only appears mysterious when viewed through the wrong progress measures. Every paper in this review, from different angles, ultimately tells this same story.
