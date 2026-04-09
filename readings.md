# Training dynamics say:
The readings are roughly ordered in a way that makes sense for learning. This is a curated list from what I find important and not exhaustive.
## Loss landscape geometry
- [Deep Learning without Poor Local Minima
](https://arxiv.org/abs/1605.07110), Kenji Kawaguchi
    - For non-bottlenecked DLNs, all local minima are global
- [The loss landscape of deep linear neural networks: a second-order analysis](https://arxiv.org/abs/2107.13289) Achour et al. 
  - First order and second order classification of critical points in DLNs loss landscapes. Classify strict and non strict saddles.
- [Pure and Spurious Critical Points: a Geometric Study of Linear Networks
](https://arxiv.org/abs/1910.01671) Matthew Trager et al.
  - Define spurious minima and count the number of connected components of the global minima
- [Geometry of fibers of the multiplication map of deep linear neural networks
](https://arxiv.org/abs/2411.19920) Simon Pepin Lehalleur et al.
  - Stratify global minima into orbits using quiver representation theory
- [The Loss Surfaces of Multilayer Networks
](https://arxiv.org/abs/1412.0233) Anna Choromanska et al
  - No local minima results in non-linear multilayer networks with some strong assumptions (spin glass models)
- [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026), Timur Garipov et al
    - Study the connectedness of global minima of loss landscapes (mode connectivity)
- [The Multilinear Structure of ReLU Networks
](https://arxiv.org/abs/1712.10132), Thomas Laurent, James von Brecht
    - Show that local minima are typically singular
- Classic result: [Neural networks and principal component analysis: Learning from examples without local minima](https://www.sciencedirect.com/science/article/abs/pii/0893608089900142) Pierre Baldi et al.
  - Original result that linear network landscapes have no spurious local minima (single hidden layer case)
- Good Review of key DLNs results: [Gradient Flow Equations for Deep Linear Neural Networks: A Survey from a Network Perspective
](https://arxiv.org/abs/2511.10362), Joel Wendin, Claudio Altafini
  - Also covers gradient flow  
## Implicit biases of gradient flow
- Saxe et al. [A mathematical theory of semantic development in deep neural networks](https://www.pnas.org/doi/10.1073/pnas.1820226116)
  - Read the supplementary materials section to understand the exact solution of gradient flow dynamics in deep linear networks
- [Saddle-to-Saddle Dynamics in Deep Linear Networks: Small Initialization Training, Symmetry, and Sparsity
](https://arxiv.org/abs/2106.15933) Arthur Jacot et al.
  - Study the Saddle to saddle dynamics in DLNs. Introduce the different regimes (lazy and rich)
- [The geometry of the deep linear network
](https://arxiv.org/abs/2411.09004) Govind Menon
  - Beautiful mathematical treatment of gradient flow in DLNs and a surprising mathematical results relating gradient flow and free energy minimization. Make implicit regularization explicit (under balanced assumptions)
- [Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures
](https://arxiv.org/abs/2512.20607) Saxe et al.
  - Simplicity bias of gradient flow from DLNs extends to non linear and transformer architectures for small initialization
- [Abide by the Law and Follow the Flow: Conservation Laws for Gradient Flows
](https://arxiv.org/abs/2307.00144)
  - Give a recipe to exhaustively derive conservations law through gradient flow
- [SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics
](https://arxiv.org/abs/2302.11055)
  - SGD builds up complex solutions by learning simpler solutions and composing them first 
- [Mixed Dynamics In Linear Networks: Unifying the Lazy and Active Regimes
](https://arxiv.org/abs/2405.17580) 
  - Unify lazy and rich in 2 layers DLNs and give conditions for transition between them
- [Neural Tangent Kernel: Convergence and Generalization in Neural Networks
](https://arxiv.org/abs/1806.07572) Arthur Jacot, Franck Gabriel, Clément Hongler
  - NTK paper which describes gradient flow in function space 
- [The Neural Race Reduction: Dynamics of Abstraction in Gated Networks
](https://arxiv.org/abs/2207.10430) Saxe et. al
  - Implicit biases toward shared representation in non linear networks as modelled by gated DLNs
- [Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks
](https://arxiv.org/abs/2506.06489) Daniel Kunin et al.
  - A theory of feature learning as a two step process between dormant and active neurons
- [From Lazy to Rich: Exact Learning Dynamics in Deep Linear Networks
](https://arxiv.org/abs/2409.14623)
    - Study transition between lazy to rich regimes in DLNs with balanceness parameters
- [Implicit Regularization in Matrix Factorization](https://arxiv.org/abs/1705.09280)
  - Gradient descent on matrix factorization converges to minimum nuclear norm
- [Gradient Descent Maximizes the Margin of Homogeneous Neural Networks
](https://arxiv.org/abs/1906.05890)
  - Implicit bias toward max-margin in classification      
- [A Convergence Analysis of Gradient Descent for Deep Linear Neural Networks
](https://arxiv.org/abs/1810.02281) Sanjeev Arora et al.
  - Convergence analysis of SGD 

## Learning rate (discrete GD)
- [Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability
](https://arxiv.org/abs/2209.15594)
  - Show that curvature stabilize around inverse of learning rate.
- [Understanding Optimization in Deep Learning with Central Flows
](https://arxiv.org/abs/2410.24206)
  - GD with discrete learning rate is equivalent to gradient flow on an effective loss
- [Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective
](https://arxiv.org/abs/2410.05192)
  - Pretraining exhibits a river valley loss landscape and give intuition about learning rate schedule (warm-up, stable, decay)
- [Optimization on multifractal loss landscapes explains a diverse range of geometrical and dynamical properties of deep learning](https://www.nature.com/articles/s41467-025-58532-9)
  - Model the landscape as multi-fractal and analyse sub and super diffusive behaviour of GD


## Stochasticity
- [On the implicit regularization of Langevin dynamics with projected noise
](https://arxiv.org/abs/2602.12257)
  - Make explicit implicit bias of stochasticity in deep linear networks with a Langevin model 
- [Beyond Implicit Bias: The Insignificance of SGD Noise in Online Learning
](https://arxiv.org/abs/2306.08590)
  - Golden Path Hypothesis: in the transient regime dominated by drift (typical of pretraining) SGD is simply a noisy deformation of GD (does not selects different basins)
- [Stochastic Collapse: How Gradient Noise Attracts SGD Dynamics Towards Simpler Subnetworks
](https://arxiv.org/abs/2306.04251)
   - Stochasticity induces a bias toward simpler solutions in deep linear networks
- [Stochastic Training is Not Necessary for Generalization
](https://arxiv.org/abs/2109.14119)
  - Make implicit regularization of stochasticity explicit
- [Implicit Bias of SGD for Diagonal Linear Networks: a Provable Benefit of Stochasticity
](https://arxiv.org/abs/2106.09524)
  - Show that stochasticity has a bias toward sparser solution relative to GD
- [Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks](https://arxiv.org/abs/1710.11029)
  - SGD minimize some free energy and can have circular currents at convergence
- [Stochastic Gradient Descent as Approximate Bayesian Inference
](https://arxiv.org/abs/1704.04289)
  - Classic paper on SGD locally approximating Bayesian inference. Although assume non-degeneracies
- [A Diffusion Theory For Deep Learning Dynamics: Stochastic Gradient Descent Exponentially Favors Flat Minima
](https://arxiv.org/abs/2002.03495)
   - Implicit bias toward flat minima
- [Implicit Regularization or Implicit Conditioning? Exact Risk Trajectories of SGD in High Dimensions
](https://arxiv.org/abs/2205.07069)
   - Show golden path hypothesis on quadratic loss in convex setup using an interesting SDE model of SGD.
- [An Empirical Model of Large-Batch Training
](https://arxiv.org/abs/1812.06162)
  - Gradient noise scale can be used to decide the optimal batch size
- [Almost Bayesian: The Fractal Dynamics of Stochastic Gradient Descent
](https://arxiv.org/abs/2503.22478)
  - Relate SGD with Bayesian training
- [Catapults in SGD: spikes in the training loss and their impact on generalization through feature learning
](https://arxiv.org/abs/2306.04815)
  - Loss spikes when training with SGD and impacts generalization
- [The Heavy-Tail Phenomenon in SGD
](https://arxiv.org/abs/2006.04740)
    - SGD noise can be heavy tailed and this induced different regularization
  
## Emergence: empirical examples
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
](https://arxiv.org/abs/2201.02177)
  - Grokking: delayed generalization on modular addition
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
  - Induction heads form during training. Important for in-context learning
- [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs
](https://arxiv.org/abs/2502.17424)
  - Emergent misalignment: fine-tuning on insecure code can generalize to misalignment behaviour on other data
- [Emergent Misalignment is Easy, Narrow Misalignment is Hard
](https://arxiv.org/abs/2602.07852)
  - By regularizing we can mitigate emergent misalignment
- [Neural Networks as Kernel Learners: The Silent Alignment Effect
](https://arxiv.org/abs/2111.00034)
  - While loss is flat student singular vector align with teacher singular vector and this can be detected with the NTK
- [Are Emergent Abilities of Large Language Models a Mirage?
](https://arxiv.org/abs/2304.15004)
  - Emergence is in the eye of the metric used to measure it
- [Training Compute-Optimal Large Language Models
](https://arxiv.org/abs/2203.15556)
  - Chinchilla scaling law paper
- [Emergent Abilities of Large Language Models
](https://arxiv.org/abs/2206.07682)
  - Show that novel capabilities emerge with more compute

## Theoretical approaches to emergence
- [Grokking as a First Order Phase Transition in Two Layer Networks, Rubin et. al
](https://arxiv.org/abs/2310.03789)
- [Blake Bordelon - Infinite limits and scaling laws of neural networks - IPAM at UCLA ](https://www.youtube.com/watch?v=WcWFFiPRslM&t=1056s)
- [A Theory for Emergence of Complex Skills in Language Models
](https://arxiv.org/abs/2307.15936)
- [On neural scaling and the quanta hypothesis
](https://ericjmichaud.com/quanta/)
- [Lecture Notes on Infinite-Width Limits of Neural Networks; Cengiz Pehlevan and Blake Bordelon](https://pehlevan.seas.harvard.edu/sites/g/files/omnuum6471/files/pehlevan/files/princeton_lecture_notes.pdf)
- [Disordered Dynamics in High Dimensions: Connections to Random Matrices and Machine Learning; Blake Bordelon, Cengiz Pehlevan](https://arxiv.org/abs/2601.01010)
- [Applications of Statistical Field Theory in Deep Learning; Zohar Ringel et. al](https://arxiv.org/abs/2502.18553)
- [Statistical Field Theory for Neural Networks, Moritz Helias, David Dahmen
](https://link.springer.com/book/10.1007/978-3-030-46444-8)
- [Lecture notes: From Gaussian processes to feature learning, Moritz Helias et al
](https://arxiv.org/abs/2602.12855
)

