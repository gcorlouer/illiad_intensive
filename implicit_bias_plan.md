# Topic 1: Implicit regularization
# Loss landscapes (20mn)
* Setup: SGD + loss + architecture
  * Mention other optimizer but we won't touch upon
* Define implicit regularization
    Contrast with weight decay as example of explicit regularization
* Loss landscapes (model + loss + data):
  * Population and empirical loss
  * Critical points: saddles (strict and non strict), spurious minima, global minima
* Some DLN slides
  *  Loss landscapes of NN are highly degenerate (link with SLT)
  *  Minima are global or saddles
  *  Global minima are connected in overparametrized
  *  Combinatorics of global minima with orbit decompositions via symmetries under GL_d
* Mention
  * 2 layer ReLU
  * Random matrix application/spin glasses 
  * See literature review on loss landscape from claude: https://claude.ai/public/artifacts/4530f831-d479-47e2-b5a3-84bfdaff1f19

## Discrete Optimization (20mn)
* Gradient Descent
  * Descent lemma: stability threshold
* Surprising phen.: Edge of stability + central flow
* Learning rate schedule
    * Warm up + River-valley landscape
* Gradient flow: NTK
  * Show NTK equation control function flow
* Stochasticity:
  * Key question: What is the role of stochasticity?
  * Implicit bias toward flatter solutions
  * Critical learning rate/Batch size and scaling of lr/B
  * Golden Path hypothesis
* Continuous models:
  * SDEs
  * Validity 
  * Fokker Planck
  * Langevin and link with SLT
  * Possible Exo Derive the stationary distribution 
    * Bias towards Flat minima
  * It's an effect at convergence: GPH
  * We limit ourselves to gradient flow
* Won’t talk about other optimizers but mention some and give references
* The question of what SGD converges to and whether converges under generic condition is still wide open (but also somehow overrated because understanding finite time and partial solutions is more relevant)

## DLN training dynamics (10mn)
* Mention the key results and we will derive them in tutorial session
* Small initialization + alignment
  * Self consistent equations
  * Saddle to saddle dynamics
  * Time scale of learning
* Large initiailzation
  * NTK 
  * Self consistent equations
  * Time scale of learning
* See effect of depth and init.
* Mixed regime: see DMFT session in the afternoon
* Mention Extensions
  * Gated DLNs
  * Simplicity bias / frequency bias. Neural networks learn low-frequency/low-complexity components first (Rahaman et al. 2019, Xu et al. 2019 "frequency principle").
  * Leap Complexity 
* Open problems


## Simplicity bias of deep learning: a worked out example with DLNs (1h30)
* Derive Gradient flow in DLNs (30mn)
  * GF equations in parameter space
  * Conserved quantities, balanced equations (could mention more general case for conservation through the flow)
  * GF in function space link with NTK
* 2 layer diagonal linear Rich regime: small init. saddle to saddle, role of depth (30mn)
  * Example: saddle to saddle dynamics in diagonal linear networks
* Stochasticity  (probably not)
  * SDE as approximate Bayesian inference. Gradient noise covariance matrix and diffusion on diagonal linear network. Link between noise, hessian, Fisher information matrix, NTK
  * * GF minimize free energy in DLNs
  * Competition energy vs entropy
  * Mention that reward is not the optimization target