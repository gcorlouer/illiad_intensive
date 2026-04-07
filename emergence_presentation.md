---
marp: true
theme: default
paginate: true
---

# Emergence during training

<!-- New slide starts here: use --- on its own line -->
---

## Mystery of emergence

Train LLMs and non-predicted capabilities emerge
*Wei et al. (2022), Figure 2: some capabilities appear abruptly once model scale crosses a threshold.*
![bg right:58% contain](emergence_slides/wei_figure2_with_legend.png)

<!-- This --- starts the next slide -->
---
## Key puzzle: hidden progress measures
New capabilities:

- Appear suddenly (not gradually)
- Are not predicted by the training signal (loss)
- Involve reorganisation of internal representations

---

## Physics analogy: phase transition
- More is different (Andersen): 
  - More compute induce novel capabilities
- Intuition: rapid change in macroscopic behaviour driven by continuous change in control parameter (compute)
  - Ex: liquid to gas or magnetisation
- A phase transition is a singularity in the Gibbs free energy

---

## Scaling laws

- Increase compute gets lower loss gets more capabilites
- Compute-optimal scaling laws (Chinchilla 2022)

![width:1100px](emergence_slides/scaling_laws_fig2.png)

---
## Mirage debate
-  Mirage: one metric's emergence is another metric continuous phenomena
-  But we do have evidence of rapid skill acquisition and qualitive change in models
*Source: Schaeffer et al. (2023), Figure 2*

![width:1050px](emergence_slides/mirage_debate_fig2.png)
---
# Empirical examples of emergence 
## Silent alignment in DLNs
- Loss plateau: alignment of student toward teacher direction 
- Can be seen with NTK
*Source: Atanasov et al. (2021), Figure 1*

![width:1050px](emergence_slides/silent_alignment_fig1.png)
---
## Sparse parity learning
- Task: SGD learns parity of a substring of bits
  - (n,k)-sparse parity string: get random n-bits string
$$ y= \Pi_{j\in k} x_j $$
  - Learner sees (x,y) must figure out k
- If SGD random: $2^{O(n)}$ steps
- But SGD not random: $n^{\Omega(k)}$ steps, polynomial (close to optimal)
---
## Sparse parity learning
- Hidden progress
*Source: Barak et al. (2022), Figure 3*

![width:1050px](emergence_slides/sparse_parity_fig3.png)
---

## Grokking

- Setup: transformer learns modular arithmetic (addition of mult)
- Observation: 
  - Delayed generalization
  - Train loss is 0 
  - test loss is high 
  - Accuracy becomes 100%
*Source: Power et al. (2022), Figure 1*

![width:1050px](emergence_slides/grokking_fig1.png)
---

## Grokking Hidden progress measures:
*Source: Nanda et al. (2023), Figure 7*

![width:1050px](emergence_slides/grokking_hidden_progress_fig7.png)

---

## Transition from memorization to generalization

- Empirical LLC detect it (not predict) 
- Interpretation: low loss basin that generalize better have lower LLC and are preferred

*Source: arXiv:2603.01192, Figure 3*

![width:1000px](emergence_slides/transition_llc_fig3.png)
---

## Various explanations
- Circuit competitions: memorising circuit vs. generalising circuit (Varma et al., 2023) [https://arxiv.org/abs/2309.02390]
  - Regularisation effect: weight decay favours efficient (generalising) solutions
- But lazy to rich transition nuance weight decay [https://arxiv.org/abs/2310.06110]: (see tutorial) 

---
### Induction Heads

- During transformer training, a specific circuit forms: induction heads
- Pattern: [A][B] ... [A] → predict [B]
Enables in-context learning

*Source: Olsson et al. (2022)*

![width:1050px](emergence_slides/induction_heads.png)

---

## Emergent misalignment

*Source: arXiv:2602.07852, Figure 1 and arXiv:2502.17424, Figure 1*

![width:1100px](emergence_slides/emergent_misalignment_fig1s.png)

---

### Emergent misalignment as a phase transition

*Source: arXiv:2506.11613v1, Figures 7, 9, 10*

![width:1000px](emergence_slides/em_phase_transition_figs.png)

---

### EM as a generalization issue

*Source: arXiv:2602.07852, Figure 5*

![width:1050px](emergence_slides/em_generalization_fig5.png)

---

# Theoretical approaches
