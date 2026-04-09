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
![bg right:58% contain](emergence_figures/emergence_capabilities.png)

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

![width:1100px](emergence_figures/scaling_laws.png)

---
## Mirage debate
-  Mirage: one metric's emergence is another metric continuous phenomena
-  But we do have evidence of rapid skill acquisition and qualitive change in models
*Source: Schaeffer et al. (2023), Figure 2*

![width:1050px](emergence_figures/emergence_mirage.png)

---
# Empirical examples of emergence 
---
## Silent alignment in DLNs

*Atanasov et al. (2021): Neural Networks as Kernel Learners: The Silent Alignment Effect*

![width:1050px](emergence_figures/silent_alignment.png)

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
*Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit*

![width:1050px](emergence_figures/hidden_progress_barak.png)

---

## Grokking

- *Alethea Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" (Figure 1)*
-  Grokking as Delayed generalization

![height:310px](emergence_figures/grokking_effect.png)

---

## Grokking Hidden progress measures:
*Source: Neel Nanda et al., "Progress measures for grokking via mechanistic interpretability" (Figure 7)*

![height:400px](emergence_figures/grokking_progress.png)

---

## Transition from memorization to generalization

- Empirical LLC detects the transition, but does not predict it
- Lower-loss basins that generalize better also tend to have lower LLC
*Ben Cullen et al., "Grokking as a Phase Transition between Competing Basins: a Singular Learning Theory Approach" (Figure 3)*

![height:280px](emergence_figures/grokking_phase_transition.png)

---
### Induction Heads

- During transformer training, a specific circuit forms: induction heads
- Pattern: [A][B] ... [A] → predict [B]
- *Catherine Olsson et al., "In-context Learning and Induction Heads"*

![height:390px](emergence_figures/induction_heads.png)

---

## Emergent misalignment

*Jan Betley et al., "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs" (Figure 1)*

![height:330px](emergence_figures/emergent_misalignment.png)

---

### Emergent misalignment as a phase transition

*Edward Turner et al., "Model Organisms for Emergent Misalignment" (Figure 10)*

![height:350px](emergence_figures/emergent_misalignment_phase_transition.png)

---

### EM as a generalization issue

*Anna Soligo et al., "Emergent Misalignment is Easy, Narrow Misalignment is Hard" (Figure 1)*

![height:390px](emergence_figures/emergent_misalignment_generalization.png)

---

### EM as a generalization issue

*Anna Soligo et al., "Emergent Misalignment is Easy, Narrow Misalignment is Hard" (Figure 5)*

![height:260px](emergence_figures/emergent_misalignment_easy_hard.png)






