# Exercises
We use the same notations as in the presentation
- Consider a diagonal linear network learning a teacher matrix $\text{diag}(s_1,...,s_d)$. Show that the student $\text{diag}(s_1,...,s_r,0,...,0)$ with $r<d$ is a saddle point.
- Derive the gradient flow equation of a Deep Linear Network
- Show that $$G_l = W_{l+1}^{\top}W_{l+1} - W_{l}W_{l}^{\top}$$ is conserved through the gradient flow
- Show that the multiplication map $\mu(\theta)=W_L...W_1$ is invariant under action of product of $GL_{d_l}$ 
- Derive the Gradient flow equation in function space (i.e. in terms of the student $W = \mu(\theta)$)