---
layout: default
title: "Chapter 2: Multi-Armed Bandits"
---

# {{ page.title }}


Consider the following problem: 
there is 1 state and $$k$$ possible actions. 
The reward for each action is given by some unknown probability
distribution, $$r(a)$$.

Let 
- $$A_t\in[k]$$ the action taken at step $t$
- $$R_t \in \mathbf{R}_{\geq 0}$$ the reward received

Our goal is to maximize the total reward in $$T$$ steps,
$$ \sum_{t=1}^{T}R_t $$. 
Had we knonw the distributions $$\{r(a)\}_{a\in [k]}$$,
a simple strategy would be to choose the best action
$$a^* = \arg\max \{r(a) a\in [k]\}$$


our best policy (by expectation) would be to set $$A_t = a^*$$
for all $$t\in [T]$$, where 

Therefore, a good idea would be to estimate $$r(a)$$ by interacting
with the environment.

Let $$Q_t\colon[k]\to\mathbf{R}$$ be out estimation for $$r(a)$$ at timestep $$t$$.
