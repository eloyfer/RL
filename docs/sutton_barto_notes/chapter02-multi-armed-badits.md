---
layout: default
title: "Chapter 2: Multi-Armed Bandits"
---

# {{ page.title }}

*Game Description*.
There are $$k$$ levers ("bandits").
If we pull lever $$i$$ we receive reward
$$r \sim \rho_i$$, where $$\rho_i$$ 
is some unknown probability distribution.
We play for $$T$$ rounds, in each round we can pull one lever.

*Formal Description*
Let $$k\in \mathbf{N}$$ and $$\rho_1,\dots,\rho_k$$ 
be probability distributions over $$\mathbf{R}$$.
A game is a sequence of actions $$a_1,\dots,a_T\in [k]$$.
Each action results in reward, $$r_t \sim \rho_{a_t}$$.
The total reward is $$R = r_1+\dots+r_T$$.
We seek to maximize $$\mathbf{E}[R]$$.

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
