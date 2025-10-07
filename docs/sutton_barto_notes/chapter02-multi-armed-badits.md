---
layout: default
title: "Chapter 2: Multi-Armed Bandits"
---

# {{ page.title }}

In the $$k$$-armed bandit problem we have $$k$$
bandits (slot-machines), each with a lever
we can pull and receive reward. The reward is a random variable
from an unknown distribution. Every bandit might have 
a different distribution.

Let 
- $$\rho_i$$ - the reward distribution of machine $$i\in[k]$$
- $$A_t\in[k]$$ - the action taken at step $$t$$
- $$R_t \sim \rho_{A_t}$$ - the reward received in step $$t$$

Our goal is to maximize the total reward in $$T$$ steps,
$$ R = \sum_{t=1}^{T}R_t $$. Notice that $$R$$ is a random variable,
so let's say we seek to maximize the expected reward, $$\mathbb{E}[R]$$.

Had we knonw the distributions $$\{\rho_i\}$$,
the best strategy would be to choose the best action
$$a^* = \arg\max \{\mathbb{E}[\rho_a] \colon a\in [k]\}$$ 
for $$T$$ steps.

However, we do not have direct access to $$\{\rho_i\}$$ and cannot
compute the means. Instead, we can interact with the bandits and
estimate the distributions.
Let $$Q_t\colon[k]\to\mathbb{R}$$ be our estimation of the means
at step $$t$$.

One way to compute $$Q_t$$ is by the *sample-average* methods:
```math
Q_t(a)
=
\frac
{\sum_{i=1}^{t}R_t \cdot\mathbb{1}_{A_i=a} }
{{\sum_{i=1}^{t}\mathbb{1}_{A_i=a} }}
```