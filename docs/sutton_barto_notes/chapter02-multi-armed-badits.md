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

We are playing for $$T$$ rounds, in each round we
choose one bandit and pull its lever.
Let 
- $$A_t$$ - the action taken at step $$t=1,\dots,T$$.
- $$R_t$$ - the reward received in step $$t$$.

Our goal is to maximize the total reward,
$$ R = \sum_{t=1}^{T}R_t $$. 

Since we do not know in advance how much each bandit
is going to give us, we can start by choosing randomly
and recording the rewards from each machine. This
is called *exploration*.

Once we have an estimate of the reward from each machine,
we can maximize our profits by choosing the one that
yields the highest reward. This is called 
*exploitation*.

One method to balance exploration and exploitation
is the $$\epsilon$$-greedy:
- with probability $$\epsilon$$, choose a random action (explore)
- with probability $$1 - \epsilon$$, choose the greedy action (exploit)

Formally, let $$\rho_1,\dots,\rho_k$$ be the reward distributions
of the bandits $$1,\dots,k$$. Then $$R_t$$ is a random variable
sampled from $$\rho_{A_t}$$.

Let $$Q_t\colon [k] \to \mathbb{R}$$ be our estimation of
the expected rewards at step $$t$$. We want that
$$\lim_{t\to\infty}Q_t(a) = \mathbb{E}[\rho_a] $$.

*sample-average methods*.

\(
Q_t(a)
=
\frac
{\sum_{i=1}^{t}R_t \cdot\mathbb{1}_{A_i=a} }
{\sum_{i=1}^{t}\mathbb{1}_{A_i=a}}
\)
This can be computed iteratively.

As an algorithm:
```
Parameters:
- eps: exploration-exploitation balance
- T: number of rounds

For a=1,...,k:
    N(a) <- 0
    Q(a) <- 0

For t=1,...,T:
    x <- Uniform([0,1])
    If x <= eps:
        A <- Uniform({1,...,k})
    Else:
        A <- argmax Q(a)
    R <- Bandit(A)
    N(A) <- N(A) + 1
    Q(A) <- Q(A) + (R - Q(A))/N(A)
```