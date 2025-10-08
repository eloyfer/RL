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

$$
\begin{equation}
Q_t(a)
=
\frac
{\sum_{i=1}^{t}R_i \cdot\mathbb{1}_{A_i=a} }
{\sum_{i=1}^{t}\mathbb{1}_{A_i=a}}
\end{equation}
$$

This can be computed iteratively. For simplicity, assume
we have only one action. Then

$$
\begin{align}
Q_t
&=
\frac
{\sum_{i=1}^{t}R_i}
{t}
\\
&=
\frac
{R_t + (t-1)Q_{t-1}}
{t}
\\
&=
Q_{t-1}
+
\frac{1}{t}
(R_t - Q_{t-1})
\end{align}
$$

In summary, we developed the following simple algorithm:
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



## Non-Stationary Problems

If the reward distribution is not stationary, 
i.e. it changes over time, it makes sense to give more weight
to the more recent observations.
The iterative update rule is then

$$
\begin{align}
Q_{n+1}
&=
Q_{n} + \alpha \cdot (R_{n} - Q_n)
\\
&=
(1-\alpha)^{n}Q_{1}
+ \alpha \sum_{i=1}^{n}(1-\alpha)^{n-i} R_{i}
\end{align}
$$

If $$\alpha \in (0,1]$$ is constant, then the weight of 
previous rewards decays exponentially as
$$(1-\alpha)^{n-i}$$.

This is called a $$weighted average$$, since the sum of weights
$$(1-\alpha)^n + \sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}= 1$$. 


If the step size $$\alpha = \alpha_n$$ is not constant, then

$$
\begin{align}
Q_{n+1}
&=
Q_{n} + \alpha_{n} \cdot (R_{n} - Q_n)
\\
&=
\alpha_{n} R_{n} + (1-\alpha_{n}) Q_n
\\
&=
\alpha_{n} R_{n} + (1-\alpha_{n}) (\alpha_{n-1} R_{n-1} + (1-\alpha_{n-1})Q_{n-1})
\\
&=
\prod_{j=1}^{n}(1-\alpha_{j}) Q_{1}
+
\sum_{i=1}^{n}\alpha_{i}\prod_{j=i+1}^{n}(1-\alpha_{j}) R_{i}
\end{align}
$$


## Optimistic Initial Values

Setting the initial values $$Q_1(a)$$ high
encourages exploration in the initial steps.
If the actual reward are lower, then each action
we visit lowers the estimation, leaving the less-explored
actions with high estimates. This turns the initial eploitation
steps into exploration steps. However, this trick is only
effective at the beginning of the run, and does not
encourage exploration in later stages.


## Unbiased Constant Step Size

Consider the following step size:

$$
\begin{align}
    \beta_n &= \alpha / o_n
    \\
    o_n &= o_{n-1} + \alpha(1 - o_{n-1}), &&o_0 = 0
\end{align}
$$

We want to express $$Q_n$$ as a weighted sum of past rewards.
Observe that

$$
\begin{align}
1 - \beta_n
&=
1 - \alpha / o_n
\\
&=
\frac{o_n - \alpha}{o_n}
\\
&=
\frac{(1-\alpha)o_{n-1}}{o_n}
\end{align}
$$

Then
$$
\begin{align}
    Q_{n+1} 
    &=
    Q_{n} + \beta_n(R_n - Q_{n})
    \\
    &=
    \prod_{j=1}^{n}(1-\beta_{j}) Q_{1}
    +
    \sum_{i=1}^{n}\beta_{i}\prod_{j=i+1}^{n}(1-\beta_{j}) R_{i}
    \\
    &=
    (1-\alpha)^{n}\frac{o_0}{o_n}
    Q_{1}
    +
    \sum_{i=1}^{n}
    \frac{\alpha}{o_i}
    (1-\alpha)^{n-i} \frac{o_{n-i}}{o_n}
    R_{i}
\end{align}
$$
