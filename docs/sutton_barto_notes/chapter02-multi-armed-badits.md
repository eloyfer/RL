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

Let $$\rho_1,\dots,\rho_k$$ be the reward distributions
of the bandits $$1,\dots,k$$. Then $$R_t$$ is a random variable
sampled from $$\rho(A_t)$$.

Initially we do not know what is the expected reward
from each action. Therefore, we need to *explore*
the environent by taking actions.

Once we have an estimate of the reward from each machine,
we can *exploit* our knowledge 
by choosing the action that maxiizes the expected reward.
<!-- 
Let us discuss some methods to balance between 
exploration and exploitation.

### $$\epsilon$$-Greedy -->

$$\epsilon$$-greedy:
- with probability $$\epsilon$$, choose a random action (explore)
- with probability $$1 - \epsilon$$, choose the greedy action (exploit)

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


## Unbiased Step Size

The constant step size induces *bias*, which is our initial value
$$Q_{1}$$, even though its weight is decreasing exponentially: 
$$(1-\alpha)^{n-1}$$.

Here is a varying step size that gets rid of the bias:

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
    (1-\alpha)^{n-i} \frac{o_{i}}{o_{n}}
    R_{i}
    \\
    &=
    \frac{1}{o_{n}}
    \sum_{i=1}^{n}
    \alpha(1-\alpha)^{n-i}R_{i}
\end{align}
$$

and indeed there is no bias.


## Upper-Confidence-Bound Action Selection

Another exploration rule is one that encourages exploration
of less-explored actions/states. For example,

$$
    A_t
    =
    \arg\max_{a}
    \left[
        Q_t(a)
        + 
        c\sqrt{\frac{\ln t}{N(a)}}
    \right]
$$

where $$c$$ is a parameter. 
If $$t$$ is large while $$N(a)$$ is small, the action $$a$$
is more likely to be selected.



## Learning to Act -- Gradient Ascent

So far we developed algorithms that learn the environment. 
The relevant information was $$\mathbb{E}[\rho(a)]$$ for all
$$a\in [k]$$.

In contrast, here the algorithm learns how to act.
Namely, it learns a distribution over actions, 
$$\pi\colon [k]\to[0,1]$$,
that seeks to maximize the expected reward:

$$
    \mathbb{E}[R_t]
    =
    \sum_{a} \pi(a) \rho(a)
$$


A convenient way to construct a distribution is using the softmax funciton:
Define an initial value $$H_1(a) = 0$$ for every action $$a$$. Then let

$$
\pi_t(a)
=
\frac
{e^{H_t(a)}}
{\sum_{a'=1}^{k}e^{H_t(a')}}
$$

and choose an action randomly by sampling $$\Pr(A_t = a) = \pi(a)$$.

Over time, we want to increase the probability of actions that
yield high rewards. We can do it using gradient ascent:

$$
H_{t+1}(a)
= 
H_{t}(a)
+
\alpha
\frac
{\partial \mathbb{E}[R_t]}
{\partial H_t(a)}
$$

Let us compute the last term.
Let $$\star = \frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)}$$.

$$
\begin{align}
\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)}
&=
\frac{\partial}{\partial H_t(a)} \sum_{a'\in [k]} \pi_t(a') \rho(a')
\\
&=
\sum_{a'\in [k]} \rho(a') \frac{\partial}{\partial H_t(a)} \pi_t(a')  
\end{align}
$$

Notice that $$\sum_{a'}\pi_t(a') = 1$$, so the derivative of the sum is $$0$$.
Hence, we can add any constant to the sum. In addition we can multiply the
$$a'$$-th term by $$\pi_t(a')/\pi_t(a')$$.

$$
\begin{align}
\star
&=
\sum_{a'\in [k]} 
(\rho(a') - B_t) 
\frac{\partial}{\partial H_t(a)} \pi_t(a')
\frac{\pi_t(a') }{\pi_t(a')}
\\
&=
\sum_{a'\in [k]} 
\pi_t(a')
\left[
    (\rho(a') - B_t) 
    \frac{\partial}{\partial H_t(a)} \pi_t(a')
    \frac{1}{\pi_t(a')}
\right]
\end{align}
$$

It can be shown that 

$$
\frac{\partial}{\partial H_t(a)} \pi_t(a') \frac{1}{\pi_t(a')}
=
\mathbf{1}_{[a = a']} - \pi_t(a)
$$

Thus

$$
\begin{align}
\star
&=
\sum_{a'\in [k]} 
\pi_t(a')
\left[
    (\rho(a') - B_t) 
    (\mathbf{1}_{[a = a']} - \pi_t(a))
\right]
\\
&=
-\pi_t(a) \sum_{a'\in [k]} \pi_t(a') (\rho(a') - B_t) 
+
\pi_t(a)(\rho(a) - B_t) 
\end{align}
$$



If the reward $$R_t$$ is high, we want to increase the probability of
taking action $$a$$, namely to increase $$H_t(a)$$. But what counts
as "high" reward?

We can keep a record of the average reward we received so far,
which can be computed exacly or incrementally with some choice
of step size $$\alpha$$. 

We update according to the rule

$$
\begin{align}
H_{t+1}(A_t) 
&= H_t(A_t) + \alpha (R_t - \overline{R}_t)(1- \pi_t(A_t))
\end{align}
$$