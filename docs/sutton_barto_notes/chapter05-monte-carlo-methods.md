---
layout: default
title: "Chapter 5: Monte Carlo Methods"
---

# {{ page.title }}


## Monte Carlo Prediction

Our goal is to evaluate $v_{\pi}(s)$, i.e. the expected discounted
return from state $s$ following the policy $\pi$.

Assume we do not know the environment's dynamics, but we
can interact with it and gather samples. Assume the
environment is episodic.

Two Monte-Carlo (MC) methods:
1. First-visit: evaluate $v_\pi(s)$ only from
the first time it is visited in an episode
2. Every-vist: evaluate $v_\pi(s)$ only from every time it
is visited during the episode.


### Algorithm: First-visit MC Policy Evaluation
```
Input: a policy $\pi$ to be evaluated
Initialize:
    $V\colon \mathcal{S}\to\RR$ - arbitrary for every $s\in\mathcal{S}$
    $Returns\colon \mathcal{S} \to []$ - an empty list for every $s\in\mathcal{S}$

Loop forevery:
    Generate an episode:
        $s_0,a_0,r_1,s_1,a_1,\dots,s_{T-1},a_{T-1},r_{T}$
    $g \leftarrow 0$
    For $t=T-1,T-2,\dots,1$:
        $g \leftarrow \gamma g + r_{t+1}$
        If $s_t \notin \{s_{0},\dots,s_{t-1\}}$:
            $Returns(s_t)$.append(g)
            $V(s_t) \leftarrow Average(Returns(s_t))$
```

$V(s)$ converges to $v_{\pi}(s)$ with standard deviation error
$1/\sqrt{n_s}$, where $n_s$ is the number of times $s$
is visited.

The every-visit method converges at the same rate but the 
analysis is more complicated.


### Comparison: MC vs. DP

- Independence between states: in DP, the value of at each
state is computed based on other states, while in MC the value
of each state is completely independent of other states
- Bootsrapping: in DP bootstrapping is possible, in MC it is not.
- Number of steps: In DP is like BFS, MC is like DFS


## MC Action-Value Estimation

If the dynamics are unknown, in order to choose good actions
we need to evaluate the actions' returns, i.e. estimate $q_{*}$.
This is the similar to the state-value evaluation, except
we estimate state-action pairs instead of states.

## MC Control

Similarly to the DP approach, also in MC
we can alternate between 
- estimating $q_{\pi}$ for a given policy $\pi$, and
- improving by $\pi'(s) = \arg\max_a q_{\pi}(s,a)$
Every step can either be carried out until convergence,
or in smaller steps, e.g. per episode.

Since here we learn from experience it is crucial to
maintain exploration continuously.

### Algorithm: MC with ES (Exploring Starts) for estimating $\pi_*$
```
Initialize:
    $\pi\colon \mathcal{S}\to\mathcal{A}$ arbitrarily
    $Q\colon \mathcal{S}\times \mathcal{A}\to\RR$ arbitrarily
    $Retruns\colon \mathcal{S}\times \mathcal{A}\to []$ empty list for every $(s,a)$

Loop forever:
    Sample $s_0,a_0$ randomly so that every $(s,a)$ have positive probability
    Generate an episode following $\pi$:
        $s_0,a_0,r_1,s_1,a_1,\dots,s_{T-1},a_{T-1},r_{T}$
    $g\leftarrow 0$
    For $t=T-1,...,0$:
        $g\leftarrow \gamma g + r_{t+1}$
        If $(s_t,a_t)\notin \{(s_0,a_0),\dots,(s_{t-1},a_{t-1})\}$:
            $Returns(s_t,a_t)$.append($g$)
            $Q(s_t,a_t)\leftarrow Average(Returns(s_t,a_t))$
            $\pi(s_t) \leftarrow \arg\max_a Q(s_t,a)$
```

*An open question*: does this algorithm converge to the optimal policy?
It is conjectured so, but has not yet been proven.

### Exercise 5.4

A more efficient algorithm:
```
Parameters:
    $\alpha \in (0,1]$ - step size

Initialize:
    $\pi\colon \mathcal{S}\to\mathcal{A}$ arbitrarily
    $Q\colon \mathcal{S}\times \mathcal{A}\to\RR$ arbitrarily

Loop forever:
    Sample $s_0,a_0$ randomly so that every $(s,a)$ have positive probability
    Generate an episode following $\pi$:
        $s_0,a_0,r_1,s_1,a_1,\dots,s_{T-1},a_{T-1},r_{T}$
    $g\leftarrow 0$
    For $t=T-1,...,0$:
        $g\leftarrow \gamma g + r_{t+1}$
        If $(s_t,a_t)\notin \{(s_0,a_0),\dots,(s_{t-1},a_{t-1})\}$:
            $Q(s_t,a_t)\leftarrow Q(s_t,a_t) + \alpha(g - Q(s_t,a_t))$
            $\pi(s_t) \leftarrow \arg\max_a Q(s_t,a)$
```


The *exploring starts* assumption can be removed by 
forcing the policy to explore, e.g. by making it
$\ve$-greedy.

*Lemma*. The $\ve$-greedy policy is the best among all $\ve$-soft policies
(where every $(s,a)$ has probability $\geq \ve/\mathcal{A}(s)$). \\
*Proof*. Let $\pi'$ be the $\ve$-greedy policy, and $\pi$ some
$\ve$-soft policy.

$$
\begin{align}
    q_{\pi}(s,\pi'(s))
    &=
    \sum_{a}\pi'(a\mid s) q_{\pi}(s,a)
    \\
    &=
    \frac{\ve}{\mathcal{A}(s)}\sum_{a} q_{\pi}(s,a)
    + (1-\ve)\max_{a} q_{\pi}(s,a)
    \\
    &\geq
    \frac{\ve}{\mathcal{A}(s)}\sum_{a} q_{\pi}(s,a)
    + 
    (1-\ve) \sum_{a} 
    \frac
    {\pi(a\mid s) - \frac{\ve}{\mathcal{A}(s)}}
    {(1-\ve)}
    q_{\pi}(s,a)
    \\
    &=
    \sum_{a} \pi(a\mid s) q_{\pi}(s,a)
    \\
    &=
    v_{\pi}(s)
\end{align}
$$

Hence, by the policy improvement theorem, $\pi' \geq \pi$. 
$\square$

It follows that the above algorithm converges to the 
optimal $\ve$-soft policy.


## Off-Policy Prediction via Importance Sampling

How can one policy learn from the experience 
of another policy?

We start from a simpler problem: 
evaluate a policy $\pi$ based on experience 
of another policy $\pi'$,
where both $\pi,\pi'$ are fixed.

For this to be possibe, we need the following assumption:

Coverage assumption
: Every $(s,a)$ that $\pi$ visits is also visited by $\pi'$,
i.e. if $\pi(a\mid s)>0$ then $\pi'(a\mid s)>0$.

For off-policy evaluation we use a technique called *importance sampling*.

Importance Sampling
: A technique for estimating expected values under one distribution,
given samples from another distribution.

We use it as follows. The probability of a 
trajectory (i.e., a state-action sequence)
starting from state $S_t$ under policy $\pi$ is

$$
\begin{multline}
    \Pr(A_t,S_{t+1},A_{t+1},\dots,S_T \mid S_t, A_{i}\sim \pi)
    =
    \\
    = 
    \pi(A_t\mid S_t)p(S_{t+1}\mid S_t, A_{t})
    \pi(A_{t+1}\mid S_{t+1})p(S_{t+2}\mid S_{t+1}, A_{t+1})
    \cdots
    \\
    = 
    \prod_{k=t}^{T-1}\pi(A_k \mid S_k) p(S_{k+1} \mid S_{k}, A_{k})
\end{multline}
$$

Thus, the relative probability of the trajectory under the
*target* and *behavior* policies is

$$
    \rho_{t\colon T-1} 
    =
    \frac
    {\prod_{k=t}^{T-1}\pi(A_k \mid S_k) p(S_{k+1} \mid S_{k}, A_{k})}
    {\prod_{k=t}^{T-1}\pi'(A_k \mid S_k) p(S_{k+1} \mid S_{k}, A_{k})}
    =
    \frac
    {\prod_{k=t}^{T-1}\pi(A_k \mid S_k)}
    {\prod_{k=t}^{T-1}\pi'(A_k \mid S_k)}
$$

We can evaluate $v_\pi$ by

$$
\begin{align}
    v_{\pi'}(s) &= \EE[G_t \mid S_t=s]
    \\
    v_{\pi}(s) 
    &= 
    % \EE_{\pi'}[G_t \mid S_t=s]
    % \\
    % \sum_{trajectory} \Pr(trajectory \mid S_t = s, A_k\sim \pi)
    % \\
    % \sum_{trajectory} \prod_{}
    % \\
    % &=
    % \EE_{\pi'}[G_t \mid S_t=s]
    % \\
    \EE[ \rho_{t\colon T-1}\cdot G_t \mid S_t=s]
\end{align}
$$


For convenience, assume the episodes are numbered continuously, so
if an episode ends at time $t$, the next one begins at time $t+1$.
Let
- $\mathcal{T}(s)$ - the time steps where $s$ was visited (first/every)
- $T(t)$ - the time of first termination after time $t$
- $\{G_t\}_{t\in \mathcal{T}(s)}$ - the returns from state $s$
- $\{\rho_{t\colon T(t)-1}\}_{t\in \mathcal{T}(s)}$ - importance ratios

There are two methods to estimate $v_{\pi}(s)$:
- Ordinary importance sampling:

    $$
    V(s) 
    =
    \frac
    {
        \sum_{t\in \mathcal{T}(s)}
        \rho_{t\colon T(t)-1} G_t
    }
    {
        \lvert \mathcal{T}(s)\rvert
    }
    $$

    Produces an unbiased estimator, but with possibly unbounded variance

- Weighted importance sampling:

    $$
    V(s) 
    =
    \frac
    {
        \sum_{t\in \mathcal{T}(s)}
        \rho_{t\colon T(t)-1} G_t
    }
    {
        \sum_{t\in \mathcal{T}(s)}
        \rho_{t\colon T(t)-1}
    }
    $$

    Biased towards $v_{\pi'}$, but the bias tends to $0$, 
    and the variance is small.


### Exercise 5.5

Since there is only one state-action pair, there
exists only one, deretministic policy. 
Let $s_0$ be the non-terminal state, and $s_1$ the terminal state.
Let $r = +1$, $T = 10$.

The probability
of the given trajectory, from time $t\in \{0,1,\dots,T-1\}$,
is

$$
\begin{multline}
\Pr(S_{t+1} = s_0, S_{t+2}=s_0,\dots, S_{T-1}=s_0, S_T=s_1 \mid S_t = s_0)=
\\
=
p^{T-1-t}(1-p)
\end{multline}
$$

and the return $G_t = T-t$. 

First visit:

$$
V(s_0) = p^{T-1}(1-p) T
$$

Every visit:

$$
\begin{align}
V(s_0) 
&= 
\sum_{t=0}^{T-1}p^{T-1-t}(1-p)(T-t)
\\
&= 
(1-p)\sum_{t=0}^{T-1} \frac{d}{dp} p^{T-t}
\\
&= 
(1-p)\frac{d}{dp} \sum_{t=1}^{T}  p^{t}
\\
&= 
(1-p)\frac{d}{dp} \frac{p(1 - p^{T})}{1-p}
\\
&= 
(1-p)
\frac{
    (1 - (T+1)p^T)(1-p) + p(1-p^{T})
}
{(1-p)^2}
\\
&=
1 - (T+1)p^{T}
+
\frac{p(1-p^T)}{1-p}
\end{align}
$$


### Incremental Implementation

- Ordinary improtance sampling: the previous methods
work fine.
- Weighted importance sampling: we need to keep track
of the weights. Let $G_1,G_2,\dots$ be the returns
from different episodes for some state $s$, and
let $W_1,W_2,\dots$ be the importance ratios.
Let $C_1,C_2,\dots$ be the cumulative sums of the
$W_i$'s, i.e. $C_n = \sum_{1 \leq i\leq n} W_i$

$$
\begin{align}
V_n
&=
\frac
{\sum_{i=1}^{n} W_i G_i}
{\sum_{i=1}^{n} W_i}
\\
&=
\frac
{\sum_{i=1}^{n-1} W_i G_i}
{C_n}
+
\frac
{W_n G_n}
{C_n}
\\
&=
\frac{C_{n-1}}{C_n} V_{n-1}
+
\frac{W_n}{C_n} G_n
\\
&=
\frac{C_{n-1}}{C_n} V_{n-1}
+
\frac{W_n}{C_n} (G_n - V_{n-1})
+ \frac{W_n}{C_n} V_{n-1} 
\\
&=
V_{n-1}
+
\frac{W_n}{C_n} (G_n - V_{n-1})
\end{align}
$$


### Algorithm: off-policy MC prediction for estimating $Q \approx q_{\pi}$

```
Input: target policy $\pi$
Initialize: for all $a\in \mathcal{A}$, $s\in \mathcal{S}$:
    $Q(s,a) \in \RR$ - arbitrary
    $C(s,a) \leftarrow 0 $ 
    
Loop forever (for each episode):
    $b \leftarrow $ some $\ve$-soft policy covering $\pi$
    Generate an episode following $b$:
        $s_0,a_0,r_1,s_1,a_1,\dots,s_{T-1},a_{T-1},r_T$
    $G \leftarrow 0$
    $W \leftarrow 1$
    For $t = T-1,T-2,\dots,0$, while $W \neq 0$:
        $G\leftarrow \gamma G + r_{t+1}$
        $C(s_t,a_t) \leftarrow C(s_t,a_t) + W$
        $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \frac{W}{C(s_t,a_t)}(G - Q(s_t,a_t))$
        $W \leftarrow \frac{\pi(a_t\mid s_t)}{b(a_t\mid s_t)}\cdot W$
```


### Algorithm: off-policy MC control for estimating $\pi \approx \pi_{*}$

Here we take $\pi$ as the greedy policy with respect to $Q$, which 
at infinity tends to $$q_{\pi_{*}}$$ and hence $$\pi \to \pi_{*}$$.

A disadvantage of this algorithm is that it usually learns 
only from tails of episodes, hence the convergence can be 
very slow.


```
Initialize: for all $a\in \mathcal{A}$, $s\in \mathcal{S}$:
    $Q(s,a) \in \RR$ - arbitrary
    $C(s,a) \leftarrow 0 $ 
    $\pi(s) \leftarrow \arg\max_{a}Q(s,a)$
    
Loop forever (for each episode):
    $b \leftarrow $ some $\ve$-soft policy covering $\pi$
    Generate an episode following $b$:
        $s_0,a_0,r_1,s_1,a_1,\dots,s_{T-1},a_{T-1},r_T$
    $G \leftarrow 0$
    $W \leftarrow 1$
    For $t = T-1,T-2,\dots,0$, while $W \neq 0$:
        $G\leftarrow \gamma G + r_{t+1}$
        $C(s_t,a_t) \leftarrow C(s_t,a_t) + W$
        $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \frac{W}{C(s_t,a_t)}(G - Q(s_t,a_t))$
        $\pi(s_t) \leftarrow \arg\max_{a}Q(s_t,a)$
        If $a_t \neq \pi(s_t)$:
            break (continue to the next episode)
        $W \leftarrow \frac{1}{b(a_t\mid s_t)}\cdot W$
```