---
layout: default
title: "Chapter 3: Finite Markov Decision Processes"
---

# {{ page.title }}

Let 
- $\mathcal{S}$ - the state space
- $\mathcal{A}$ - the action space
- $\mathcal{R}$ - the reward space

A Markov Decision Process (MDP) a sequence of random variables
1. Environemnet presents a state $S_{t}\in \mathcal{S}$
2. Agent performs an action $A_{t} \in \mathcal{A}$
3. Environment presents a reward $R_{t+1}\in \mathcal{R}$ and the next state $S_{t+1}$

*Markov property*
: The future depends only on the current state, and not on previous states. 

Assuming Markov property, the dynamics of the MDP is 
described by the function 

$$
\begin{align}
    & p\colon \mathcal{S}\times \mathcal{R}\times \mathcal{S}\times\mathcal{A}\to[0,1]
    \\
    & p(s',r \mid s,a) = \Pr(S_t = s', R_t = r \mid S_{t-1} = s, A_{t-1} = a)
\end{align}
$$

We distinguish two types of agent-environment interactions:
1. **Episodic**: the interaction has a terminal state, and is comprised of a sequence of episodes,

    $$
        \{(S_{t,1},A_{t,1},R_{t,1})\}_{0\leq t\leq T_1},
        \dots,
        \{(S_{t,N},A_{t,N},R_{t,N})\}_{0\leq t\leq T_N}
    $$

    By convention, the reward after an episode terminates is set to $0$: 
    
    $$R_{t,i} = 0 \quad\text{if}\quad t > T_{i}$$

2. **Continuous**: the interaction continues forever, $\{(S_{t},A_{t},R_{t})\}_{0\leq t\leq \infty}$

*Total Return*
: The return or total return is the 
sum of current and future rewrards, possibly discounted:

    $$
        G_t 
        = 
        \sum_{i\geq t+1} \gamma^{i-t-1} R_{i}
        =
        R_{t+1} + \gamma G_{t+1}
    $$

    where $\gamma \in [0,1]$ is the *discount rate* parameter.


## Policies and Value Functions

Policy
: A policy $\pi$ maps every state to a probability distribution over the actions.
$\pi(a\mid s)$ is the probability that action $a\in \mathcal{A}$ 
is chosen when in state $s\in \mathcal{S}$.

Notice that

$$
\begin{align}
    \EE_{\pi}[R_{t+1}\mid S_t] 
    &= 
    \sum_{a\in \mathcal{A}}\pi(a\mid S_t) \EE[R_t \mid S_t, a]
    \\
    &=
    \sum_{a\in \mathcal{A}}
    \pi(a\mid S_t)
    \sum_{s\in \mathcal{S}, r\in \mathcal{R}} 
    r\cdot
    p(s, r\mid S_t, a)
\end{align}
$$

Value Function
: The value function measures how "good" a state is, i.e. the expected return. 
This depends on the policy
and on the discount parameter.

The *state-value function for policy $\pi$*: 

$$
v_{\pi}(s)
=
\EE_{\pi}[G_t \mid S_t = s]
=
\EE_{\pi}
\left[
    \sum_{i=t+1}^{\infty} \gamma^{i-t-1} R_{i} \mid S_t = s
\right]
$$

The *action-value function for policy $\pi$*: estimates the return 
from the current state, if taking action $a$ and then acting
according to policy $\pi$:

$$
q_{\pi}(s,a)
=
\EE_{\pi}[G_t \mid S_t = s, A_t=a]
=
\EE_{\pi}
\left[
    \sum_{i=t+1}^{\infty} \gamma^{i-t-1} R_{i} \mid S_t = s, A_t=a
\right]
$$


Relations between $v_{\pi}(s)$ and $q_{\pi}(s,a)$:

$$
\begin{align}
    v_{\pi}(s) 
    &= 
    \EE_{a\sim \pi(a\mid S_t)} [q_{\pi}(s,a)]
    &&=
    \sum_{a\in \mathcal{A}} \pi(a\mid S_t)\cdot q_{\pi}(s,a)
    \\
    q_{\pi}(s,a)
    &=
    \EE_{s',r\sim p(s',r\mid s,a)}
    \left[
        r + \gamma\cdot v_{\pi}(s')
    \right]
    &&=
    \sum_{r\in \mathcal{R}, s'\in \mathcal{S}}
    p(s',r\mid s,a) 
    \left[
        r + \gamma\cdot v_{\pi}(s')
    \right]
\end{align}
$$


The **Bellman equation** for $v_{\pi}$: 

$$
\begin{align}
    v_{\pi}(s)
    &=
    \EE_{\pi}[G_t \mid S_t = s]
    \\
    &=
    \EE_{\pi}[R_t + \gamma G_{t+1} \mid S_t = s]
    \\
    &=
    \sum_{a,s',r} \pi(a\mid s) p(s',r\mid S_t,a) (r + \gamma v_{\pi}(s'))
\end{align}
$$

Note that the Bellman equation is linear in $v_\pi$.

The state-value function $v_{\pi}$ is the unique solution to
its Bellman equation.


### Grid World Example

```
Parameters:
- n,m - grid size
- P_1,...,P_k - spcial origin points
- P'_1,...,P'_k - spcial destination points
- R_1,...,R_k - rewards for special points

State space: S = {1,...,n}x{1,...,n}
Action space: A = {(1,0),(-1,0),(0,1),(0,-1)}
Transition rules:
Let 
    s = (i,j)
    s' = (i',j')
If s = P_i for some i in {1,...,k}:
    r = R_i
    s' = P'_i
Else if (i = 1 and a=(-1,0)) or
        (i = n and a=(1,0)) or
        (j = 1 and a=(0,-1)) or
        (j = m and a=(0,1)):
    r = -1
    s' = s
Else:
    r = 0
    s' = s + a
```

The Bellman equations for the random policy:

$$
v(i,j)
=
\begin{cases}
    R_k + \gamma v_{\pi}(P'_k) & (i,j) = P_k
    \\
    -\frac{1}{2} + \frac{\gamma}{4}(2v(1,1) + v(1,2) + v(2,1)) & (i,j) = (1,1)
    \\
    -\frac{1}{2} + \frac{\gamma}{4}(2v(m,n) + v(m-1,n) + v(m,n-1)) & (i,j) = (m,n)
    \\
    -\frac{1}{4} + \frac{\gamma}{4}(v(i,j) + v(i+1,j) + v(i,j+1), v(i,j-1)) & i=1, 2\leq j \leq n-1
    \\
    -\frac{1}{4} + \frac{\gamma}{4}(v(i,j) + v(i+1,j) + v(i-1,j), v(i,j+1)) & j=1, 2\leq i \leq m-1
    \\
    -\frac{1}{4} + \frac{\gamma}{4}(v(i,j) + v(i-1,j) + v(i-1,j), v(i,j+1)) & i=m, 2\leq j \leq n-1
    \\
    -\frac{1}{4} + \frac{\gamma}{4}(v(i,j) + v(i+1,j) + v(i-1,j), v(i,j-1)) & j=n, 2\leq i \leq m-1
    \\
    \frac{\gamma}{4}(v(i+1,j) + v(i-1,j), v(i,j+1), v(i,j-1)) & 2\leq i,j \leq m-1
\end{cases}
$$


### Exercise 3.17
The Bellman equation for $q_{\pi}(s,a)$ can be derived
either directly by recursion, or
from the Bellman equation for $v_{\pi}$ and the 
relation between $v_{\pi}$ and $q_{\pi}$.

$$
\begin{align}
q_{\pi}(s,a)
&=
\EE_{\pi}[G_t \mid S_{t}=s, A_{t}=a]
\\
&=
\EE_{\pi}[R_{t+1} +\gamma \cdot G_{t+1} \mid S_{t}=s, A_{t}=a]
\\
&=
\sum_{s',r}p(s',r\mid s,a)
\left[r + \gamma\cdot \EE_{\pi}[G_{t+1} \mid S_{t+1}=s']\right]
\\
&=
\sum_{s',r}p(s',r\mid s,a)
\left[
    r + 
    \gamma\cdot 
    \sum_{a'\in \mathcal{A}}
    \pi(a'\mid s')
    \EE_{\pi}[G_{t+1} \mid S_{t+1}=s',A_{t+1}=a']
\right]
\\
&=
\sum_{s',r}p(s',r\mid s,a)
\left[
    r + 
    \gamma\cdot 
    \sum_{a'\in \mathcal{A}}
    \pi(a'\mid s')
    q_{\pi}(s',a')
\right]
\\
&=
\EE[R_{t+1}\mid S_t = s, A_{t}=a]
+
\gamma\cdot
\EE_{s',r\sim p(s',r\mid s,a)}
\left[
    \sum_{a'\in \mathcal{A}}
    \pi(a'\mid s')
    q_{\pi}(s',a')
\right]
\end{align}
$$


## Optimal Policies and Optimal Value Functions

In finite MDPs, we can define a partial order over the policies:

$$
    \pi \geq \pi'
    \iff
    v_{\pi}(s) \geq v_{\pi'}(s) \quad \forall s\in \mathcal{S}
$$

A policy is *optimal* if it is better than all other policies.
We denote such a policy by $\pi_{*}$ (it might not be unique).

*Claim*. there exists an optimal policy.  
*Proof*. TODO

We define the optimal state-value function and optimal action-value function:

$$
\begin{align}
    v_*(s) &= \max_{\pi} v_{\pi}(s) && \forall s\in\mathcal{S}
    \\
    q_*(s,a) &= \max_{\pi} q_{\pi}(s,a) && \forall s\in\mathcal{S},a\in\mathcal{A}
\end{align}
$$


### Bellman optimality equation

The optimal state-value function assumes the best action is taken
from each state, hence it can be expressed using the optimal 
action-value function:

$$
\begin{align}
    v_{*}(s)
    &=
    \max_{a} q_{*}(s,a)
    \\
    &=
    \max_{a} \EE_{\pi_{*}} [G_t \mid S_{t}=s, A_{t}=a]
    \\
    &=
    \max_{a} \EE_{\pi_{*}} [R_{t+1} + \gamma \cdot G_{t+1} \mid S_{t}=s, A_{t}=a]
    \\
    &=
    \max_{a} \EE_{\pi_{*}} [R_{t+1} + \gamma \cdot v_*(S_{t+1}) \mid S_{t}=s, A_{t}=a]
    \\
    &=
    \max_{a} 
    \sum_{s',r}p(s',r\mid s,a)\cdot [r + \gamma \cdot v_{*}(s')]
\end{align}
$$

For the action-value function:

$$
\begin{align}
    q_{*}(s,a)
    &=
    \EE
    \left[
        R_{t+1} + \gamma \cdot \max_{a'}q_{*}(S_{t+1},a') 
        \mid 
        S_{t} = s, A_{t} = 1
    \right]
    \\
    &=
    \sum_{s',r}
    p(s',r\mid s,a)\cdot 
    [r + \gamma\cdot \max_{a'}q_{*}(s',a') ]
\end{align}
$$

The Bellman optimality equation has a unique solution.
If the dynamics of the MDP are known, i.e. $p(s',r\mid s,a)$,
it is possible to find the solution. 
However, the equations are non-linear.

Once a solution is found, it is easy to define an optimal policy
$\pi_{*}$, by taking a greedy step from every state.
That is, choosing an action $a$ which maximizes
the optimality equation.