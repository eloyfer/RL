---
layout: default
title: "Chapter 4: Dynamic Programming"
---

# {{ page.title }}

Dynamic Programmind (DP) is a collection of algorithms used
to solve RL tasks, i.e. find good policies. They are not 
practical but important for theoretical research.

## Iterative Policy Evaluation (Prediction)

Given a policy $\pi$ we wish to evaluate it, i.e. find its
value function $v_\pi$:

$$
v_{\pi}(s)
=
\EE_{\pi}[R_{t+1}+\gamma v_{\pi}(S_{t+1}) \mid S_t = s]
\quad \forall s\in \mathcal{S}
$$

Recall that $v_{\pi}$ exists if either $\gamma < 1$
or the environment is episodic.

If the environment dynamics are known, $v_\pi$ can be
obtained by solving the set of linear equations.
Alternatively, it can be found iteratively as follows:
fix some $v_0\colon \mathcal{S}\to \RR$, then update 

$$
\begin{align}
v_{k+1} 
&=
\EE_{\pi}[R_{t+1}+\gamma v_{k}(S_{t+1}) \mid S_t = s]
\quad \forall s\in \mathcal{S}
\\
&=
\sum_{a}\pi(a\mid s)\sum_{s',r} p(s',r\mid s,a) [r + \gamma v_k(s')] 
\end{align}
$$

Observe that $v_{\pi}$ is a fixed point. It can be
shown that $v_k\to v_{\pi}$ as $k\to\infty$, 
if $v_\pi$ exists.

This update rule is coined *expected update*, because
it is based on the expectation of the next state, rather
than a sample of the next state.


## Policy Improvement

In this subsection we consider deterministic policies for
simplicity, but the results extend to non-deterministic
policies as well.

*Policy improvement theorem*. Let $\pi,\pi'$ be two
deterministic policies such that 

$$
    q_{\pi}(s,\pi'(s)) \geq v_{\pi}(s)
    \quad
    \forall s\in \mathcal{S}
$$

Then, $v_{\pi'}(s) \geq v_{\pi}(s)$ for all $s\in \mathcal{S}$

In other words, if taking the action of $\pi'$ and continuing according
to $\pi$ leads to higher expected return, then the value of $\pi'$
is higher.

*Proof*. 

$$
\begin{align}
v_{\pi}(s)
&\leq
q_{\pi}(s,\pi'(s))
\\
&=
\EE[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s, A_t = \pi'(s)]
\\
&=
\EE_{\pi'}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s]
\\
&\leq
\EE_{\pi'}[R_{t+1} + \gamma q_{\pi}(S_{t+1}, \pi'(S_{t+1})) \mid S_t = s]
\\
&=
\EE_{\pi'}[R_{t+1} + \gamma  R_{t+2} + \gamma^2 v_{\pi}(S_{t+2})\mid S_t = s]
\\
&\leq
\EE_{\pi'}[R_{t+1} + \gamma  R_{t+2} + \gamma^2 R_{t+2}+\dots\mid S_t = s]
\\
&= 
v_{\pi'}(s)
\end{align}
$$

Given a policy $\pi$, we can define a new policy
$\pi'(s) = \arg\max q_{\pi}(s,a)$ for all $s\in\mathcal{S}$.
This algorithms is called *policy improvement*.

By the above theorem, $\pi'$ is at least as good as $\pi$. If
$\pi'$ does not improve $\pi$, namely $v_{\pi'}=v_{\pi}$,
then, $\pi$ is a solution to the Bellmans optimality equation,
and therefore optimal.

It follows that the policy improvement algorithm 
strcitly improves a policy, unless it is optimal.


## Policy Iteration

We can combine the two algorithms, policy evaluation
and policy improvement, and repeat them to iteratively
improve a policy. Since the MDP is finite and thus
has finitely many policies, the process converges
to the optimal policy.

### Algorithm: Policy iteration
1. Init:
    - $V\colon \mathcal{S}\to \RR$,
    - $\pi\colon \mathcal{S}\to\mathcal{A}$
2. Run policy evaluation: $V \leftarrow PolicyEval(\pi)$
3. Run policy improvement:
    - stab $\leftarrow$ True
    - For $s\in \mathcal{S}$:
        - $a_0 = \pi(s)$
        - $\pi(s) \leftarrow \arg\max_{a} \sum_{s',r} p(s',r\mid s,a) [r + \gamma V(s')]$
        - If $\pi(s)\neq a_0$:
            - stab $\leftarrow$ False
    - If stab:
        - Return $V,\pi$
    - Else:
        - Go to 2


## Value Iteration

The Bellman optimality equation can be taken as an update
rule to produce the sequence $v_0,v_1,\dots,$.
It merges the policy evaluation and policy improvement
into one step:

$$
v_{k+1}(s)
=
\max_{a} \EE[R_{t+1} + \gamma v_{k}(S_{t+1}) \mid S_{t} = s, A_t = a]
$$

It can be shown that if 
$v_{\*}$ 
exists then 
$v_{k}\to v_{\*}$
as 
$k\to\infty$.

Similarly, one can write a recurrence formula for $q_{k}(s,a)$:

$$
q_{k+1}(s,a) = 
\EE[R_{t+1} 
+
\gamma \max_{a'} q_{k}(S_{t+1},a')
\mid S_t = s, A_t = a]
$$


### Generalized Policy Iteration

The updates in the policy iteration need not be in
a fixed order, and the result converges to
optimal if every state is visited infinitely many times.

The evaluation and improvement steps can be seen as
competing: 
- evaluation may change the optimal action, making the policy not greedy
- improvement changes the policy and thus incur error in the value estimation

Using previous estimation of the value function to evaluate 
the new policy is called **bootstrapping**.