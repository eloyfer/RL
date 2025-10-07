---
layout: default
title: "Chapter 2: Multi-Armed Bandits"
---


Consider the following problem: 
there is only 1 state with $k$ possible actions.

Let 
- $A_t\in[k]$ the action taken at step $t$
- $R_t$ the reward received

Let $q_*\colon[k]\to\R$ the expected reward for each action,
$$
    q_*(a) = \mathbb{E}[R_t \mid A_t = a]
$$
