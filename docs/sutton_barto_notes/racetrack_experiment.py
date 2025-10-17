import numpy as np
import racetrack
from racetrack import RaceTrackEnv, GamePlay, plot_track, RaceTrackRandomPolicy, \
                    OffPolicyMCControlLearner, off_polilcy_MC_control, policy_evaluation

from racetrack import ValueIterationLearner
from itertools import product
from copy import deepcopy
from tqdm import tqdm

N = 15
k = 7
track = np.zeros([N,N], dtype=np.int32)
track[0] = 1
track[:,-1] = 2
track[:k,N-k:] = -1

plot_track(track, show=True)

env = RaceTrackEnv(track)
gp = GamePlay(env)

pi_eps = [0] #, 0.2, 0.4]
b_eps = [1] # [0.3, 0.5, 0.7, 1]


vi_learner = ValueIterationLearner(env)
for i in range(50):
    vi_learner.learn(10)
    print(len(vi_learner.next_state_probs))
    mean, std = policy_evaluation(gp, vi_learner.policy, 15, 2000)
    print(mean, std)

experiment_data = {}
for eps1, eps2 in product(pi_eps, b_eps):
    print(eps1, eps2)
    cur_data = {"mean": [], "std": []}
    learner = OffPolicyMCControlLearner(env, discount_factor=1)
    learner.gp.set_render_pause(0.01)
    learner.policy.set_eps(0)
    T = []
    for i in range(1,15001):
        # b_policy = deepcopy(learner.policy)
        # b_policy.set_eps(eps2)
        b_policy = RaceTrackRandomPolicy()
        learner.learn_episode(b_policy)
        # learner.learn_episode(learner.policy)
        T.append(len(learner.gp.cur_episode))
        if i % 1000 == 0:
            print(i, sum(len(vals) for vals in learner.Q.values()))

            print("max episode length: ", max(T))

            learner.policy.set_eps(0)
            mean, std = policy_evaluation(learner.gp, learner.policy, 15, 2000)
            print(mean, std)
            cur_data["mean"].append(mean)
            cur_data["std"].append(std)
            # learner.policy.set_eps(0.2)

            # learner.gp.play_episode(learner.policy, render=True, early_stop=2000)


            # print("max Q:", max(max(vals.values()) for key,vals in learner.Q.items()))
    
    experiment_data[(eps1,eps2)] = cur_data

