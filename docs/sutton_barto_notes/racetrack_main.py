
import numpy as np
import racetrack
from racetrack import RaceTrackEnv, GamePlay, plot_track, RaceTrackRandomPolicy, \
                    OffPolicyMCControlLearner, off_polilcy_MC_control

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

learner = OffPolicyMCControlLearner(env, discount_factor=1)
learner.policy.set_eps(0.2)
for i in range(1,4001):
    learner.learn_episode(learner.policy)
    if i % 100 == 0:
        print(i, sum(len(vals) for vals in learner.Q.values()))

learner.policy.set_eps(0)
for i in range(5):
    learner.gp.play_episode(learner.policy, render=True)