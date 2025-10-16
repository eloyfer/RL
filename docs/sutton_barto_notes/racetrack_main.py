
import numpy as np
import racetrack
from racetrack import RaceTrackEnv, GamePlay, plot_track, RaceTrackRandomPolicy, \
                    OffPolicyMCControlLearner, off_polilcy_MC_control

from tqdm import tqdm


N = 3
k = 1
track = np.zeros([N,N], dtype=np.int32)
track[0] = 1
track[:,-1] = 2
track[:k,N-k:] = -1

plot_track(track, show=True)

env = RaceTrackEnv(track)
gp = GamePlay(env)

learner = OffPolicyMCControlLearner(env)
for i in range(1,1001):
    learner.learn_episode()
    if i % 100 == 0:
        print(i, sum(len(vals) for vals in learner.Q.values()))

for key,vals in learner.Q.items():
    print(key)
    print(vals)

learner.gp.play_episode(learner.policy, render=True)


# import pdb; pdb.set_trace()

# # policy = RaceTrackRandomPolicy()
# # gp.play_episode(policy, render=False)
# policy = off_polilcy_MC_control(env)
# gp = GamePlay(env)
# gp.play_episode(policy, render=True)
