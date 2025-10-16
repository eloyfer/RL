from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("error")



def plot_track(track, fig=None, ax=None, show=False):
    # track = track.T
    if fig is None:
        m,n = track.shape
        figscale = 5
        figsize = (figscale, figscale * n/m)
        fig, ax = plt.subplots(figsize=figsize)
    if ax is None:
        ax = fig.get_axes()[0]
    X, Y = np.nonzero(track == 0)
    ax.scatter(X, Y, marker='s', color="white", edgecolors='k', s=100)
    X, Y = np.nonzero(track == 1)
    ax.scatter(X, Y, marker='o', color='r')
    X, Y = np.nonzero(track == 2)
    ax.scatter(X, Y, marker='o', color='g')
    # ax.axis('off')
    if show:
        plt.show(block=False)
    else:
        return fig, ax
    # plt.close(fig)

def get_intersecting_tiles(track, pos0, pos1, visualize=False):
    x0, y0 = pos0
    x1, y1 = pos1

    # compute line equation a*x + b*y + c = 0
    if x0 == x1:
        a = 1
        b = 0
        c = -x0
    elif y0 == y1:
        a = 0
        b = 1
        c = -y0
    else:
        a = y1-y0
        b = -(x1-x0)
        c = -(y1*b + x1*a)

    # print("line equation:")
    # print(f"{a}*x + {b}*y + {c} = 0")
    
    # find intersecting squares
    X, Y = np.meshgrid(range(x0,x1+1), range(y0,y1+1))
    dist_to_line = np.abs(a * X + b * Y + c) / (a**2 + b**2)
    l_inf_dist = np.maximum(np.abs(a*dist_to_line), np.abs(b*dist_to_line))
    intersecting_idx = l_inf_dist <= 0.5
    Xi = X[intersecting_idx]
    Yi = Y[intersecting_idx]

    if visualize:

        fig, ax = plot_track(track)
        # plot trajectory
        X, Y = np.meshgrid(range(x0,x1+1), range(y0,y1+1))
        ax.scatter(X, Y, marker='o', color='yellow')
        ax.arrow(x0,y0, x1-x0, y1-y0)
        ax.scatter(Xi, Yi, marker='o', color='blue')

        plt.show()
    
    return Xi, Yi




ACTIONS = list(product([-1,0,1],[-1,0,1]))


class State:

    def __init__(self, px, py, vx, vy):
        self.data = tuple(map(int,(px, py, vx, vy)))
    
    def get_position(self):
        return self.data[:2]

    def get_position_array(self):
        return np.array(self.get_position(), dtype=np.int32)
    
    def get_velocity(self):
        return self.data[2:]

    def get_velocity_array(self):
        return np.array(self.get_velocity(), dtype=np.int32)

    # def set_position(self, px, py):
    #     self.data = (px, py) + self.get_velocity()
    
    # def set_velocity(self, vx, vy):
    #     self.data = self.get_posistion() + (vx, vy)
    
    # def __call__(self):
    #     return self.data

    def __repr__(self):
        return str(self.data)
    
    def __hash__(self):
        return hash(self.data)

    def __eq__(self, other):
        return other.data == self.data
    
    # def copy(self):
    #     return State(*self.data)

class RaceTrackEnv:

    def __init__(self, track, seed=None):
        self.track = track
        self.start_positions = list(zip(*np.nonzero(track == 1)))
        self.finish_positions = list(zip(*np.nonzero(track == 2)))
        self.height = track.shape[0]
        self.width = track.shape[1]
        self.rng = np.random.default_rng(seed)
        self.velocity = None
        self.position = None
        self.end_state = True
        self.start_state()
    
    def set_position(self, x, y):
        self.position = np.array([x,y], dtype=np.int32)
    
    def set_velocity(self, vx, vy):
        self.velocity = np.array([vx,vy], dtype=np.int32)

    def start_state(self):
        self.set_velocity(0,0)
        px, py = self.rng.choice(self.start_positions)
        self.set_position(px, py)
        self.end_state = False
    
    def episode_end(self):
        return self.end_state
    
    @staticmethod
    def project_state(state, action):
        pos = state.get_position_array()
        vel = state.get_velocity_array()
        action = np.array(action, dtype=np.int32)
        new_vel = vel + action
        if new_vel.min() < 0 or new_vel.max() > 5 or new_vel.sum() <= 0:
            new_vel = vel
        new_pos = pos + new_vel
        return State(new_pos[0], new_pos[1], new_vel[0], new_vel[1])
        
    def next_state(self, action):
        assert len(action) == 2
        assert set(action) <= set([-1,1,0])

        proj_st = RaceTrackEnv.project_state(self.get_state(), action)

        new_pos = proj_st.get_position_array()

        Xi, Yi = get_intersecting_tiles(self.track, self.position, new_pos)

        intersect_tiles = list(zip(Xi,Yi))
        def dist_to_current_pos(p1):
            p0x, p0y = self.position
            p1x, p1y = p1
            return ((p1x-p0x)**2 + (p1y-p0y)**2)**(1/2)
        intersect_tiles.sort(key=dist_to_current_pos)
        
        for x,y in intersect_tiles:
            
            if x >= self.height or y >= self.width:
                self.start_state()
                return -1
            
            elif (x,y) in self.finish_positions:
                self.end_state = True
                self.set_position(x,y)
                return 0
            
            elif self.track[x,y] == -1:
                self.start_state()
                return -1

        else:
            x, y = new_pos
            self.set_position(x,y)
            vx, vy = proj_st.get_velocity()
            self.set_velocity(vx, vy)
            return -1
        
    def get_state(self):
        px, py = self.position
        vx, vy = self.velocity
        return State(px, py, vx, vy)

    def render_init(self):
        plt.ion()
        self.fig, self.ax = plot_track(self.track)
        self.artists = []
        self.cur_line, = self.ax.plot([],[], "x", color="blue")
        self.lines = []
        plt.show()
    
    def render_update(self, st, at):
        """
        If velocity is positive:
        - Move the current position
        - Add new projected path
        - Recolor previous projected paths
        """

        p0x, p0y = st.get_position()
        self.cur_line.set_xdata([p0x])
        self.cur_line.set_ydata([p0y])

        proj_st = RaceTrackEnv.project_state(st, at)
        if sum(proj_st.get_velocity()) > 0:
            p1x, p1y = proj_st.get_position()
            new_line, = self.ax.plot([p0x, p1x], [p0y, p1y], zorder=0)
            self.lines.append(new_line)

            # update colors
            L = len(self.lines)
            cmap = colormaps.get_cmap('viridis')
            for i, line in enumerate(self.lines):
                line.set_color(cmap((i+0.5)/L))
        plt.pause(0.2)
    
class EpisodeStep:

    def __init__(self, reward, state, action):
        self.action = action
        self.reward = reward
        self.state = state
    
    def __repr__(self):
        return f"rt={self.reward} st={self.state}, at={self.action}"
    
    def __hash__(self):
        hash((self.state, self.action))

class RaceTrackPolicy:

    def __call__(self, state):
        raise NotImplementedError()

class RaceTrackRandomPolicy(RaceTrackPolicy):

    def __call__(self, state):
        return [1/len(ACTIONS)] * len(ACTIONS)

class RaceTrackGreedyPolicy(RaceTrackPolicy):

    def __init__(self):
        self.state_action = dict()
    
    def set_action(self, state, action):
        self.state_action[state] = action
    
    def __call__(self, state):
        p = np.zeros(len(ACTIONS), dtype=np.float32)
        a = self.state_action.get(state, (0,0))
        ai = ACTIONS.index(a)
        p[ai] = 1
        return p

class RaceTrackEpsilonGreedyPolicy(RaceTrackGreedyPolicy):

    def __init__(self, eps=0):
        self.eps = eps
        super().__init__()
    
    def set_eps(self, eps):
        self.eps = eps
    
    def __call__(self, state):
        p = super().__call__(state)
        p = p * (1 - self.eps) + self.eps / len(p)
        return p

class GamePlay:

    def __init__(self, env, seed=None):
        self.rng = np.random.default_rng(seed)
        self.env = env
        self.cur_episode = []
    
    def render_init(self):
        plt.ion()
        self.fig, self.ax = plot_track(self.env.track)
        self.artists = []
        self.cur_line, = self.ax.plot([],[], "x", color="blue")
        self.lines = []
        plt.show()
    
    def render_update(self, st, at):
        """
        If velocity is positive:
        - Move the current position
        - Add new projected path
        - Recolor previous projected paths
        """

        print(self.cur_episode[-1])

        p0x, p0y = st.get_position()
        self.cur_line.set_xdata([p0x])
        self.cur_line.set_ydata([p0y])

        proj_st = RaceTrackEnv.project_state(st, at)
        if sum(proj_st.get_velocity()) > 0:
            p1x, p1y = proj_st.get_position()
            new_line, = self.ax.plot([p0x, p1x], [p0y, p1y], zorder=0)
            self.lines.append(new_line)

            # update colors
            L = len(self.lines)
            cmap = colormaps.get_cmap('viridis')
            for i, line in enumerate(self.lines):
                line.set_color(cmap((i+0.5)/L))
        
        plt.pause(0.2)
    
    def play_episode(self, policy, render=False):

        self.cur_episode = []

        self.env.start_state()

        if render:
            self.render_init()

        episode = []
        rt = None
        while not self.env.episode_end():
            st = self.env.get_state()
            at = ACTIONS[self.rng.choice(len(ACTIONS), p=policy(st))]
            self.cur_episode.append(EpisodeStep(rt, st, at))

            if render:
                self.render_update(st, at)
            
            rt = self.env.next_state(at)
        
        st = self.env.get_state()
        at = (0,0)
        self.cur_episode.append(EpisodeStep(rt, st, at))

        if render:
            self.render_update(st, at)
        
        plt.show(block=True)

        return self.cur_episode

def dict_argmax(d):
    best_key = None
    best_val = None
    for key,val in d.items():
        if best_val is None or val > best_val:
            best_key = key
            best_val = val
    return best_key


class OffPolicyMCControlLearner:

    def __init__(self, env, discount_factor=0.9):
        self.discount_factor = discount_factor
        self.Q = dict() # state*action -> value
        self.C = dict() # state*action -> value
        self.policy = RaceTrackEpsilonGreedyPolicy(eps=0)
        self.gp = GamePlay(env)
    
    def learn_episode(self, b_policy=None):
        if b_policy is None:
            b_policy = RaceTrackRandomPolicy()

        C = self.C
        Q = self.Q
        policy = self.policy

        episode = self.gp.play_episode(b_policy)
        G = 0
        W = 1
        T = len(episode)
        for t in range(T-2, -1, -1):
            G = episode[t+1].reward + self.discount_factor*G
            st = episode[t].state
            at = episode[t].action
            if st not in C:
                C[st] = dict()
            if st not in Q:
                Q[st] = dict()
            
            C[st][at] = C[st].get(at,0) + W
            # print(f'C[st][at]: {C[st][at]}')
            # if C[st][at] == 0:
            #     import pdb; pdb.set_trace()
            Qstat = Q[st].get(at, 0)
            Q[st][at] = Qstat + W/C[st][at] * (G - Qstat)
            policy.set_action(st, dict_argmax(Q[st]))

            p1 = policy(st)[ACTIONS.index(at)]
            p0 = b_policy(st)[ACTIONS.index(at)]
            
            if p1 == 0:
                break
            else:
                W = p1 / p0 * W
        
        self.Q = Q
        self.C = C
        self.policy = policy


def off_polilcy_MC_control(racetrack_env, discount_factor=0.9, seed=None):
    Q = dict() # state*action -> value
    C = dict() # state*action -> value
    policy = RaceTrackGreedyPolicy()

    gp = GamePlay(racetrack_env, seed)

    for _ in tqdm(range(10000)):
        b_policy = RaceTrackRandomPolicy()
        episode = gp.play_episode(b_policy)
        G = 0
        W = 1
        T = len(episode)
        for t in range(T-2, -1, -1):
            G = episode[t+1].reward + discount_factor*G
            st = episode[t].state
            at = episode[t].action
            if st not in C:
                C[st] = {a: 0 for a in ACTIONS}
            if st not in Q:
                Q[st] = {a: 0 for a in ACTIONS}
            # if st not in policy:
            #     policy.set_action(st, 0)
            
            C[st][at] += W
            Q[st][at] += W/C[st][at] * (G - Q[st][at])
            policy.set_action(st, dict_argmax(Q[st]))

            p1 = policy(st)[ACTIONS.index(at)]
            p0 = b_policy(st)[ACTIONS.index(at)]
            
            if p1 == 0:
                break
            else:
                W = p1 / p0 * W
    
    return policy

