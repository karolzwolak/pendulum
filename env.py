import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Env(gym.Env):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(sim.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )  # One float: direction & magnitude

    def reset(self, seed=None, options=None):
        self.sim.reset()
        obs = self.sim.state()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        # Action is a float in a 1D array
        force = float(action[0])
        obs, reward, done = self.sim.step(force)
        return np.array(obs, dtype=np.float32), reward, done, False, {}

    def state(self):
        return self.sim.state()
