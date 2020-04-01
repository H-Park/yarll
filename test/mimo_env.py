import gym
import numpy as np
from gym.spaces import Discrete


class MIMOEnv(gym.Env):
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self):
        self.action_space = Discrete(5)

        self.observation_space = Discrete(32)

        self.counter = 0
        self.action_mask = [1, 1, 1, 1, 1]

    def reset(self):
        self.counter = 0
        self.action_mask = [1, 1, 1, 1, 1]
        return self.state()

    def step(self, action: int):
        action_mask = [1, 1, 1, 1, 1]
        if self.action_mask[action] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self.action_mask, action))
        action_mask[action] = 0

        self.counter += 1
        self.action_mask = action_mask
        return self.state(), 0, self.finish(), {}

    def render(self, mode='human'):
        pass

    def finish(self):
        return self.counter == 250

    def state(self):
        tmp = np.reshape(np.array([*range(32)]), self.observation_space.n)
        obs = tmp / 100
        return obs
