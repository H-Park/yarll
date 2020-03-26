import gym
import numpy as np
from gym.spaces import Discrete


class MIMOEnv(gym.Env):
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self):
        self.action_space = [Discrete(3), Discrete(5)]

        self.observation_shape = [(1, 10, 10), (3, 5, 5)]
        self.observation_space = [gym.spaces.Box(low=0, high=1, shape=self.observation_shape[0], dtype=np.float16),
                                  gym.spaces.Box(low=0, high=1, shape=self.observation_shape[1], dtype=np.float16)]

        self.counter = 0
        self.valid_actions = [[1, 1, 1], [1, 1, 1, 1, 1]]

    def reset(self):
        self.counter = 0
        self.valid_actions = [[1, 1, 1], [1, 1, 1, 1, 1]]
        return self.state()

    def step(self, action: int):
        valid_actions = [[1, 1, 1], [1, 1, 1, 1, 1]]
        if self.valid_actions[action] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self.valid_actions, action))
        valid_actions[action] = 0

        self.counter += 1
        self.valid_actions = [[1, 1, 1], [1, 1, 1, 1, 1]]
        return self.state(), 0, self.finish(), {'action_mask': self.valid_actions}

    def render(self, mode='human'):
        pass

    def finish(self):
        return self.counter == 250

    def state(self):
        tmp = np.reshape(np.array([*range(100)]), self.observation_shape)
        obs = tmp / 100
        return obs