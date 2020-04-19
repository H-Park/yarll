import numpy as np
from typing import Optional
import torch

from gym import Env, Space
from gym.spaces import Discrete, MultiDiscrete, Box


class IdentityEnv(Env):
    def __init__(self,
                 dim: Optional[int] = None,
                 space: Optional[Space] = None,
                 ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param dim: the size of the action and observation dimension you want
            to learn. Provide at most one of `dim` and `space`. If both are
            None, then initialization proceeds with `dim=1` and `space=None`.
        :param space: the action and observation space. Provide at most one of
            `dim` and `space`.
        :param ep_length: the length of each episode in timesteps
        """
        if space is None:
            if dim is None:
                self.dim = 5
            space = Discrete(self.dim)
        else:
            assert dim is None, "arguments for both 'dim' and 'space' provided: at most one allowed"

        self.action_space = self.observation_space = space
        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def reset(self):
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state

    def step(self, action: torch.Tensor):
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _choose_next_state(self):
        self.state = torch.tensor(self.observation_space.sample(), dtype=torch.long)

    def _get_reward(self, action: torch.Tensor):
        return 1 if torch.all(self.state.eq(action)) else 0

    def render(self, mode='human'):
        pass


class IdentityEnvBox(IdentityEnv):
    def __init__(self, low=-1, high=1, eps=0.05, ep_length=100):
        """
        Identity environment for testing purposes

        :param low: (float) the lower bound of the box dim
        :param high: (float) the upper bound of the box dim
        :param eps: (float) the epsilon bound for correct value
        :param ep_length: (int) the length of each episode in timesteps
        """
        space = Box(low=low, high=high, shape=(1,), dtype=np.float32)
        super().__init__(ep_length=ep_length, space=space)
        self.eps = eps

    def step(self, action):
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _get_reward(self, action):
        return 1 if (self.state - self.eps) <= action <= (self.state + self.eps) else 0


class IdentityEnvMultiDiscrete(IdentityEnv):
    def __init__(self, dim=3, ep_length=100):
        """
        Identity environment for testing purposes

        :param dim: (int) the size of the dimensions you want to learn
        :param ep_length: (int) the length of each episode in timesteps
        """
        self.dim = dim
        space = MultiDiscrete([self.dim, self.dim])
        super().__init__(ep_length=ep_length, space=space)
