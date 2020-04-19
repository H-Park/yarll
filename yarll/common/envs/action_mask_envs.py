import gym
from gym.spaces import Discrete, MultiDiscrete

import torch


class DiscreteMaskEnv(gym.Env):
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self):
        self.action_space = Discrete(5)
        self.observation_space = Discrete(3)
        self.current_step = 0
        self._action_mask = torch.ones(self.action_space.n)

    def reset(self):
        self.current_step = 0
        self._action_mask = torch.ones(self.action_space.n)
        self._choose_next_state()
        return self.state

    def step(self, action: int):
        action_mask = torch.ones(self.action_space.n)
        if self.action_mask[action] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self.action_mask, action))
        action_mask[action] = 0

        self.current_step += 1
        self._action_mask = action_mask
        self._choose_next_state()
        return self.state, 0, self.finish(), {"action_mask": self.action_mask}

    def render(self, mode='human'):
        pass

    def finish(self):
        return self.current_step == 250

    def _choose_next_state(self):
        self.state = torch.tensor(self.observation_space.sample(), dtype=torch.long)

    @property
    def action_mask(self):
        return self._action_mask


class MultiDiscreteMaskEnv(gym.Env):
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self):
        self.action_space = MultiDiscrete([2, 3, 4])

        self.observation_space = MultiDiscrete([4, 5])

        self.current_step = 0
        self._valid_actions1 = torch.ones(self.action_space.nvec[0])
        self._valid_actions2 = torch.ones(self.action_space.nvec[0], self.action_space.nvec[1])
        self._valid_actions3 = torch.ones(self.action_space.nvec[0],
                                          self.action_space.nvec[1],
                                          self.action_space.nvec[2])
        self._action_mask = [self._valid_actions1, self._valid_actions2, self._valid_actions3]

    def reset(self):
        self._valid_actions1 = torch.ones(self.action_space.nvec[0])
        self._valid_actions2 = torch.ones(self.action_space.nvec[0], self.action_space.nvec[1])
        self._valid_actions3 = torch.ones(self.action_space.nvec[0],
                                          self.action_space.nvec[1],
                                          self.action_space.nvec[2])
        self._action_mask = [self._valid_actions1, self._valid_actions2, self._valid_actions3]
        self.current_step = 0
        self._choose_next_state()
        return self.state

    def step(self, actions):
        valid_actions1 = torch.ones(self.action_space.nvec[0])
        valid_actions2 = torch.ones(self.action_space.nvec[0], self.action_space.nvec[1])
        valid_actions3 = torch.ones(self.action_space.nvec[0],
                                    self.action_space.nvec[1],
                                    self.action_space.nvec[2])

        if self._action_mask[0][actions[0]] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self._action_mask[0], actions))
        else:
            valid_actions1[actions[0]] = 0
        if self._action_mask[1][actions[0]][actions[1]] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self._action_mask[1][actions[0]], actions))
        else:
            valid_actions2[0][actions[1]] = 0
            valid_actions2[1][actions[1]] = 0
        if self._action_mask[2][actions[0]][actions[1]][actions[2]] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self._action_mask[2][actions[0][actions[2]]], actions))
        else:
            valid_actions3[0][0][actions[2]] = 0
            valid_actions3[0][1][actions[2]] = 0
            valid_actions3[0][2][actions[2]] = 0
            valid_actions3[1][0][actions[2]] = 0
            valid_actions3[1][1][actions[2]] = 0
            valid_actions3[1][2][actions[2]] = 0

        self._action_mask = [valid_actions1, valid_actions2, valid_actions3]
        self._choose_next_state()
        self.current_step += 1

        return self.state, 0, self.finish(), {"action_mask": self._action_mask}

    def render(self, mode='human'):
        pass

    def finish(self):
        return self.current_step == 250

    def _choose_next_state(self):
        self.state = torch.tensor(self.observation_space.sample(), dtype=torch.long)
