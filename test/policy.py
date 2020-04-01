import torch
from gym.spaces import Discrete
import torch.nn.functional as F


class RLPolicy(torch.nn.Module):
    def __init__(self, obs_space: Discrete, ac_space: Discrete):
        super(RLPolicy, self).__init__()
        self.linear1 = torch.nn.Linear(obs_space.n, 16)
        self.linear2 = torch.nn.Linear(16, 16)
        self.linear3 = torch.nn.Linear(16, ac_space.n)

    def forward(self, x, action_mask):
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.sigmoid(self.linear3(x))
        _, indices = F.softmax(x * action_mask, dim=0).max(0)
        return indices
