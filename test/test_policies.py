import torch
from gym.spaces import Discrete, MultiDiscrete
import numpy as np
from yarll.deepq import DQNPolicy


class DiscretePolicy(DQNPolicy):
    def __init__(self, obs_space: Discrete, ac_space: Discrete):
        super(DiscretePolicy, self).__init__()

        self.linear1 = torch.nn.Linear(obs_space.n, 8)
        self.linear2 = torch.nn.Linear(8, 8)
        self.linear3 = torch.nn.Linear(8, ac_space.n)

    def forward(self, x, action_mask=None):
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.sigmoid(self.linear3(x))
        if action_mask is not None:
            _, indices = torch.softmax(x * action_mask, dim=-1).max(-1)
        else:
            _, indices = torch.softmax(x, dim=-1).max(-1)
        return indices

    def apply_mask(self, logits, action_mask=None):
        if action_mask is None:
            return torch.tensor(logits)
        else:
            return np.random.choice(np.nonzero(np.array(action_mask.cpu()))[0])


class MultiDiscretePolicy(DQNPolicy):
    def __init__(self, obs_space: MultiDiscrete, ac_space: MultiDiscrete):
        super(MultiDiscretePolicy, self).__init__()
        self.linear1 = []
        for space_dim in obs_space.nvec:
            self.linear1.append(torch.nn.Linear(space_dim, 8))

        self.linear2 = torch.nn.Linear(len(self.linear1) * 8, 8)
        self.output = []
        for space_dim in ac_space.nvec:
            self.output.append(torch.nn.Linear(8, space_dim))

    def forward(self, *args, action_mask=None):
        x1 = self.linear1[0](args[0][0])
        x2 = self.linear1[1](args[0][1])
        concat = torch.cat([x1, x2])
        x = self.linear2(concat)
        actions = []
        for i, output in enumerate(self.output):
            if action_mask is not None:
                if i == 0:
                    _, action = torch.softmax(torch.sigmoid(output(x)) * action_mask[i], dim=0).max(-1)
                    actions.append(action)
                else:
                    sliced_action_mask = action_mask[i]
                    for j, index in enumerate(actions):
                        sliced_action_mask = sliced_action_mask[index]
                    _, action = torch.softmax(torch.sigmoid(output(x)) * sliced_action_mask, dim=0).max(-1)
                    actions.append(action)
            else:
                _, action = torch.softmax(output(x), dim=0).max(0)
                actions.append(action)
        return torch.stack(actions)

    def apply_mask(self, logits, action_mask):
        if action_mask is None:
            if isinstance(logits, list):
                masked = []
                for logit in logits:
                    masked.append(torch.tensor(logit))
                return torch.stack(masked)
            return torch.tensor(logits)
        else:
            actions = []
            for mask in action_mask:
                if len(actions) == 0:
                    actions.append(np.random.choice(np.nonzero(np.array(mask.cpu()))[0]))
                else:
                    for action in actions:
                        mask = mask[action]
                    actions.append(np.random.choice(np.nonzero(np.array(mask.cpu()))[0]))
            return actions


class RecurrentMaskPolicy(DQNPolicy):
    def __init__(self, obs_space: Discrete, ac_space: Discrete):
        super(RecurrentMaskPolicy, self).__init__()

        self.lstm1 = torch.nn.LSTM(obs_space.n, 8)
        self.linear1 = torch.nn.Linear(8, 8)
        self.linear2 = torch.nn.Linear(8, ac_space.n)

    def forward(self, x, action_mask=None):
        x = x.view(1, 1, -1)
        x, _ = self.lstm1(x)
        x = self.linear1(x[:, -1, :])
        x = torch.sigmoid(self.linear2(x))[0]
        if action_mask is not None:
            _, indices = torch.softmax(x * action_mask, dim=0).max(0)
        else:
            _, indices = torch.softmax(x, dim=0).max(0)
        return indices

    def apply_mask(self, logits, action_mask=None):
        if action_mask is None:
            return logits
        else:
            return np.random.choice(np.nonzero(np.array(action_mask.cpu()))[0])
