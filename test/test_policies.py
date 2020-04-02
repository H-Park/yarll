import torch
from gym.spaces import Discrete, MultiDiscrete


class DiscretePolicy(torch.nn.Module):
    def __init__(self, obs_space: Discrete, ac_space: Discrete):
        super(DiscretePolicy, self).__init__()
        self.input_one_hot = torch.eye(obs_space.n)

        self.linear1 = torch.nn.Linear(obs_space.n, 16)
        self.linear2 = torch.nn.Linear(16, 16)
        self.linear3 = torch.nn.Linear(16, ac_space.n)

    def forward(self, x, action_mask=None):
        x = self.input_one_hot[x-1].float()
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.sigmoid(self.linear3(x))
        if action_mask is not None:
            _, indices = torch.softmax(x * action_mask, dim=0).max(0)
        else:
            _, indices = torch.softmax(x, dim=0).max(0)
        return indices


class MultiDiscretePolicy(torch.nn.Module):
    def __init__(self, obs_space: MultiDiscrete, ac_space: MultiDiscrete):
        super(MultiDiscretePolicy, self).__init__()
        self.linear1 = []
        self.inputs_one_hot = []
        for space_dim in obs_space.nvec:
            self.inputs_one_hot.append(torch.eye(int(space_dim)))
            self.linear1.append(torch.nn.Linear(space_dim, 16))

        self.linear = torch.nn.Linear(len(self.linear1) * 16, 16)
        self.output = []
        for space_dim in ac_space.nvec:
            self.output.append(torch.nn.Linear(16, space_dim))

    def forward(self, x, action_mask=None):
        inputs = []
        for i, linear in enumerate(self.linear1):
            inputs.append(linear(self.inputs_one_hot[i][x[i] - 1]))
        concat = torch.cat(inputs)
        x = self.linear(concat)
        actions = []
        for output in self.output:
            _, action = torch.sigmoid(output(x)).max(0)
            actions.append(action)
        return actions
