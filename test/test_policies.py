import torch
from gym.spaces import Discrete, MultiDiscrete


class IdentityPolicy(torch.nn.Module):
    def __init__(self, obs_space: MultiDiscrete, ac_space: MultiDiscrete):
        super(IdentityPolicy, self).__init__()
        self.linear1 = []
        self.inputs_one_hot = []
        for space_dim in obs_space.nvec:
            self.inputs_one_hot.append(torch.eye(int(space_dim)))
            self.linear1.append(torch.nn.Linear(space_dim, 4))

        self.linear = torch.nn.Linear(len(self.linear1) * 4, 4)
        self.output = []
        for space_dim in ac_space.nvec:
            self.output.append(torch.nn.Linear(4, space_dim))

    def forward(self, x, action_mask=None):
        inputs = []
        for i, linear in enumerate(self.linear1):
            inputs.append(linear(self.inputs_one_hot[i][x[i] - 1]))
        concat = torch.cat(inputs)
        x = self.linear(concat)
        actions = []
        for output in self.output:
            _, action = torch.softmax(output(x), dim=0).max(0)
            actions.append(action)
        return actions


class DiscreteMaskPolicy(torch.nn.Module):
    def __init__(self, obs_space: Discrete, ac_space: Discrete):
        super(DiscreteMaskPolicy, self).__init__()

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


class RecurrentMaskPolicy(torch.nn.Module):
    def __init__(self, obs_space: Discrete, ac_space: Discrete):
        super(RecurrentMaskPolicy, self).__init__()
        self.hidden_size = 32
        self.num_layers = 2
        self.input_one_hot = torch.eye(obs_space.n)

        self.rnn1 = torch.nn.RNN(obs_space.n, self.hidden_size, self.num_layers, batch_first=True)
        self.linear1 = torch.nn.Linear(self.hidden_size, ac_space.n)

    def forward(self, x, action_mask=None):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        x = x.view(1, 1, 1)
        x, _ = self.rnn1(x, hidden)
        x = self.linear1(x[:, -1, :])
        x = torch.sigmoid(self.linear1(x))
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
        for i, output in enumerate(self.output):
            if action_mask is not None:
                if i == 0:
                    _, action = torch.softmax(torch.sigmoid(output(x)) * action_mask[i], dim=0).max(0)
                    actions.append(action)
                else:
                    sliced_action_mask = action_mask[i]
                    for j, index in enumerate(actions):
                        sliced_action_mask = sliced_action_mask[index]
                    _, action = torch.softmax(torch.sigmoid(output(x)) * sliced_action_mask, dim=0).max(0)
                    actions.append(action)
            else:
                _, action = torch.softmax(output(x), dim=0).max(0)
                actions.append(action)
        return actions
