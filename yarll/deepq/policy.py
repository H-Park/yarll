import torch
from abc import ABC, abstractmethod


class DQNPolicy(torch.nn.Module):
    def __init__(self):
        super(DQNPolicy, self).__init__()

    @abstractmethod
    def apply_mask(self, logits, action_mask):
        """
        Given an action, apply the action mask. Note: This is used for masking actions produced by exploration
        and NOT from the policy
        :param logits: the raw logits to be masked
        :param action_mask: The action mask(s) to apply
        :return: a valid action according to action_mask
        """
        pass
