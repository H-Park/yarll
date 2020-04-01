from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """
    The base policy object

    :param policy: [(torch.nn.Module] The RL policy
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batches to run (n_envs * n_steps)
    """

    def __init__(self, policy, n_env, n_steps, n_batch):
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_batch
        self._policy = policy

    @property
    def policy(self):
        return self._policy

    @abstractmethod
    def step(self, obs, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) actions
        """
        raise NotImplementedError

    @abstractmethod
    def proba_step(self, obs, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError


class ActorCriticPolicy(BasePolicy):
    """
    Policy object that implements actor critic

    :param policy: [(torch.nn.Module] The RL policy
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    """

    def __init__(self, policy, n_env, n_steps, n_batch):
        super(ActorCriticPolicy, self).__init__(policy, n_env, n_steps, n_batch)

    @abstractmethod
    def step(self, obs, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) actions
        """
        raise NotImplementedError

    @abstractmethod
    def value(self, obs, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError
