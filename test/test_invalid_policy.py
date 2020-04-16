import pytest

from gym.spaces import Discrete
from yarll import DQN
from yarll.common.envs import IdentityEnv, InvalidEnv
from yarll.common.envs.vec_env.dummy_vec_env import DummyVecEnv
from test.test_policies import DiscreteMaskPolicy

MODEL_LIST = [
    DQN
]


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_env_policy_compatibility(model_class):
    """
    Tests the compatibility check between the environment with the policy

    :param model_class: (BaseRLModel) A RL Algorithm
    """
    env = DummyVecEnv([IdentityEnv])
    policy = DiscreteMaskPolicy(env.observation_space, Discrete(13))
    with pytest.raises(Exception):
        model_class(policy, env)


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_env_compatibility(model_class):
    """
    Tests the compatibility check between the environment with the policy

    :param model_class: (BaseRLModel) A RL Algorithm
    """
    env = DummyVecEnv([InvalidEnv])
    policy = DiscreteMaskPolicy(env.observation_space, Discrete(13))
    with pytest.raises(Exception):
        model_class(policy, env)