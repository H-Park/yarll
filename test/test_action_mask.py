import pytest


from yarll import DQN
from yarll.common.envs import DiscreteMaskEnv, MultiDiscreteMaskEnv
from yarll.common.envs.vec_env.dummy_vec_env import DummyVecEnv
from yarll.common.evaluation import evaluate_policy
from test.test_policies import DiscretePolicy, MultiDiscretePolicy

import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

MODEL_LIST = [
    DQN
    # A2C,
    # PPO,
    # SAC
]


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_action_mask_discrete(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    with a multidiscrete action space

    :param model_class: (BaseRLModel) A RL Algorithm
    """
    env = DummyVecEnv([DiscreteMaskEnv])
    policy = DiscretePolicy(env.observation_space, env.action_space)
    model = model_class(policy, env)
    model.learn(total_timesteps=128)
    evaluate_policy(model, DiscreteMaskEnv(), n_eval_episodes=5)


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_action_mask_multidiscrete(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    with a multidiscrete action space

    :param model_class: (BaseRLModel) A RL Algorithm
    """
    env = DummyVecEnv([MultiDiscreteMaskEnv])
    policy = MultiDiscretePolicy(env.observation_space, env.action_space)
    model = model_class(policy, env)
    model.learn(total_timesteps=128)
    evaluate_policy(model, MultiDiscreteMaskEnv(), n_eval_episodes=5)

