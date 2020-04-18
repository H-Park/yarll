import pytest

from yarll import DQN
from yarll.common.envs import IdentityEnv, IdentityEnvMultiDiscrete
from yarll.common.envs.vec_env.dummy_vec_env import DummyVecEnv
from yarll.common.evaluation import evaluate_policy
from test.test_policies import IdentityPolicy

MODEL_LIST = [
    DQN
    # A2C,
    # PPO,
    # SAC
]


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_identity_multidiscrete(model_class):
    """
    Test if the algorithm with a given policy
    can learn an identity transformation (i.e. return observation as an action)
    with a multidiscrete action space

    :param model_class: (BaseRLModel) A RL Algorithm
    """
    env = DummyVecEnv([IdentityEnvMultiDiscrete])
    policy = IdentityPolicy(env.observation_space, env.action_space).cuda()
    model = model_class(policy, env, tensorboard_log="run/")
    model.learn(total_timesteps=10000)
    evaluate_policy(model, IdentityEnvMultiDiscrete(), n_eval_episodes=4)

