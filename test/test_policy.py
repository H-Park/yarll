from test.mimo_env import MIMOEnv
from test.policy import RLPolicy
from yarll.deepq import DQN

env = MIMOEnv()
policy = RLPolicy(env.observation_space, env.action_space)

model = DQN(policy, env, verbose=1)

model.learn(100)

