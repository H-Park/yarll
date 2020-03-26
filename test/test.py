from yarll import DQN
from test.mimo_env import MIMOEnv

policy = [[4, 4], [8, 8], 16, 16, [32, 32], [64, 64]]

env = MIMOEnv()
model = DQN(policy, env)

model.learn(1000)
