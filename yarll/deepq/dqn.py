from functools import partial

import torch
import numpy as np
import gym

from yarll import logger
from yarll.common import OffPolicyRLModel, SetVerbosity, TensorboardWriter
from yarll.common.vec_env import VecEnv
from yarll.common.schedules import LinearSchedule
from yarll.common.buffers import ReplayBuffer, PrioritizedReplayBuffer


class DQN(OffPolicyRLModel):
    """
    The DQN model class.
    DQN paper: https://arxiv.org/abs/1312.5602
    Dueling DQN: https://arxiv.org/abs/1511.06581
    Double-Q Learning: https://arxiv.org/abs/1509.06461
    Prioritized Experience Replay: https://arxiv.org/abs/1511.05952

    :param policy: (DQNPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) size of the replay buffer
    :param exploration_fraction: (float) fraction of entire training period over which the exploration rate is
            annealed
    :param exploration_final_eps: (float) final value of random action probability
    :param exploration_initial_eps: (float) initial value of random action probability
    :param train_freq: (int) update the model every `train_freq` steps. set to None to disable printing
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param double_q: (bool) Whether to enable Double-Q learning or not.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param prioritized_replay: (bool) if True prioritized replay buffer will be used.
    :param prioritized_replay_alpha: (float)alpha parameter for prioritized replay buffer.
        It determines how much prioritization is used, with alpha=0 corresponding to the uniform case.
    :param prioritized_replay_beta0: (float) initial value of beta for prioritized replay buffer
    :param prioritized_replay_beta_iters: (int) number of iterations over which beta will be annealed from initial
            value to 1.0. If set to None equals to max_timesteps.
    :param prioritized_replay_eps: (float) epsilon to add to the TD errors when updating priorities.
    :param param_noise: (bool) Whether or not to apply noise to the parameters of the policy.
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    """
    def __init__(self, policy, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
                 exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True,
                 learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6, param_noise=False,  verbose=0, tensorboard_log=None, seed=None):

        # TODO: replay_buffer refactoring
        super(DQN, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  requires_vec_env=False, seed=seed)

        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log
        self.double_q = double_q

        self.policy = policy
        self.update_target = None
        self.act = None
        self.replay_buffer = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None
        self.summary = None

    def setup_model(self):
        pass

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        self._setup_learn()

        # Create the replay buffer
        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
            if self.prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = total_timesteps
            else:
                prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                initial_p=self.prioritized_replay_beta0,
                                                final_p=1.0)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)
            self.beta_schedule = None

        if replay_wrapper is not None:
            assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                          initial_p=self.exploration_initial_eps,
                                          final_p=self.exploration_final_eps)

        episode_rewards = [0.0]
        episode_successes = []

        callback.on_training_start(locals(), globals())
        callback.on_rollout_start()

        reset = True
        obs = self.env.reset()
        # Retrieve unnormalized observation for saving into the buffer
        if self._vec_normalize_env is not None:
            obs_ = self._vec_normalize_env.get_original_obs().squeeze()

        for _ in range(total_timesteps):
            # Take action and update exploration to the newest value
            kwargs = {}
            if not self.param_noise:
                update_eps = self.exploration.value(self.num_timesteps)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = \
                    -np.log(1. - self.exploration.value(self.num_timesteps) +
                            self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True

            action = self.policy.forward(torch.from_numpy(np.array(obs)).float(),
                                         action_mask=torch.from_numpy(np.array(self.env.action_mask)).float())
            env_action = action
            reset = False
            new_obs, rew, done, info = self.env.step(env_action)

            self.num_timesteps += 1

            # Stop training if return value is False
            if callback.on_step() is False:
                break

            # Store only the unnormalized version
            if self._vec_normalize_env is not None:
                new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                reward_ = self._vec_normalize_env.get_original_reward().squeeze()
            else:
                # Avoid changing the original ones
                obs_, new_obs_, reward_ = obs, new_obs, rew
            # Store transition in the replay buffer.
            self.replay_buffer.add(obs_, action, reward_, new_obs_, float(done))
            obs = new_obs
            # Save the unnormalized observation
            if self._vec_normalize_env is not None:
                obs_ = new_obs_

            # TODO: Log to tensorboard

            episode_rewards[-1] += reward_
            if done:
                maybe_is_success = info.get('is_success')
                if maybe_is_success is not None:
                    episode_successes.append(float(maybe_is_success))
                if not isinstance(self.env, VecEnv):
                    obs = self.env.reset()
                episode_rewards.append(0.0)
                reset = True

            # Do not train if the warmup phase is not over
            # or if there are not enough samples in the replay buffer
            can_sample = self.replay_buffer.can_sample(self.batch_size)
            if can_sample and self.num_timesteps > self.learning_starts \
                    and self.num_timesteps % self.train_freq == 0:

                callback.on_rollout_end()
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                # pytype:disable=bad-unpacking
                if self.prioritized_replay:
                    assert self.beta_schedule is not None, \
                           "BUG: should be LinearSchedule when self.prioritized_replay True"
                    experience = self.replay_buffer.sample(self.batch_size,
                                                           beta=self.beta_schedule.value(self.num_timesteps),
                                                           env=self._vec_normalize_env)
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size,
                                                                                            env=self._vec_normalize_env)
                    weights, batch_idxes = np.ones_like(rewards), None
                # pytype:enable=bad-unpacking

                # TODO: TRAIN STEP

                if self.prioritized_replay:
                    new_priorities = np.abs(0) + self.prioritized_replay_eps
                    assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
                    self.replay_buffer.update_priorities(batch_idxes, new_priorities)
            #
            #     callback.on_rollout_start()
            #
            # if can_sample and self.num_timesteps > self.learning_starts and \
            #         self.num_timesteps % self.target_network_update_freq == 0:
            #     # Update target network periodically.
            #     self.update_target(sess=self.sess)

            if len(episode_rewards[-101:-1]) == 0:
                mean_100ep_reward = -np.inf
            else:
                mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

            num_episodes = len(episode_rewards)
            if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                logger.record_tabular("steps", self.num_timesteps)
                logger.record_tabular("episodes", num_episodes)
                if len(episode_successes) > 0:
                    logger.logkv("success rate", np.mean(episode_successes[-100:]))
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring",
                                      int(100 * self.exploration.value(self.num_timesteps)))
                logger.dump_tabular()

        callback.on_training_end()
        return self

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)

        actions, _, _ = self.policy.forward(observation)

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return self.params

    def save(self, save_path, cloudpickle=False):
        # params
        data = {
            "double_q": self.double_q,
            "param_noise": self.param_noise,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "prioritized_replay": self.prioritized_replay,
            "prioritized_replay_eps": self.prioritized_replay_eps,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "prioritized_replay_alpha": self.prioritized_replay_alpha,
            "prioritized_replay_beta0": self.prioritized_replay_beta0,
            "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
        }

        params_to_save = self.get_parameters()

        self.save(save_path)
