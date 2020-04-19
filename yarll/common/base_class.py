import os
import glob
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from typing import Union, List, Callable, Optional

import gym

import torch
from torch.utils.tensorboard import SummaryWriter

from yarll.common.misc_util import set_global_seeds, verify_env_policy
from yarll.common.runners import AbstractEnvRunner
from yarll.common.envs.vec_env import VecEnvWrapper, VecEnv, VecNormalize, unwrap_vec_normalize
from yarll.common.envs.vec_env.dummy_vec_env import DummyVecEnv
from yarll.common.callbacks import BaseCallback, CallbackList, ConvertCallback
from yarll import logger


class BaseRLModel(ABC):
    """
    The base RL model

    :param policy: (torch.nn.Module) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    """

    def __init__(self, policy, env, verbose=0, *, requires_vec_env, seed=None):
        self.policy = policy
        self.env = env
        self.verbose = verbose
        self._requires_vec_env = requires_vec_env
        self.observation_space = None
        self.action_space = None
        self.n_envs = None
        self._vectorize_action = False
        self.num_timesteps = 0
        self.seed = seed
        self.episode_reward = None
        self.ep_info_buf = None
        self.obs_transformation = None
        self.ac_transformation = None

        if env is not None:
            if isinstance(env, str):
                if self.verbose >= 1:
                    print("Creating environment from the given name, wrapped in a DummyVecEnv.")
                self.env = env = DummyVecEnv([lambda: gym.make(env)])

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.set_up_space_transformations()
            if requires_vec_env:
                if isinstance(env, VecEnv):
                    self.n_envs = env.num_envs
                else:
                    # The model requires a VecEnv
                    # wrap it in a DummyVecEnv to avoid error
                    self.env = DummyVecEnv([lambda: env])
                    if self.verbose >= 1:
                        print("Wrapping the env in a DummyVecEnv.")
                    self.n_envs = 1
            else:
                if isinstance(env, VecEnv):
                    if env.num_envs == 1:
                        self.env = _UnvecWrapper(env)
                        self._vectorize_action = True
                    else:
                        raise ValueError("Error: the model requires a non vectorized environment or a single vectorized"
                                         " environment.")
                self.n_envs = 1

        # Get VecNormalize object if it exists
        self._vec_normalize_env = unwrap_vec_normalize(self.env)

        verify_env_policy(self, self.env)

    def get_env(self):
        """
        returns the current environment (can be None if not defined)

        :return: (Gym Environment) The current environment
        """
        return self.env

    def get_vec_normalize_env(self) -> Optional[VecNormalize]:
        """
        Return the ``VecNormalize`` wrapper of the training env
        if it exists.

        :return: Optional[VecNormalize] The ``VecNormalize`` env.
        """
        return self._vec_normalize_env

    def set_env(self, env):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.

        :param env: (Gym Environment) The environment for learning a policy
        """
        if env is None and self.env is None:
            if self.verbose >= 1:
                print("Loading a model without an environment, "
                      "this model cannot be trained until it has a valid environment.")
            return
        elif env is None:
            raise ValueError("Error: trying to replace the current environment with None")

        # sanity checking the environment
        assert self.observation_space == env.observation_space, \
            "Error: the environment passed must have at least the same observation space as the model was trained on."
        assert self.action_space == env.action_space, \
            "Error: the environment passed must have at least the same action space as the model was trained on."
        if self._requires_vec_env:
            assert isinstance(env, VecEnv), \
                "Error: the environment passed is not a vectorized environment, however {} requires it".format(
                    self.__class__.__name__)
            assert not self.policy.recurrent or self.n_envs == env.num_envs, \
                "Error: the environment passed must have the same number of environments as the model was trained on." \
                "This is due to the Lstm policy not being capable of changing the number of environments."
            self.n_envs = env.num_envs
        else:
            # for models that dont want vectorized environment, check if they make sense and adapt them.
            # Otherwise tell the user about this issue
            if isinstance(env, VecEnv):
                if env.num_envs == 1:
                    env = _UnvecWrapper(env)
                    self._vectorize_action = True
                else:
                    raise ValueError("Error: the model requires a non vectorized environment or a single vectorized "
                                     "environment.")
            else:
                self._vectorize_action = False

            self.n_envs = 1

        self.env = env
        self._vec_normalize_env = unwrap_vec_normalize(env)

        # Invalidated by environment change.
        self.episode_reward = None
        self.ep_info_buf = None

    def _init_num_timesteps(self, reset_num_timesteps=True):
        """
        Initialize and resets num_timesteps (total timesteps since beginning of training)
        if needed. Mainly used logging and plotting (tensorboard).

        :param reset_num_timesteps: (bool) Set it to false when continuing training
            to not create new plotting curves in tensorboard.
        :return: (bool) Whether a new tensorboard log needs to be created
        """
        if reset_num_timesteps:
            self.num_timesteps = 0

        new_tb_log = self.num_timesteps == 0
        return new_tb_log

    @abstractmethod
    def setup_model(self):
        """
        Create all the functions and tensorflow graphs necessary to train the model
        """
        pass

    def _init_callback(self,
                      callback: Union[None, Callable, List[BaseCallback], BaseCallback]
                      ) -> BaseCallback:
        """
        :param callback: (Union[None, Callable, List[BaseCallback], BaseCallback])
        :return: (BaseCallback)
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)
        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        callback.init_callback(self)
        return callback

    def set_random_seed(self, seed: Optional[int]) -> None:
        """
        :param seed: (Optional[int]) Seed for the pseudo-random generators. If None,
            do not change the seeds.
        """
        # Ignore if the seed is None
        if seed is None:
            return
        # Seed python, numpy and tf random generator
        set_global_seeds(seed)
        if self.env is not None:
            self.env.seed(seed)
            # Seed the action space
            # useful when selecting random actions
            self.env.action_space.seed(seed)
        self.action_space.seed(seed)

    def _setup_learn(self):
        """
        Check the environment.
        """
        if self.env is None:
            raise ValueError("Error: cannot train the model without a valid environment, please set an environment with"
                             "set_env(self, env) method.")
        if self.episode_reward is None:
            self.episode_reward = torch.zeros((self.n_envs,))
        if self.ep_info_buf is None:
            self.ep_info_buf = deque(maxlen=100)

    @abstractmethod
    def get_parameter_list(self):
        """
        Get Pytorch Variables of model's parameters

        This includes all variables necessary for continuing training (saving / loading).

        :return: (list) List of tensorflow Variables
        """
        pass

    def get_parameters(self):
        """
        Get current model parameters as dictionary of variable name -> ndarray.

        :return: (OrderedDict) Dictionary of variable name -> ndarray of model's parameters.
        """
        return_dictionary = OrderedDict((name, param.data) for name, param in self.policy.named_parameters())
        return return_dictionary

    def pretrain(self, dataset, n_epochs=10, learning_rate=1e-4,
                 adam_epsilon=1e-8, val_interval=None):
        """
        Pretrain a model using behavior cloning:
        supervised learning given an expert dataset.

        NOTE: only Box and Discrete spaces are supported for now.

        :param dataset: (ExpertDataset) Dataset manager
        :param n_epochs: (int) Number of iterations on the training set
        :param learning_rate: (float) Learning rate
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param val_interval: (int) Report training and validation losses every n epochs.
            By default, every 10th of the maximum number of epochs.
        :return: (BaseRLModel) the pretrained model
        """

        # TODO: THIS
        return self

    @abstractmethod
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run",
              reset_num_timesteps=True):
        """
        Return a trained model.

        :param total_timesteps: (int) The total number of samples to train on
        :param callback: (Union[callable, [callable], BaseCallback])
            function called at every steps with state of the algorithm.
            It takes the local and global variables. If it returns False, training is aborted.
            When the callback inherits from BaseCallback, you will have access
            to additional stages of the training (training start/end),
            please read the documentation for more details.
        :param log_interval: (int) The number of timesteps before logging.
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param reset_num_timesteps: (bool) whether or not to reset the current timestep number (used in logging)
        :return: (BaseRLModel) the trained model
        """
        pass

    @abstractmethod
    def predict(self, observation, action_mask=None):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param action_mask: (np.ndarray) the action mask
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        pass

    def load_parameters(self, load_path_or_dict, exact_match=True):
        """
        Load model parameters from a file or a dictionary

        Dictionary keys should be tensorflow variable names, which can be obtained
        with ``get_parameters`` function. If ``exact_match`` is True, dictionary
        should contain keys for all model's parameters, otherwise RunTimeError
        is raised. If False, only variables included in the dictionary will be updated.

        This does not load agent's hyper-parameters.

        .. warning::
            This function does not update trainer/optimizer variables (e.g. momentum).
            As such training after using this function may lead to less-than-optimal results.

        :param load_path_or_dict: (str or file-like or dict) Save parameter location
            or dict of parameters as variable.name -> ndarrays to be loaded.
        :param exact_match: (bool) If True, expects load dictionary to contain keys for
            all variables in the model. If False, loads parameters only for variables
            mentioned in the dictionary. Defaults to True.
        """
        # TODO: THIS
        pass

    @abstractmethod
    def save(self, save_path, policy):
        """
        Save the current parameters to file

        :param save_path: (str or file-like) The save location
        :param policy: (torch.nn.Module) The RL policy
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, load_path, env=None):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Environment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        """
        raise NotImplementedError()

    @staticmethod
    def _is_vectorized_observation(observation, observation_space):
        """
        For every observation type, detects and validates the shape,
        then returns whether or not the observation is vectorized.

        :param observation: (np.ndarray) the input observation to validate
        :param observation_space: (gym.spaces) the observation space
        :return: (bool) whether the given observation is vectorized or not
        """
        if isinstance(observation_space, gym.spaces.Box):
            if observation.shape == observation_space.shape:
                return False
            elif observation.shape[1:] == observation_space.shape:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for ".format(observation.shape) +
                                 "Box environment, please use {} ".format(observation_space.shape) +
                                 "or (n_env, {}) for the observation shape."
                                 .format(", ".join(map(str, observation_space.shape))))
        elif isinstance(observation_space, gym.spaces.Discrete):
            if observation.shape == ():  # A numpy array of a number, has shape empty tuple '()'
                return False
            elif len(observation.shape) == 1:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for ".format(observation.shape) +
                                 "Discrete environment, please use (1,) or (n_env, 1) for the observation shape.")
        elif isinstance(observation_space, gym.spaces.MultiDiscrete):
            if observation.shape == (len(observation_space.nvec),):
                return False
            elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for MultiDiscrete ".format(observation.shape) +
                                 "environment, please use ({},) or ".format(len(observation_space.nvec)) +
                                 "(n_env, {}) for the observation shape.".format(len(observation_space.nvec)))
        elif isinstance(observation_space, gym.spaces.MultiBinary):
            if observation.shape == (observation_space.n,):
                return False
            elif len(observation.shape) == 2 and observation.shape[1] == observation_space.n:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for MultiBinary ".format(observation.shape) +
                                 "environment, please use ({},) or ".format(observation_space.n) +
                                 "(n_env, {}) for the observation shape.".format(observation_space.n))
        else:
            raise ValueError("Error: Cannot determine if the observation is vectorized with the space type {}."
                             .format(observation_space))

    def set_up_space_transformations(self):
        self.set_up_obs_transformation()
        self.set_up_ac_transformation()

    def set_up_ac_transformation(self):
        if not isinstance(self.action_space, List):
            self.action_space = [self.action_space]
        self.ac_transformation = []
        for ac_space in self.action_space:
            if isinstance(ac_space, gym.spaces.Discrete):
                self.ac_transformation.append(torch.eye(ac_space.n))
            elif isinstance(ac_space, gym.spaces.MultiDiscrete):
                transformations = []
                for discrete in ac_space.nvec:
                    transformations.append(torch.eye(int(discrete)))  # nvec is a list of numpy values
                self.ac_transformation.append(transformations)

    def set_up_obs_transformation(self):
        if not isinstance(self.observation_space, List):
            self.observation_space = [self.observation_space]
        self.obs_transformation = []
        for obs_space in self.observation_space:
            if isinstance(obs_space, gym.spaces.Discrete):
                self.obs_transformation.append(torch.eye(obs_space.n))
            elif isinstance(obs_space, gym.spaces.MultiDiscrete):
                transformations = []
                for discrete in obs_space.nvec:
                    transformations.append(torch.eye(int(discrete)))  # nvec is a list of numpy values
                self.obs_transformation.append(transformations)

    def transform_observation(self, obs):
        if not isinstance(obs, List):
            obs = [obs]
        transformed = []
        for i, obs_space in enumerate(self.observation_space):
            if isinstance(obs_space, gym.spaces.Discrete):
                transformed.append(self.obs_transformation[i][obs[i].long() - 1])
            elif isinstance(obs_space, gym.spaces.MultiDiscrete):
                transformed.append([self.obs_transformation[i][j][index.long() - 1] for j, index in enumerate(obs[i])])
        return transformed

    def transform_action(self, action):
        if not isinstance(action, List):
            action = [action]
        transformed = []
        for i, ac_space in enumerate(self.action_space):
            if isinstance(ac_space, gym.spaces.Discrete):
                transformed.append(self.ac_transformation[i][action[i] - 1])
            elif isinstance(ac_space, gym.spaces.MultiDiscrete):
                transformed.append([self.ac_transformation[i][j][index - 1]
                                    for j, index in enumerate(action[i])])
        return transformed


class ActorCriticRLModel(BaseRLModel):
    """
    The base class for Actor critic model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    """

    def __init__(self, policy, env, _init_setup_model, verbose=0, requires_vec_env=False, seed=None):
        super(ActorCriticRLModel, self).__init__(policy, env, verbose=verbose, requires_vec_env=requires_vec_env,
                                                 seed=seed)

        self.initial_state = None
        self.policy = None
        self.proba_step = None
        self.params = None
        self._runner = None

    def _make_runner(self) -> AbstractEnvRunner:
        """Builds a new Runner.

        Lazily called whenever `self.runner` is accessed and `self._runner is None`.
        """
        raise NotImplementedError("This model is not configured to use a Runner")

    @property
    def runner(self) -> AbstractEnvRunner:
        if self._runner is None:
            self._runner = self._make_runner()
        return self._runner

    def set_env(self, env):
        self._runner = None  # New environment invalidates `self._runner`.
        super().set_env(env)

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def learn(self, total_timesteps, callback=None,
              log_interval=100, tb_log_name="run", reset_num_timesteps=True):
        pass

    def predict(self, observation, action_mask=None):
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy.forward(observation, action_mask)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = torch.clamp(actions, min=self.action_space.low, max=self.action_space.high)

        if not vectorized_env:
            clipped_actions = clipped_actions[0]

        return clipped_actions

    def get_parameter_list(self):
        return self.policy.named_parameters()

    @abstractmethod
    def save(self, save_path, policy):
        pass

    @classmethod
    def load(cls, load_path, env=None):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Environment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        """
        pass


class OffPolicyRLModel(BaseRLModel):
    """
    The base class for off policy RL model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param replay_buffer: (ReplayBuffer) the type of replay buffer
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    """

    def __init__(self, policy, env, replay_buffer=None, _init_setup_model=False, verbose=0, *,
                 requires_vec_env=False, seed=None):
        super(OffPolicyRLModel, self).__init__(policy, env, verbose=verbose, requires_vec_env=requires_vec_env,
                                               seed=seed)

        self.replay_buffer = replay_buffer

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def learn(self, total_timesteps, callback=None,
              log_interval=100, tb_log_name="run", reset_num_timesteps=True, replay_wrapper=None):
        pass

    @abstractmethod
    def predict(self, observation, action_mask=None):
        pass

    @abstractmethod
    def save(self, save_path, policy):
        pass

    @classmethod
    def load(cls, load_path, env=None):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Environment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        """
    pass


class _UnvecWrapper(VecEnvWrapper):
    def __init__(self, venv):
        """
        Unvectorize a vectorized environment, for vectorized environment that only have one environment

        :param venv: (VecEnv) the vectorized environment to wrap
        """
        super().__init__(venv)
        assert venv.num_envs == 1, "Error: cannot unwrap a environment wrapper that has more than one environment."

    def seed(self, seed=None):
        return self.venv.env_method('seed', seed)

    def compute_reward(self, achieved_goal, desired_goal, _info):
        return float(self.venv.env_method('compute_reward', achieved_goal, desired_goal, _info)[0])

    @staticmethod
    def unvec_obs(obs):
        """
        :param obs: (Union[np.ndarray, dict])
        :return: (Union[np.ndarray, dict])
        """
        if not isinstance(obs, dict):
            return obs[0]
        obs_ = OrderedDict()
        for key in obs.keys():
            obs_[key] = obs[key][0]
        del obs
        return obs_

    def reset(self):
        return self.unvec_obs(self.venv.reset())

    def step_async(self, actions):
        self.venv.step_async([actions])

    def step_wait(self):
        obs, rewards, dones, information = self.venv.step_wait()
        return self.unvec_obs(obs), float(rewards[0]), dones[0], information[0]

    def render(self, mode='human'):
        return self.venv.render(mode=mode)


class SetVerbosity:
    def __init__(self, verbose=0):
        """
        define a region of code for certain level of verbosity

        :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        """
        self.verbose = verbose

    def __enter__(self):
        self.tf_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
        self.log_level = logger.get_level()
        self.gym_level = gym.logger.MIN_LEVEL

        if self.verbose <= 1:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if self.verbose <= 0:
            logger.set_level(logger.DISABLED)
            gym.logger.set_level(gym.logger.DISABLED)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose <= 1:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.tf_level

        if self.verbose <= 0:
            logger.set_level(self.log_level)
            gym.logger.set_level(self.gym_level)


class TensorboardWriter:
    def __init__(self, policy, tensorboard_log_path, tb_log_name, new_tb_log=True):
        """
        Create a Tensorboard writer for a code segment, and saves it to the log directory as its own run

        :param policy: (Tensorflow Graph) the model graph
        :param tensorboard_log_path: (str) the save path for the log (can be None for no logging)
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param new_tb_log: (bool) whether or not to create a new logging folder for tensorbaord
        """
        self.policy = policy
        self.tensorboard_log_path = tensorboard_log_path
        self.tb_log_name = tb_log_name
        self.writer = None
        self.new_tb_log = new_tb_log

    def __enter__(self):
        if self.tensorboard_log_path is not None:
            latest_run_id = self._get_latest_run_id()
            if self.new_tb_log:
                latest_run_id = latest_run_id + 1
            save_path = os.path.join(self.tensorboard_log_path, "{}_{}".format(self.tb_log_name, latest_run_id))
            self.writer = SummaryWriter(save_path)
        return self.writer

    def _get_latest_run_id(self):
        """
        returns the latest run number for the given log name and log path,
        by finding the greatest number in the directories.

        :return: (int) latest run number
        """
        max_run_id = 0
        for path in glob.glob("{}/{}_[0-9]*".format(self.tensorboard_log_path, self.tb_log_name)):
            file_name = path.split(os.sep)[-1]
            ext = file_name.split("_")[-1]
            if self.tb_log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
                max_run_id = int(ext)
        return max_run_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            # self.writer.add_graph(self.policy)
            self.writer.flush()
