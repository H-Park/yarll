import random

import gym
import numpy as np
import torch
from yarll.common.evaluation import evaluate_policy


def zipsame(*seqs):
    """
    Performs a zip function, but asserts that all zipped elements are of the same size

    :param seqs: a list of arrays that are zipped together
    :return: the zipped arguments
    """
    length = len(seqs[0])
    assert all(len(seq) == length for seq in seqs[1:])
    return zip(*seqs)


def set_global_seeds(seed):
    """
    set the seed for python random, tensorflow, numpy and gym spaces

    :param seed: (int) the seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def mpi_rank_or_zero():
    """
    Return the MPI rank if mpi is installed. Otherwise, return 0.
    :return: (int)
    """
    try:
        import mpi4py
        return mpi4py.MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0


def flatten_lists(listoflists):
    """
    Flatten a python list of list

    :param listoflists: (list(list))
    :return: (list)
    """
    return [el for list_ in listoflists for el in list_]


def verify_env_policy(policy, env: gym.Env):
    # verify Env's observation space matches the action space
    env.reset()
    if isinstance(env.action_space, list):
        action = None
        for ac_space in env.action_space:
            action.append(ac_space.sample())
    else:
        action = env.action_space.sample()
    try:
        env.step(action)
    except:
        raise Exception("The environment provided does not accept a proper action! Make sure env.step() expects the "
                        "same type / dimension as env.action_space.sample().")

    # verify Policy and Env compatibility
    try:
        evaluate_policy(policy, env, n_eval_episodes=1)
    except:
        raise Exception("The provided policy does not provide a valid action according to the provided "
                        "environment!")


def total_episode_reward_logger(rew_acc, rewards, masks, writer, steps):
    """
    calculates the cumulated episode reward, and prints to tensorflow log the output
    :param rew_acc: (np.array float) the total running reward
    :param rewards: (np.array float) the rewards
    :param masks: (np.array bool) the end of episodes
    :param writer: (TensorFlow Session.writer) the writer to log to
    :param steps: (int) the current timestep
    :return: (np.array float) the updated total running reward
    :return: (np.array float) the updated total running reward
    """
    for env_idx in range(rewards.shape[0]):
        dones_idx = np.sort(np.argwhere(masks[env_idx]))

        if len(dones_idx) == 0:
            rew_acc[env_idx] += sum(rewards[env_idx])
        else:
            rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
            writer.add_scalar("Episode rewards", rew_acc[env_idx], steps + dones_idx[0, 0])
            for k in range(1, len(dones_idx[:, 0])):
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k-1, 0]:dones_idx[k, 0]])
                writer.add_scalar("Episode rewards", rew_acc[env_idx], steps + dones_idx[k, 0])
            rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])
    return rew_acc