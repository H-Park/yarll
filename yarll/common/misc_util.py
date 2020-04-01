import random

import gym
import numpy as np
import torch

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
