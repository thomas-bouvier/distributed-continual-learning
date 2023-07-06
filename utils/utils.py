import math
import torch
import wandb

from six import string_types


def get_device(use_cuda=True):
    """
    This function returns the device being used for computation.

    Args:
        use_cuda (bool): Whether or not to use CUDA. Default is True.

    Returns:
        str: Device being used for computation.
    """
    return "cuda" if use_cuda else "cpu"


def move_cuda(item, use_cuda=True, cuda_device=-1):
    """
    Moves the given item to CUDA and returns it, if `use_cuda` is available.

    Args:
        item (object): any object to move to potentially gpu
        use_cuda (bool): if we want to use gpu or cpu.

    Returns:
        object: the same object, potentially moved to gpu.
    """
    if use_cuda:
        if cuda_device != -1:
            item = item.to(torch.device(f"cuda:{cuda_device}"))
        else:
            item = item.cuda()
    return item


def synchronize_cuda(use_cuda=True):
    """
    This function synchronizes the current CUDA stream if use_cuda is True.

    Args:
        use_cuda (bool): Whether or not to synchronize CUDA. Default is True.
    """
    if use_cuda:
        torch.cuda.current_stream().synchronize()


def eval_func(f, x):
    """
    This function evaluates a function f at x.

    Args:
        f (str or function): The function to evaluate.
        x (float or torch.Tensor): The input to the function.

    Returns:
        float or torch.Tensor: The output of the function.
    """
    if isinstance(f, string_types):
        f = eval(f)
    return f(x)
