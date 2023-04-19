import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import wandb
import numpy as np

from bisect import bisect
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


def find_2d_idx(c, idx):
    """
    This function finds the 2D index from an accumulated list of lengths.

    Args:
        c (list): The accumulated list of lengths.
        idx (int): The index to find the 2D index of.

    Returns:
        tuple: The 2D index.
    """
    i1 = bisect(c, idx)
    i2 = (idx - c[i1 - 1]) if i1 > 0 else idx
    if torch.is_tensor(i2):
        i2 = i2.item()
    return (i1, i2)


def plot_representatives(rep_values, rep_labels, num_cols):
    """
    This function plots representative images.

    Args:
        rep_values (list): The representative images.
        rep_labels (list): The labels for the representative images.
        num_cols (int): The number of columns to display.

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    num_candidates = len(rep_values)
    fig, ax = plt.subplots(num_candidates // num_cols, num_cols)
    for j in range(num_candidates // num_cols):
        for k in range(num_cols):
            ax[j, k].imshow(rep_values[j * num_cols + k].T,
                            interpolation='none')
            ax[j, k].set_title(rep_labels[j * num_cols + k].item())
            ax[j, k].axis('off')
    return fig


def display(filename, outputs, columns=2, captions=None):
    rows = int(math.ceil(len(outputs) / columns))
    fig = plt.figure()
    fig.set_size_inches(16, 6 * rows)
    gs = gridspec.GridSpec(rows, columns)
    row = 0
    col = 0
    for i in range(len(outputs)):
        plt.subplot(gs[i])
        plt.axis("off")
        if captions is not None:
            plt.title(captions[i])
        plt.imshow(np.array(outputs[i].permute(1, 2, 0).cpu()))
    path = f"{filename}.jpg"
    plt.savefig(path)
    plt.close()
    wandb.save(path)