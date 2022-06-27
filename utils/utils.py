import torch
import matplotlib.pyplot as plt

from bisect import bisect
from six import string_types


def get_device(use_cuda=True):
    if use_cuda:
        return "cuda"
    return "cpu"


def move_cuda(item, use_cuda=True, cuda_device=-1):
    """
    Moves the given item to CUDA and returns it, if `use_cuda` is available.

    Args:
        item (object): any object to move to potentially gpu
        use_cuda (bool): if we want to use gpu or cpu.
    Returns
        object: the same object, potentially moved to gpu.
    """
    if use_cuda:
        if cuda_device != -1:
            item = item.to(torch.device(f"cuda:{cuda_device}"))
        else:
            item = item.cuda()
    return item


def synchronize_cuda(use_cuda=True):
    if use_cuda:
        torch.cuda.synchronize()


def eval_func(f, x):
    if isinstance(f, string_types):
        f = eval(f)
    return f(x)


# Find 2D index from accumulated list of lengths
def find_2d_idx(c, idx):
    i1 = bisect(c, idx)
    i2 = (idx - c[i1 - 1]) if i1 > 0 else idx
    if torch.is_tensor(i2):
        i2 = i2.item()
    return (i1, i2)


def plot_representatives(rep_values, rep_labels, num_cols):
    num_candidates = len(rep_values)
    fig, ax = plt.subplots(num_candidates // num_cols, num_cols)
    for j in range(num_candidates // num_cols):
        for k in range(num_cols):
            ax[j, k].imshow(rep_values[j * num_cols + k].T,
                            interpolation='none')
            ax[j, k].set_title(rep_labels[j * num_cols + k].item())
            ax[j, k].axis('off')
    return fig
