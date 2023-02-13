import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import wandb
import numpy as np

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