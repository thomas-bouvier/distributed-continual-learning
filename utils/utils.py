import torch

def move_cuda(item, use_cuda=True, **kw):
    """
    Moves the given item to CUDA and returns it, if `use_cuda` is available.

    Args:
        item (object): any object to move to potentially gpu
        use_cuda (bool): if we want to use gpu or cpu.
    Returns
        object: the same object, potentially moved to gpu.
    """

    if use_cuda and torch.cuda.is_available():
        item = item.cuda()
    return item
