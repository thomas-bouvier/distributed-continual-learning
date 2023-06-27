import horovod.torch as hvd
import nvtx
import torch

from utils.log import PerformanceResultsLog
from utils.utils import synchronize_cuda


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = torch.tensor(0.0)
        self.avg = torch.tensor(0.0)
        self.sum = torch.tensor(0.0)
        self.count = torch.tensor(0.0)

    def update(self, val, n=1):
        val = val.detach().cpu() if isinstance(val, torch.Tensor) else torch.tensor(val)
        self.val = hvd.allreduce(val, name=self.name)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class MeasureTime:
    def __init__(self, name, batch, perf_metrics: PerformanceResultsLog = None, cuda=True, dummy=False):
        self.batch = batch
        self.name = name
        self.perf_metrics = perf_metrics
        self.cuda = cuda
        self.dummy = dummy
        self.rng = None

    def __enter__(self):
        if not self.dummy:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            synchronize_cuda(self.cuda)
            self.start.record()
            self.rng = nvtx.start_range(message=self.name)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if not self.dummy:
            nvtx.end_range(self.rng)
            self.end.record()
            synchronize_cuda(self.cuda)
            time = self.start.elapsed_time(self.end)

            if self.perf_metrics is not None:
                self.perf_metrics.add(self.batch, time, key=self.name)


def get_timer(name, batch, perf_metrics=None, previous_iteration=False, cuda=True, dummy=False):
    if previous_iteration:
        batch -= 1
    return MeasureTime(name, batch, perf_metrics=perf_metrics, cuda=cuda, dummy=dummy)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
