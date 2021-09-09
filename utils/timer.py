import time
import torch
import horovod.torch as hvd

def hprint(*args, rank=0):
    if hvd.rank() == rank:
        print(*args)


def gpu_stats():
    if torch.__version__ == '1.2.0':
        hprint()
        hprint('Allocated:', torch.cuda.memory_allocated() / 1048576, 'MiB')
        hprint('Reserved:', torch.cuda.memory_cached() / 1048576, 'MiB')
        hprint()
    else:
        stats = torch.cuda.memory_stats()
        hprint()
        #hprint('Active:', stats['active_bytes.all.current'] / 1048576, 'MiB')
        hprint('Allocated:', stats['allocated_bytes.all.current'] / 1048576, 'MiB')
        hprint('Reserved:', stats['reserved_bytes.all.current'] / 1048576, 'MiB')
        hprint()


class Timer():

    training = True

    def __init__(self):
        self.times = {}
        self.lock = False


    def set_training(self, training):
        self.training = training


    def start(self, name):
        if self.lock:
            return

        if not self.training:
            return

        if name in self.times.keys():
            hprint(f"Error: Timer '{name}' has already started.")
        else :
            self.times[name] = time.time()


    def end(self, name):
        if self.lock:
            return

        if not self.training:
            return

        if name not in self.times.keys():
            hprint(f"Error: Timer '{name}' doesn't exist yet.")
        else:
            self.times[name] = time.time() - self.times[name]


    def lock(self):
        self.lock = True


    def unlock(self):
        self.lock = False


    def retrieve(self):
        return self.times