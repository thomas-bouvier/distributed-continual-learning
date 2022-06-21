import collections
import numpy as np
import types
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import torch

from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali.pipeline import Pipeline
from random import shuffle


"""https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/frameworks/pytorch/pytorch-external_input.html
"""
class ExternalInputIterator(object):
    def __init__(self, taskset, task_id, batch_size, device_id, num_gpus):
        raw_samples = taskset.get_raw_samples()
        self.task_id = task_id
        self.batch_size = batch_size
        # Whole taskset size
        self.data_set_len = len(raw_samples[0])

        # Since we aren't using DistributedSampler with DALI to reduce the
        # number of samples per rank, we manually adjust the length of the DALI
        # pipeline when running distributed training
        self.samples = raw_samples[0][self.data_set_len * device_id // num_gpus:
                                      self.data_set_len * (device_id + 1) // num_gpus]
        self.target = raw_samples[1][self.data_set_len * device_id // num_gpus:
                                     self.data_set_len * (device_id + 1) // num_gpus]
        self.n = len(self.samples)

    def __iter__(self):
        self.i = 0
        # shuffle(self.files)
        return self

    def __next__(self):
        inputs = []
        target = []
        task_ids = []
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            inputs.append(np.fromfile(self.samples[self.i % self.n],
                            dtype=np.uint8))
            target.append(torch.tensor([int(self.target[self.i % self.n])], dtype=torch.uint8))
            task_ids.append(torch.tensor([self.task_id], dtype=torch.uint8))
            self.i += 1
        return (inputs, target, task_ids)

    def __len__(self):
        return self.data_set_len

    next = __next__


class DaliDataLoader(object):
    def __init__(self, dataset, task_id, batch_size=1, num_workers=1, **kwargs):
        self.eii = ExternalInputIterator(dataset, task_id, batch_size, 0, 1)
        self.batch_size = batch_size

        pipeline = self.get_pipeline(batch_size,
                                     num_threads=num_workers or 1, device_id=0,
                                     external_data=self.eii)

        self.iterator = DALIGenericIterator(
            pipeline, ["x", "y", "t"], last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL, prepare_first_batch=True)

    def get_pipeline(self, batch_size, num_threads, device_id, external_data):
        use_gpu = True
        decoder_device, device = ("mixed", "gpu") if use_gpu else ("cpu", "cpu")
        pipeline = Pipeline(batch_size, num_threads,
                            device_id, seed=12 + device_id)

        with pipeline:
            inputs, target, task_ids = fn.external_source(
                source=external_data, num_outputs=3, dtype=types.UINT8)
            images = fn.decoders.image(
                inputs, device=decoder_device, output_type=types.RGB)
            images = fn.random_resized_crop(
                images, device=device, size=224, random_area=[0.08, 1.25])
            images = fn.crop_mirror_normalize(images, device=device,
                                              dtype=types.FLOAT,
                                              output_layout=types.NCHW,
                                              mean=[0.485 * 255, 0.456 *
                                                    255, 0.406 * 255],
                                              std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
            images = fn.random.coin_flip(images, probability=0.5)
            output = fn.cast(images, dtype=types.FLOAT)
            pipeline.set_outputs(output, target, task_ids)
        return pipeline

    def __len__(self):
        return len(self.eii) // self.batch_size

    def __iter__(self):
        # A token is a list [{'x': , 'y': , 't': }]
        for token in self.iterator:
            x = token[0]['x']
            y = token[0]['y'].squeeze()
            t = token[0]['t'].squeeze()
            yield x, y, t
