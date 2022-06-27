import collections
import numpy as np
import types
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import torch

from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali.pipeline import Pipeline
from random import shuffle


"""
https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/frameworks/pytorch/pytorch-external_input.html
https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/parallel_external_source.html
"""

# TODO: pipeline validation

class ExternalInputCallable:
    def __init__(self, taskset, task_id, batch_size, shard_id, num_shards):
        raw_samples = taskset.get_raw_samples()
        self.task_id = task_id
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards

        self.files = raw_samples[0]
        self.labels = raw_samples[1]

        # If the dataset size is not divisible by number of shards, the trailing samples will
        # be omitted.
        self.shard_size = len(self.files) // num_shards
        self.shard_offset = self.shard_size * shard_id
        # If the shard size is not divisible by the batch size, the last incomplete batch
        # will be omitted.
        self.full_iterations = self.shard_size // batch_size
        self.perm = None  # permutation of indices
        # so that we don't have to recompute the `self.perm` for every sample
        self.last_seen_epoch = None

    def __call__(self, sample_info):
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            self.perm = np.random.default_rng(
                seed=42 + sample_info.epoch_idx).permutation(len(self.files))
        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]

        with open(self.files[sample_idx], 'rb') as f:
            encoded_img = np.frombuffer(f.read(), dtype=np.uint8)
        label = np.int64([self.labels[sample_idx]])
        return encoded_img, label, np.int32([self.task_id])


class DaliDataLoader(object):
    def __init__(self, dataset, task_id, cuda, batch_size=1, num_workers=1,
                 device_id=1, shard_id=0, num_shards=1, precision=32, **kwargs):
        decoder_device, device = ("mixed", "gpu") if cuda else ("cpu", "cpu")
        img_type = types.FLOAT16 if precision == 16 else types.FLOAT

        self.external_data = ExternalInputCallable(
            dataset, task_id, batch_size, shard_id, num_shards)

        @pipeline_def(batch_size=batch_size, num_threads=num_workers or 1,
                      device_id=device_id, py_num_workers=4, py_start_method='spawn')
        def callable_pipeline():
            inputs, target, task_ids = fn.external_source(
                source=self.external_data, num_outputs=3, dtype=[types.UINT8, types.INT64, types.INT32],
                batch=False, parallel=True)
            images = fn.decoders.image(
                inputs, device=decoder_device, output_type=types.RGB)
            images = fn.random_resized_crop(
                images, device=device, size=224, random_area=[0.08, 1.25])
            images = fn.crop_mirror_normalize(images, device=device,
                                              dtype=img_type,
                                              output_layout=types.NCHW,
                                              mean=[0.485 * 255, 0.456 *
                                                    255, 0.406 * 255],
                                              std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
            images = fn.random.coin_flip(images, probability=0.5)
            images = fn.cast(images, dtype=img_type)
            return images, target, task_ids

        pipeline = callable_pipeline()
        pipeline.build()

        self.iterator = DALIGenericIterator(
            pipeline, ["x", "y", "t"], last_batch_padded=True,
            prepare_first_batch=True, auto_reset=True)

    def __len__(self):
        return self.external_data.full_iterations

    def __iter__(self):
        for token in self.iterator:
            x = token[0]['x']
            y = token[0]['y'].squeeze()
            t = token[0]['t'].squeeze()
            yield x, y, t
