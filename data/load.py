import random

import numpy as np
import torch

from continuum.tasks import TaskType
from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali.pipeline import Pipeline
from tqdm import tqdm


class ExternalInputCallable:
    """
    https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/frameworks/pytorch/pytorch-external_input.html
    https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/parallel_external_source.html
    """

    def __init__(self, taskset, task_id, batch_size, shard_id, num_shards):
        self.task_id = task_id
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.data_type = taskset.data_type

        raw_samples = taskset.get_raw_samples()
        files = np.array(raw_samples[0])
        labels = np.array(raw_samples[1])
        self.data_set_len = len(files)

        # If the dataset size is not divisible by number of shards, the trailing samples will
        # be omitted.
        shard_size = len(files) // num_shards
        self.shard_offset = shard_size * shard_id

        # perform a permutation for shards to be iid
        perm = np.random.permutation(self.data_set_len)
        self.files = files[perm]
        self.labels = labels[perm]

        # If the shard size is not divisible by the batch size, the last incomplete batch
        # will be omitted.
        self.n = shard_size
        self.full_iterations = self.n // batch_size

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
                seed=42 + sample_info.epoch_idx
            ).permutation(len(self.files))
        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]

        if self.data_type == TaskType.IMAGE_PATH:
            with open(self.files[sample_idx], "rb") as f:
                encoded_img = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            encoded_img = self.files[sample_idx]

        label = np.int64([self.labels[sample_idx]])

        return encoded_img, label, np.int32([self.task_id])


class DaliDataLoader:
    def __init__(
        self,
        taskset,
        task_id,
        batch_size=1,
        num_workers=1,
        device_id=1,
        shard_id=0,
        num_shards=1,
        precision=32,
        training=True,
        **kwargs,
    ):
        cuda = torch.cuda.is_available()
        decoder_device, device = ("mixed", "gpu") if cuda else ("cpu", "cpu")
        crop_size = 224
        val_size = 256
        img_type = types.FLOAT16 if precision == 16 else types.FLOAT

        # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
        preallocate_width_hint = 5980 if decoder_device == "mixed" else 0
        preallocate_height_hint = 6430 if decoder_device == "mixed" else 0

        self.external_data = ExternalInputCallable(
            taskset, task_id, batch_size, shard_id, num_shards
        )

        @pipeline_def(
            batch_size=batch_size,
            num_threads=num_workers or 1,
            device_id=device_id,
            py_num_workers=4,
            py_start_method="spawn",
        )
        def callable_pipeline():
            images, target, task_ids = fn.external_source(
                source=self.external_data,
                num_outputs=3,
                dtype=[types.UINT8, types.INT64, types.INT32],
                batch=False,
                parallel=True,
            )

            if training:
                images = fn.decoders.image_random_crop(
                    images,
                    device=decoder_device,
                    output_type=types.RGB,
                    device_memory_padding=device_memory_padding,
                    host_memory_padding=host_memory_padding,
                    preallocate_width_hint=preallocate_width_hint,
                    preallocate_height_hint=preallocate_height_hint,
                    random_aspect_ratio=[0.8, 1.25],
                    random_area=[0.1, 1.0],
                    num_attempts=100,
                )
                images = fn.resize(
                    images,
                    device=device,
                    resize_x=crop_size,
                    resize_y=crop_size,
                    interp_type=types.INTERP_TRIANGULAR,
                )
                mirror = fn.random.coin_flip(probability=0.5)
            else:
                images = fn.decoders.image(
                    inputs, device=decoder_device, output_type=types.RGB
                )
                images = fn.resize(
                    images,
                    device=device,
                    size=val_size,
                    mode="not_smaller",
                    interp_type=types.INTERP_TRIANGULAR,
                )
                mirror = False

            images = fn.crop_mirror_normalize(
                images.gpu() if cuda else images,
                dtype=img_type,
                output_layout="CHW",
                crop=(crop_size, crop_size),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                mirror=mirror,
            )

            if cuda:
                target = target.gpu()
                task_ids = task_ids.gpu()

            return images, target, task_ids

        self.pipeline = callable_pipeline()
        self.pipeline.build()

        self.iterator = DALIGenericIterator(
            self.pipeline,
            ["x", "y", "t"],
            last_batch_padded=True,
            prepare_first_batch=True,
            auto_reset=True,
        )

    def release(self):
        del self.pipeline, self.iterator
        # dali.backend.ReleaseUnusedMemory()

    def __len__(self):
        return self.external_data.full_iterations

    def __iter__(self):
        for token in self.iterator:
            x = token[0]["x"]
            y = token[0]["y"].squeeze()
            t = token[0]["t"].squeeze()
            yield [x, y, t]


class PtychoExternalInputCallable:
    def __init__(
        self,
        taskset,
        task_id,
        batch_size,
        shard_id,
        num_shards,
        shuffle=False,
        training=True,
    ):
        self.task_id = task_id
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.shuffle = shuffle and training

        self.diff_data, self.ampli_data, self.phase_data = taskset
        num_samples = len(self.diff_data)

        # Preparing the train/validation sets
        random.seed(42)
        num_samples_eval = int(0.1 * num_samples)
        self.random_indices = random.sample(range(num_samples), num_samples_eval)

        if training:
            train_indices = []
            for j in range(num_samples):
                if j not in self.random_indices:
                    train_indices.append(j)
            # Shuffle inside the current task
            if self.shuffle:
                random.shuffle(train_indices)
            self.random_indices = train_indices

        self.full_iterations = len(self.random_indices) // batch_size

        self.perm = None  # permutation of indices
        # so that we don't have to recompute the `self.perm` for every sample
        self.last_seen_epoch = None

    def __call__(self, sample_info):
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration
        # From my experiments, shuffling at every epoch doesn't increase the acc.
        # Same effect as shuffling only once. Still doing it.
        if self.shuffle:
            if self.last_seen_epoch != sample_info.epoch_idx:
                self.last_seen_epoch = sample_info.epoch_idx
                self.perm = np.random.default_rng(
                    seed=42 + sample_info.epoch_idx
                ).permutation(len(self.random_indices))
            sample_idx = self.perm[sample_info.idx_in_epoch]
        else:
            sample_idx = sample_info.idx_in_epoch

        return (
            self.diff_data[self.random_indices[sample_idx]],
            self.ampli_data[self.random_indices[sample_idx]],
            self.phase_data[self.random_indices[sample_idx]],
            np.int32([self.task_id]),
        )


class PtychoDaliDataLoader:
    def __init__(
        self,
        taskset,
        task_id,
        batch_size=1,
        num_workers=1,
        device_id=1,
        shard_id=0,
        num_shards=1,
        precision=32,
        training=True,
        **kwargs,
    ):
        cuda = torch.cuda.is_available()
        decoder_device, device = ("mixed", "gpu") if cuda else ("cpu", "cpu")
        crop_size = 128

        file_paths = taskset.get_raw_samples()[0]
        diffraction_paths = [f"{p}/cropped_exp_diffr_data.npy" for p in file_paths]
        rspace_paths = [f"{p}/patched_psi.npy" for p in file_paths]

        task_diff_data = []
        task_ampli_data = []
        task_phase_data = []
        for i, _ in enumerate(
            tqdm(
                file_paths,
                desc=f"Loading {len(file_paths)} perspectives for task {task_id+1}",
            )
        ):
            # Calculating the phase and amplitude from the real-space data
            rspace_data = np.load(rspace_paths[i])
            ampli_data = np.abs(rspace_data)
            phase_data = np.angle(rspace_data)

            # Filtering out void patterns
            av_vals = np.mean(phase_data**2, axis=(1, 2))
            mean_phsqr_val = 0.02
            idx = np.argwhere(av_vals >= mean_phsqr_val)

            shard_size = len(idx) // num_shards
            shard_offset = shard_id * shard_size

            # Concatenating scan position(s) for this task
            diff_data = np.load(diffraction_paths[i])
            task_diff_data.extend(
                diff_data[idx][shard_offset : shard_offset + shard_size]
            )
            task_ampli_data.extend(
                ampli_data[idx][shard_offset : shard_offset + shard_size]
            )
            task_phase_data.extend(
                phase_data[idx][shard_offset : shard_offset + shard_size]
            )

        task_diff_data = np.array(task_diff_data, dtype=np.float32)
        task_ampli_data = np.array(task_ampli_data, dtype=np.float32)
        task_phase_data = np.array(task_phase_data, dtype=np.float32)
        taskset = (task_diff_data, task_ampli_data, task_phase_data)

        # Not used yet because of https://github.com/NVIDIA/DALI/issues/5265
        # In the future, maybe the two pipelines should be chained?
        # - https://github.com/NVIDIA/DALI/issues/3702
        # - https://github.com/NVIDIA/DALI/issues/5070
        @pipeline_def(batch_size=1, num_threads=1, device_id=device_id)
        def input_pipeline():
            """
            This pipeline reads scan positions from the disk. One scan perspective
            (containing many diffraction patterns) is returned at every
            iteration.

            https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/numpy_reader.html
            https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.readers.numpy.html
            """
            file_paths = taskset.get_raw_samples()[0]
            diffraction_paths = [f"{p}/cropped_exp_diffr_data.npy" for p in file_paths]
            rspace_paths = [f"{p}/patched_psi.npy" for p in file_paths]

            # GPU Direct Storage :)
            # TODO: use ROIs to manage shards
            diff_data = fn.readers.numpy(
                device="gpu",
                files=diffraction_paths,
                shard_id=shard_id,
                num_shards=num_shards,
            )
            # This npy file contains complex numbers, unfortunately not
            # supported by DALI (https://github.com/NVIDIA/DALI/issues/5265).
            rspace_data = fn.readers.numpy(
                device="gpu",
                files=rspace_paths,
                shard_id=shard_id,
                num_shards=num_shards,
            )

            return diff_data, rspace_data

        # input_pipeline = input_pipeline()
        # input_pipeline.build()

        self.external_data = PtychoExternalInputCallable(
            taskset, task_id, batch_size, shard_id, num_shards, training=training
        )

        @pipeline_def(
            batch_size=batch_size,
            num_threads=num_workers or 1,
            device_id=device_id,
            py_num_workers=1,
            py_start_method="spawn",
        )
        def callable_pipeline():
            """
            This pipeline read individual diffraction patterns from a scan
            position read by a previous pipeline.
            """
            images, amp, ph, task_ids = fn.external_source(
                source=self.external_data,
                num_outputs=4,
                dtype=[types.FLOAT, types.FLOAT, types.FLOAT, types.INT32],
                batch=False,
                parallel=False,
            )

            # Cropping down from 256x256 to 128x128
            """
            if training:
                images = fn.decoders.image(
                    images, device=decoder_device, output_type=types.RGB
                )
                images = fn.resize(
                    images,
                    device=device,
                    resize_x=crop_size,
                    resize_y=crop_size,
                    interp_type=types.INTERP_TRIANGULAR,
                )
                mirror = fn.random.coin_flip(probability=0.5)
            else:
                images = fn.decoders.image(
                    inputs, device=decoder_device, output_type=types.RGB
                )
                images = fn.resize(
                    images,
                    device=device,
                    size=val_size,
                    mode="not_smaller",
                    interp_type=types.INTERP_TRIANGULAR,
                )
                mirror = False
            """

            if cuda:
                images = images.gpu()
                amp = amp.gpu()
                ph = ph.gpu()
                task_ids = task_ids.gpu()

            return images, amp, ph, task_ids

        self.pipeline = callable_pipeline()
        self.pipeline.build()

        self.iterator = DALIGenericIterator(
            self.pipeline,
            ["x", "amp", "ph", "t"],
            last_batch_padded=True,
            prepare_first_batch=True,
            auto_reset=True,
        )

    def release(self):
        del self.pipeline, self.iterator
        # dali.backend.ReleaseUnusedMemory()

    def __len__(self):
        return self.external_data.full_iterations

    def __iter__(self):
        for token in self.iterator:
            x = token[0]["x"]
            amp = token[0]["amp"]
            ph = token[0]["ph"]
            t = token[0]["t"].squeeze()
            y = torch.zeros_like(t)
            yield [x, y, amp, ph, t]
