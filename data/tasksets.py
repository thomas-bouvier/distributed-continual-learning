import numpy as np
import torch

from continuum.tasks import TaskType
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Tuple, Union, Optional, List


class DiffractionArrayTaskSet(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_amp: np.ndarray,
        y_ph: np.ndarray,
        t: np.ndarray,
        trsf: Union[transforms.Compose, List[transforms.Compose]],
        target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]],
        bounding_boxes: Optional[np.ndarray] = None,
    ):
        self._x, self._y_amp, self._y_ph, self._t = x, y_amp, y_ph, t

        # if task index are not provided t is always -1
        if self._t is None:
            self._t = -1 * np.ones_like(y, dtype=np.int64)

        self.trsf = trsf
        self.data_type = TaskType.TENSOR
        self.target_trsf = target_trsf
        self.data_type = TaskType.TENSOR
        self.bounding_boxes = bounding_boxes

        self._to_tensor = transforms.ToTensor()
        # self.data_type = TaskType.IMAGE_ARRAY

    def _transform_y(self, y_amp, y_ph, t):
        """Array of all classes contained in the current task."""
        for i, (y_amp_, y_ph_, t_) in enumerate(zip(y_amp, y_ph, t)):
            y_amp[i] = self.get_task_target_trsf(t_)(y_amp_)
            y_ph[i] = self.get_task_target_trsf(t_)(y_ph_)
        return y_amp, y_ph

    @property
    def nb_classes(self):
        """The number of classes contained in the current task."""
        return len(self.get_classes())

    def get_classes(self) -> List[int]:
        return [0]

    def concat(self, *task_sets):
        """Concat others task sets.

        :param task_sets: One or many task sets.
        """
        for task_set in task_sets:
            if task_set.data_type != self.data_type:
                raise Exception(
                    f"Invalid data type {task_set.data_type} != {self.data_type}, "
                    "all concatenated tasksets must be of the same type."
                )

            self.add_samples(task_set._x, task_set._y_amp, task_set._y_ph, task_set._t)

    def add_samples(
        self,
        x: np.ndarray,
        y_amp: np.ndarray,
        y_ph: np.ndarray,
        t: Union[None, np.ndarray] = None,
    ):
        """Add memory for rehearsal.

        :param x: Sampled data chosen for rehearsal.
        :param y_amp: The associated amplitude patterns of `x_memory`.
        :param y_ph: The associated phase patterns of `x_memory`
        :param t: The associated task ids. If not provided, they will be
                         defaulted to -1.
        """
        self._x = np.concatenate((self._x, x))
        self._y_amp = np.concatenate((self._y_amp, y_amp))
        self._y_ph = np.concatenate((self._y_ph, y_ph))
        if t is not None:
            self._t = np.concatenate((self._t, t))
        else:
            self._t = np.concatenate((self._t, -1 * np.ones(len(x))))

    def __len__(self) -> int:
        """The amount of images in the current task."""
        return self._y_amp.shape[0]

    def get_random_samples(self, nb_samples):
        nb_tot_samples = self._x.shape[0]
        indexes = np.random.randint(0, nb_tot_samples, nb_samples)
        return self.get_samples(indexes)

    def get_samples(self, indexes):
        samples, targets_amp, targets_ph, tasks = [], [], [], []

        for index in indexes:
            # we need to use __getitem__ to have the transform used
            sample, y_amp, y_ph, t = self[index]
            samples.append(sample)
            targets_amp.append(y_amp)
            targets_ph.append(y_ph)
            tasks.append(t)

        return (
            _tensorize_list(samples),
            _tensorize_list(targets_amp),
            _tensorize_list(targets_ph),
            _tensorize_list(tasks),
        )

    def get_raw_samples(self, indexes=None):
        """Get samples without preprocessing, for split train/val for example."""
        if indexes is None:
            return self._x, self._y_amp, self._y_ph, self._t
        return (
            self._x[indexes],
            self._y_amp[indexes],
            self._y_ph[indexes],
            self._t[indexes],
        )

    def get_sample(self, index: int) -> np.ndarray:
        """Returns the tensor corresponding to the given `index`.

        :param index: Index to query the image.
        :return: A Pillow image.
        """
        x = self._x[index]
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return x

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        x = self.get_sample(index)
        y_amp = self._y_amp[index]
        y_ph = self._y_ph[index]
        t = self._t[index]

        if self.target_trsf is not None:
            y_amp = self.get_task_target_trsf(t)(y_amp)
            y_ph = self.get_task_target_trsf(t)(y_ph)

        return x, 0, y_amp, y_ph, t

    def get_task_trsf(self, t: int):
        if isinstance(self.trsf, list):
            return self.trsf[t]
        return self.trsf

    def get_task_target_trsf(self, t: int):
        if isinstance(self.target_trsf, list):
            return self.target_trsf[t]
        return self.target_trsf
