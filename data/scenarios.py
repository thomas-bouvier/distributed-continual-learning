import warnings
import numpy as np

from data.tasksets import DiffractionPathTaskSet

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import _BaseScenario
from typing import Callable, List, Optional, Union


class ReconstructionIncremental(_BaseScenario):
    """Continual Loader, generating instance incremental consecutive tasks.

    Scenario: Classes are always the same but instances change (NI scenario)

    :param cl_dataset: A continual dataset.
    :param nb_tasks: The scenario number of tasks.
    :param transformations: A list of transformations applied to all tasks. If
                            it's a list of list, then the transformation will be
                            different per task.
    :param random_seed: A random seed to init random processes.
    """

    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_tasks: Optional[int] = None,
        transformations: Union[List[Callable], List[List[Callable]]] = None,
        random_seed: int = 1,
    ):
        self.cl_dataset = cl_dataset
        self._nb_tasks = self._setup(nb_tasks)
        super().__init__(
            cl_dataset=cl_dataset,
            nb_tasks=self._nb_tasks,
            transformations=transformations,
        )

        self._random_state = np.random.RandomState(seed=random_seed)

    def _setup(self, nb_tasks: Optional[int]) -> int:
        x, t = self.cl_dataset.get_data()

        if (
            nb_tasks is not None and nb_tasks > 0
        ):  # If the user wants a particular nb of tasks
            task_ids = self._split_dataset(x, nb_tasks)
            self.dataset = (x, task_ids)
        elif (
            t is not None
        ):  # Otherwise use the default task ids if provided by the dataset
            self.dataset = (x, t)
            nb_tasks = len(np.unique(t))
        else:
            raise Exception(
                f"The dataset ({self.cl_dataset}) doesn't provide task ids, "
                f"you must then specify a number of tasks, not ({nb_tasks}."
            )

        return nb_tasks

    def __getitem__(self, task_index: Union[int, slice]):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A train PyTorch's Datasets.
        """
        if isinstance(task_index, slice) and isinstance(self.trsf, list):
            raise ValueError(
                f"You cannot select multiple task ({task_index}) when you have a "
                "different set of transformations per task"
            )

        x, t, _, data_indexes = self._select_data_by_task(task_index)

        return DiffractionPathTaskSet(
            x,
            t,
            trsf=self.trsf[task_index] if isinstance(self.trsf, list) else self.trsf,
            target_trsf=None,
            bounding_boxes=self.cl_dataset.bounding_boxes,
        )

    def _split_dataset(self, y, nb_tasks):
        tasks_ids = np.repeat(
            np.arange(nb_tasks),
            np.shape(y)[0] // nb_tasks,
        )
        if np.shape(y)[0] % nb_tasks != 0:
            tasks_ids = np.concatenate(
                (tasks_ids, np.arange(nb_tasks)[: np.shape(y)[0] % nb_tasks])
            )
        return tasks_ids

    def _select_data_by_task(
        self, task_index: Union[int, slice, np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, Union[int, List[int]]]:
        """Selects a subset of the whole data for a given task.

        This class returns the "task_index" in addition of the x, y, t data.
        This task index is either an integer or a list of integer when the user
        used a slice. We need this variable when in segmentation to disentangle
        samples with multiple task ids.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A tuple of numpy array being resp. (1) the data, (2) the targets,
                 (3) task ids, and (4) the actual task required by the user.
        """

        # conversion of task_index into a list

        if isinstance(task_index, slice):
            start = task_index.start if task_index.start is not None else 0
            stop = task_index.stop if task_index.stop is not None else len(self) + 1
            step = task_index.step if task_index.step is not None else 1
            task_index = list(range(start, stop, step))
            if len(task_index) == 0:
                raise ValueError(
                    f"Invalid slicing resulting in no data (start={start}, end={stop}, step={step})."
                )

        if isinstance(task_index, np.ndarray):
            task_index = list(task_index)

        x, t = self.dataset  # type: ignore

        if isinstance(task_index, list):
            task_index = [
                t if t >= 0 else self._handle_negative_indexes(t, len(self))
                for t in task_index
            ]
            if len(t.shape) == 2:
                data_indexes = np.unique(np.where(t[:, task_index] == 1)[0])
            else:
                data_indexes = np.where(np.isin(t, task_index))[0]
        else:
            if task_index < 0:
                task_index = self._handle_negative_indexes(task_index, len(self))

            if len(t.shape) == 2:
                data_indexes = np.where(t[:, task_index] == 1)[0]
            else:
                data_indexes = np.where(t == task_index)[0]

        selected_x = x[data_indexes]
        selected_t = t[data_indexes]

        return (
            selected_x,
            selected_t,
            task_index,
            data_indexes,
        )

    def _handle_negative_indexes(self, index: int, total_len: int) -> int:
        if index < 0:
            index = index % total_len
        return index
