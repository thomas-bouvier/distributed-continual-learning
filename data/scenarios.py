import warnings
import numpy as np

from data.tasksets import DiffractionArrayTaskSet

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
        x, y, t = self.cl_dataset.get_data()

        if (
            nb_tasks is not None and nb_tasks > 0
        ):  # If the user wants a particular nb of tasks
            task_ids = _split_dataset(y, nb_tasks)
            self.dataset = (x, y, task_ids)
        elif (
            t is not None
        ):  # Otherwise use the default task ids if provided by the dataset
            self.dataset = (x, y, t)
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

        x, y, t, _, data_indexes = self._select_data_by_task(task_index)

        return DiffractionArrayTaskSet(
            x,
            y,
            t,
            trsf=self.trsf[task_index] if isinstance(self.trsf, list) else self.trsf,
            target_trsf=None,
            bounding_boxes=self.cl_dataset.bounding_boxes,
        )


def _split_dataset(y, nb_tasks):
    tasks_ids = np.repeat(np.arange(nb_tasks), np.shape(y)[0] // nb_tasks)
    if np.shape(y)[0] % nb_tasks != 0:
        tasks_ids = np.concatenate(
            (tasks_ids, np.arange(nb_tasks)[: np.shape(y)[0] % nb_tasks])
        )
    return tasks_ids
