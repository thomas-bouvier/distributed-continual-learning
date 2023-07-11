import numpy as np

from continuum.tasks import BaseTaskSet, TaskType
from torchvision import transforms
from typing import Union, Optional, List


class DiffractionArrayTaskSet(BaseTaskSet):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        trsf: Union[transforms.Compose, List[transforms.Compose]],
        target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]],
        bounding_boxes: Optional[np.ndarray] = None,
    ):
        super().__init__(x, y, t, trsf, target_trsf, bounding_boxes=bounding_boxes)
        self.data_type = TaskType.IMAGE_ARRAY

    def get_classes(self) -> List[int]:
        return [1]
