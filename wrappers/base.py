import abc
import torch.nn as nn

from abc import abstractmethod

class Agent(nn.Module, metaclass=abc.ABCMeta):
    '''
    Abstract module which is inherited by each and every continual learning algorithm.
    '''
    def __init__(self, model, config, cuda=False, state_dict=None):
        super(Agent, self).__init__()
        self.model = model
        self.config = config
        self.cuda = cuda

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def forward(self, inputs):
        return self.model(inputs)

    @abstractmethod
    def before_all_tasks(self):
        pass

    @abstractmethod
    def after_all_tasks(self):
        pass

    @abstractmethod
    def before_every_task(self, task_id, train_taskset):
        pass

    @abstractmethod
    def after_every_task(self):
        pass

    @abstractmethod
    def before_every_epoch(self, i_epoch):
        pass

    @abstractmethod
    def after_every_epoch(self):
        pass

    @abstractmethod
    def before_every_batch(self, i_batch, input, target):
        pass

    @abstractmethod
    def after_every_batch(self):
        pass

    def get_state_dict(self):
        return self.model.state_dict()