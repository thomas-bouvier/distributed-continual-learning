import models

from .base import Agent

class basic_agent(Agent):
    def __init__(self, model, config, cuda=False, state_dict=None):
        super(basic_agent, self).__init__(model, config, cuda, state_dict)

    def before_all_tasks(self):
        pass

    def after_all_tasks(self):
        pass

    def before_every_task(self, task_id, train_taskset):
        pass

    def after_every_task(self):
        pass

    def before_every_epoch(self, i_epoch):
        pass

    def after_every_epoch(self):
        pass

    def before_every_batch(self, i_batch, input, target):
        pass

    def after_every_batch(self):
        pass


def basic(model_config, agent_config=None, cuda=False):
    model_name = model_config.pop('model', 'resnet')
    model = models.__dict__[model_name]

    return basic_agent(model(model_config), agent_config, cuda)
