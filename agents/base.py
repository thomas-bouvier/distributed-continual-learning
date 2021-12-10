import mlflow
import torch

from meters import AverageMeter, accuracy
from utils.utils import move_cuda

class Agent():
    def __init__(self, model, config, optimizer, criterion, cuda, log_interval, state_dict=None):
        super(Agent, self).__init__()
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.criterion = criterion
        self.cuda = cuda
        self.log_interval = log_interval
        self.epoch = 0
        self.training_steps = 0

        # Move model to GPU.
        move_cuda(self.model, cuda)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)


    def _step(self, i_batch, inputs_batch, target_batch, training=False, average_output=False, chunk_batch=1):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.training_steps)

        for i, (inputs, target) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                                 target_batch.chunk(chunk_batch, dim=0))):
            inputs, target = move_cuda(inputs, self.cuda), move_cuda(target, self.cuda)

            output = self.model(inputs)
            loss = self.criterion(output, target)

            if training:
                # accumulate gradient
                loss.backward()
                # SGD step
                self.optimizer.step()
                self.training_steps += 1

            outputs.append(output.detach())
            total_loss += float(loss)

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss


    """
    Forward pass for the current epoch
    """
    def loop(self, data_loader, average_output=False, training=False):
        meters = {metric: AverageMeter()
                  for metric in ['loss', 'prec1', 'prec5']}

        for i_batch, item in enumerate(data_loader):
            inputs = item[0] # x
            target = item[1] # y

            output, loss = self._step(i_batch,
                                      inputs,
                                      target,
                                      training=training,
                                      average_output=average_output)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))

            mlflow.log_metrics({
                'loss': float(loss),
                'prec1': float(prec1),
                'prec5': float(prec5)
            }, step=self.epoch)

            if i_batch % self.log_interval == 0 or i_batch == len(data_loader) - 1:
                print('{phase}: epoch: {0} [{1}/{2}]\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t'
                             'Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t'
                             .format(
                                 self.epoch+1, i_batch, len(data_loader),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 meters=meters))

        meters = {name: meter.avg for name, meter in meters.items()}
        meters['error1'] = 100. - meters['prec1']
        meters['error5'] = 100. - meters['prec5']

        return meters


    def train(self, data_loader, average_output=False):
        # switch to train mode
        self.model.train()
        return self.loop(data_loader, average_output=average_output, training=True)


    def validate(self, data_loader, average_output=False):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.loop(data_loader, average_output=average_output, training=False)


    def before_all_tasks(self, taskets):
        pass


    def after_all_tasks(self):
        pass


    def before_every_task(self, task_id, train_taskset):
        pass


    def after_every_task(self):
        pass


    def get_state_dict(self):
        return self.model.state_dict()


def base(agent_config, model_config, model, optimizer, criterion, cuda, log_interval):
    return Agent(model, agent_config, optimizer, criterion, cuda, log_interval)
