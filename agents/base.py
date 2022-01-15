import copy
import logging
import tensorwatch
import torch

from torch.utils.tensorboard import SummaryWriter
from utils.meters import AverageMeter, accuracy
from utils.utils import move_cuda

class Agent():
    def __init__(self, model, config, optimizer, criterion, cuda, log_interval,
                 state_dict=None):
        super(Agent, self).__init__()
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.criterion = criterion
        self.cuda = cuda
        self.log_interval = log_interval
        self.epoch = 0
        self.training_steps = 0
        self.writer = None
        self.watcher = None
        self.streams = {}

        # Move model to GPU.
        move_cuda(self.model, cuda)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)
            self.model_snapshot = copy.deepcopy(state_dict)
        else:
            self.model_snapshot = copy.deepcopy(self.model.state_dict())

    """
    Forward pass for the current epoch
    """
    def loop(self, data_regime, average_output=False, training=False):
        meters = {metric: AverageMeter()
                  for metric in ['loss', 'prec1', 'prec5']}

        for i_batch, item in enumerate(data_regime.get_loader()):
            torch.cuda.nvtx.range_push(f"Batch {i_batch}")
            inputs = item[0] # x
            target = item[1] # y

            output, loss = self._step(i_batch,
                                      inputs,
                                      target,
                                      training=training,
                                      average_output=average_output)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, min(self.model.num_classes, 5)))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))

            if i_batch % self.log_interval == 0 or i_batch == len(data_regime.get_loader()):
                logging.info('{phase}: epoch: {0} [{1}/{2}]\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t'
                             'Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t'
                             .format(
                                 self.epoch+1, i_batch, len(data_regime.get_loader()),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 meters=meters))

                prefix='train' if training else 'val'
                if self.writer is not None:
                    self.writer.add_scalar(f"{prefix}_loss", meters['loss'].avg, self.training_steps)
                    self.writer.add_scalar(f"{prefix}_prec1", meters['prec1'].avg, self.training_steps)
                    if training:
                        self.writer.add_scalar('lr', self.optimizer.get_lr()[0], self.training_steps)
                    self.writer.flush()
                if self.watcher is not None:
                    self.observe(trainer=self,
                                model=self.model,
                                optimizer=self.optimizer,
                                data=(inputs, target))
                    self.stream_meters(meters, prefix=prefix)
                    if training:
                        self.write_stream('lr',
                                         (self.training_steps, self.optimizer.get_lr()[0]))
            torch.cuda.nvtx.range_pop()

        meters = {name: meter.avg for name, meter in meters.items()}
        meters['error1'] = 100. - meters['prec1']
        meters['error5'] = 100. - meters['prec5']

        return meters

    def _step(self, i_batch, inputs_batch, target_batch, training=False,
              average_output=False, chunk_batch=1):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.training_steps)

        for i, (inputs, target) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                                 target_batch.chunk(chunk_batch, dim=0))):
            torch.cuda.nvtx.range_push(f"Chunk {i}")

            torch.cuda.nvtx.range_push("Copy to device")
            inputs, target = move_cuda(inputs, self.cuda), move_cuda(target, self.cuda)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Forward pass")
            output = self.model(inputs)
            loss = self.criterion(output, target)
            torch.cuda.nvtx.range_pop()

            if training:
                # accumulate gradient
                loss.backward()
                # SGD step
                torch.cuda.nvtx.range_push("Optimizer step")
                self.optimizer.step()
                torch.cuda.nvtx.range_pop()
                self.training_steps += 1

            outputs.append(output.detach())
            total_loss += float(loss)

            torch.cuda.nvtx.range_pop()

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss

    def train(self, data_regime, average_output=False):
        # switch to train mode
        self.model.train()
        self.write_stream('epoch', (self.training_steps, self.epoch))
        return self.loop(data_regime, average_output=average_output, training=True)

    def validate(self, data_regime, average_output=False):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.loop(data_regime, average_output=average_output, training=False)

    def before_all_tasks(self, train_data_regime, validate_data_regime):
        pass

    def after_all_tasks(self):
        pass

    def before_every_task(self, task_id, train_data_regime, validate_data_regime):
        # Distribute the data
        torch.cuda.nvtx.range_push("Distribute dataset")
        train_data_regime.get_loader(True)
        validate_data_regime.get_loader(True)
        torch.cuda.nvtx.range_pop()

        if task_id > 0:
            if self.config.get('reset_state_dict', False):
                self.model.load_state_dict(copy.deepcopy(self.model_snapshot))
            self.optimizer.reset()

    def after_every_task(self):
        pass

    def set_tensorboard_writer(self, save_path, dummy=False):
        if dummy:
            return False
        self.writer = SummaryWriter(log_dir=save_path)
        return True

    def set_tensorwatch_watcher(self, filename, port=0, dummy=False):
        if dummy:
            return False
        self.watcher = tensorwatch.Watcher(filename=filename, port=port)
        self.get_stream('train_loss')
        self.get_stream('val_loss')
        self.get_stream('train_prec1')
        self.get_stream('val_prec1')
        self.get_stream('lr')
        self.watcher.make_notebook()
        return True

    def get_state_dict(self):
        return self.model.state_dict()

    def get_stream(self, name, **kwargs):
        if self.watcher is None:
            return None
        if name not in self.streams.keys():
            self.streams[name] = self.watcher.create_stream(name=name, **kwargs)
        return self.streams[name]

    def observe(self, **kwargs):
        if self.watcher is None:
            return False
        self.watcher.observe(**kwargs)
        return True

    def stream_meters(self, meters_dict, prefix=None):
        if self.watcher is None:
            return False
        for name, value in meters_dict.items():
            if prefix is not None:
                name = '_'.join([prefix, name])
            value = value.val
            stream = self.get_stream(name)
            if stream is None:
                continue
            stream.write((self.training_steps, value))
        return True

    def write_stream(self, name, values):
        stream = self.get_stream(name)
        if stream is not None:
            stream.write(values)


def base(model, agent_config, optimizer, criterion, cuda, log_interval):
    return Agent(model, agent_config, optimizer, criterion, cuda, log_interval)
