from meters import AverageMeter, accuracy

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F

class Trainer(object):
    def __init__(self, model, optimizer, criterion, cuda, log_interval):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cuda = cuda
        self.log_interval = log_interval
        self.epoch = 0
        self.training_steps = 0


    def _step(self, i_batch, inputs_batch, target_batch, training=False, average_output=False, chunk_batch=1):
        outputs = []
        total_loss = 0

        #self.timer.start(f"epoch_{self.epoch }-batch_{i_batch}")

        if training:
            #self.timer.start(f"epoch_{self.epoch}-batch_{i_batch}-zero_grad")
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.training_steps)
            #self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}-zero_grad")

        for i, (inputs, target) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                                 target_batch.chunk(chunk_batch, dim=0))):
            if self.cuda:
                #self.timer.start(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-move_batch_to_gpu")
                inputs, target = inputs.cuda(), target.cuda()
                #self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-move_batch_to_gpu")

            #self.timer.start(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-forward_pass")
            output = self.model(inputs)
            #self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-forward_pass")

            #self.timer.start(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-compute_loss")
            loss = self.criterion(output, target)
            #self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-compute_loss")

            if training:
                #self.timer.start(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-backward_pass")
                # accumulate gradient
                loss.backward()
                #self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-backward_pass")

            if training:
                #self.timer.start(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-optimizer_step")
                # SGD step
                self.optimizer.step()
                self.training_steps += 1
                #self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-optimizer_step")

            outputs.append(output.detach())
            total_loss += float(loss)

        #self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}")

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss


    def forward(self, data_loader, training=False, average_output=False):
        meters = {metric: AverageMeter()
                  for metric in ['loss', 'prec1', 'prec5']}

        for i_batch, (inputs, target) in enumerate(data_loader):
            #self.timer.start(f"start_epoch_{self.epoch}-batch_{i_batch}")
            output, loss = self._step(i_batch,
                                      inputs,
                                      target,
                                      training=training,
                                      average_output=average_output)
            #self.timer.start(f"end_epoch_{self.epoch}-batch_{i_batch}")

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
                                 self.epoch, i_batch, len(data_loader),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 meters=meters))

        meters = {name: meter.avg for name, meter in meters.items()}
        meters['error1'] = 100. - meters['prec1']
        meters['error5'] = 100. - meters['prec5']

        return meters


    def train(self, data_loader, average_output=False):
        # switch to train mode
        self.model.train()
        return self.forward(data_loader, average_output=average_output, training=True)


    def validate(self, data_loader, average_output=False):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.forward(data_loader, average_output=average_output, training=False)