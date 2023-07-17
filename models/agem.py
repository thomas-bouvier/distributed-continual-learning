__all__['Agem']

class Agem(ContinualLearner):

    def __init__(
        self,
        backbone: nn.Module,
        optimizer_regime,
        use_amp,
        batch_size,
        config,
        buffer_config,
        batch_metrics=None,
    ):
        super(Agem, self).__init__(
            backbone,
            optimizer_regime,
            use_amp,
            batch_size,
            config,
            buffer_config,
            batch_metrics,
        )

        self.use_memory_buffer = True

        # Implémentation via mammoth à voir pour grad_xy et grad_er
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

    def before_all_tasks(self, train_data_regime):
        self.buffer = Buffer(
            train_data_regime.total_num_classes,
            train_data_regime.sample_shape,
            self.batch_size,
            cuda=self._is_on_cuda(),
            **self.buffer_config,
        )

        x, y, _ = next(iter(train_data_regime.get_loader(0)))
        self.buffer.add_data(x, y, dict(batch=-1))

    def before_every_task(self, task_id, train_data_regime):
        super().before_every_task(task_id, train_data_regime)

        if task_id > 0:
            self.buffer.enable_augmentations()

    def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1

    def overwrite_grad(params, newgrad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = sum(grad_dims[:count + 1])
                this_grad = newgrad[begin: end].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
            count += 1

    def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
        corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
        return gxy - corr * ger

    def train_one_step(self, x, y, meters, step):

        with get_timer(
            "train",
            step["batch"],
            perf_metrics=self.perf_metrics,
            dummy=not measure_performance(step),
        ):
            self.optimizer_regime.update(step["epoch"], step["batch"])
            self.optimizer_regime.zero_grad()

            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.backbone(x)
                loss = self.criterion(outputs,y)

            # Loss Backwards
            self.scaler.scale(loss.sum() / loss.size(0)).backward()

            if not self.buffer.is_empty():
                store_grad(self.parameters,self.grad_xy,self.grad_dims)  # A voir pour le self.parameters()

                buf_x, buf_y = self.buffer.__get_data(step) # step ou step-1 ?
                self.optimizer_regime.zero_grad() # self.net.zero_grad()

                buf_outputs = self.backbone(buf_x)
                buf_loss = self.criterion(buf_outputs,buf_y)
                self.scaler.scale(buf_loss.sum() / buf_loss.size(0)).backward()

                store_grad(self.parameters,self.grad_er,self.grad_dims) # A voir pour le self.parameters()

                dot_prod = torch.dot(self.grad_xy,self.grad_er)
                
                if dot_prod.item() < 0:
                    g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                    overwrite_grad(self.parameters, g_tilde, self.grad_dims)
                else:
                    overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

            self.optimizer_regime.optimizer.synchronize()
            with self.optimizer_regime.optimizer.skip_synchronize():
                self.scaler.step(self.optimizer_regime.optimizer)
                self.scaler.update()

            # Measure accuracy and record metrics
            prec1, prec5 = accuracy(output, aug_y, topk=(1, 5))
            meters["loss"].update(loss.sum() / aug_y.size(0))
            meters["prec1"].update(prec1, aug_x.size(0))
            meters["prec5"].update(prec5, aug_x.size(0))
            meters["num_samples"].update(aug_x.size(0))
            meters["local_rehearsal_size"].update(self.buffer.get_size())