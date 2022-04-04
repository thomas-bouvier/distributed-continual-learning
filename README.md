# distributed-continual-learning

Distributed continual learning with Horovod + PyTorch.

## Installation

```
# In your conda env:
pip install -r requirements.txt
pip uninstall horovod
HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 pip install --no-cache-dir horovod[pytorch]
horovodrun --check-build
```

Datasets should be added in `datasets/`, or using the following:

```
ln -s /home/tbouvier/Documents/datasets/ datasets
```

## Usage

Parameters defined in the `config.yaml` override CLI parameters.

| Parameter name | Required | Description | Possible values |
|---|---|---|---|
| `--model` | Yes | DL model to instanciate  | `mnistnet`, `resnet` |
| `--model-config` |   | Model-specific parameters  | Res-Net models accept `"{'depth': 50}"` |
| `--agent` | Default: `base` | DL model to instanciate  | `base`, `nil`, `icarl` |
| `--agent-config` |   | Agent-specific parameters  | `"{'reset_state_dict': True}"` allows to reset the model internal state between tasks |
| `--tasksets-config` |   | Scenario configuration, as defined in the [`continuum` package](https://continuum.readthedocs.io/en/latest/_tutorials/scenarios/scenarios.html)  | Class-incremental scenario with 2 tasks: `"{'scenario': 'class', 'initial_increment': 5, 'increment': 5}"` allows to reset the model internal state between tasks<br>Instance-incremental scenario (domain) with 2 tasks: `"{'scenario': 'domain', 'num_tasks': 5}"`<br>`"{'concatenate_tasksets': True}"` allows to concatenate previous tasksets before next task |
| `--dataset` |   | Dataset  | `mnist`, `cifar10`, `cifar100`, `tinyimagenet`, `imagenet`, `imagenet_blurred` |
## Examples

### Deep learning

Usual deep learning can be done using this project:

```
python main.py --model candlenet --dataset candle
python main.py --model mnistnet --dataset mnist
python main.py --model resnet --dataset cifar10
python main.py --model resnet --model-config "{'depth': 50}" --dataset cifar100
python main.py --model resnet --model-config "{'depth': 101}" --dataset tinyimagenet
```

### Continual learning

```
python main.py --model mnistnet --dataset mnist --tasksets-config "{'scenario': 'class', 'initial_increment': 5, 'increment': 5}"
python main.py --model resnet --model-config "{'depth': 18}" --dataset cifar10 --tasksets-config "{'scenario': 'class', 'initial_increment': 4, 'increment': 3}"
python main.py --model resnet --agent nil --dataset cifar100 --tasksets-config "{'scenario': 'domain', 'num_tasks': 5}"
python main.py --model resnet --model-config "{'depth': 50}" --agent icarl --dataset tinyimagenet --tasksets-config "{'scenario': 'domain', 'num_tasks': 5}"
```

#### Scratch baseline

```
python main.py --model <model> --dataset <dataset> --agent-config "{'reset_state_dict': True}" --tasksets-config "{<..tasksets-config, 'concatenate_tasksets': True>}"
```

#### Transfer baseline

```
python main.py --model <model> --dataset <dataset> --tasksets-config "{<..tasksets-config, 'concatenate_tasksets': True>}"
```

#### Naive baseline

```
python main.py --model <model> --dataset <dataset> --tasksets-config "{<tasksets-config>}"
```

#### NIL implementations

**nil**: default implementation, without parallel memory management

**nil_v1**: synchronous parallel memory management, representatives are synchronized using `Queue`s. The memory manager waits until the main process sends candidates, and the main process waits until the memory manager fills the shared buffer with new representatives. *First implementation v1*.

**nil_v2**: asynchronous version of **nil_v1**. The main process doesn’t wait for the memory manager to fill the shared buffer. If the episodic memory update takes longer than the training time, the process is sped up with the drawback of reusing the same representatives multiple times in a row. *First implementation v2*.

**nil_v3**: management of the episodic memory including the creation of batches from the taskset, their copy on the GPU, the selection of candidates and the concatenation of batches with selected representatives.

**nil_v4**: **nil_v3** including batch sampling. *Second implementation*.

**nil_global**: strategy where representatives are shared between workers
#### iCaRL implementations

**icarl**: default implementation, without parallel memory management

**icarl_v1**: parallel memory management but very little impact observed
## Credits

Some parts of the project are inspired by https://github.com/eladhoffer/convNet.pytorch.

Thanks to Hugo Chaugier for his initial work on continual learning.
