# distributed-continual-learning

Distributed continual learning with Horovod + PyTorch.

## Installation

Datasets should be added in `datasets/`, or using the following:

```
ln -s /home/tbouvier/Documents/datasets/ datasets
```

## Neomem

Make sure to install [Neomem](https://gitlab.inria.fr/Kerdata/Kerdata-Codes/distributed-rehearsal-buffer). Simlink it using `ln -s ../distributed-rehearsal-buffer cpp_loader`.

## Usage

Parameters defined in the `config.yaml` override CLI parameters. However, values for `agent_config`, `model_config`, `tasksets_config` will be concatenated with those defined by CLI, instead of override them ;

Values for `optimizer_regime` will override regimes defined by `models/` in Python.

| Parameter name | Required | Description | Possible values |
|---|---|---|---|
| `--model` | Yes | DL model to instanciate  | `mnistnet`, `resnet` |
| `--model-config` |   | Model-specific parameters  | Res-Net models accept `"{'depth': 50}"` |
| `--agent` | Default: `base` | DL model to instanciate  | `base`, `nil` |
| `--agent-config` |   | Agent-specific parameters  | `"{'reset_state_dict': True}"` allows to reset the model internal state between tasks |
| `--tasksets-config` |   | Scenario configuration, as defined in the [`continuum` package](https://continuum.readthedocs.io/en/latest/tutorials/scenarios/scenarios.html)  | Class-incremental scenario with 2 tasks: `"{'scenario': 'class', 'initial_increment': 5, 'increment': 5}"` allows to reset the model internal state between tasks<br>Instance-incremental scenario (domain) with 2 tasks: `"{'scenario': 'domain', 'num_tasks': 5}"`<br>`"{'concatenate_tasksets': True}"` allows to concatenate previous tasksets before next task |
| `--dataset` |   | Dataset  | `mnist`, `cifar10`, `cifar100`, `tinyimagenet`, `imagenet`, `imagenet_blurred` |
## Examples

### Deep learning

Usual deep learning can be done using this project:

```
python main.py --model candlenet --dataset candle
python main.py --model mnistnet --dataset mnist
python main.py --model resnet18 --dataset cifar10
python main.py --model resnet50 --dataset cifar100
python main.py --model resnet50 --dataset imagenet_blurred
python main.py --model resnet101 --dataset tinyimagenet
```

### Continual learning

```
python main.py --model mnistnet --dataset mnist --tasksets-config "{'scenario': 'class', 'initial_increment': 5, 'increment': 5}"
python main.py --model resnet18 --dataset cifar10 --tasksets-config "{'scenario': 'class', 'initial_increment': 4, 'increment': 3}"
python main.py --model resnet18 --agent nil --dataset cifar100 --tasksets-config "{'scenario': 'domain', 'num_tasks': 5}"
python main.py --model resnet18 --agent nil --agent-config "{'rehearsal_size': 100}" --dataset cifar10 --tasksets-config "{'scenario': 'class', 'initial_increment': 4, 'increment': 3}"
python main.py --model resnet18 --agent nil --agent-config "{'rehearsal_size': 100}" --dataset imagenet100small --tasksets-config "{'scenario': 'class', 'initial_increment': 40, 'increment': 30}"
python main.py --model resnet50 --agent nil --dataset tinyimagenet --tasksets-config "{'scenario': 'domain', 'num_tasks': 5}"
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

**nil**: global sampling in Python, very inefficient

**nil_cpp**: global sampling with C++ backend

**nil_local**: local sampling in Python, not correct as representatives are not shared between workers
