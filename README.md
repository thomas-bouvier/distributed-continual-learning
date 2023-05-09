# distributed-continual-learning

This is a PyTorch + Horovod implementation of the continual learning experiments with deep neural networks described in the following article:

* [Three types of incremental learning](https://www.nature.com/articles/s42256-022-00568-3) (2022, *Nature Machine Intelligence*)

Continual learning approaches implemented here are based on rehearsal, which is delegated in a separate high-performance C++ backend [Neomem](https://gitlab.inria.fr/Kerdata/Kerdata-Codes/neomem).

This repository supports experiments in the academic continual learning setting, whereby a classification-based problem is split up into multiple, non-overlapping tasks that must be learned sequentially.

Some Python code comes from the [convNet.pytorch repository](https://github.com/eladhoffer/convNet.pytorch/tree/master).

## Installation

The current version of the code has been tested with Python 3.10.0 with the following package versions:

* `pytorch 1.13.1`
* `torchvision 0.14.1`
* `horovod 0.26.1`
* `continuum 1.2.7`

Make sure to install [Neomem](https://gitlab.inria.fr/Kerdata/Kerdata-Codes/distributed-rehearsal-buffer) if using a `cpp` implementation for rehearsal. Simlink it using `ln -s ../distributed-rehearsal-buffer cpp_loader`.

Further Python-packages used are listed in requirements.txt. Assuming Python and pip are set up, these packages can be installed using:

```bash
pip install -r requirements.txt
```

In an HPC environment, we strongly advise to use Spack to manage dependencies.

## Usage

Parameters defined in the `config.yaml` override CLI parameters. However, values for `agent_config`, `model_config`, `tasksets_config` will be concatenated with those defined by CLI, instead of override them ;

Values for `optimizer_regime` will override regimes defined by `models/` in Python.

| Parameter name | Required | Description | Possible values |
|---|---|---|---|
| `--model` | Yes | DL model to instanciate  | `mnistnet`, `resnet18`, `resnet50`, `efficientnetv2`, `convnext` |
| `--model-config` |   | Model-specific parameters  | ConvNext requires `"{'lr_min': 1e-6}"` |
| `--agent` | Default: `base` | DL model to instanciate  | `base`, `nil` |
| `--agent-config` |   | Agent-specific parameters  | `"{'reset_state_dict': True}"` allows to reset the model internal state between tasks |
| `--tasksets-config` |   | Scenario configuration, as defined in the [`continuum` package](https://continuum.readthedocs.io/en/latest/tutorials/scenarios/scenarios.html)  | Class-incremental scenario with 2 tasks: `"{'scenario': 'class', 'initial_increment': 5, 'increment': 5}"` allows to reset the model internal state between tasks<br>Instance-incremental scenario (domain) with 2 tasks: `"{'scenario': 'domain', 'num_tasks': 5}"`<br>`"{'concatenate_tasksets': True}"` allows to concatenate previous tasksets before next task |
| `--dataset` |   | Dataset  | `mnist`, `cifar10`, `cifar100`, `tinyimagenet`, `imagenet`, `imagenet_blurred` |

## Examples

### Deep learning

Usual deep learning can be done using this project:

```
python main.py --model mnistnet --dataset mnist
python main.py --model resnet18 --dataset cifar100
python main.py --model resnet50 --dataset tinyimagenet
python main.py --model efficientnetv2 --dataset imagenet_blurred
```

### Continual learning

```
python main.py --model mnistnet --dataset mnist --tasksets-config "{'scenario': 'class', 'initial_increment': 5, 'increment': 5}"
python main.py --model resnet18 --dataset cifar10 --tasksets-config "{'scenario': 'class', 'initial_increment': 4, 'increment': 3}"
python main.py --model resnet18 --agent nil --dataset cifar100 --tasksets-config "{'scenario': 'domain', 'num_tasks': 5}"
python main.py --model resnet18 --agent nil --agent-config "{'rehearsal_ratio': 100}" --dataset cifar10 --tasksets-config "{'scenario': 'class', 'initial_increment': 4, 'increment': 3}"
python main.py --model resnet18 --agent nil --agent-config "{'rehearsal_ratio': 100}" --dataset imagenet100small --tasksets-config "{'scenario': 'class', 'initial_increment': 40, 'increment': 30}"
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

**nil_cpp**: global sampling with Neomem

**nil_cpp_cat**: global sampling with Neomem, variant of `nil_cpp`

**nil_local**: local sampling in Python, not correct as representatives are not shared between workers
