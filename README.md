# distributed-continual-learning

This is a PyTorch + Horovod implementation of the continual learning experiments with deep neural networks described in the following article:

* [Three types of incremental learning](https://www.nature.com/articles/s42256-022-00568-3) (2022, *Nature Machine Intelligence*)

Continual learning approaches implemented here are based on rehearsal, which is delegated in a separate high-performance C++ backend [Neomem](https://gitlab.inria.fr/Kerdata/Kerdata-Codes/neomem).

This repository primarily supports experiments in the academic continual learning setting, whereby a classification-based problem is split up into multiple, non-overlapping tasks that must be learned sequentially (class-incremental scenario). Instance-incremental scenarios are supported too.

Some Python code has been inspired by the [mammoth](https://github.com/aimagelab/mammoth) and [convNet.pytorch](https://github.com/eladhoffer/convNet.pytorch/tree/master) repositories.

## Installation

The current version of the code has been tested with Python 3.10 with the following package versions:

* `pytorch 2.0.1`
* `timm 0.9.2`
* `horovod 0.28.1`
* `continuum 1.2.7`

Make sure to install [Neomem](https://gitlab.inria.fr/Kerdata/Kerdata-Codes/neomem) to benefit from global sampling of representatives. Simlink it using `ln -s ../neomem cpp_loader`. If not available, this code will fallback to a Python, local only, low performance rehearsal buffer implementation.

Further Python packages used are listed in requirements.txt. Assuming Python and pip are set up, these packages can be installed using:

```bash
pip install -r requirements.txt
```

In an HPC environment, we strongly advise to use [Spack](https://github.com/spack/spack) to manage dependencies.

## Usage

Parameters defined in the `config.yaml` override CLI parameters. However, values for `backbone_config`, `buffer_config`, `tasksets_config` will be concatenated with those defined by CLI, instead of override them ;

Values for `optimizer_regime` will override regimes defined by `backbone/` in Python.

| Parameter name | Required | Description | Possible values |
|---|---|---|---|
| `--backbone` | Yes | DL backbone model to instanciate  | `mnistnet`, `resnet18`, `resnet50`, `mobilenetv3`, `efficientnetv2`, `convnext`, `ghostnet`, `ptychonn` |
| `--backbone-config` |   | Backbone-specific parameters  | `"{'lr': 0.01, 'lr_min': 1e-6, }"` |
| `--model` | Default: `Vanilla` | Continual Learning strategy | `Vanilla`, `Er`, `Agem`, `Der`, `Derpp` |
| `--model-config` |   | Reset strategies and CL model-specific parameters | `"{'reset_state_dict': True}"` allows to reset the model internal state between tasks<br>`"{'alpha': 0.2}"` is needed for Der model<br>`"{'alpha': 0.2, 'beta': 0.8}"` are needed for Derpp model |
| `--buffer-config` |   | Rehearsal buffer parameters  |  `"{'rehearsal_ratio': 20}"` sets the proportion of the input dataset to be stored in the rehearsal buffer |
| `--tasksets-config` |   | Scenario configuration, as defined in the [`continuum` package](https://continuum.readthedocs.io/en/latest/tutorials/scenarios/scenarios.html)  | Class-incremental scenario with 2 tasks: `"{'scenario': 'class', 'initial_increment': 5, 'increment': 5}"`<br>Instance-incremental scenario with 2 tasks: `"{'scenario': 'instance', 'num_tasks': 5}"`<br>`"{'concatenate_tasksets': True}"` allows to concatenate previous tasksets before next task |
| `--dataset` |   | Dataset  | `mnist`, `cifar10`, `cifar100`, `tinyimagenet`, `imagenet`, `imagenet_blurred`, `ptycho` |

### WandB sweeps

To run a hyperparameter search, first adapt the `sweep.py` (located in this directory) file if needed. Then, configure your optimization objective in `sweep.yaml`.

Make sure you exported your WandB API key before running anything `export WANDB_API_KEY=key` and set `WANDB_MODE=run`. Once you are ready, execute the `sweep_launcher.sh <hostname> [<existing_sweep_id>]` script on the master machine, not forgetting to provide the address of the current machine e.g., `chifflot-7.lille.grid5000.fr:1`.

To stop a sweep run, go to the online WandB dashboard and click "Stop run". To stop the whole sweep process, `ps aux | grep 'wandb agent'` on the machine and kill the process.

## Continual Learning Strategies

Specific implementations have to be selected using `--buffer-config "{'implementation': <implementation>}"`. ER with implementation `standard` was used in the paper.

| Approach | Name | Available Implementations |
|---|---|---|
| Experience Replay (ER) | `Er` | `standard`, `flyweight`, `python` |
| Averaged (A-GEM) | `Agem` | `python` |
| Dark Experience Replay (DER) | `Der` | `python` |
| Dark Experience Replay ++ (DER++) | `Derpp` | `python` |

### Baselines

#### From Scratch

```
python main.py --backbone <backbone_model> --dataset <dataset> --model Vanilla --model-config "{'reset_state_dict': True}" --tasksets-config "{<..tasksets-config, 'concatenate_tasksets': True>}"
```

#### Incremental

```
python main.py --backbone <backbone_model> --dataset <dataset> --model Vanilla --tasksets-config "{<tasksets-config>}"
```

## Examples

### Deep learning

Usual deep learning can be done using this project. Model `Vanilla` will be instanciated by default:

```
python main.py --backbone mnistnet --dataset mnist
python main.py --backbone resnet18 --dataset cifar100
python main.py --backbone resnet50 --dataset tinyimagenet
python main.py --backbone efficientnetv2 --dataset imagenet_blurred
```

### Continual learning

```
python main.py --backbone mnistnet --dataset mnist --tasksets-config "{'scenario': 'class', 'initial_increment': 5, 'increment': 5}"
python main.py --backbone resnet18 --dataset cifar10 --tasksets-config "{'scenario': 'class', 'initial_increment': 4, 'increment': 3}"
python main.py --backbone resnet18 --model Er --dataset cifar100 --tasksets-config "{'scenario': 'instance', 'num_tasks': 5}"
python main.py --backbone resnet18 --model Der --buffer-config "{'rehearsal_ratio': 20}" --dataset cifar10 --tasksets-config "{'scenario': 'class', 'initial_increment': 4, 'increment': 3}"
python main.py --backbone resnet18 --model Derpp --buffer-config "{'rehearsal_ratio': 20}" --dataset imagenet100small --tasksets-config "{'scenario': 'class', 'initial_increment': 40, 'increment': 30}"
python main.py --backbone resnet50 --model Agem --dataset tinyimagenet --tasksets-config "{'scenario': 'instance', 'num_tasks': 5}"
```
