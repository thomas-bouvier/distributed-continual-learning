# distributed-deep-learning

## Usage

```
# In your conda env:
pip install -r requirements.txt
pip uninstall horovod
HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 pip install --no-cache-dir horovod[pytorch]
horovodrun --check-build
```

Datasets should be added in `data/`, or using the following:

```
ln -s /home/tbouvier/Dev/horovod-e2clab-examples/artifacts/data/ data
```

## Examples

### Deep learning

```
python main.py --model mnistnet --dataset mnist
python main.py --model resnet --dataset cifar10
python main.py --model resnet --model-config "{'depth': 48}" --dataset cifar100
```

### Continual learning

```
python main.py --model mnistnet --dataset cifar100 --dataset-config "{'scenario': 'class', 'initial_increment': 10, 'increment': 2}" --continual
python main.py --model resnet --model-config "{'depth': 101}" --dataset tinyimagenet --dataset-config "{'scenario': 'domain', 'nb_tasks': 5}" --continual
python main.py --model resnet --agent nil --dataset mnist --dataset-config "{'scenario': 'domain', 'nb_tasks': 5}" --continual
python main.py --model resnet --model-config "{'depth': 101}" --agent icarl --dataset tinyimagenet --dataset-config "{'scenario': 'domain', 'nb_tasks': 5}" --continual
```

Change the network interface if needed: `HOROVOD_GLOO_IFACE=wlo1`.

#### Scenarios

##### Naive baseline

```
python main.py --model mnistnet --dataset cifar100 --dataset-config "{'scenario': 'class', 'initial_increment': 10, 'increment': 2}" --continual
python main.py --model resnet --model-config "{'depth': 101}" --dataset tinyimagenet --dataset-config "{'scenario': 'domain', 'nb_tasks': 5}" --continual
python main.py --model resnet --agent nil --dataset mnist --dataset-config "{'scenario': 'domain', 'nb_tasks': 5}" --continual
python main.py --model resnet --model-config "{'depth': 101}" --agent icarl --dataset tinyimagenet --dataset-config "{'scenario': 'domain', 'nb_tasks': 5}" --continual
```

## Credits

Inspired by https://github.com/eladhoffer/convNet.pytorch.

```
@inproceedings{hoffer2018fix,
  title={Fix your classifier: the marginal value of training the last weight layer},
  author={Elad Hoffer and Itay Hubara and Daniel Soudry},
  booktitle={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=S1Dh8Tg0-},
}
```