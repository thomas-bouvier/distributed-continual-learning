# distributed-deep-learning

Distributed deep/continual learning with Horovod + PyTorch.

## Usage

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

## Examples

### Deep learning

```
python main.py --model candlenet --dataset candle
python main.py --model mnistnet --dataset mnist
python main.py --model resnet --dataset cifar10
python main.py --model resnet --model-config "{'depth': 50}" --dataset cifar100
```

### Continual learning

```
python main.py --model mnistnet --dataset mnist --tasksets-config "{'scenario': 'class', 'initial_increment': 5, 'increment': 5}"
python main.py --model resnet --model-config "{'depth': 101}" --dataset cifar10 --tasksets-config "{'scenario': 'class', 'initial_increment': 4, 'increment': 3}"
python main.py --model resnet --agent nil --dataset cifar100 --tasksets-config "{'scenario': 'domain', 'num_tasks': 5}"
python main.py --model resnet --model-config "{'depth': 101}" --agent icarl --dataset tinyimagenet --tasksets-config "{'scenario': 'domain', 'num_tasks': 5}"
```

Change the network interface if needed: `HOROVOD_GLOO_IFACE=wlo1`.

## Credits

Inspired by https://github.com/eladhoffer/convNet.pytorch.

Thanks to Hugo Chaugier for his initial work on continual learning.
