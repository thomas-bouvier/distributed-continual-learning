# distributed-deep-learning

## Usage

```
pip install -r requirements.txt
pip uninstall horovod
HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod[pytorch]
```

Datasets should be added in `data/`, or using the following:

```
ln -s /home/tbouvier/Dev/horovod-e2clab-examples/artifacts/data/ data
```

## Examples

```
python main.py --model mnistnet --dataset mnist
python main.py --model resnet --dataset cifar10
python main.py --model resnet --dataset cifar100
```

Change the network interface if needed: `HOROVOD_GLOO_IFACE=wlo1`.

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