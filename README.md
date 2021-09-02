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
python main.py --model mnistnet --dataset-name MNIST
```

Change the network interface if needed: `HOROVOD_GLOO_IFACE=wlo1`.

## Credits

Inspired by https://github.com/eladhoffer/convNet.pytorch.