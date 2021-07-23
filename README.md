# distributed-deep-learning

```
pip install -r requirements.txt
pip uninstall horovod
HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod[pytorch]
```

Datasets should be added in `data/`, or using the following:

```
ln -s /home/tbouvier/Dev/horovod-e2clab-examples/artifacts/datasets/ data
```