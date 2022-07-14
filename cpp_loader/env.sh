#!/bin/bash

export TORCH_USE_RTLD_GLOBAL=YES
# Bogdan
#export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/torch/lib
# Thomas
export LD_LIBRARY_PATH=$HOME/.conda/envs/horovod-py39/lib/python3.9/site-packages/torch/lib:/opt/libfabric/lib:/opt/libfabric/lib/libfabric:$LD_LIBRARY_PATH
