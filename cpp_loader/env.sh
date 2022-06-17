#!/bin/bash

export TORCH_USE_RTLD_GLOBAL=YES
# Bogdan
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/torch/lib
# Thomas
#export LD_LIBRARY_PATH=$HOME/.conda/envs/horovod/lib/python3.8/site-packages/torch/lib
# container on g5k
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.8/dist-packages/torch/