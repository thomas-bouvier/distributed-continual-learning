#!/bin/bash

HOST="$1"
SWEEP_ID="$2"

cd /root/distributed-continual-learning/ && sed -i "s/host = .*/host = \"$HOST\"/g" "sweep.py"

cd /root/distributed-continual-learning/ && git config --global --add safe.directory /root/distributed-continual-learning

if [[ ! -v SWEEP_ID ]]; then
    cd /root/distributed-continual-learning/ && wandb sweep sweep.yaml --project distributed-continual-learning 2>&1 | eval $(grep -o "wandb agent .*")
else
    cd /root/distributed-continual-learning/ && wandb agent $SWEEP_ID
fi
