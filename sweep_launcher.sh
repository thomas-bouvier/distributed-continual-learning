#!/bin/bash

HOST="$1"
WANDB_PROJECT="$2"
SWEEP_CONF="$3"
SWEEP_ID="$4"

# Check if WANDB_MODE is set
if [ -z "$WANDB_MODE" ]; then
  echo "Error: WANDB_MODE is not set. Please set it to 'run'."
  exit 1
fi

# Check if WANDB_MODE is set to "run"
if [ "$WANDB_MODE" != "run" ]; then
  echo "Error: WANDB_MODE is set to '$WANDB_MODE'. Please set it to 'run'."
  exit 1
fi

# Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
  echo "Error: WANDB_API_KEY is not set. Please set it to your API key."
  exit 1
fi

# Check if SPACK_ENV is set
if [ -z "$SPACK_ENV" ]; then
  echo "Error: SPACK_ENV is not set. Please activate your environment."
  exit 1
fi

# Check if HOST is set
if [ -z "$HOST" ]; then
  echo "Error: HOST is not set. Please provide it with ./sweep_launcher <host> <wandb_project> <sweep_conf>."
  exit 1
fi

# Check if WANDB_PROJECT is set
if [ -z "$WANDB_PROJECT" ]; then
  echo "Error: WANDB_PROJECT is not set. Please provide it with ./sweep_launcher <host> <wandb_project> <sweep_conf>."
  exit 1
fi

# Check if SWEEP_CONF is set
if [ -z "$SWEEP_CONF" ]; then
  echo "Error: SWEEP_CONF is not set. This corresponds to a key in sweep.py. Please provide it with ./sweep_launcher <host> <wandb_project> <sweep_conf>."
  exit 1
fi

cd /root/distributed-continual-learning/ && sed -i "s/host = .*/host = \"$HOST\"/g" "sweep.py"
cd /root/distributed-continual-learning/ && sed -i "s/sweep_conf = .*/sweep_conf = \"$SWEEP_CONF\"/g" "sweep.py"

cd /root/distributed-continual-learning/ && git config --global --add safe.directory /root/distributed-continual-learning

if [ -z "$SWEEP_ID" ]; then
    cd /root/distributed-continual-learning/ && wandb sweep sweep.yaml --project $WANDB_PROJECT 2>&1 | eval $(grep -o "wandb agent .*")
else
    cd /root/distributed-continual-learning/ && wandb agent $SWEEP_ID
fi
