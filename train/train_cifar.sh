#!/bin/bash
script_dir="$(command dirname -- "${0}")"
LOG="$script_dir/xnornet-cifar-train-log.txt.`date +'%Y-%m-%d_%H-%M-%S'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python -u $script_dir/cifar10_train.py
