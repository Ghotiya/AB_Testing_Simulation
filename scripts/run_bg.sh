#!/usr/bin/env bash
# Run a command detached so it can't disturb other jobs on this machine:
# nohup, lowest CPU priority, no GPU visible, single-threaded math libraries.
# Usage: scripts/run_bg.sh <name> <command...>   -> logs/<name>.log, logs/<name>.pid
set -u
cd "$(dirname "$0")/.."
# Project environment: this project's own uv venv (see requirements.txt / requirements.lock).
source .venv/bin/activate
mkdir -p logs
name=$1
shift
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
nohup nice -n 19 "$@" > "logs/$name.log" 2>&1 &
echo "$!" > "logs/$name.pid"
echo "started $name pid $! -> logs/$name.log"
