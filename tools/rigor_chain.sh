#!/bin/bash
# Full rigor chain on thh: exp6 (ETTm2, push-forward target) -> exp7 (baselines) -> exp8 (validation)
export SLB_THREADS=4
cd ~/repos/stochastic-latent-bounds
P=~/repos/slb-venv/bin/python
$P experiments/exp6_grid.py > /tmp/exp6.log 2>&1
$P experiments/exp7_baselines.py > /tmp/exp7.log 2>&1
$P experiments/exp8_validation.py > /tmp/exp8.log 2>&1
echo "chain done $(date)" > /tmp/chain_done.txt
