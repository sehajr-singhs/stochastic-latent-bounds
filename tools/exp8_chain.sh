#!/bin/bash
# exp8 launcher: waits for exp6/exp7 to finish, then runs the validation suite
cd ~/repos/stochastic-latent-bounds
while pgrep -f "exp7_baselines.py" > /dev/null || pgrep -f "exp6_grid.py" > /dev/null; do
  sleep 60
done
exec ~/repos/slb-venv/bin/python experiments/exp8_validation.py > /tmp/exp8.log 2>&1
