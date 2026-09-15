#!/bin/bash
# Full experiment pipeline on thh. Checkpointed: safe to re-run.
cd ~/repos/stochastic-latent-bounds
PY=~/repos/slb-venv/bin/python
mkdir -p results logs
echo "=== tests ==="
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/ -q 2>&1 | tail -2
echo "=== exp3 ==="
$PY experiments/exp3_tightness.py
echo "=== exp1 ==="
$PY experiments/exp1_main.py
echo "=== exp2 ==="
$PY experiments/exp2_shadow.py
echo "=== figures + site ==="
$PY tools/figures.py
$PY tools/site.py
echo "=== ALL DONE ==="
