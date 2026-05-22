#!/bin/bash
# Full pipeline: run baselines → run DOMOO → process results
# Prerequisites: Off-MOO-Bench installed and baseline results available

conda activate off-moo

# Run DOMOO main experiments (5 seeds)
bash scripts/domoo_main.sh

# Run DOMOO ablation studies
bash scripts/domoo_ablation1.sh
bash scripts/domoo_ablation2.sh
bash scripts/domoo_ablation3.sh
bash scripts/domoo_ablation4.sh
bash scripts/domoo_ablation5.sh

# Process results (requires baseline results in ./result/)
python DOMOO/process_result_hv.py
python DOMOO/process_result_igdoff.py
# Processed results in ./dst/
