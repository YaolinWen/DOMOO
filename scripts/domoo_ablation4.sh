#!/bin/bash
# Ablation: w.o. PSMG (no PSL model generation, use surrogate only)
seeds="1000 2000 3000 4000 5000"
tasks="mo_hopper_v2 in1kmop7 re24 portfolio dtlz3"
model="DOMOO"; train_modes="Vallina"
MAX_JOBS=3; AVAILABLE_GPUS="1 2 3"; MAX_RETRIES=1

cd "$(dirname "$0")/.."
source /opt/conda/bin/activate off-moo 2>/dev/null || source activate off-moo
get_gpu() { local g=($AVAILABLE_GPUS); echo ${g[(($1 % ${#g[@]}))]}; }
check() { while [ "$(jobs -p|wc -l)" -ge "$MAX_JOBS" ]; do sleep 1; done; }

job=0
for seed in $seeds; do for task in $tasks; do for train_mode in $train_modes; do
    check; gpu=$(get_gpu $job); ((job++))
    CUDA_VISIBLE_DEVICES=$gpu python DOMOO/domoo.py --model=$model --train_mode=$train_mode --task=$task --retrain_model=False --seed=$seed --results_dir=./ablation/no_psl --no_psl=True &
done; done; done
wait && echo "ALL DONE"
