#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:4"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G


cd "${MAPS_PROJECT_DIR}"

num_gpus=4
num_processes=$((num_gpus*1))
task_suite_names=(
    "libero_90"
    "libero_goal"
    "libero_object"
    "libero_spatial"
    "libero_10"
)

name=""

# srun -u ${PYTHON_BIN} -m scripts.merge_lora \
#     --use_minivla False \
#     --use_prismatic_vlm True \
#     --vlm_path pretrained_models/prism-dinosiglip-224px+7b \
#     --lora_finetuned_checkpoint_dir ${name}

for task_suite_name in "${task_suite_names[@]}"; do
    srun -u ${PYTHON_BIN} -m scripts.eval_libero \
        --num-trials-per-task 10 \
        --num-gpus $num_gpus \
        --num-processes $num_processes \
        --task-suite-name $task_suite_name \
        --pretrained-checkpoint $name \
        --save-root results/osmesa/$name \
        --center-crop true \
        --use-l1-regression \
        --use-proprio \
        --num-images-in-input 2 \
        --use-pro-version \
        --use-minivlm \
done
