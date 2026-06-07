#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G


cd "${MAPS_PROJECT_DIR}"

name=""

# srun -u ${PYTHON_BIN} -m scripts.merge_lora \
#     --use_minivla False \
#     --use_prismatic_vlm True \
#     --vlm_path pretrained_models/prism-dinosiglip-224px+7b \
#     --lora_finetuned_checkpoint_dir ${name}
    
srun -u ${PYTHON_BIN} -m  experiments.robot.simpler.run_simpler_eval \
    --model_family openvla \
    --pretrained_checkpoint $name \
    --task_suite_name simpler_widowx \
    --center_crop True \
    --use_proprio False \
    --num_trials_per_task 50 \
