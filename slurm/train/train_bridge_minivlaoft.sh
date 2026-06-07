#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G


cd "${MAPS_PROJECT_DIR}"

data_name=bridge

srun -u ${PYTHON_BIN} -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  scripts/train.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/modified_oxe_rlds \
  --dataset_name $data_name \
  --run_root_dir outputs/bridge/minivlaoft \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --use_lora False \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 100000 \
  --max_steps 100000 \
  --save_freq 20000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_project "$data_name" \
  --run_id_note VLA-OFT--BRIDGE--$current_time \
  --freeze_vlm False \
  --freeze_language False \
  --unfreeze_last_llm_layer False \
  --unfreeze_early_language False \
  --freeze_vision False \
  --freeze_dino False \
  --unfreeze_dino False \
  --weight_decay_scheduler layerwise_decay \
  --optimizer SPD \
  --weight_decay 3.0 \