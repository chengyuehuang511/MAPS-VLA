#!/bin/bash

#SBATCH --nodes=2
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G

set -euo pipefail

# --- Your env ---

# Optional NCCL nic (set to your cluster's high-speed NIC if needed, e.g., ib0)
# export NCCL_SOCKET_IFNAME=ib0

# RDZV endpoint = first hostname in the allocation
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
GPUS_PER_NODE=8
current_time=$(date +%Y%m%d-%H%M%S)

cd "${MAPS_PROJECT_DIR}"
data_name=bridge

# Launch exactly one torchrun *per node*
srun -N "$SLURM_NNODES" -n "$SLURM_NNODES" --ntasks-per-node=1 -u bash -lc "
  $PYTHON_BIN -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/train.py \
    --vlm_path pretrained_models/prism-dinosiglip-224px+7b \
    --config_file_path pretrained_models/configs-openvla-7b \
    --data_root_dir data/modified_oxe_rlds \
    --dataset_name $data_name \
    --run_root_dir outputs/bridge/openvlaoft \
    --use_film False \
    --num_images_in_input 1 \
    --use_proprio False \
    --use_lora True \
    --use_fz False \
    --use_minivlm False \
    --use_prismatic_vlm True \
    --image_aug True \
    --num_steps_before_decay 100000 \
    --max_steps 100000 \
    --save_freq 5000 \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training False \
    --batch_size 4 \
    --grad_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --lora_rank 32 \
    --use_pro_version True \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_project "openvla-oft-$data_name" \
    --run_id_note OPENVLA-OFT--BRIDGE--$current_time \
    --freeze_vlm False \
    --freeze_language False \
    --unfreeze_last_llm_layer False \
    --freeze_vision False \
    --freeze_dino False \
    --weight_decay_scheduler layerwise_decay \
    --optimizer SPD \
    --weight_decay 0.8 \
"