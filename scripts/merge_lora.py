
"""
Loads a checkpoint that only has a LoRA adapter (no merged model) and merges the adapter
into the base VLA-Adapter model. Saves the final checkpoint in the same directory.

Usage:
python vla-scripts/merge_lora_weights_and_save.py \
    --base_checkpoint openvla/openvla-7b \
    --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT/DIR/

python -m vla-scripts.merge_lora_weights_and_save \
    --use_minivla True \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --lora_finetuned_checkpoint_dir outputs/calvin_new/configs+calvin_abc_rlds+b8+lr-0.0002+AdamW+wd-0+x-action_queries+lora-r64+dropout-0.0--image_aug--VLA-Adapter--CALVIN-ABC----50000_chkpt

python -m vla-scripts.merge_lora_weights_and_save \
    --use_minivla True \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --lora_finetuned_checkpoint_dir outputs/calvin_new/configs+calvin_abc_rlds+b8+lr-0.0002+AdamW+wd-0+x-action_queries+lora-r64+dropout-0.0--image_aug--VLA-Adapter--CALVIN-ABC----20000_chkpt

python -m vla-scripts.merge_lora_weights_and_save \
    --use_minivla True \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --lora_finetuned_checkpoint_dir outputs/calvin_new/configs+calvin_abc_rlds+b8+lr-0.0002+AdamW+wd-0+x-action_queries+lora-r64+dropout-0.0--image_aug--VLA-Adapter--CALVIN-ABC----45000_chkpt

python -m vla-scripts.merge_lora_weights_and_save \
    --use_minivla True \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --lora_finetuned_checkpoint_dir outputs/calvin_new/configs+calvin_abc_rlds+b8+lr-0.0002+AdamW+wd-0+x-action_queries+lora-r64+dropout-0.0--image_aug--VLA-Adapter--CALVIN-ABC----25000_chkpt

python -m vla-scripts.merge_lora_weights_and_save \
    --use_minivla True \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --lora_finetuned_checkpoint_dir outputs/calvin_new/configs+calvin_abc_rlds+b8+lr-0.0002+AdamW+wd-0+x-action_queries+lora-r64+dropout-0.0--image_aug--VLA-Adapter--CALVIN-ABC----30000_chkpt

python -m vla-scripts.merge_lora_weights_and_save \
    --use_minivla True \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --lora_finetuned_checkpoint_dir outputs/calvin_new/configs+calvin_abc_rlds+b8+lr-0.0002+AdamW+wd-0+x-action_queries+lora-r64+dropout-0.0--image_aug--VLA-Adapter--CALVIN-ABC----35000_chkpt

python -m vla-scripts.merge_lora_weights_and_save \
    --use_minivla True \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --lora_finetuned_checkpoint_dir outputs/calvin_new/configs+calvin_abc_rlds+b8+lr-0.0002+AdamW+wd-0+x-action_queries+lora-r64+dropout-0.0--image_aug--VLA-Adapter--CALVIN-ABC----40000_chkpt

python -m vla-scripts.merge_lora_weights_and_save \
    --use_minivla True \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --lora_finetuned_checkpoint_dir outputs/calvin_new/configs+calvin_abc_rlds+b8+lr-0.0001+AdamW+wd-0+x-action_queries+lora-r64+dropout-0.0--image_aug--VLA-Adapter--CALVIN-ABC----40000_chkpt

python -m vla-scripts.merge_lora_weights_and_save \
    --use_minivla True \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --lora_finetuned_checkpoint_dir outputs/calvin_overcap/configs+calvin_abc_rlds+b8+lr-0.0001+SPD+wd-0.7+x-action_queries+lora-r64+dropout-0.0+layerwise_decay--image_aug--VLA-Adapter--CALVIN-ABC----40000_chkpt

"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models import load, load_vla

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ConvertConfig:
    # fmt: off

    base_checkpoint: Union[str, Path] = ""                   # Base model checkpoint path/dir (either openvla/openvla-7b or whichever model you fine-tuned / resumed training from)
    lora_finetuned_checkpoint_dir: Union[str, Path] = ""     # Checkpoint directory containing the LoRA adapter
    vlm_path: Union[str, Path] = "" 
    use_minivla: bool = False                        # 
    use_prismatic_vlm: bool = False                        #


    # fmt: on


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if cfg.use_minivla or cfg.use_prismatic_vlm:
        hf_token = ''
        hf_token_path = '.hf_token'
        if os.path.exists(hf_token_path):
            with open(hf_token_path, 'r') as f:
                hf_token = f.read().strip()
        if cfg.use_minivla:
            if 'prism-qwen25-extra-dinosiglip-224px-0_5b' in cfg.vlm_path:
                vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
            else:
                vlm = load_vla(
                    cfg.vlm_path,
                    hf_token=hf_token,
                    load_for_training=True,
                    )
            config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
        elif cfg.use_prismatic_vlm:
            if 'prism-dinosiglip-224px+7b' in cfg.vlm_path:
                vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
            else:
                vlm = load_vla(
                    cfg.vlm_path,
                    hf_token=hf_token,
                    load_for_training=True,
                    )
            config = AutoConfig.from_pretrained("pretrained_models/configs-openvla-7b/config.json")
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device)
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
        ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict
        
        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
        # torch.save(vla.state_dict()['module.base_model.model.action_queries.weight'], checkpoint_dir / f"action_queries--{checkpoint_name_suffix}")
        step = cfg.lora_finetuned_checkpoint_dir.split('--')[-1].replace('_chkpt', '')
        checkpoint_path = os.path.join(cfg.lora_finetuned_checkpoint_dir, f"action_queries--{step}_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            action_queries = torch.load(checkpoint_path, weights_only=True, map_location=vla.device)
            RAW_STATE_DICT['action_queries.weight'] = action_queries
    
        missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
    else:
        # Load Model using HF AutoClasses
        print(f"Loading base model: {cfg.base_checkpoint}")
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.base_checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)

    # Load LoRA weights and merge into base model, then save final checkpoint
    print("Merging LoRA weights into base model...")
    start_time = time.time()
    merged_vla = PeftModel.from_pretrained(vla, os.path.join(cfg.lora_finetuned_checkpoint_dir, "lora_adapter")).to(
        device
    )
    merged_vla = merged_vla.merge_and_unload()
    merged_vla.save_pretrained(cfg.lora_finetuned_checkpoint_dir)
    print(f"\nMerging complete! Time elapsed (sec): {time.time() - start_time}")
    print(f"\nSaved merged model checkpoint at:\n{cfg.lora_finetuned_checkpoint_dir}")


if __name__ == "__main__":
    main()
