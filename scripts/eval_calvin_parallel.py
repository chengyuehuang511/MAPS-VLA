import os
import json
import time
import copy
import argparse
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, deque
from typing import Optional, Union, List, Tuple

import numpy as np
from termcolor import colored
from moviepy import ImageSequenceClip

import torch
from omegaconf import OmegaConf
import hydra
from tqdm.auto import tqdm

# --- Project imports from VLA-Adapter codebase ---
# These match the original evaluate_calvin.py so this file can live alongside it
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    get_model,
)
from vla_evaluation import DualSystemCalvinEvaluation
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)

# Calvin wrapper (same as original script)
from calvin_env_wrapper import CalvinEnvWrapperRaw

os.environ["FFMPEG_BINARY"] = "auto-detect"
os.environ.setdefault("CALVIN_ROOT", "calvin")
CALVIN_ROOT = os.environ["CALVIN_ROOT"]

DEVICE0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ----------------------
# Config
# ----------------------
@dataclass
class GenerateConfig:
    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "../outputs/calvin-abc"
    use_minivla: bool = False
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_x0_prediction: bool = False
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = False
    num_open_loop_steps: int = 8
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Calvin
    calvin_path: str = "calvin"
    with_depth: bool = True
    with_gripper: bool = True
    with_cfg: bool = True
    enrich_lang: bool = False

    # Eval
    ep_len: int = 360
    num_sequences: int = 1000
    save_root: str = "./evaluation_results/calvin"
    fps: int = 50

    # Parallelism
    num_gpus: int = 8
    num_processes: int = 32

    # Misc
    seed: int = 7
    run_id_note: Optional[str] = None


# ----------------------
# Utility helpers
# ----------------------

def _check_free_gpus() -> List[int]:
    """Return indices of GPUs with < ~1GB used (rough heuristic)."""
    try:
        used_lines = os.popen(
            "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader"
        ).readlines()
        used = [int(x.strip()) for x in used_lines]
        return [i for i, m in enumerate(used) if m < 1000]
    except Exception:
        # Fall back to a single device if nvidia-smi isn't available
        return [0]


def _set_gpu(gpu_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Torch will pick GPU:0 of the visible set
    torch.cuda.empty_cache()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


# ----------------------
# Env & rollout
# ----------------------

def make_env(dataset_path: str, observation_space: dict, device: torch.device):
    val_folder = Path(dataset_path) / "validation"
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    normalized = action.copy()
    orig_low, orig_high = 0.0, 1.0
    normalized[..., -1] = 2 * (normalized[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        sign = np.sign(normalized[..., -1])
        sign = np.array(sign)
        sign[sign == 0.0] = 1
        sign[sign == -0.0] = -1
        normalized[..., -1] = sign
    return normalized


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    inv = action.copy()
    inv[..., -1] *= -1.0
    return inv


def process_action(action: np.ndarray, model_family: str) -> np.ndarray:
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def rollout_hi3(env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len, fps):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    img_dict = {"static": [], "gripper": []}

    for step in range(80):
        action_buffers = [None, None, None]
        action_buffers[0] = model.step(obs, lang_annotation, 0)
        action = process_action(action_buffers[0][0], "openvla")
        obs, *_ = env.step(action.tolist())
        img_dict["static"].append(copy.deepcopy(obs["rgb_obs"]["rgb_static"]))
        img_dict["gripper"].append(copy.deepcopy(obs["rgb_obs"]["rgb_gripper"]))
        current_info = env.get_info()
        if len(task_oracle.get_task_info_for_set(start_info, current_info, {subtask})) > 0:
            _save_clips(img_dict, eval_dir, sequence_i, subtask_i, subtask, True, fps)
            return True

        action_buffers[1] = model.step(obs, lang_annotation, 1)
        action = (action_buffers[0][1] + action_buffers[1][0]) / 2
        action = process_action(action, "openvla")
        obs, *_ = env.step(action.tolist())
        img_dict["static"].append(copy.deepcopy(obs["rgb_obs"]["rgb_static"]))
        img_dict["gripper"].append(copy.deepcopy(obs["rgb_obs"]["rgb_gripper"]))
        current_info = env.get_info()
        if len(task_oracle.get_task_info_for_set(start_info, current_info, {subtask})) > 0:
            _save_clips(img_dict, eval_dir, sequence_i, subtask_i, subtask, True, fps)
            return True

        action_buffers[2] = model.step(obs, lang_annotation, 2)
        action = (action_buffers[0][2] + action_buffers[1][1] + action_buffers[2][0]) / 3
        action = process_action(action, "openvla")
        obs, *_ = env.step(action.tolist())
        img_dict["static"].append(copy.deepcopy(obs["rgb_obs"]["rgb_static"]))
        img_dict["gripper"].append(copy.deepcopy(obs["rgb_obs"]["rgb_gripper"]))
        current_info = env.get_info()
        if len(task_oracle.get_task_info_for_set(start_info, current_info, {subtask})) > 0:
            _save_clips(img_dict, eval_dir, sequence_i, subtask_i, subtask, True, fps)
            return True

        for t in range(2, 7):
            action = (action_buffers[0][t] + action_buffers[1][t - 1] + action_buffers[2][t - 2]) / 3
            action = process_action(action, "openvla")
            obs, *_ = env.step(action.tolist())
            img_dict["static"].append(copy.deepcopy(obs["rgb_obs"]["rgb_static"]))
            img_dict["gripper"].append(copy.deepcopy(obs["rgb_obs"]["rgb_gripper"]))
            current_info = env.get_info()
            if len(task_oracle.get_task_info_for_set(start_info, current_info, {subtask})) > 0:
                _save_clips(img_dict, eval_dir, sequence_i, subtask_i, subtask, True, fps)
                return True

        action = (action_buffers[1][7] + action_buffers[2][6]) / 2
        action = process_action(action, "openvla")
        obs, *_ = env.step(action.tolist())
        img_dict["static"].append(copy.deepcopy(obs["rgb_obs"]["rgb_static"]))
        img_dict["gripper"].append(copy.deepcopy(obs["rgb_obs"]["rgb_gripper"]))
        current_info = env.get_info()
        if len(task_oracle.get_task_info_for_set(start_info, current_info, {subtask})) > 0:
            _save_clips(img_dict, eval_dir, sequence_i, subtask_i, subtask, True, fps)
            return True

        action = action_buffers[2][7]
        action = process_action(action, "openvla")
        obs, *_ = env.step(action.tolist())
        img_dict["static"].append(copy.deepcopy(obs["rgb_obs"]["rgb_static"]))
        img_dict["gripper"].append(copy.deepcopy(obs["rgb_obs"]["rgb_gripper"]))
        current_info = env.get_info()
        if len(task_oracle.get_task_info_for_set(start_info, current_info, {subtask})) > 0:
            _save_clips(img_dict, eval_dir, sequence_i, subtask_i, subtask, True, fps)
            return True

    _save_clips(img_dict, eval_dir, sequence_i, subtask_i, subtask, False, fps)
    return False


def _save_clips(img_dict, eval_dir, sequence_i, subtask_i, subtask, success: bool, fps: int):
    status = "succ" if success else "fail"
    for key in img_dict.keys():
        clip = ImageSequenceClip(img_dict[key], fps=fps)
        clip.write_videofile(
            os.path.join(
                eval_dir, f"{sequence_i}-{subtask_i}-{subtask}-{key}-{status}.mp4"
            ),
            fps=fps,
            codec="libx264",
            bitrate="5000k",
        )


# ----------------------
# Eval core (single process)
# ----------------------

def _init_model(cfg: GenerateConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = get_model(cfg)
    model.set_version("Pro")  # keep parity with original script

    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    noisy_action_projector = (
        get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None
    )
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    eva = DualSystemCalvinEvaluation(
        model,
        proprio_projector,
        noisy_action_projector,
        action_head,
        processor,
        use_x0_prediction=cfg.use_x0_prediction,
    )
    return eva


def _print_and_save(results: List[int], sequences: List[Tuple], eval_result_path: str, task_name: Optional[str] = None, epoch: Optional[int] = None):
    current_data = {}
    avg_seq_len = np.mean(results) if len(results) > 0 else 0.0
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}

    cnt_success = Counter()
    cnt_fail = Counter()
    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {task: {"success": cnt_success[task], "total": total[task]} for task in total}

    data = {"avg_seq_len": float(avg_seq_len), "chain_sr": chain_sr, "task_info": task_info}
    current_data[epoch if epoch is not None else 0] = data

    os.makedirs(os.path.dirname(eval_result_path), exist_ok=True)
    with open(eval_result_path, "w") as f:
        json.dump(current_data, f)


# ----------------------
# Parallel worker
# ----------------------

def _worker(gpu_id: int, proc_idx: int, cfg: GenerateConfig, sequences: List, ret_list: mp.managers.ListProxy, base_save_dir: str):
    try:
        _set_gpu(gpu_id)
        eva = _init_model(cfg)

        # Calvin env & assets
        observation_space = {
            "rgb_obs": ["rgb_static", "rgb_gripper"],
            "depth_obs": ["depth_static", "depth_gripper"],
            "state_obs": ["robot_obs"],
            "actions": ["rel_actions"],
            "language": ["language"],
        }
        eval_dir = get_log_dir(os.path.join(base_save_dir, f"proc{proc_idx}"))
        env = make_env(os.path.join(CALVIN_ROOT, "dataset/task_ABC_D"), observation_space, torch.device("cuda:0"))

        # Task oracle & annotations
        conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
        task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        task_oracle = hydra.utils.instantiate(task_cfg)
        if cfg.enrich_lang:
            with open("./vla-scripts/enrich_lang_annotations.json", "r") as f:
                val_annotations = json.load(f)
        else:
            val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

        results = []
        # tqdm only if single process to avoid messy bars; otherwise simple loop
        iterator = sequences if len(sequences) > 1 else tqdm(sequences)
        for seq_i, (initial_state, eval_sequence) in enumerate(iterator):
            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            success_counter = 0
            for subtask_i, subtask in enumerate(eval_sequence):
                ok = rollout_hi3(
                    env,
                    eva,
                    task_oracle,
                    subtask,
                    val_annotations,
                    debug=False,
                    eval_dir=eval_dir,
                    subtask_i=subtask_i,
                    sequence_i=seq_i,
                    ep_len=cfg.ep_len,
                    fps=cfg.fps,
                )
                if ok:
                    success_counter += 1
                else:
                    break
            results.append(success_counter)

        # push per-process summary to shared list
        ret_list.append({
            "proc_idx": proc_idx,
            "gpu": gpu_id,
            "results": results,
        })

    except Exception as e:
        # Capture errors per process
        err_path = os.path.join(base_save_dir, f"error_proc{proc_idx}.log")
        with open(err_path, "w") as f:
            f.write(str(e) + "\n")
            f.write(traceback.format_exc())


# ----------------------
# Orchestrator
# ----------------------

def split_evenly(items: List, num_bins: int) -> List[List]:
    bins: List[List] = [[] for _ in range(num_bins)]
    for i, item in enumerate(items):
        bins[i % num_bins].append(item)
    return bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", type=str, default="../outputs/calvin-abc")
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--num_processes", type=int, default=32)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--save_root", type=str, default="./evaluation_results/calvin")
    parser.add_argument("--enrich_lang", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = GenerateConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_sequences=args.num_sequences,
        num_processes=args.num_processes,
        num_gpus=args.num_gpus,
        save_root=args.save_root,
        enrich_lang=args.enrich_lang,
        seed=args.seed,
    )

    run_id = f"{_timestamp()}_{Path(cfg.pretrained_checkpoint).name}"
    save_dir = os.path.join(cfg.save_root, run_id)
    os.makedirs(save_dir, exist_ok=True)

    # Determine GPUs to use
    gpus = _check_free_gpus()
    if cfg.num_gpus < len(gpus):
        gpus = gpus[: cfg.num_gpus]
    if len(gpus) == 0:
        gpus = [0]

    # Build full sequence list and shard across processes
    full_sequences = get_sequences(cfg.num_sequences)
    shards = split_evenly(full_sequences, cfg.num_processes)

    manager = mp.Manager()
    ret_list = manager.list()
    procs: List[mp.Process] = []

    for proc_idx, shard in enumerate(shards):
        gpu = gpus[proc_idx % len(gpus)]
        p = mp.Process(
            target=_worker,
            args=(gpu, proc_idx, cfg, shard, ret_list, save_dir),
        )
        procs.append(p)

    for p in procs:
        p.start()
    for p in procs:
        p.join()

    # Aggregate
    all_results: List[int] = []
    proc_records = list(ret_list)
    # Reconstruct the same ordering length to compute success chains
    # (order doesn't affect count_success computation as long as lengths match)
    for rec in proc_records:
        all_results.extend(rec["results"])

    # Persist summary
    eval_result_path = os.path.join(save_dir, "result.json")
    _print_and_save(all_results, full_sequences, eval_result_path, task_name="calvin-parallel")

    # Console summary
    chain = count_success(all_results)
    print(f"Average successful sequence length: {np.mean(all_results) if all_results else 0.0:.3f}")
    print("Success rates for i instructions in a row:")
    for i, sr in enumerate(chain, start=1):
        print(f"  {i}: {sr * 100:.1f}%")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
