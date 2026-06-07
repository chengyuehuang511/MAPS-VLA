"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/simpler/run_simpler_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ simpler_widowx ... ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm

sys.path.append('.')
import wandb
from experiments.robot.simpler.simpler_benchmark import get_benchmark
from experiments.robot.simpler.simpler_utils import (
    convert_maniskill,
    get_simpler_dummy_action,
    get_simpler_env,
    get_simpler_img,
)

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    save_rollout_video,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    hf_token: str = Path(".hf_token")                       # Model family
    pretrained_checkpoint: Union[str, Path] = "pretrained/minivla"     # Pretrained checkpoint path
    vlm_checkpoint: Optional[Union[str, Path]] = None  # (For Prismatic only) VLM checkpoint path
    image_sequence_len: int = 1
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    obs_history: int = 1                             # Number of images to pass in from history
    use_wrist_image: bool = False                    # Use wrist images (doubles the number of input images)
    
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_vq_action_tokenizer: bool = False            # If True, uses VQ-VAE-based action tokenizer
    use_minivlm: bool = True                         # If True, uses minivlm
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input
    num_open_loop_steps: int = 5                     # Number of actions to execute open-loop before requerying policy

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "simpler_widowx"          # Task suite.
    initial_states_type: str = "eval"
    #                                       Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 0                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    prefix: str = ''

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "prismatic"        # Name of W&B project to log to (use default!)
    wandb_entity: Optional[str] = None          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    save_version: str = "vla-adapter"                # version of the model

    # fmt: on


import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_to_rpy(qx, qy, qz, qw):
    # Quaternion -> roll, pitch, yaw (XYZ convention)
    r = R.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    return roll, pitch, yaw

def extract_gripper(qpos, mode="mean", bounds=None):
    """
    qpos: full agent qpos (len>=8 for WidowX; last two are finger joints)
    mode: 'mean' | 'sum' | 'right' | 'left'
    bounds: (min_width, max_width) for optional [0,1] normalization if using 'sum' or 'mean'
    """
    q_l, q_r = float(qpos[-2]), float(qpos[-1])

    if mode == "mean":
        g = 0.5 * (q_l + q_r)
    elif mode == "sum":
        g = q_l + q_r
    elif mode == "left":
        g = q_l
    elif mode == "right":
        g = q_r
    else:
        raise ValueError("unknown gripper mode")

    # Optional normalization to [0,1]
    if bounds is not None:
        g = (g - bounds[0]) / max(1e-8, (bounds[1] - bounds[0]))
        g = float(np.clip(g, 0.0, 1.0))
    return float(g)

def simplerenv_to_bridge7(obs, normalize_gripper=False, gripper_bounds=(0.0, 0.04), gripper_mode="mean"):
    """
    Convert SimplERenv obs to Bridge/WidowX 7D state: [x, y, z, roll, pitch, yaw, gripper]
    - Position/quaternion from obs['extra']['tcp_pose'] (xyz + quat)
    - Gripper from obs['agent']['qpos'] last two entries (finger joints)
    """
    tcp = np.asarray(obs['extra']['tcp_pose'], dtype=np.float32)  # [x,y,z,qx,qy,qz,qw]
    x, y, z = tcp[:3].tolist()
    qx, qy, qz, qw = tcp[3:].tolist()
    roll, pitch, yaw = quat_to_rpy(qx, qy, qz, qw)

    qpos = np.asarray(obs['agent']['qpos'], dtype=np.float32)
    if normalize_gripper:
        grip = extract_gripper(qpos, mode=gripper_mode, bounds=gripper_bounds)
    else:
        # raw finger value (mean of the two finger joints)
        grip = extract_gripper(qpos, mode=gripper_mode, bounds=None)

    state7 = np.array([x, y, z, roll, pitch, yaw, grip], dtype=np.float32)
    return state7

# Example:
# state = simplerenv_to_bridge7(obs, normalize_gripper=True, gripper_bounds=(0.0, 0.04))
# print(state)  # -> [x, y, z, roll, pitch, yaw, gripper]


@draccus.wrap()
def eval_simpler(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    if cfg.model_family == "prismatic":
        cfg.unnorm_key = "bridge_dataset"
    else:
        cfg.unnorm_key = "bridge_orig"

    # Load model
    model = get_model(cfg)
    print("use_minivlm:", cfg.use_minivlm)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        print("Loading proprio projector...")
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=7,  # 7-dimensional proprio for WidowX
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression:
        print("Loading action head...")
        action_head = get_action_head(cfg, model.llm_dim)
        
    # Load noisy action projector if using diffusion
    noisy_action_projector = None

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family in ["openvla", "prismatic"]:
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"{cfg.prefix}EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize SIMPLER task suite

    task_suite = get_benchmark(cfg.task_suite_name)()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default SIMPLER envs
        if cfg.initial_states_type == "eval":
            seeds = itertools.count(1000)
        elif cfg.initial_states_type == "train":
            seeds = itertools.count(0)
        else:
            raise ValueError("Unsupported initial states type")

        # Initialize LIBERO environment and task description
        env = get_simpler_env(task, cfg.model_family)
        task_description = env.get_language_instruction()

        # Start episodes
        task_episodes, task_successes = 0, 0
        for _ in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment to specified seed (initial state)
            obs, reset_info = env.reset(seed=next(seeds))

            # Setup
            t = 0
            replay_images = []
            replay_wrist_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            elif cfg.task_suite_name.startswith("simpler"):
                max_steps = 150  # data is at 5Hz, so this is 30 seconds
            else:
                raise NotImplementedError

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    obs, reward, done, trunc, info = env.step(get_simpler_dummy_action(cfg.model_family))
                    t += 1
                    continue

                # Get preprocessed image
                img = get_simpler_img(env, obs, resize_size)

                # Save preprocessed image for replay video
                replay_images.append(img)

                """from libero"""
                def prepare_observation(obs, resize_size):

                    """Prepare observation for policy input."""
                    # Get preprocessed images
                    img = get_simpler_img(env, obs, resize_size)
                    # wrist_img = get_simpler_img(env, obs, resize_size, key="robot0_eye_in_hand_image")

                    # Prepare observations dict
                    observation = {
                        "full_image": img,
                        # "wrist_image": wrist_img,
                        # "state": obs["extra"]["tcp_pose"],
                        'state': simplerenv_to_bridge7(obs, normalize_gripper=False, gripper_bounds=(0.0, 0.04)),
                        # obs["agent"]["qpos"][:7]
                        # np.concatenate(
                        #     (obs["agent"]["qpos"][:3], quat2axisangle(obs["agent"]["qpos"][3:7]), obs["agent"]["qpos"][-1:])
                        # )
                    }
                    # print("""obs["agent"]["qpos"][:7]""")
                    # print('obs', obs)
                    # print("obs['extra'] keys:", obs["extra"].keys())
                    # print("state shape:", observation["state"].shape)

                    return observation, img  # Return both processed observation and original image for replay
                observation, img = prepare_observation(obs, resize_size)
                replay_images.append(img)

                # # use_wrist_image
                # if cfg.use_wrist_image:
                #     raise NotImplementedError
                #     # wrist_img = get_simpler_img(obs, resize_size, key="robot0_eye_in_hand_image")
                #     # replay_wrist_images.append(wrist_img)

                # # buffering #obs_history images, optionally
                # image_history = replay_images[-cfg.obs_history :]
                # if len(image_history) < cfg.obs_history:
                #     image_history.extend([replay_images[-1]] * (cfg.obs_history - len(image_history)))

                # # same but for optional wrist images
                # if cfg.use_wrist_image:
                #     wrist_image_history = replay_wrist_images[-cfg.obs_history :]
                #     if len(wrist_image_history) < cfg.obs_history:
                #         wrist_image_history.extend(
                #             [replay_wrist_images[-1]] * (cfg.obs_history - len(wrist_image_history))
                #         )
                #     # interleaved images [... image_t, wrist_t ...]
                #     image_history = [val for tup in zip(image_history, wrist_image_history) for val in tup]

                # # Prepare observations dict
                # # Note: OpenVLA does not take proprio state as input
                # observation = {
                #     "full_image": image_history,
                #     "state": obs["extra"]["tcp_pose"],
                # }

                # Query model to get action
                # action = get_action(
                #     cfg,
                #     model,
                #     observation,
                #     task_description,
                #     processor=processor,
                # )
                action = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    action_tokenizer=None,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm
                )

                # TODO figure out if below is libero only
                # # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                # action = normalize_gripper_action(action, binarize=True)
                # # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                # # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                # if cfg.model_family in ["openvla", "prismatic"]:
                #     action = invert_gripper_action(action)

                # Execute action in environment
                if isinstance(action, list):
                    for a in action:
                        obs, reward, done, trunc, info = env.step(convert_maniskill(a))
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1
                    if done:
                        break
                else:
                    obs, reward, done, trunc, info = env.step(convert_maniskill(action))
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                # except Exception as e:
                #     print(f"Caught exception: {e}")
                #     log_file.write(f"Caught exception: {e}\n")
                #     break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Save at most 5 successes and at most 5 failures
            if cfg.use_wandb and ((done and task_successes < 5) or (not done and task_episodes - task_successes < 5)):
                group = "success" if done else "failure"
                idx = task_successes if done else task_episodes - task_successes
                wandb.log(
                    {f"{task_description}/{group}/{idx}": wandb.Video(np.array(replay_images).transpose(0, 3, 1, 2))}
                )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_simpler()
