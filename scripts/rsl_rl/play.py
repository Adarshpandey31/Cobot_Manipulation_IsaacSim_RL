# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import argparse
import inspect
import os
import sys
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from rsl_rl.algorithms import PPO

# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum play steps. Use -1 for infinite.")

# Pickup debug logging
parser.add_argument("--pickup_debug", action="store_true", default=False, help="Enable pickup debug CSV logging.")
parser.add_argument(
    "--debug_csv",
    type=str,
    default="debug_logs/pickup_policy_debug.csv",
    help="CSV path for pickup debug logging.",
)
parser.add_argument("--debug_env_id", type=int, default=0, help="Environment index to log.")
parser.add_argument("--debug_hover_height", type=float, default=0.18)
parser.add_argument("--debug_grasp_height_offset", type=float, default=0.035)
parser.add_argument("--debug_close_threshold", type=float, default=0.10)
parser.add_argument(
    "--debug_target_quat",
    type=float,
    nargs=4,
    default=(0.0, 1.0, 0.0, 0.0),
    help="Target TCP quaternion [w x y z] used for orientation debug.",
)

# append RSL-RL cli arguments
# NOTE: this already adds --checkpoint, --load_run, etc.
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse known args and leave Hydra overrides separate
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Imports after app launch
# -----------------------------------------------------------------------------

import cobot.tasks  # noqa: F401
import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import scripts/pickup_debug_logger.py
SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

try:
    from pickup_debug_logger import PickupDebugLogger
except Exception:
    PickupDebugLogger = None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Play with RSL-RL agent."""

    # -------------------------------------------------------------------------
    # Task/checkpoint setup
    # -------------------------------------------------------------------------
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set environment seed and device
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # -------------------------------------------------------------------------
    # Create Isaac environment
    # -------------------------------------------------------------------------
    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        raw_env = gym.wrappers.RecordVideo(raw_env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    agent_cfg_dict = agent_cfg.to_dict()

    # Keep only PPO keys supported by installed rsl_rl version.
    # Preserve algorithm["class_name"] because OnPolicyRunner expects it.
    if "algorithm" in agent_cfg_dict:
        original_algo_cfg = agent_cfg_dict["algorithm"]
        algo_class_name = original_algo_cfg.get("class_name", "PPO")

        ppo_sig = inspect.signature(PPO.__init__)
        valid_algo_keys = set(ppo_sig.parameters.keys())
        valid_algo_keys.discard("self")
        valid_algo_keys.discard("actor_critic")
        valid_algo_keys.discard("device")
        valid_algo_keys.discard("multi_gpu_cfg")

        filtered_algo_cfg = {"class_name": algo_class_name}
        filtered_algo_cfg.update({k: v for k, v in original_algo_cfg.items() if k in valid_algo_keys})

        removed_algo_keys = sorted(set(original_algo_cfg.keys()) - set(filtered_algo_cfg.keys()))
        if removed_algo_keys:
            print("[INFO] Removed unsupported PPO algorithm keys for play:", removed_algo_keys)

        print("[INFO] Final PPO algorithm keys for play:", sorted(filtered_algo_cfg.keys()))
        agent_cfg_dict["algorithm"] = filtered_algo_cfg

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)

    # obtain trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # extract normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # -------------------------------------------------------------------------
    # Pickup debug logger
    # -------------------------------------------------------------------------
    pickup_logger = None
    if args_cli.pickup_debug:
        if PickupDebugLogger is None:
            raise RuntimeError(
                "Could not import PickupDebugLogger. Make sure this file exists:\nscripts/pickup_debug_logger.py"
            )

        pickup_logger = PickupDebugLogger(
            csv_path=args_cli.debug_csv,
            env_id=args_cli.debug_env_id,
            hover_height=args_cli.debug_hover_height,
            grasp_height_offset=args_cli.debug_grasp_height_offset,
            close_threshold=args_cli.debug_close_threshold,
            target_quat=tuple(args_cli.debug_target_quat),
        )
        print(f"[INFO] Pickup debug logging enabled: {args_cli.debug_csv}")

    # -------------------------------------------------------------------------
    # Play loop
    # -------------------------------------------------------------------------
    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        if pickup_logger is not None:
            try:
                pickup_logger.log(env.unwrapped, actions, timestep)
            except Exception as err:
                print(f"[PICKUP-DEBUG ERROR] step={timestep}: {err}")

        timestep += 1

        if args_cli.video and timestep >= args_cli.video_length:
            break

        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # -------------------------------------------------------------------------
    # Close
    # -------------------------------------------------------------------------
    if pickup_logger is not None:
        pickup_logger.close()
        print(f"[INFO] Pickup debug CSV saved to: {args_cli.debug_csv}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
