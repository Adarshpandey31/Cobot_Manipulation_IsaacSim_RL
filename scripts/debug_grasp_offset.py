import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug UR10e Robotiq grasp-center / TCP offsets.")
parser.add_argument("--task", type=str, default="Template-Cobot-RL-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import cobot.tasks  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # Step a few times so body poses and frame transformer data are updated.
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    for _ in range(5):
        env.step(actions)

    robot = env.unwrapped.scene["robot"]
    body_names = robot.data.body_names

    print("\n================ BODY NAMES DEBUG ================")
    for i, name in enumerate(body_names):
        if "wrist" in name or "finger" in name or "robotiq" in name or "gripper" in name:
            print(i, name)
    print("==================================================\n")

    def has_body(name: str) -> bool:
        return name in body_names

    def idx(name: str) -> int:
        if name not in body_names:
            raise RuntimeError(
                f"Body name '{name}' not found.\n"
                f"Available body names:\n" + "\n".join(body_names)
            )
        return body_names.index(name)

    wrist_idx = idx("wrist_3_link")
    wrist_pos = robot.data.body_pos_w[:, wrist_idx, :]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx, :]

    print("\n================ WRIST DEBUG ================")
    print("wrist_pos =", wrist_pos[0].detach().cpu().numpy().tolist())
    print("wrist_quat =", wrist_quat[0].detach().cpu().numpy().tolist())
    print("=============================================\n")

    # Main check: ee_frame must be offset from wrist if GRASP_CENTER_OFFSET is active.
    if "ee_frame" in env.unwrapped.scene.keys():
        ee_frame = env.unwrapped.scene["ee_frame"]
        ee_pos = ee_frame.data.target_pos_w[..., 0, :]

        ee_offset_world = ee_pos - wrist_pos
        ee_offset_in_wrist, _ = subtract_frame_transforms(
            wrist_pos,
            wrist_quat,
            ee_pos,
        )

        print("\n================ EE FRAME / TCP DEBUG ================")
        print("ee_frame_pos =", ee_pos[0].detach().cpu().numpy().tolist())
        print("ee_minus_wrist_world =", ee_offset_world[0].detach().cpu().numpy().tolist())
        print("ee_offset_in_wrist_frame =", ee_offset_in_wrist[0].detach().cpu().numpy().tolist())
        print("Expected approx local offset if configured: [0.0, 0.0, 0.18]")
        print("======================================================\n")
    else:
        print("\n[WARNING] ee_frame not found in scene. Check joint_pos_env_cfg.py\n")

    # Old finger-origin check: useful only to confirm these body origins are not reliable TCP points.
    if has_body("left_outer_finger") and has_body("right_outer_finger"):
        left_outer_idx = idx("left_outer_finger")
        right_outer_idx = idx("right_outer_finger")

        left_outer_pos = robot.data.body_pos_w[:, left_outer_idx, :]
        right_outer_pos = robot.data.body_pos_w[:, right_outer_idx, :]
        outer_mid_w = 0.5 * (left_outer_pos + right_outer_pos)

        outer_offset_in_wrist, _ = subtract_frame_transforms(
            wrist_pos,
            wrist_quat,
            outer_mid_w,
        )

        print("\n================ OUTER FINGER BODY-ORIGIN DEBUG ================")
        print("outer_mid_w =", outer_mid_w[0].detach().cpu().numpy().tolist())
        print("outer_mid_offset_in_wrist =", outer_offset_in_wrist[0].detach().cpu().numpy().tolist())
        print("If this is near [0,0,0], these body origins are not actual fingertip/TCP points.")
        print("===============================================================\n")
    else:
        print("\n[INFO] Outer finger bodies not found; skipping outer midpoint debug.\n")

    if has_body("left_inner_finger") and has_body("right_inner_finger"):
        left_inner_idx = idx("left_inner_finger")
        right_inner_idx = idx("right_inner_finger")

        left_inner_pos = robot.data.body_pos_w[:, left_inner_idx, :]
        right_inner_pos = robot.data.body_pos_w[:, right_inner_idx, :]
        inner_mid_w = 0.5 * (left_inner_pos + right_inner_pos)

        inner_offset_in_wrist, _ = subtract_frame_transforms(
            wrist_pos,
            wrist_quat,
            inner_mid_w,
        )

        print("\n================ INNER FINGER BODY-ORIGIN DEBUG ================")
        print("inner_mid_w =", inner_mid_w[0].detach().cpu().numpy().tolist())
        print("inner_mid_offset_in_wrist =", inner_offset_in_wrist[0].detach().cpu().numpy().tolist())
        print("If this is near [0,0,0], these body origins are not actual fingertip/TCP points.")
        print("===============================================================\n")
    else:
        print("\n[INFO] Inner finger bodies not found; skipping inner midpoint debug.\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()