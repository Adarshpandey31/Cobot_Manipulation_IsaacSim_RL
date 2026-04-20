import argparse
from enum import Enum, auto

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Local UR10e pick-and-place test")
parser.add_argument("--task", type=str, default="Template-Cobot-v0", help="Task name to launch.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import cobot.tasks  # noqa: F401


class Phase(Enum):
    REST = auto()
    APPROACH_ABOVE_PICK = auto()
    APPROACH_PICK = auto()
    GRASP = auto()
    LIFT = auto()
    MOVE_ABOVE_PLACE = auto()
    LOWER_TO_PLACE = auto()
    RELEASE = auto()
    RETRACT = auto()
    DONE = auto()
    FAIL = auto()


def print_phase(phase: Phase):
    print("\n" + "=" * 90)
    print(f"[PHASE] {phase.name}")
    print("=" * 90)


def reached_xyz(current_pos: torch.Tensor, target_pos: torch.Tensor, threshold: float = 0.015) -> bool:
    err = torch.norm(current_pos - target_pos, dim=1)
    return bool(torch.all(err < threshold).item())


def get_ur10e_downward_quat(num_envs: int, device: torch.device) -> torch.Tensor:
    desired_quat = torch.zeros((num_envs, 4), device=device)
    desired_quat[:, 1] = 1.0
    return desired_quat


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env_cfg.episode_length_s = 20.0
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    dt = env_cfg.sim.dt * env_cfg.decimation

    GRIPPER_OPEN = 1.0
    GRIPPER_CLOSE = -1.0

    desired_quat = get_ur10e_downward_quat(num_envs, device)

    # ------------------------------------------------------------------
    # Inspect action layout so this script doesn't hardcode wrong shape.
    # This was the source of your 8 vs 7 crash.
    # ------------------------------------------------------------------
    am = env.unwrapped.action_manager
    print(am)
    print("active_terms =", am.active_terms)
    print("term_dims    =", am.action_term_dim)
    print("total_dim    =", am.total_action_dim)

    term_dims = am.action_term_dim

    # Supported layouts for this scripted EE-pose agent:
    #   [7]    -> absolute IK arm only  (x,y,z,qw,qx,qy,qz)
    #   [7, 1] -> absolute IK arm + gripper
    if term_dims == [7]:
        arm_has_pose_action = True
        gripper_available = False
        print("[WARN] This env exposes only 7 actions: EE pose only, no gripper action.")
        print("[WARN] Arm motion can be debugged, but grasp/pick will intentionally stop at GRASP.")
    elif term_dims == [7, 1]:
        arm_has_pose_action = True
        gripper_available = True
        print("[INFO] Detected pose IK arm action + gripper action.")
    else:
        raise RuntimeError(
            "zero_agent.py expects an absolute pose IK action layout of [7] or [7, 1]. "
            f"Got term_dims={term_dims}. This script is not compatible with the current task/action config."
        )

    def build_action(target_pos: torch.Tensor, target_quat: torch.Tensor, gripper_value: float) -> torch.Tensor:
        if gripper_available:
            gripper = torch.full((num_envs, 1), gripper_value, device=device)
            return torch.cat([target_pos, target_quat, gripper], dim=-1)
        else:
            return torch.cat([target_pos, target_quat], dim=-1)

    # Read current state immediately after reset
    ee_frame_sensor = env.unwrapped.scene["ee_frame"]
    init_ee_pos = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
    init_ee_quat = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

    # SAFE first action: hold current pose/orientation
    actions = build_action(init_ee_pos, init_ee_quat, GRIPPER_OPEN)

    rest_time = 0.40
    grasp_hold_time = 1.00
    release_hold_time = 0.60
    done_hold_time = 0.80

    pick_hover_offset_z = 0.12
    lift_height_above_grasp = 0.20
    place_hover_z = 0.20
    min_required_object_lift = 0.025

    phase = Phase.REST
    prev_phase = None
    phase_time = 0.0

    grasp_object_pos = None
    place_pos = None
    place_hover_pos = None
    lifted_successfully = False
    fail_reason = ""

    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(actions)

            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            ee_pos = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            ee_quat = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

            object_data = env.unwrapped.scene["object"].data
            object_pos = object_data.root_pos_w.clone() - env.unwrapped.scene.env_origins

            if phase != prev_phase:
                print_phase(phase)
                if phase == Phase.FAIL and fail_reason:
                    print(f"[WARN] {fail_reason}")
                prev_phase = phase
                phase_time = 0.0

            target_pos = ee_pos.clone()
            target_quat = ee_quat.clone()

            # Only meaningful when gripper exists.
            gripper_value = GRIPPER_OPEN

            if phase == Phase.REST:
                target_pos = ee_pos.clone()
                target_quat = ee_quat.clone()
                gripper_value = GRIPPER_OPEN

                if phase_time >= rest_time:
                    phase = Phase.APPROACH_ABOVE_PICK

            elif phase == Phase.APPROACH_ABOVE_PICK:
                target_pos = object_pos.clone()
                target_pos[:, 2] = object_pos[:, 2] + pick_hover_offset_z
                target_quat = desired_quat.clone()
                gripper_value = GRIPPER_OPEN

                if reached_xyz(ee_pos, target_pos, threshold=0.02):
                    phase = Phase.APPROACH_PICK

            elif phase == Phase.APPROACH_PICK:
                target_pos = object_pos.clone()
                target_quat = desired_quat.clone()
                gripper_value = GRIPPER_OPEN

                if reached_xyz(ee_pos, target_pos, threshold=0.012):
                    grasp_object_pos = object_pos.clone()

                    place_pos = torch.zeros((num_envs, 3), device=device)
                    place_pos[:, 0] = 0.68
                    place_pos[:, 1] = -0.22
                    place_pos[:, 2] = grasp_object_pos[:, 2]

                    place_hover_pos = place_pos.clone()
                    place_hover_pos[:, 2] = grasp_object_pos[:, 2] + place_hover_z

                    phase = Phase.GRASP

            elif phase == Phase.GRASP:
                target_pos = grasp_object_pos.clone()
                target_quat = desired_quat.clone()
                gripper_value = GRIPPER_CLOSE

                if not gripper_available:
                    fail_reason = (
                        "This environment exposes only a 7D EE pose action and no gripper action. "
                        "Arm motion is working, but grasp cannot be executed until gripper_action is active in the env."
                    )
                    phase = Phase.FAIL
                elif phase_time >= grasp_hold_time:
                    phase = Phase.LIFT

            elif phase == Phase.LIFT:
                target_pos = grasp_object_pos.clone()
                target_pos[:, 2] = grasp_object_pos[:, 2] + lift_height_above_grasp
                target_quat = desired_quat.clone()
                gripper_value = GRIPPER_CLOSE

                object_lift_delta = (object_pos[:, 2] - grasp_object_pos[:, 2]).mean().item()

                if object_lift_delta >= min_required_object_lift and reached_xyz(ee_pos, target_pos, threshold=0.025):
                    lifted_successfully = True
                    phase = Phase.MOVE_ABOVE_PLACE
                elif phase_time >= 2.5:
                    fail_reason = (
                        f"Object did not follow during lift. "
                        f"object_lift_delta={object_lift_delta:.4f}, expected >= {min_required_object_lift:.4f}"
                    )
                    phase = Phase.FAIL

            elif phase == Phase.MOVE_ABOVE_PLACE:
                target_pos = place_hover_pos.clone()
                target_quat = desired_quat.clone()
                gripper_value = GRIPPER_CLOSE

                if reached_xyz(ee_pos, target_pos, threshold=0.02):
                    phase = Phase.LOWER_TO_PLACE

            elif phase == Phase.LOWER_TO_PLACE:
                target_pos = place_pos.clone()
                target_quat = desired_quat.clone()
                gripper_value = GRIPPER_CLOSE

                if reached_xyz(ee_pos, target_pos, threshold=0.012):
                    phase = Phase.RELEASE

            elif phase == Phase.RELEASE:
                target_pos = place_pos.clone()
                target_quat = desired_quat.clone()
                gripper_value = GRIPPER_OPEN

                if phase_time >= release_hold_time:
                    phase = Phase.RETRACT

            elif phase == Phase.RETRACT:
                target_pos = place_hover_pos.clone()
                target_quat = desired_quat.clone()
                gripper_value = GRIPPER_OPEN

                if reached_xyz(ee_pos, target_pos, threshold=0.02):
                    phase = Phase.DONE

            elif phase == Phase.DONE:
                target_pos = place_hover_pos.clone()
                target_quat = desired_quat.clone()
                gripper_value = GRIPPER_OPEN

                if phase_time >= done_hold_time:
                    if lifted_successfully:
                        print("[CHECK] Pick-and-place finished successfully.")
                    else:
                        print("[WARN] Reached DONE pose, but object was never confirmed as lifted.")
                    break

            elif phase == Phase.FAIL:
                target_pos = ee_pos.clone()
                target_quat = ee_quat.clone()
                gripper_value = GRIPPER_OPEN
                print("[CHECK] Stopping because grasp/lift failed.")
                break

            actions = build_action(target_pos, target_quat, gripper_value)
            phase_time += dt

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()