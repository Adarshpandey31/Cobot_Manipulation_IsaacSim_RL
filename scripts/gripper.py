import argparse
import math
from enum import Enum, auto

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR10e gripper-only diagnostic for cube grasping")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import cobot.tasks  # noqa: F401

from isaaclab.controllers import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.sim.utils import get_current_stage, resolve_prim_pose


class Phase(Enum):
    HOME = auto()
    MOVE_ABOVE_CUBE = auto()
    DESCEND_SMALL_STEPS = auto()
    CLOSE_ONLY = auto()
    HOLD = auto()
    LIFT_SMALL = auto()
    DONE = auto()
    FAIL = auto()


OUTER_LEFT = "left_outer_finger"
OUTER_RIGHT = "right_outer_finger"
INNER_LEFT = "left_inner_finger"
INNER_RIGHT = "right_inner_finger"
EE_BODY = "wrist_3_link"
BASE_BODY = "base_link"
GRIPPER_JOINT = "finger_joint"
CUBE_PATH = "/World/envs/env_0/Cube"


def banner(phase: Phase):
    messages = {
        Phase.HOME: "[INFO] Settling robot at home posture with gripper open.",
        Phase.MOVE_ABOVE_CUBE: "[INFO] Moving above cube while keeping finger midpoint centered.",
        Phase.DESCEND_SMALL_STEPS: "[INFO] Descending in tiny steps while keeping finger midpoint centered.",
        Phase.CLOSE_ONLY: "[INFO] Closing gripper only. No lift yet.",
        Phase.HOLD: "[INFO] Holding the grasp pose to see if the cube stays trapped.",
        Phase.LIFT_SMALL: "[INFO] Lifting only 1-2 cm to test whether the grasp is physically real.",
        Phase.DONE: "[INFO] Diagnostic finished.",
        Phase.FAIL: "[WARN] Diagnostic failed.",
    }
    print("\n" + "=" * 100)
    print(f"[PHASE] {phase.name}")
    print(messages[phase])
    print("=" * 100)


def get_vertical_down_quat(device: torch.device) -> torch.Tensor:
    angle = math.pi
    qw = math.cos(angle / 2.0)
    qx = math.sin(angle / 2.0)
    return torch.tensor([[qw, qx, 0.0, 0.0]], dtype=torch.float32, device=device)


def get_cube_pose(device: torch.device) -> torch.Tensor:
    stage = get_current_stage()
    cube_prim = stage.GetPrimAtPath(CUBE_PATH)
    cube_pos, _ = resolve_prim_pose(cube_prim)
    return torch.tensor(cube_pos, dtype=torch.float32, device=device).unsqueeze(0)


def clamp_workspace(target_pos: torch.Tensor) -> torch.Tensor:
    out = target_pos.clone()
    out[:, 0] = torch.clamp(out[:, 0], 0.35, 1.00)
    out[:, 1] = torch.clamp(out[:, 1], -0.60, 0.60)
    out[:, 2] = torch.clamp(out[:, 2], 0.82, 1.35)
    return out


def limit_cartesian_step(current_pos: torch.Tensor, target_pos: torch.Tensor, max_step_xy: float, max_step_z: float) -> torch.Tensor:
    delta = target_pos - current_pos
    out = current_pos.clone()
    out[:, 0] += torch.clamp(delta[:, 0], -max_step_xy, max_step_xy)
    out[:, 1] += torch.clamp(delta[:, 1], -max_step_xy, max_step_xy)
    out[:, 2] += torch.clamp(delta[:, 2], -max_step_z, max_step_z)
    return out


def smooth_ik_joint_targets(current_arm_joint_pos: torch.Tensor, desired_joint_targets: torch.Tensor, phase: Phase) -> torch.Tensor:
    if phase in [Phase.DESCEND_SMALL_STEPS, Phase.LIFT_SMALL]:
        alpha = 0.06
        max_joint_step = 0.03
    else:
        alpha = 0.055
        max_joint_step = 0.028
    delta = alpha * (desired_joint_targets - current_arm_joint_pos)
    delta = torch.clamp(delta, -max_joint_step, max_joint_step)
    return current_arm_joint_pos + delta


def pose_reached(current_pos: torch.Tensor, target_pos: torch.Tensor, xy_tol: float, z_tol: float) -> bool:
    xy_error = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=1)
    z_error = torch.abs(current_pos[:, 2] - target_pos[:, 2])
    return bool(torch.all(xy_error < xy_tol).item() and torch.all(z_error < z_tol).item())


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    print("[INFO] Environment created successfully")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Num envs: {num_envs}")

    body_name_to_idx = {name: i for i, name in enumerate(robot.data.body_names)}
    joint_name_to_idx = {name: i for i, name in enumerate(robot.data.joint_names)}

    required_body_names = [EE_BODY, BASE_BODY, OUTER_LEFT, OUTER_RIGHT, INNER_LEFT, INNER_RIGHT]
    required_joint_names = [GRIPPER_JOINT]
    for name in required_body_names:
        if name not in body_name_to_idx:
            raise RuntimeError(f"Required body '{name}' not found. Available: {robot.data.body_names}")
    for name in required_joint_names:
        if name not in joint_name_to_idx:
            raise RuntimeError(f"Required joint '{name}' not found. Available: {robot.data.joint_names}")

    ee_idx = body_name_to_idx[EE_BODY]
    base_idx = body_name_to_idx[BASE_BODY]
    outer_left_idx = body_name_to_idx[OUTER_LEFT]
    outer_right_idx = body_name_to_idx[OUTER_RIGHT]
    inner_left_idx = body_name_to_idx[INNER_LEFT]
    inner_right_idx = body_name_to_idx[INNER_RIGHT]
    finger_joint_idx = joint_name_to_idx[GRIPPER_JOINT]
    arm_joint_ids = [joint_name_to_idx[name] for name in [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]]

    print("\n===== BODY INDICES USED =====")
    print(f"{EE_BODY}: {ee_idx}")
    print(f"{OUTER_LEFT}: {outer_left_idx}")
    print(f"{OUTER_RIGHT}: {outer_right_idx}")
    print(f"{INNER_LEFT}: {inner_left_idx}")
    print(f"{INNER_RIGHT}: {inner_right_idx}")
    print("\n===== JOINT INDICES USED =====")
    print(f"{GRIPPER_JOINT}: {finger_joint_idx}")

    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.03, "joint_weight": 0.6},
    )
    ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=1, device=device)
    vertical_quat = get_vertical_down_quat(device)

    home_q = torch.tensor([[0.00, -1.20, 1.40, -1.60, -1.57, 0.00, 0.03]], dtype=torch.float32, device=device).repeat(num_envs, 1)

    full_open = 0.03
    close_cmd = 0.23
    hover_offset = 0.17
    # Start a little higher than your current grasp height to avoid palm/front interference.
    grasp_height_offset = 0.115
    descend_step = 0.0020
    tiny_lift_total = 0.020
    tiny_lift_step = 0.0020
    hold_steps = 120
    close_hold_steps = 120
    xy_tol = 0.005
    z_tol = 0.004

    phase = Phase.HOME
    prev_phase = None
    sim_step = 0
    align_hold_counter = 0
    close_counter = 0
    hold_counter = 0
    lift_counter = 0
    fail_reason = ""

    desired_finger_xy = None
    locked_grasp_z = None
    cube_z_at_close = None
    cube_xy_at_close = None
    lift_start_cube_z = None
    lift_start_ee_z = None
    target_lift_z = None

    def compute_midpoints(body_pos_w: torch.Tensor):
        outer_mid = 0.5 * (body_pos_w[:, outer_left_idx, :] + body_pos_w[:, outer_right_idx, :])
        inner_mid = 0.5 * (body_pos_w[:, inner_left_idx, :] + body_pos_w[:, inner_right_idx, :])
        return outer_mid, inner_mid

    banner(Phase.HOME)
    for step in range(100):
        env.step(home_q)
        if (step + 1) % 25 == 0:
            print(f"[DEBUG] HOME settling step {step + 1}/100")

    while simulation_app.is_running():
        simulation_app.update()
        sim_step += 1

        ee_pos = robot.data.body_pos_w[:, ee_idx, :]
        ee_quat = robot.data.body_quat_w[:, ee_idx, :]
        joint_pos = robot.data.joint_pos
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_idx - base_idx, :, :][:, :, arm_joint_ids]
        body_pos_w = robot.data.body_pos_w
        outer_mid, inner_mid = compute_midpoints(body_pos_w)
        cube = get_cube_pose(device)

        if phase != prev_phase:
            banner(phase)
            if phase == Phase.FAIL and fail_reason:
                print(f"[WARN] {fail_reason}")
                print("=" * 100)
            prev_phase = phase

        action = torch.zeros_like(joint_pos[:, :7])
        target_pos = ee_pos.clone()
        desired_gripper = full_open

        # Use outer finger midpoint as primary grasp center, but keep inner midpoint visible in logs.
        grasp_mid = outer_mid

        if phase == Phase.MOVE_ABOVE_CUBE:
            desired_finger_xy = cube[:, :2].clone()
            wrist_xy_offset = grasp_mid[:, :2] - ee_pos[:, :2]
            target_pos[:, :2] = desired_finger_xy - wrist_xy_offset
            target_pos[:, 2] = cube[:, 2] + hover_offset

            finger_xy_error = torch.norm(grasp_mid[:, :2] - cube[:, :2], dim=1).mean().item()
            if sim_step % 10 == 0:
                print(
                    f"[DEBUG] MOVE_ABOVE_CUBE | finger_xy_error={finger_xy_error:.5f} | "
                    f"outer_mid=({outer_mid[0,0].item():.4f},{outer_mid[0,1].item():.4f},{outer_mid[0,2].item():.4f}) | "
                    f"inner_mid=({inner_mid[0,0].item():.4f},{inner_mid[0,1].item():.4f},{inner_mid[0,2].item():.4f}) | "
                    f"cube=({cube[0,0].item():.4f},{cube[0,1].item():.4f},{cube[0,2].item():.4f})"
                )
            if finger_xy_error < xy_tol:
                align_hold_counter += 1
            else:
                align_hold_counter = 0
            if align_hold_counter >= 8:
                print("[CHECK] Finger midpoint aligned above cube center.")
                locked_grasp_z = cube[:, 2] + grasp_height_offset
                phase = Phase.DESCEND_SMALL_STEPS

        elif phase == Phase.DESCEND_SMALL_STEPS:
            wrist_xy_offset = grasp_mid[:, :2] - ee_pos[:, :2]
            target_pos[:, :2] = desired_finger_xy - wrist_xy_offset
            target_pos[:, 2] = torch.maximum(ee_pos[:, 2] - descend_step, locked_grasp_z)

            finger_xy_error = torch.norm(grasp_mid[:, :2] - cube[:, :2], dim=1).mean().item()
            z_error = torch.abs(ee_pos[:, 2] - locked_grasp_z).mean().item()
            if sim_step % 5 == 0:
                print(
                    f"[DEBUG] DESCEND_SMALL_STEPS | finger_xy_error={finger_xy_error:.5f} | "
                    f"ee_z={ee_pos[0,2].item():.4f} | target_z={target_pos[0,2].item():.4f} | "
                    f"outer_mid=({outer_mid[0,0].item():.4f},{outer_mid[0,1].item():.4f},{outer_mid[0,2].item():.4f}) | "
                    f"cube=({cube[0,0].item():.4f},{cube[0,1].item():.4f},{cube[0,2].item():.4f})"
                )
            if z_error < z_tol:
                print("[CHECK] Reached diagnostic grasp height with gripper open.")
                close_counter = 0
                phase = Phase.CLOSE_ONLY

        elif phase == Phase.CLOSE_ONLY:
            wrist_xy_offset = grasp_mid[:, :2] - ee_pos[:, :2]
            target_pos[:, :2] = desired_finger_xy - wrist_xy_offset
            target_pos[:, 2] = locked_grasp_z
            desired_gripper = close_cmd
            close_counter += 1

            if close_counter % 10 == 0:
                print(
                    f"[DEBUG] CLOSE_ONLY | counter={close_counter}/{close_hold_steps} | "
                    f"finger_joint={joint_pos[0, finger_joint_idx].item():.5f} | "
                    f"outer_mid=({outer_mid[0,0].item():.4f},{outer_mid[0,1].item():.4f},{outer_mid[0,2].item():.4f}) | "
                    f"inner_mid=({inner_mid[0,0].item():.4f},{inner_mid[0,1].item():.4f},{inner_mid[0,2].item():.4f}) | "
                    f"cube=({cube[0,0].item():.4f},{cube[0,1].item():.4f},{cube[0,2].item():.4f})"
                )

            if close_counter >= close_hold_steps:
                cube_z_at_close = cube[:, 2].clone().detach()
                cube_xy_at_close = cube[:, :2].clone().detach()
                hold_counter = 0
                print(
                    f"[CHECK] Close-only window complete. finger_joint={joint_pos[0, finger_joint_idx].item():.5f} | "
                    f"cube_z={cube[0,2].item():.4f}"
                )
                phase = Phase.HOLD

        elif phase == Phase.HOLD:
            wrist_xy_offset = grasp_mid[:, :2] - ee_pos[:, :2]
            target_pos[:, :2] = desired_finger_xy - wrist_xy_offset
            target_pos[:, 2] = locked_grasp_z
            desired_gripper = close_cmd
            hold_counter += 1

            cube_lift_delta = (cube[:, 2] - cube_z_at_close).mean().item()
            cube_xy_shift = torch.norm(cube[:, :2] - cube_xy_at_close, dim=1).mean().item()
            if hold_counter % 10 == 0:
                print(
                    f"[DEBUG] HOLD | counter={hold_counter}/{hold_steps} | "
                    f"finger_joint={joint_pos[0, finger_joint_idx].item():.5f} | "
                    f"cube_z={cube[0,2].item():.4f} | cube_lift_delta={cube_lift_delta:.4f} | cube_xy_shift={cube_xy_shift:.4f}"
                )

            if hold_counter >= hold_steps:
                lift_start_cube_z = cube[:, 2].clone().detach()
                lift_start_ee_z = ee_pos[:, 2].clone().detach()
                target_lift_z = ee_pos[:, 2].clone().detach()
                lift_counter = 0
                print("[CHECK] Hold window complete. Starting tiny lift.")
                phase = Phase.LIFT_SMALL

        elif phase == Phase.LIFT_SMALL:
            wrist_xy_offset = grasp_mid[:, :2] - ee_pos[:, :2]
            target_pos[:, :2] = desired_finger_xy - wrist_xy_offset
            target_lift_z = torch.minimum(
                target_lift_z + tiny_lift_step,
                lift_start_ee_z + tiny_lift_total,
            )
            target_pos[:, 2] = target_lift_z
            desired_gripper = close_cmd
            lift_counter += 1

            cube_lift_delta = (cube[:, 2] - lift_start_cube_z).mean().item()
            ee_lift_delta = (ee_pos[:, 2] - lift_start_ee_z).mean().item()
            if lift_counter % 5 == 0:
                print(
                    f"[DEBUG] LIFT_SMALL | counter={lift_counter} | "
                    f"ee_z={ee_pos[0,2].item():.4f} | target_z={target_pos[0,2].item():.4f} | "
                    f"cube_z={cube[0,2].item():.4f} | ee_lift_delta={ee_lift_delta:.4f} | cube_lift_delta={cube_lift_delta:.4f} | "
                    f"finger_joint={joint_pos[0, finger_joint_idx].item():.5f}"
                )

            # Finish after the target height is reached and held briefly.
            if ee_lift_delta >= tiny_lift_total - 0.003:
                print(
                    f"[RESULT] Tiny lift complete. ee_lift_delta={ee_lift_delta:.4f} | "
                    f"cube_lift_delta={cube_lift_delta:.4f}"
                )
                phase = Phase.DONE
            elif lift_counter >= 120:
                fail_reason = (
                    f"Tiny lift timed out. ee_lift_delta={ee_lift_delta:.4f}, cube_lift_delta={cube_lift_delta:.4f}"
                )
                phase = Phase.FAIL

        elif phase == Phase.DONE:
            target_pos = ee_pos.clone()
            desired_gripper = close_cmd

        elif phase == Phase.FAIL:
            target_pos = ee_pos.clone()
            desired_gripper = close_cmd

        if phase not in [Phase.HOME, Phase.DONE, Phase.FAIL]:
            target_pos = clamp_workspace(target_pos)
            target_pos = limit_cartesian_step(ee_pos, target_pos, max_step_xy=0.008, max_step_z=0.006)
            ik_command = torch.cat([target_pos, vertical_quat], dim=1)
            ik_controller.set_command(ik_command)
            desired_joint_targets = ik_controller.compute(ee_pos, ee_quat, jacobian, joint_pos[:, arm_joint_ids])
            action[:, :6] = smooth_ik_joint_targets(joint_pos[:, arm_joint_ids], desired_joint_targets, phase)
            action[:, 6] = desired_gripper
        else:
            action[:, :6] = joint_pos[:, arm_joint_ids]
            action[:, 6] = desired_gripper

        env.step(action)

        if phase == Phase.HOME:
            # This phase is handled before the loop.
            phase = Phase.MOVE_ABOVE_CUBE

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()