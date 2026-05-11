import argparse
from enum import IntEnum

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR10e robust scripted pick-and-place demo")
parser.add_argument("--task", type=str, default="Template-Cobot-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episode_length_s", type=float, default=80.0)
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--single_cycle", action="store_true", default=False)
parser.add_argument("--print_every", type=int, default=100)

# Important tuning values
parser.add_argument("--grasp_z_offset", type=float, default=-0.006)
parser.add_argument("--regrasp_step", type=float, default=0.004)
parser.add_argument("--max_retries", type=int, default=5)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cobot.tasks  # noqa: F401
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


class Phase(IntEnum):
    REST = 0
    APPROACH_ABOVE_PICK = 1
    APPROACH_PICK = 2
    SETTLE_AT_PICK = 3
    GRASP = 4
    LIFT = 5
    MOVE_ABOVE_PLACE = 6
    LOWER_TO_PLACE = 7
    RELEASE = 8
    RETRACT = 9
    DONE = 10


PHASE_NAMES = {
    Phase.REST: "REST",
    Phase.APPROACH_ABOVE_PICK: "APPROACH_ABOVE_PICK",
    Phase.APPROACH_PICK: "APPROACH_PICK",
    Phase.SETTLE_AT_PICK: "SETTLE_AT_PICK",
    Phase.GRASP: "GRASP",
    Phase.LIFT: "LIFT",
    Phase.MOVE_ABOVE_PLACE: "MOVE_ABOVE_PLACE",
    Phase.LOWER_TO_PLACE: "LOWER_TO_PLACE",
    Phase.RELEASE: "RELEASE",
    Phase.RETRACT: "RETRACT",
    Phase.DONE: "DONE",
}


def reached_mask(current_pos: torch.Tensor, target_pos: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.norm(current_pos - target_pos, dim=1) < threshold


def get_ur10e_downward_quat(num_envs: int, device: torch.device) -> torch.Tensor:
    # Isaac Lab quaternion action format: [w, x, y, z]
    quat = torch.zeros((num_envs, 4), device=device)
    quat[:, 1] = 1.0
    return quat


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    dt = env_cfg.sim.dt * env_cfg.decimation

    GRIPPER_OPEN = 1.0
    GRIPPER_CLOSE = -1.0

    desired_quat = get_ur10e_downward_quat(num_envs, device)

    am = env.unwrapped.action_manager
    print(am)
    print("active_terms =", am.active_terms)
    print("term_dims    =", am.action_term_dim)
    print("total_dim    =", am.total_action_dim)

    term_dims = am.action_term_dim
    if term_dims == [7]:
        gripper_available = False
        print("[WARN] Only 7D arm action found. No gripper action available.")
    elif term_dims == [7, 1]:
        gripper_available = True
        print("[INFO] Detected absolute IK arm action + gripper action.")
    else:
        raise RuntimeError(
            f"Expected action layout [7] or [7, 1], but got {term_dims}. "
            "Use Template-Cobot-v0 with absolute IK pose control."
        )

    def build_action(target_pos: torch.Tensor, target_quat: torch.Tensor, gripper_value: torch.Tensor) -> torch.Tensor:
        if gripper_available:
            if gripper_value.ndim == 1:
                gripper_value = gripper_value.unsqueeze(-1)
            return torch.cat([target_pos, target_quat, gripper_value], dim=-1)
        return torch.cat([target_pos, target_quat], dim=-1)

    def gripper_vec(value: float) -> torch.Tensor:
        return torch.full((num_envs,), value, device=device)

    def advance(mask: torch.Tensor, next_phase: Phase):
        if torch.any(mask):
            phase[mask] = int(next_phase)
            phase_time[mask] = 0.0

    # Initial state
    ee_frame = env.unwrapped.scene["ee_frame"]
    init_ee_pos = ee_frame.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
    init_ee_quat = ee_frame.data.target_quat_w[..., 0, :].clone()

    actions = build_action(init_ee_pos, init_ee_quat, gripper_vec(GRIPPER_OPEN))

    # Timing
    rest_time = 0.40
    settle_pick_time = 0.25
    grasp_hold_time = 1.60
    release_hold_time = 0.70
    done_hold_time = 0.80

    # Motion
    pick_hover_offset_z = 0.13
    lift_height = 0.22
    place_hover_z = 0.20
    min_required_lift = 0.025

    # Per-env states
    phase = torch.full((num_envs,), int(Phase.REST), dtype=torch.long, device=device)
    phase_time = torch.zeros(num_envs, device=device)
    cycle_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    retry_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    grasp_z_offset = torch.full((num_envs,), args_cli.grasp_z_offset, device=device)

    grasp_target_pos = torch.zeros((num_envs, 3), device=device)
    object_z_at_grasp = torch.zeros(num_envs, device=device)

    place_pos = torch.zeros((num_envs, 3), device=device)
    place_hover_pos = torch.zeros((num_envs, 3), device=device)

    step = 0

    print("\n[INFO] Robust multi-robot zero_agent started.")
    print("[INFO] Fix added: per-env lift check + automatic regrasp retry.")
    print(f"[INFO] Initial grasp_z_offset = {args_cli.grasp_z_offset}")
    print(f"[INFO] max_retries = {args_cli.max_retries}, regrasp_step = {args_cli.regrasp_step}\n")

    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(actions)

            ee_frame = env.unwrapped.scene["ee_frame"]
            ee_pos = ee_frame.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            ee_quat = ee_frame.data.target_quat_w[..., 0, :].clone()

            obj = env.unwrapped.scene["object"]
            object_pos = obj.data.root_pos_w.clone() - env.unwrapped.scene.env_origins

            target_pos = ee_pos.clone()
            target_quat = ee_quat.clone()
            grip = gripper_vec(GRIPPER_OPEN)

            # ----------------------------------------------------------
            # REST
            # ----------------------------------------------------------
            m = phase == int(Phase.REST)
            if torch.any(m):
                target_pos[m] = ee_pos[m]
                target_quat[m] = ee_quat[m]
                grip[m] = GRIPPER_OPEN
                advance(m & (phase_time >= rest_time), Phase.APPROACH_ABOVE_PICK)

            # ----------------------------------------------------------
            # MOVE ABOVE PICK
            # ----------------------------------------------------------
            m = phase == int(Phase.APPROACH_ABOVE_PICK)
            if torch.any(m):
                above_pick = object_pos.clone()
                above_pick[:, 2] = object_pos[:, 2] + pick_hover_offset_z

                target_pos[m] = above_pick[m]
                target_quat[m] = desired_quat[m]
                grip[m] = GRIPPER_OPEN

                reached = reached_mask(ee_pos, above_pick, 0.025)
                timeout = phase_time > 4.0
                advance(m & (reached | timeout), Phase.APPROACH_PICK)

            # ----------------------------------------------------------
            # DESCEND TO PICK
            # ----------------------------------------------------------
            m = phase == int(Phase.APPROACH_PICK)
            if torch.any(m):
                pick_pos = object_pos.clone()
                pick_pos[:, 2] = object_pos[:, 2] + grasp_z_offset

                target_pos[m] = pick_pos[m]
                target_quat[m] = desired_quat[m]
                grip[m] = GRIPPER_OPEN

                reached = reached_mask(ee_pos, pick_pos, 0.014)
                soft_timeout = (phase_time > 4.0) & reached_mask(ee_pos, pick_pos, 0.030)

                done = m & (reached | soft_timeout)

                if torch.any(done):
                    grasp_target_pos[done] = pick_pos[done]
                    object_z_at_grasp[done] = object_pos[done, 2]

                    c = cycle_count % 4

                    p = torch.zeros((num_envs, 3), device=device)
                    p[:, 0] = 0.68
                    p[:, 1] = -0.22
                    p[:, 2] = object_pos[:, 2]

                    p[c == 1, 0] = 0.58
                    p[c == 1, 1] = 0.22

                    p[c == 2, 0] = 0.72
                    p[c == 2, 1] = 0.12

                    p[c == 3, 0] = 0.55
                    p[c == 3, 1] = -0.18

                    place_pos[done] = p[done]
                    place_hover_pos[done] = p[done]
                    place_hover_pos[done, 2] = p[done, 2] + place_hover_z

                advance(done, Phase.SETTLE_AT_PICK)

            # ----------------------------------------------------------
            # SETTLE BEFORE CLOSING
            # ----------------------------------------------------------
            m = phase == int(Phase.SETTLE_AT_PICK)
            if torch.any(m):
                target_pos[m] = grasp_target_pos[m]
                target_quat[m] = desired_quat[m]
                grip[m] = GRIPPER_OPEN

                advance(m & (phase_time >= settle_pick_time), Phase.GRASP)

            # ----------------------------------------------------------
            # CLOSE GRIPPER
            # ----------------------------------------------------------
            m = phase == int(Phase.GRASP)
            if torch.any(m):
                target_pos[m] = grasp_target_pos[m]
                target_quat[m] = desired_quat[m]
                grip[m] = GRIPPER_CLOSE if gripper_available else GRIPPER_OPEN

                advance(m & (phase_time >= grasp_hold_time), Phase.LIFT)

            # ----------------------------------------------------------
            # LIFT AND VERIFY PER ROBOT
            # ----------------------------------------------------------
            m = phase == int(Phase.LIFT)
            if torch.any(m):
                lift_pos = grasp_target_pos.clone()
                lift_pos[:, 2] = grasp_target_pos[:, 2] + lift_height

                target_pos[m] = lift_pos[m]
                target_quat[m] = desired_quat[m]
                grip[m] = GRIPPER_CLOSE if gripper_available else GRIPPER_OPEN

                lift_delta = object_pos[:, 2] - object_z_at_grasp

                lifted = lift_delta > min_required_lift
                lift_reached = reached_mask(ee_pos, lift_pos, 0.040)

                success = m & lifted & (phase_time > 0.8)
                advance(success, Phase.MOVE_ABOVE_PLACE)

                failed_lift = m & (~lifted) & (phase_time > 2.2)

                retry = failed_lift & (retry_count < args_cli.max_retries)
                if torch.any(retry):
                    retry_count[retry] += 1
                    grasp_z_offset[retry] -= args_cli.regrasp_step

                    print(
                        "[REGRASP] envs:",
                        torch.nonzero(retry).flatten().detach().cpu().tolist(),
                        "new_offsets:",
                        grasp_z_offset[retry].detach().cpu().tolist(),
                    )

                    # Open, go back above cube, and try again lower
                    grip[retry] = GRIPPER_OPEN
                    advance(retry, Phase.APPROACH_ABOVE_PICK)

                # After max retries, keep trying instead of stopping demo.
                final_retry = failed_lift & (retry_count >= args_cli.max_retries)
                if torch.any(final_retry):
                    grasp_z_offset[final_retry] -= args_cli.regrasp_step * 0.5
                    grip[final_retry] = GRIPPER_OPEN
                    advance(final_retry, Phase.APPROACH_ABOVE_PICK)

            # ----------------------------------------------------------
            # MOVE ABOVE PLACE
            # ----------------------------------------------------------
            m = phase == int(Phase.MOVE_ABOVE_PLACE)
            if torch.any(m):
                target_pos[m] = place_hover_pos[m]
                target_quat[m] = desired_quat[m]
                grip[m] = GRIPPER_CLOSE if gripper_available else GRIPPER_OPEN

                reached = reached_mask(ee_pos, place_hover_pos, 0.035)
                timeout = phase_time > 5.0
                advance(m & (reached | timeout), Phase.LOWER_TO_PLACE)

            # ----------------------------------------------------------
            # LOWER TO PLACE
            # ----------------------------------------------------------
            m = phase == int(Phase.LOWER_TO_PLACE)
            if torch.any(m):
                target_pos[m] = place_pos[m]
                target_quat[m] = desired_quat[m]
                grip[m] = GRIPPER_CLOSE if gripper_available else GRIPPER_OPEN

                reached = reached_mask(ee_pos, place_pos, 0.022)
                timeout = phase_time > 4.0
                advance(m & (reached | timeout), Phase.RELEASE)

            # ----------------------------------------------------------
            # RELEASE
            # ----------------------------------------------------------
            m = phase == int(Phase.RELEASE)
            if torch.any(m):
                target_pos[m] = place_pos[m]
                target_quat[m] = desired_quat[m]
                grip[m] = GRIPPER_OPEN

                advance(m & (phase_time >= release_hold_time), Phase.RETRACT)

            # ----------------------------------------------------------
            # RETRACT
            # ----------------------------------------------------------
            m = phase == int(Phase.RETRACT)
            if torch.any(m):
                target_pos[m] = place_hover_pos[m]
                target_quat[m] = desired_quat[m]
                grip[m] = GRIPPER_OPEN

                reached = reached_mask(ee_pos, place_hover_pos, 0.035)
                timeout = phase_time > 4.0
                advance(m & (reached | timeout), Phase.DONE)

            # ----------------------------------------------------------
            # DONE
            # ----------------------------------------------------------
            m = phase == int(Phase.DONE)
            if torch.any(m):
                target_pos[m] = place_hover_pos[m]
                target_quat[m] = desired_quat[m]
                grip[m] = GRIPPER_OPEN

                if args_cli.single_cycle:
                    if torch.all(phase == int(Phase.DONE)):
                        print("[CHECK] All robots completed one pick-place cycle.")
                        break
                else:
                    repeat = m & (phase_time >= done_hold_time)
                    if torch.any(repeat):
                        cycle_count[repeat] += 1
                        retry_count[repeat] = 0
                        grasp_z_offset[repeat] = args_cli.grasp_z_offset
                    advance(repeat, Phase.APPROACH_ABOVE_PICK)

            actions = build_action(target_pos, target_quat, grip)

            phase_time += dt
            step += 1

            if args_cli.print_every > 0 and step % args_cli.print_every == 0:
                counts = []
                for ph in Phase:
                    n = int(torch.sum(phase == int(ph)).item())
                    if n > 0:
                        counts.append(f"{PHASE_NAMES[ph]}={n}")

                lifted_now = object_pos[:, 2] - object_z_at_grasp
                lifted_count = int(torch.sum(lifted_now > min_required_lift).item())

                print(f"[STEP {step:05d}] " + " | ".join(counts) + f" | lifted_now={lifted_count}/{num_envs}")

            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                print("[INFO] max_steps reached. Closing.")
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
