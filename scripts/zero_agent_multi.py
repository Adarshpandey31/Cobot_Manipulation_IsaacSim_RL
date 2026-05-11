import argparse
from enum import IntEnum

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR10e scripted multi-env pick-and-place demo.")
parser.add_argument("--task", type=str, default="Template-Cobot-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--episode_length_s", type=float, default=60.0)
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--keep_random_object", action="store_true", default=False)
parser.add_argument("--no_repeat", action="store_true", default=False)
parser.add_argument("--grasp_z_offset", type=float, default=0.0)
parser.add_argument("--place_radius", type=float, default=0.18)
parser.add_argument("--print_every", type=int, default=100)

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
    ABOVE_PICK = 1
    DESCEND_PICK = 2
    CLOSE_GRIPPER = 3
    LIFT = 4
    ABOVE_PLACE = 5
    LOWER_PLACE = 6
    OPEN_GRIPPER = 7
    RETRACT = 8
    DONE = 9


PHASE_NAMES = {
    Phase.REST: "REST",
    Phase.ABOVE_PICK: "ABOVE_PICK",
    Phase.DESCEND_PICK: "DESCEND_PICK",
    Phase.CLOSE_GRIPPER: "CLOSE_GRIPPER",
    Phase.LIFT: "LIFT",
    Phase.ABOVE_PLACE: "ABOVE_PLACE",
    Phase.LOWER_PLACE: "LOWER_PLACE",
    Phase.OPEN_GRIPPER: "OPEN_GRIPPER",
    Phase.RETRACT: "RETRACT",
    Phase.DONE: "DONE",
}


def reached_mask(current_pos: torch.Tensor, target_pos: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.norm(current_pos - target_pos, dim=1) < threshold


def get_ur10e_downward_quat(num_envs: int, device: torch.device) -> torch.Tensor:
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

    # For presentation: keep cubes at a stable, reachable pose.
    # Your old random reset is useful for testing, but bad for a clean 16-robot demo.
    if not args_cli.keep_random_object:
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_object_position"):
            env_cfg.events.reset_object_position = None
            print("[INFO] Disabled reset_object_position for stable presentation demo.")

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
    if term_dims == [7, 1]:
        gripper_available = True
        print("[INFO] Detected absolute IK pose action + gripper action.")
    elif term_dims == [7]:
        gripper_available = False
        print("[WARN] Only 7D IK action found. Gripper action is missing.")
    else:
        raise RuntimeError(f"This scripted demo expects action layout [7, 1] or [7]. Got term_dims={term_dims}")

    def build_action(target_pos: torch.Tensor, target_quat: torch.Tensor, gripper_cmd: torch.Tensor) -> torch.Tensor:
        if gripper_available:
            return torch.cat([target_pos, target_quat, gripper_cmd.unsqueeze(-1)], dim=-1)
        return torch.cat([target_pos, target_quat], dim=-1)

    ee_frame = env.unwrapped.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
    ee_quat = ee_frame.data.target_quat_w[..., 0, :].clone()

    home_pos = ee_pos.clone()
    home_quat = ee_quat.clone()

    phase = torch.full((num_envs,), Phase.REST, dtype=torch.long, device=device)
    phase_time = torch.zeros(num_envs, device=device)
    cycle_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    grasp_pos = torch.zeros((num_envs, 3), device=device)
    place_pos = torch.zeros((num_envs, 3), device=device)
    place_hover_pos = torch.zeros((num_envs, 3), device=device)

    target_pos = home_pos.clone()
    target_quat = home_quat.clone()
    gripper_cmd = torch.full((num_envs,), GRIPPER_OPEN, device=device)
    actions = build_action(target_pos, target_quat, gripper_cmd)

    rest_time = 0.35
    close_hold_time = 0.85
    open_hold_time = 0.60
    done_hold_time = 0.50

    pick_hover_offset_z = 0.16
    lift_height = 0.23
    place_hover_z = 0.18

    step = 0

    def advance(mask: torch.Tensor, next_phase: Phase):
        if torch.any(mask):
            phase[mask] = int(next_phase)
            phase_time[mask] = 0.0

    def compute_place_from_object(object_pos_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = torch.arange(num_envs, device=device, dtype=torch.float32)
        cyc = cycle_count.to(torch.float32)

        # Four place directions, changing after every cycle.
        angle = 0.5 * torch.pi * ((env_ids + cyc) % 4.0)
        dx = args_cli.place_radius * torch.cos(angle)
        dy = args_cli.place_radius * torch.sin(angle)

        p = object_pos_local.clone()
        p[:, 0] = torch.clamp(p[:, 0] + dx, 0.38, 0.78)
        p[:, 1] = torch.clamp(p[:, 1] + dy, -0.32, 0.32)
        p[:, 2] = object_pos_local[:, 2] + args_cli.grasp_z_offset

        ph = p.clone()
        ph[:, 2] = p[:, 2] + place_hover_z
        return p, ph

    print("\n[INFO] Multi-robot independent phase controller started.")
    print("[INFO] Each environment now has its own phase, timer, grasp pose, and place pose.\n")

    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(actions)

            ee_frame = env.unwrapped.scene["ee_frame"]
            ee_pos = ee_frame.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            ee_quat = ee_frame.data.target_quat_w[..., 0, :].clone()

            object_data = env.unwrapped.scene["object"].data
            object_pos = object_data.root_pos_w.clone() - env.unwrapped.scene.env_origins

            target_pos = ee_pos.clone()
            target_quat = desired_quat.clone()
            gripper_cmd = torch.full((num_envs,), GRIPPER_OPEN, device=device)

            # ------------------------------------------------------------------
            # REST
            # ------------------------------------------------------------------
            m = phase == Phase.REST
            if torch.any(m):
                target_pos[m] = home_pos[m]
                target_quat[m] = home_quat[m]
                gripper_cmd[m] = GRIPPER_OPEN
                advance(m & (phase_time > rest_time), Phase.ABOVE_PICK)

            # ------------------------------------------------------------------
            # Move above cube
            # ------------------------------------------------------------------
            m = phase == Phase.ABOVE_PICK
            if torch.any(m):
                above = object_pos.clone()
                above[:, 2] = object_pos[:, 2] + pick_hover_offset_z
                target_pos[m] = above[m]
                gripper_cmd[m] = GRIPPER_OPEN

                done = reached_mask(ee_pos, above, 0.025)
                advance(m & done, Phase.DESCEND_PICK)

            # ------------------------------------------------------------------
            # Vertical descent to grasp pose
            # ------------------------------------------------------------------
            m = phase == Phase.DESCEND_PICK
            if torch.any(m):
                g = object_pos.clone()
                g[:, 2] = object_pos[:, 2] + args_cli.grasp_z_offset
                target_pos[m] = g[m]
                gripper_cmd[m] = GRIPPER_OPEN

                done = reached_mask(ee_pos, g, 0.014)
                just_done = m & done
                if torch.any(just_done):
                    grasp_pos[just_done] = g[just_done]
                    pp, php = compute_place_from_object(object_pos)
                    place_pos[just_done] = pp[just_done]
                    place_hover_pos[just_done] = php[just_done]
                advance(just_done, Phase.CLOSE_GRIPPER)

            # ------------------------------------------------------------------
            # Close gripper at grasp pose
            # ------------------------------------------------------------------
            m = phase == Phase.CLOSE_GRIPPER
            if torch.any(m):
                target_pos[m] = grasp_pos[m]
                gripper_cmd[m] = GRIPPER_CLOSE if gripper_available else GRIPPER_OPEN
                advance(m & (phase_time > close_hold_time), Phase.LIFT)

            # ------------------------------------------------------------------
            # Lift vertically
            # ------------------------------------------------------------------
            m = phase == Phase.LIFT
            if torch.any(m):
                lift = grasp_pos.clone()
                lift[:, 2] = grasp_pos[:, 2] + lift_height
                target_pos[m] = lift[m]
                gripper_cmd[m] = GRIPPER_CLOSE if gripper_available else GRIPPER_OPEN

                done = reached_mask(ee_pos, lift, 0.030)
                timeout = phase_time > 3.0
                advance(m & (done | timeout), Phase.ABOVE_PLACE)

            # ------------------------------------------------------------------
            # Move above place pose
            # ------------------------------------------------------------------
            m = phase == Phase.ABOVE_PLACE
            if torch.any(m):
                target_pos[m] = place_hover_pos[m]
                gripper_cmd[m] = GRIPPER_CLOSE if gripper_available else GRIPPER_OPEN

                done = reached_mask(ee_pos, place_hover_pos, 0.030)
                advance(m & done, Phase.LOWER_PLACE)

            # ------------------------------------------------------------------
            # Lower to place pose
            # ------------------------------------------------------------------
            m = phase == Phase.LOWER_PLACE
            if torch.any(m):
                target_pos[m] = place_pos[m]
                gripper_cmd[m] = GRIPPER_CLOSE if gripper_available else GRIPPER_OPEN

                done = reached_mask(ee_pos, place_pos, 0.018)
                advance(m & done, Phase.OPEN_GRIPPER)

            # ------------------------------------------------------------------
            # Open gripper / release
            # ------------------------------------------------------------------
            m = phase == Phase.OPEN_GRIPPER
            if torch.any(m):
                target_pos[m] = place_pos[m]
                gripper_cmd[m] = GRIPPER_OPEN
                advance(m & (phase_time > open_hold_time), Phase.RETRACT)

            # ------------------------------------------------------------------
            # Retract up
            # ------------------------------------------------------------------
            m = phase == Phase.RETRACT
            if torch.any(m):
                target_pos[m] = place_hover_pos[m]
                gripper_cmd[m] = GRIPPER_OPEN

                done = reached_mask(ee_pos, place_hover_pos, 0.030)
                advance(m & done, Phase.DONE)

            # ------------------------------------------------------------------
            # Done / repeat
            # ------------------------------------------------------------------
            m = phase == Phase.DONE
            if torch.any(m):
                target_pos[m] = place_hover_pos[m]
                gripper_cmd[m] = GRIPPER_OPEN

                if not args_cli.no_repeat:
                    repeat_mask = m & (phase_time > done_hold_time)
                    if torch.any(repeat_mask):
                        cycle_count[repeat_mask] += 1
                    advance(repeat_mask, Phase.ABOVE_PICK)

            actions = build_action(target_pos, target_quat, gripper_cmd)

            phase_time += dt
            step += 1

            if args_cli.print_every > 0 and step % args_cli.print_every == 0:
                counts = []
                for ph in Phase:
                    c = int(torch.sum(phase == ph).item())
                    if c > 0:
                        counts.append(f"{PHASE_NAMES[ph]}={c}")
                print(f"[STEP {step:05d}] " + " | ".join(counts))

            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                print("[INFO] max_steps reached. Closing demo.")
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
