import argparse
import csv
import json
import math
import os
import sys
import select
import termios
import tty
from datetime import datetime

import torch

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Manual joint control + full debug recorder for robot + gripper")
parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--log_every", type=int, default=60, help="Print detailed state every N sim steps")
parser.add_argument("--save_dir", type=str, default="manual_debug_outputs", help="Directory to save logs/waypoints")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab / project imports after app launch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import cobot.tasks  # ensure your custom tasks are registered

from isaaclab.sim.utils import get_current_stage, resolve_prim_pose

# -----------------------------------------------------------------------------
# Joint names from your robot
# -----------------------------------------------------------------------------
ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_JOINT = "finger_joint"
CONTROLLED_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]

FINGER_BODY_NAMES = [
    "left_outer_finger",
    "right_outer_finger",
    "left_inner_finger",
    "right_inner_finger",
]

GRIPPER_DEBUG_JOINTS = [
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
]

WRIST_BODY = "wrist_3_link"
BASE_BODY = "base_link"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_key():
    dr, _, _ = select.select([sys.stdin], [], [], 0.05)
    if dr:
        return sys.stdin.read(1)
    return None


def format_vec3(v):
    return f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})"


def quat_to_str(q):
    return f"({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f})"


def safe_prim_pose(prim_path, device):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        return None, None
    pos, quat = resolve_prim_pose(prim)
    pos_t = torch.tensor(pos, dtype=torch.float32, device=device).unsqueeze(0)
    quat_t = torch.tensor(quat, dtype=torch.float32, device=device).unsqueeze(0)
    return pos_t, quat_t


def print_help():
    print("\n================ MANUAL CONTROL + DEBUG =================")
    print("Select joint:")
    print("  1 -> shoulder_pan_joint")
    print("  2 -> shoulder_lift_joint")
    print("  3 -> elbow_joint")
    print("  4 -> wrist_1_joint")
    print("  5 -> wrist_2_joint")
    print("  6 -> wrist_3_joint")
    print("  7 -> finger_joint")
    print("")
    print("Move selected joint:")
    print("  j -> decrease selected joint")
    print("  k -> increase selected joint")
    print("")
    print("Quick gripper:")
    print("  o -> gripper open")
    print("  c -> gripper close to 0.20")
    print("")
    print("Debug / logging:")
    print("  p -> print current full debug snapshot")
    print("  s -> save current waypoint snapshot to JSON")
    print("  l -> toggle periodic terminal logging on/off")
    print("  r -> reset env and reload current joint targets")
    print("  h -> show help")
    print("  q -> quit")
    print("========================================================\n")


def print_joint_targets(joint_targets, joint_name_to_idx):
    print("\n===== CURRENT TARGETS =====")
    for name in CONTROLLED_JOINTS:
        idx = joint_name_to_idx[name]
        print(f"{name}: {joint_targets[0, idx].item():.4f}")
    print("================================\n")


def make_save_paths(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_dir, f"manual_debug_{stamp}.csv")
    jsonl_path = os.path.join(save_dir, f"manual_debug_{stamp}.jsonl")
    waypoints_path = os.path.join(save_dir, f"manual_waypoints_{stamp}.json")
    return csv_path, jsonl_path, waypoints_path


def create_csv_writer(csv_path, controlled_joint_names, gripper_debug_joint_names):
    f = open(csv_path, "w", newline="")
    fieldnames = [
        "step",
        "sim_time",
        "selected_joint",
        "cube_x", "cube_y", "cube_z",
        "wrist_x", "wrist_y", "wrist_z",
        "outer_mid_x", "outer_mid_y", "outer_mid_z",
        "inner_mid_x", "inner_mid_y", "inner_mid_z",
        "outer_mid_to_cube_xy",
        "inner_mid_to_cube_xy",
        "outer_mid_to_cube_3d",
        "inner_mid_to_cube_3d",
        "outer_finger_gap",
        "inner_finger_gap",
    ]
    for name in controlled_joint_names:
        fieldnames += [
            f"target_{name}",
            f"actual_{name}",
            f"error_{name}",
        ]
    for name in gripper_debug_joint_names:
        if name not in controlled_joint_names:
            fieldnames += [f"actual_{name}"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def build_debug_snapshot(
    step_count,
    sim_time,
    selected_joint_name,
    robot,
    joint_targets,
    joint_name_to_idx,
    body_name_to_idx,
    cube_pos_w,
):
    joint_pos = robot.data.joint_pos[0].detach().cpu()
    body_pos_w = robot.data.body_pos_w[0].detach().cpu()
    body_quat_w = robot.data.body_quat_w[0].detach().cpu()

    wrist_idx = body_name_to_idx[WRIST_BODY]
    left_outer_idx = body_name_to_idx["left_outer_finger"]
    right_outer_idx = body_name_to_idx["right_outer_finger"]
    left_inner_idx = body_name_to_idx["left_inner_finger"]
    right_inner_idx = body_name_to_idx["right_inner_finger"]

    wrist_pos = body_pos_w[wrist_idx]
    wrist_quat = body_quat_w[wrist_idx]

    left_outer = body_pos_w[left_outer_idx]
    right_outer = body_pos_w[right_outer_idx]
    left_inner = body_pos_w[left_inner_idx]
    right_inner = body_pos_w[right_inner_idx]

    outer_mid = 0.5 * (left_outer + right_outer)
    inner_mid = 0.5 * (left_inner + right_inner)

    cube_pos = cube_pos_w[0].detach().cpu() if cube_pos_w is not None else None

    snapshot = {
        "step": int(step_count),
        "sim_time": float(sim_time),
        "selected_joint": selected_joint_name,
        "wrist_pos": wrist_pos.tolist(),
        "wrist_quat": wrist_quat.tolist(),
        "left_outer_finger_pos": left_outer.tolist(),
        "right_outer_finger_pos": right_outer.tolist(),
        "left_inner_finger_pos": left_inner.tolist(),
        "right_inner_finger_pos": right_inner.tolist(),
        "outer_mid_pos": outer_mid.tolist(),
        "inner_mid_pos": inner_mid.tolist(),
    }

    if cube_pos is not None:
        snapshot["cube_pos"] = cube_pos.tolist()
        snapshot["outer_mid_to_cube_xy"] = float(torch.norm(outer_mid[:2] - cube_pos[:2]).item())
        snapshot["inner_mid_to_cube_xy"] = float(torch.norm(inner_mid[:2] - cube_pos[:2]).item())
        snapshot["outer_mid_to_cube_3d"] = float(torch.norm(outer_mid - cube_pos).item())
        snapshot["inner_mid_to_cube_3d"] = float(torch.norm(inner_mid - cube_pos).item())
    else:
        snapshot["cube_pos"] = None
        snapshot["outer_mid_to_cube_xy"] = None
        snapshot["inner_mid_to_cube_xy"] = None
        snapshot["outer_mid_to_cube_3d"] = None
        snapshot["inner_mid_to_cube_3d"] = None

    snapshot["outer_finger_gap"] = float(torch.norm(left_outer - right_outer).item())
    snapshot["inner_finger_gap"] = float(torch.norm(left_inner - right_inner).item())

    joint_debug = {}
    for name in CONTROLLED_JOINTS:
        idx = joint_name_to_idx[name]
        target = float(joint_targets[0, idx].item())
        actual = float(joint_pos[idx].item())
        joint_debug[name] = {
            "target": target,
            "actual": actual,
            "error": target - actual,
        }

    passive_debug = {}
    for name in GRIPPER_DEBUG_JOINTS:
        idx = joint_name_to_idx[name]
        passive_debug[name] = float(joint_pos[idx].item())

    snapshot["joint_debug"] = joint_debug
    snapshot["gripper_joint_positions"] = passive_debug
    return snapshot


def print_debug_snapshot(snapshot):
    print("\n================ DEBUG SNAPSHOT ================")
    print(f"step={snapshot['step']} | sim_time={snapshot['sim_time']:.4f} | selected_joint={snapshot['selected_joint']}")

    if snapshot["cube_pos"] is not None:
        print(f"cube_pos         = {format_vec3(snapshot['cube_pos'])}")
        print(
            f"outer_mid->cube  = xy:{snapshot['outer_mid_to_cube_xy']:.5f} | 3d:{snapshot['outer_mid_to_cube_3d']:.5f}"
        )
        print(
            f"inner_mid->cube  = xy:{snapshot['inner_mid_to_cube_xy']:.5f} | 3d:{snapshot['inner_mid_to_cube_3d']:.5f}"
        )
    else:
        print("cube_pos         = not found at /World/envs/env_0/Cube")

    print(f"wrist_pos        = {format_vec3(snapshot['wrist_pos'])}")
    print(f"wrist_quat       = {quat_to_str(snapshot['wrist_quat'])}")

    print(f"left_outer       = {format_vec3(snapshot['left_outer_finger_pos'])}")
    print(f"right_outer      = {format_vec3(snapshot['right_outer_finger_pos'])}")
    print(f"outer_mid        = {format_vec3(snapshot['outer_mid_pos'])}")
    print(f"outer_finger_gap = {snapshot['outer_finger_gap']:.5f}")

    print(f"left_inner       = {format_vec3(snapshot['left_inner_finger_pos'])}")
    print(f"right_inner      = {format_vec3(snapshot['right_inner_finger_pos'])}")
    print(f"inner_mid        = {format_vec3(snapshot['inner_mid_pos'])}")
    print(f"inner_finger_gap = {snapshot['inner_finger_gap']:.5f}")

    print("\n----- Controlled joint target / actual / error -----")
    for name in CONTROLLED_JOINTS:
        d = snapshot["joint_debug"][name]
        print(f"{name:28s} target={d['target']:+.5f} | actual={d['actual']:+.5f} | error={d['error']:+.5f}")

    print("\n----- Gripper-related actual joint positions -----")
    for name in GRIPPER_DEBUG_JOINTS:
        val = snapshot["gripper_joint_positions"][name]
        print(f"{name:32s} actual={val:+.5f}")
    print("==================================================\n")


def snapshot_to_csv_row(snapshot):
    row = {
        "step": snapshot["step"],
        "sim_time": snapshot["sim_time"],
        "selected_joint": snapshot["selected_joint"],
    }

    cube = snapshot["cube_pos"] if snapshot["cube_pos"] is not None else [float("nan")] * 3
    wrist = snapshot["wrist_pos"]
    outer_mid = snapshot["outer_mid_pos"]
    inner_mid = snapshot["inner_mid_pos"]

    row.update({
        "cube_x": cube[0], "cube_y": cube[1], "cube_z": cube[2],
        "wrist_x": wrist[0], "wrist_y": wrist[1], "wrist_z": wrist[2],
        "outer_mid_x": outer_mid[0], "outer_mid_y": outer_mid[1], "outer_mid_z": outer_mid[2],
        "inner_mid_x": inner_mid[0], "inner_mid_y": inner_mid[1], "inner_mid_z": inner_mid[2],
        "outer_mid_to_cube_xy": snapshot["outer_mid_to_cube_xy"],
        "inner_mid_to_cube_xy": snapshot["inner_mid_to_cube_xy"],
        "outer_mid_to_cube_3d": snapshot["outer_mid_to_cube_3d"],
        "inner_mid_to_cube_3d": snapshot["inner_mid_to_cube_3d"],
        "outer_finger_gap": snapshot["outer_finger_gap"],
        "inner_finger_gap": snapshot["inner_finger_gap"],
    })

    for name in CONTROLLED_JOINTS:
        d = snapshot["joint_debug"][name]
        row[f"target_{name}"] = d["target"]
        row[f"actual_{name}"] = d["actual"]
        row[f"error_{name}"] = d["error"]

    for name in GRIPPER_DEBUG_JOINTS:
        if name not in CONTROLLED_JOINTS:
            row[f"actual_{name}"] = snapshot["gripper_joint_positions"][name]

    return row


def save_waypoints_json(path, waypoints):
    with open(path, "w") as f:
        json.dump(waypoints, f, indent=2)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cpu",
        num_envs=args_cli.num_envs,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    scene = env.unwrapped.scene
    sim = env.unwrapped.sim
    sim_dt = sim.get_physics_dt()
    device = env.unwrapped.device

    print("\n===== BODY NAMES =====")
    for i, name in enumerate(robot.data.body_names):
        print(i, name)

    print("\n===== JOINT NAMES =====")
    for i, name in enumerate(robot.data.joint_names):
        print(i, name)

    joint_name_to_idx = {name: i for i, name in enumerate(robot.data.joint_names)}
    body_name_to_idx = {name: i for i, name in enumerate(robot.data.body_names)}

    print("\n===== CONTROLLED JOINTS =====")
    for name in CONTROLLED_JOINTS:
        if name not in joint_name_to_idx:
            raise ValueError(f"Joint '{name}' not found in robot.data.joint_names")
        print(f"{name}: {joint_name_to_idx[name]}")

    print("\n===== FINGER BODY INDICES =====")
    for name in FINGER_BODY_NAMES:
        if name not in body_name_to_idx:
            raise ValueError(f"Body '{name}' not found in robot.data.body_names")
        print(f"{name}: {body_name_to_idx[name]}")

    print("\n===== GRIPPER DEBUG JOINTS =====")
    for name in GRIPPER_DEBUG_JOINTS:
        if name not in joint_name_to_idx:
            raise ValueError(f"Joint '{name}' not found in robot.data.joint_names")
        print(f"{name}: {joint_name_to_idx[name]}")

    joint_targets = robot.data.joint_pos.clone()

    selected_joint = 0
    arm_step = 0.01
    gripper_step = 0.01
    periodic_logging = True
    step_count = 0
    sim_time = 0.0

    csv_path, jsonl_path, waypoints_path = make_save_paths(args_cli.save_dir)
    csv_file, csv_writer = create_csv_writer(csv_path, CONTROLLED_JOINTS, GRIPPER_DEBUG_JOINTS)
    jsonl_file = open(jsonl_path, "w")
    waypoints = []

    print_help()
    print(f"[INFO] Selected joint: {CONTROLLED_JOINTS[selected_joint]}")
    print("[INFO] Focus terminal window, then use keyboard controls.")
    print(f"[INFO] CSV log   : {csv_path}")
    print(f"[INFO] JSONL log : {jsonl_path}")
    print(f"[INFO] Waypoints : {waypoints_path}")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        while simulation_app.is_running():
            key = get_key()

            if key is not None:
                if key == "q":
                    print("[INFO] Quitting manual control.")
                    break

                elif key == "h":
                    print_help()

                elif key in ["1", "2", "3", "4", "5", "6", "7"]:
                    selected_joint = int(key) - 1
                    print(f"[INFO] Selected joint: {CONTROLLED_JOINTS[selected_joint]}")

                elif key == "j":
                    joint_name = CONTROLLED_JOINTS[selected_joint]
                    joint_idx = joint_name_to_idx[joint_name]
                    step = gripper_step if joint_name == GRIPPER_JOINT else arm_step
                    joint_targets[:, joint_idx] -= step
                    print(f"[MOVE] {joint_name} -> {joint_targets[0, joint_idx].item():.4f}")

                elif key == "k":
                    joint_name = CONTROLLED_JOINTS[selected_joint]
                    joint_idx = joint_name_to_idx[joint_name]
                    step = gripper_step if joint_name == GRIPPER_JOINT else arm_step
                    joint_targets[:, joint_idx] += step
                    print(f"[MOVE] {joint_name} -> {joint_targets[0, joint_idx].item():.4f}")

                elif key == "o":
                    joint_idx = joint_name_to_idx[GRIPPER_JOINT]
                    joint_targets[:, joint_idx] = 0.0
                    print(f"[GRIPPER] open -> {joint_targets[0, joint_idx].item():.4f}")

                elif key == "c":
                    joint_idx = joint_name_to_idx[GRIPPER_JOINT]
                    joint_targets[:, joint_idx] = 0.20
                    print(f"[GRIPPER] close -> {joint_targets[0, joint_idx].item():.4f}")

                elif key == "p":
                    cube_pos_w, _ = safe_prim_pose("/World/envs/env_0/Cube", device)
                    snapshot = build_debug_snapshot(
                        step_count, sim_time, CONTROLLED_JOINTS[selected_joint],
                        robot, joint_targets, joint_name_to_idx, body_name_to_idx, cube_pos_w
                    )
                    print_debug_snapshot(snapshot)

                elif key == "s":
                    cube_pos_w, _ = safe_prim_pose("/World/envs/env_0/Cube", device)
                    snapshot = build_debug_snapshot(
                        step_count, sim_time, CONTROLLED_JOINTS[selected_joint],
                        robot, joint_targets, joint_name_to_idx, body_name_to_idx, cube_pos_w
                    )
                    waypoints.append(snapshot)
                    save_waypoints_json(waypoints_path, waypoints)
                    print(f"[SAVE] Waypoint saved. total_waypoints={len(waypoints)} -> {waypoints_path}")

                elif key == "l":
                    periodic_logging = not periodic_logging
                    print(f"[INFO] periodic_logging={periodic_logging}")

                elif key == "r":
                    env.reset()
                    joint_targets = robot.data.joint_pos.clone()
                    print("[INFO] Environment reset.")
                    print_joint_targets(joint_targets, joint_name_to_idx)

            robot.set_joint_position_target(joint_targets)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_dt)
            simulation_app.update()

            step_count += 1
            sim_time += sim_dt

            cube_pos_w, _ = safe_prim_pose("/World/envs/env_0/Cube", device)
            snapshot = build_debug_snapshot(
                step_count, sim_time, CONTROLLED_JOINTS[selected_joint],
                robot, joint_targets, joint_name_to_idx, body_name_to_idx, cube_pos_w
            )

            csv_writer.writerow(snapshot_to_csv_row(snapshot))
            jsonl_file.write(json.dumps(snapshot) + "\n")

            if periodic_logging and step_count % max(1, args_cli.log_every) == 0:
                print_debug_snapshot(snapshot)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        csv_file.close()
        jsonl_file.close()
        save_waypoints_json(waypoints_path, waypoints)
        env.close()


if __name__ == "__main__":
    main()