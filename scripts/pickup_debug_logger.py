from __future__ import annotations

import csv
from pathlib import Path

import torch


class PickupDebugLogger:
    def __init__(
        self,
        csv_path: str,
        env_id: int = 0,
        hover_height: float = 0.18,
        grasp_height_offset: float = 0.070,
        close_threshold: float = 0.10,
        target_quat=(0.0, 1.0, 0.0, 0.0),
    ):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.env_id = env_id
        self.hover_height = hover_height
        self.grasp_height_offset = grasp_height_offset
        self.close_threshold = close_threshold
        self.target_quat = target_quat

        self.file = open(self.csv_path, "w", newline="")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "step",
                "phase",
                "cube_x",
                "cube_y",
                "cube_z",
                "tcp_x",
                "tcp_y",
                "tcp_z",
                "wrist_x",
                "wrist_y",
                "wrist_z",
                "xy_dist",
                "tcp_minus_cube_z",
                "hover_z",
                "grasp_z",
                "hover_z_error",
                "grasp_z_error",
                "finger_pos",
                "is_closed",
                "close_too_high",
                "ori_error_rad",
                "tcp_qw",
                "tcp_qx",
                "tcp_qy",
                "tcp_qz",
                "a0",
                "a1",
                "a2",
                "a3",
                "a4",
                "a5",
                "a6",
            ],
        )
        self.writer.writeheader()

    def _quat_error(self, q, target):
        # Isaac quaternion format: [w, x, y, z]
        target_t = torch.tensor(target, device=q.device, dtype=q.dtype)
        dot = torch.abs(torch.sum(q * target_t))
        dot = torch.clamp(dot, -1.0, 1.0)
        return float((2.0 * torch.acos(dot)).detach().cpu())

    def _phase_name(self, xy_dist, tcp_z, cube_z, finger_pos):
        hover_z = cube_z + self.hover_height
        grasp_z = cube_z + self.grasp_height_offset
        closed = finger_pos > self.close_threshold

        if closed and tcp_z > grasp_z + 0.03:
            return "BAD_CLOSE_TOO_HIGH"

        if xy_dist < 0.06 and abs(tcp_z - hover_z) < 0.07 and not closed:
            return "HOVER_ABOVE_OPEN"

        if xy_dist < 0.06 and (grasp_z + 0.03) < tcp_z < (hover_z + 0.03) and not closed:
            return "DESCENDING_OPEN"

        if xy_dist < 0.055 and abs(tcp_z - grasp_z) < 0.035 and not closed:
            return "AT_GRASP_OPEN"

        if xy_dist < 0.055 and abs(tcp_z - grasp_z) < 0.035 and closed:
            return "AT_GRASP_CLOSED"

        if closed:
            return "CLOSED_OTHER"

        return "SEARCHING"

    def log(self, env, actions, step: int):
        scene = env.scene

        robot = scene["robot"]
        obj = scene["object"]
        ee_frame = scene["ee_frame"]

        env_id = self.env_id

        cube_pos = obj.data.root_pos_w[env_id, :3]
        tcp_pos = ee_frame.data.target_pos_w[env_id, 0, :]
        tcp_quat = ee_frame.data.target_quat_w[env_id, 0, :]

        body_names = robot.data.body_names
        wrist_idx = body_names.index("wrist_3_link")
        wrist_pos = robot.data.body_pos_w[env_id, wrist_idx, :]

        joint_names = robot.data.joint_names
        finger_idx = joint_names.index("finger_joint")
        finger_pos = float(robot.data.joint_pos[env_id, finger_idx].detach().cpu())

        xy_dist = float(torch.norm(tcp_pos[:2] - cube_pos[:2]).detach().cpu())
        tcp_minus_cube_z = float((tcp_pos[2] - cube_pos[2]).detach().cpu())

        cube_z = float(cube_pos[2].detach().cpu())
        tcp_z = float(tcp_pos[2].detach().cpu())

        hover_z = cube_z + self.hover_height
        grasp_z = cube_z + self.grasp_height_offset

        hover_z_error = tcp_z - hover_z
        grasp_z_error = tcp_z - grasp_z

        is_closed = finger_pos > self.close_threshold
        close_too_high = is_closed and (tcp_z > grasp_z + 0.03)

        ori_error = self._quat_error(tcp_quat, self.target_quat)

        phase = self._phase_name(xy_dist, tcp_z, cube_z, finger_pos)

        # actions shape can be [num_envs, action_dim]
        a = actions[env_id].detach().cpu().flatten().tolist()
        a = a + [0.0] * (7 - len(a))

        row = {
            "step": step,
            "phase": phase,
            "cube_x": float(cube_pos[0].detach().cpu()),
            "cube_y": float(cube_pos[1].detach().cpu()),
            "cube_z": cube_z,
            "tcp_x": float(tcp_pos[0].detach().cpu()),
            "tcp_y": float(tcp_pos[1].detach().cpu()),
            "tcp_z": tcp_z,
            "wrist_x": float(wrist_pos[0].detach().cpu()),
            "wrist_y": float(wrist_pos[1].detach().cpu()),
            "wrist_z": float(wrist_pos[2].detach().cpu()),
            "xy_dist": xy_dist,
            "tcp_minus_cube_z": tcp_minus_cube_z,
            "hover_z": hover_z,
            "grasp_z": grasp_z,
            "hover_z_error": hover_z_error,
            "grasp_z_error": grasp_z_error,
            "finger_pos": finger_pos,
            "is_closed": int(is_closed),
            "close_too_high": int(close_too_high),
            "ori_error_rad": ori_error,
            "tcp_qw": float(tcp_quat[0].detach().cpu()),
            "tcp_qx": float(tcp_quat[1].detach().cpu()),
            "tcp_qy": float(tcp_quat[2].detach().cpu()),
            "tcp_qz": float(tcp_quat[3].detach().cpu()),
            "a0": a[0],
            "a1": a[1],
            "a2": a[2],
            "a3": a[3],
            "a4": a[4],
            "a5": a[5],
            "a6": a[6],
        }

        self.writer.writerow(row)

        if step % 25 == 0:
            print(
                f"[PICKUP-DEBUG] step={step:04d} phase={phase} "
                f"xy={xy_dist:.3f} z_tcp-cube={tcp_minus_cube_z:.3f} "
                f"grasp_z_err={grasp_z_error:.3f} finger={finger_pos:.3f} "
                f"ori_err={ori_error:.3f} close_high={int(close_too_high)}"
            )

    def close(self):
        self.file.flush()
        self.file.close()
