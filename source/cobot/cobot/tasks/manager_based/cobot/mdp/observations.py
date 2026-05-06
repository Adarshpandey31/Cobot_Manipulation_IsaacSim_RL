from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.math import combine_frame_transforms

def object_to_goal_position(
    env,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    return des_pos_w - obj.data.root_pos_w[:, :3]

def gripper_opening(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    joint_names = robot.data.joint_names
    idx = joint_names.index("finger_joint")
    return robot.data.joint_pos[:, idx:idx+1]
    
def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w)
    return ee_pos_b


def ee_to_object_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    return object_pos_w - ee_pos_w

def gripper_midpoint_position(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]

    body_names = robot.data.body_names
    left_idx = body_names.index("left_outer_finger")
    right_idx = body_names.index("right_outer_finger")

    left_pos = robot.data.body_pos_w[:, left_idx, :]
    right_pos = robot.data.body_pos_w[:, right_idx, :]
    gripper_mid_w = 0.5 * (left_pos + right_pos)

    root_pos_w = robot.data.root_pos_w[:, :3]
    root_quat_w = robot.data.root_quat_w

    gripper_mid_b, _ = subtract_frame_transforms(
        root_pos_w, root_quat_w, gripper_mid_w
    )

    return gripper_mid_b    

def gripper_midpoint_to_object(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    body_names = robot.data.body_names
    left_idx = body_names.index("left_outer_finger")
    right_idx = body_names.index("right_outer_finger")

    left_pos = robot.data.body_pos_w[:, left_idx, :]
    right_pos = robot.data.body_pos_w[:, right_idx, :]
    gripper_mid_w = 0.5 * (left_pos + right_pos)

    object_pos_w = obj.data.root_pos_w[:, :3]

    root_pos_w = robot.data.root_pos_w[:, :3]
    root_quat_w = robot.data.root_quat_w

    gripper_mid_b, _ = subtract_frame_transforms(
        root_pos_w, root_quat_w, gripper_mid_w
    )

    object_pos_b, _ = subtract_frame_transforms(
        root_pos_w, root_quat_w, object_pos_w
    )

    return object_pos_b - gripper_mid_b