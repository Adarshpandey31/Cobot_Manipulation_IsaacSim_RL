from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ======================================================================================
# Core helper: use ee_frame as the real grasp center / TCP
# ======================================================================================


def get_grasp_center_pos_w(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return virtual TCP / grasp-center position in world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]


def get_grasp_center_quat_w(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return virtual TCP / grasp-center quaternion in world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[..., 0, :]


def _get_finger_position(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_joint_name: str = "finger_joint",
) -> torch.Tensor:
    """Return Robotiq main finger joint position."""
    robot = env.scene[robot_cfg.name]
    finger_idx = robot.data.joint_names.index(finger_joint_name)
    return robot.data.joint_pos[:, finger_idx]


def _get_last_gripper_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return last scalar gripper action.

    For current BinaryJointPositionAction:
        positive action -> open
        negative action -> close
    """
    if hasattr(env.action_manager, "action"):
        actions = env.action_manager.action
    elif hasattr(env.action_manager, "_action"):
        actions = env.action_manager._action
    else:
        # Safe fallback. This should normally not happen.
        obj: RigidObject = env.scene["object"]
        return torch.zeros(obj.data.root_pos_w.shape[0], device=obj.data.root_pos_w.device)

    return actions[:, -1]


def _orientation_error(
    tcp_quat: torch.Tensor,
    target_quat: tuple[float, float, float, float],
) -> torch.Tensor:
    """Quaternion orientation error in radians.

    Isaac quaternion format: [w, x, y, z].
    """
    target = torch.tensor(target_quat, device=tcp_quat.device, dtype=tcp_quat.dtype).unsqueeze(0)
    target = target.repeat(tcp_quat.shape[0], 1)
    target = target / torch.clamp(torch.norm(target, dim=1, keepdim=True), min=1e-8)

    tcp_quat = tcp_quat / torch.clamp(torch.norm(tcp_quat, dim=1, keepdim=True), min=1e-8)

    dot = torch.abs(torch.sum(tcp_quat * target, dim=1))
    dot = torch.clamp(dot, 0.0, 1.0)
    return 2.0 * torch.acos(dot)


# ======================================================================================
# Orientation rewards
# ======================================================================================


def tcp_orientation_alignment(
    env: ManagerBasedRLEnv,
    target_quat: tuple[float, float, float, float],
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward TCP orientation matching scripted vertical grasp orientation."""
    tcp_quat = get_grasp_center_quat_w(env, ee_frame_cfg)
    angle_error = _orientation_error(tcp_quat, target_quat)
    return 1.0 - torch.tanh(angle_error / std)


def tcp_vertical_grasp_pose_oriented(
    env: ManagerBasedRLEnv,
    std_xy: float,
    std_z: float,
    grasp_height_offset: float,
    target_quat: tuple[float, float, float, float],
    ori_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward correct XYZ grasp pose and correct gripper orientation."""
    pos_reward = tcp_vertical_grasp_pose(
        env=env,
        std_xy=std_xy,
        std_z=std_z,
        grasp_height_offset=grasp_height_offset,
        object_cfg=object_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )

    ori_reward = tcp_orientation_alignment(
        env=env,
        target_quat=target_quat,
        std=ori_std,
        ee_frame_cfg=ee_frame_cfg,
    )

    return pos_reward * ori_reward


def close_gripper_at_oriented_vertical_grasp_pose(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    grasp_height_offset: float,
    target_quat: tuple[float, float, float, float],
    ori_threshold: float,
    close_threshold: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closed gripper only when TCP pose and orientation are correct."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)
    tcp_quat = get_grasp_center_quat_w(env, ee_frame_cfg)

    target_z = object_pos[:, 2] + grasp_height_offset

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    z_dist = torch.abs(tcp_pos[:, 2] - target_z)
    angle_error = _orientation_error(tcp_quat, target_quat)

    finger_pos = _get_finger_position(env, robot_cfg)

    at_xy = xy_dist < xy_threshold
    at_z = z_dist < z_threshold
    oriented = angle_error < ori_threshold
    closed = finger_pos > close_threshold

    return (at_xy & at_z & oriented & closed).float()


# ======================================================================================
# Basic object / TCP rewards
# ======================================================================================


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary reward: object z height is above threshold."""
    obj: RigidObject = env.scene[object_cfg.name]
    return (obj.data.root_pos_w[:, 2] > minimal_height).float()


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Fine reaching reward between object and virtual TCP."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    grasp_pos_w = get_grasp_center_pos_w(env, ee_frame_cfg)

    distance = torch.norm(object_pos_w - grasp_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def ee_pregrasp_distance(
    env: ManagerBasedRLEnv,
    std: float,
    hover_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward virtual TCP for reaching a point above the object."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    grasp_pos_w = get_grasp_center_pos_w(env, ee_frame_cfg)

    target = object_pos_w.clone()
    target[:, 2] += hover_height

    distance = torch.norm(target - grasp_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def object_xy_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """XY alignment reward between object and virtual TCP."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    grasp_pos_w = get_grasp_center_pos_w(env, ee_frame_cfg)

    distance_xy = torch.norm(object_pos_w[:, :2] - grasp_pos_w[:, :2], dim=1)
    return 1.0 - torch.tanh(distance_xy / std)


# ======================================================================================
# Goal tracking rewards, kept for later place/lift-to-goal stage
# ======================================================================================


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward object for being near commanded target pose after it is lifted."""
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)

    distance = torch.norm(des_pos_w - obj.data.root_pos_w[:, :3], dim=1)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height

    return lifted.float() * (1.0 - torch.tanh(distance / std))


def object_lifted_and_near_goal(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Compatibility wrapper for goal reward."""
    return object_goal_distance(env, std, minimal_height, command_name, robot_cfg, object_cfg)


# ======================================================================================
# Grasp-center reaching rewards
# ======================================================================================


def grasp_center_object_distance_coarse(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Coarse reaching reward using virtual TCP/grasp center."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    grasp_pos_w = get_grasp_center_pos_w(env, ee_frame_cfg)

    distance = torch.norm(object_pos_w - grasp_pos_w, dim=1)
    return torch.exp(-distance / std)


def grasp_center_object_distance_fine(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Fine reaching reward using virtual TCP/grasp center."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    grasp_pos_w = get_grasp_center_pos_w(env, ee_frame_cfg)

    distance = torch.norm(object_pos_w - grasp_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def grasp_center_xy_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """XY alignment reward using virtual TCP/grasp center."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    grasp_pos_w = get_grasp_center_pos_w(env, ee_frame_cfg)

    distance_xy = torch.norm(object_pos_w[:, :2] - grasp_pos_w[:, :2], dim=1)
    return 1.0 - torch.tanh(distance_xy / std)


def grasp_center_pregrasp_distance(
    env: ManagerBasedRLEnv,
    std: float,
    hover_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Coarse pregrasp reward using virtual TCP/grasp center."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    grasp_pos_w = get_grasp_center_pos_w(env, ee_frame_cfg)

    target = object_pos_w.clone()
    target[:, 2] += hover_height

    distance = torch.norm(target - grasp_pos_w, dim=1)
    return torch.exp(-distance / std)


def grasp_center_grasp_pose_reward(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward TCP for being centered at object pose."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    grasp_pos_w = get_grasp_center_pos_w(env, ee_frame_cfg)

    xy_dist = torch.norm(object_pos_w[:, :2] - grasp_pos_w[:, :2], dim=1)
    z_dist = torch.abs(object_pos_w[:, 2] - grasp_pos_w[:, 2])

    xy_score = torch.exp(-xy_dist / xy_std)
    z_score = torch.exp(-z_dist / z_std)

    return xy_score * z_score


# ======================================================================================
# Scripted-style vertical pickup shaping
# ======================================================================================


def tcp_hover_above_object(
    env: ManagerBasedRLEnv,
    std_xy: float,
    std_z: float,
    hover_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward TCP at hover position above cube center."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)

    target_z = object_pos[:, 2] + hover_height

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    z_dist = torch.abs(tcp_pos[:, 2] - target_z)

    xy_score = torch.exp(-xy_dist / std_xy)
    z_score = torch.exp(-z_dist / std_z)

    return xy_score * z_score


def tcp_vertical_grasp_pose(
    env: ManagerBasedRLEnv,
    std_xy: float,
    std_z: float,
    grasp_height_offset: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward TCP at lower vertical grasp pose."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)

    target_z = object_pos[:, 2] + grasp_height_offset

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    z_dist = torch.abs(tcp_pos[:, 2] - target_z)

    xy_score = torch.exp(-xy_dist / std_xy)
    z_score = torch.exp(-z_dist / std_z)

    return xy_score * z_score


def tcp_vertical_descent_progress(
    env: ManagerBasedRLEnv,
    std_xy: float,
    hover_height: float,
    grasp_height_offset: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward vertical descent inside the column above the cube."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)

    hover_z = object_pos[:, 2] + hover_height
    grasp_z = object_pos[:, 2] + grasp_height_offset

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    xy_score = torch.exp(-xy_dist / std_xy)

    denom = torch.clamp(hover_z - grasp_z, min=1e-4)
    descent_progress = torch.clamp((hover_z - tcp_pos[:, 2]) / denom, 0.0, 1.0)

    not_too_low = tcp_pos[:, 2] > (object_pos[:, 2] - 0.02)

    return xy_score * descent_progress * not_too_low.float()


def low_side_approach_penalty(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    safe_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty for approaching low from the side instead of vertically."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)

    low = tcp_pos[:, 2] < (object_pos[:, 2] + safe_height)
    side = xy_dist > xy_threshold

    return (low & side).float()


# ======================================================================================
# Gripper position rewards
# ======================================================================================


def open_gripper_above_object(
    env: ManagerBasedRLEnv,
    std_xy: float,
    std_z: float,
    hover_height: float,
    close_position: float = 0.72,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward gripper staying open while TCP is above cube."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)

    target_z = object_pos[:, 2] + hover_height

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    z_dist = torch.abs(tcp_pos[:, 2] - target_z)

    xy_score = torch.exp(-xy_dist / std_xy)
    z_score = torch.exp(-z_dist / std_z)

    finger_pos = _get_finger_position(env, robot_cfg)

    open_score = 1.0 - torch.clamp(finger_pos / close_position, 0.0, 1.0)

    return xy_score * z_score * open_score


def soft_close_gripper_at_vertical_grasp_pose(
    env: ManagerBasedRLEnv,
    std_xy: float,
    std_z: float,
    grasp_height_offset: float,
    close_position: float = 0.72,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Soft reward for actual finger closing only at correct vertical grasp pose."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)

    target_z = object_pos[:, 2] + grasp_height_offset

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    z_dist = torch.abs(tcp_pos[:, 2] - target_z)

    xy_score = torch.exp(-xy_dist / std_xy)
    z_score = torch.exp(-z_dist / std_z)

    finger_pos = _get_finger_position(env, robot_cfg)
    closed_score = torch.clamp(finger_pos / close_position, 0.0, 1.0)

    return xy_score * z_score * closed_score


def close_gripper_at_vertical_grasp_pose(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    grasp_height_offset: float,
    close_threshold: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Hard reward for closed gripper at correct vertical grasp pose."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)

    target_z = object_pos[:, 2] + grasp_height_offset

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    z_dist = torch.abs(tcp_pos[:, 2] - target_z)

    finger_pos = _get_finger_position(env, robot_cfg)

    at_grasp_xy = xy_dist < xy_threshold
    at_grasp_z = z_dist < z_threshold
    closed = finger_pos > close_threshold

    return (at_grasp_xy & at_grasp_z & closed).float()


def close_too_high_penalty(
    env: ManagerBasedRLEnv,
    grasp_height_offset: float,
    height_margin: float,
    close_threshold: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty if gripper closes while TCP is still above grasp height."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)

    correct_grasp_z = object_pos[:, 2] + grasp_height_offset
    too_high = tcp_pos[:, 2] > (correct_grasp_z + height_margin)

    finger_pos = _get_finger_position(env, robot_cfg)
    closed = finger_pos > close_threshold

    return (too_high & closed).float()


def open_until_vertical_grasp_pose(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    grasp_height_offset: float,
    close_position: float = 0.72,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward gripper staying open until TCP reaches actual grasp pose."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)

    target_z = object_pos[:, 2] + grasp_height_offset

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    z_dist = torch.abs(tcp_pos[:, 2] - target_z)

    at_grasp_pose = (xy_dist < xy_threshold) & (z_dist < z_threshold)

    finger_pos = _get_finger_position(env, robot_cfg)
    open_score = 1.0 - torch.clamp(finger_pos / close_position, 0.0, 1.0)

    return (~at_grasp_pose).float() * open_score


def close_before_vertical_grasp_penalty(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    grasp_height_offset: float,
    close_threshold: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty if gripper closes before TCP reaches correct grasp pose."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)

    target_z = object_pos[:, 2] + grasp_height_offset

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    z_dist = torch.abs(tcp_pos[:, 2] - target_z)

    at_grasp_pose = (xy_dist < xy_threshold) & (z_dist < z_threshold)

    finger_pos = _get_finger_position(env, robot_cfg)
    closed = finger_pos > close_threshold

    return (closed & (~at_grasp_pose)).float()


# ======================================================================================
# NEW: Gripper action sign rewards
# These solve the current log issue:
#   AT_GRASP_OPEN, finger=0.000, action[-1] positive.
# ======================================================================================


def close_action_at_vertical_grasp_pose(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    grasp_height_offset: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward negative gripper action when TCP is at the vertical grasp pose.

    For current BinaryJointPositionAction:
        negative action -> close
        positive action -> open
    """
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)

    target_z = object_pos[:, 2] + grasp_height_offset

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    z_dist = torch.abs(tcp_pos[:, 2] - target_z)

    at_grasp = (xy_dist < xy_threshold) & (z_dist < z_threshold)

    gripper_action = _get_last_gripper_action(env)
    close_action_score = torch.clamp(-gripper_action, 0.0, 1.0)

    return at_grasp.float() * close_action_score


def open_action_at_vertical_grasp_penalty(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    grasp_height_offset: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty for positive/open gripper action when TCP is already at grasp pose."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)

    target_z = object_pos[:, 2] + grasp_height_offset

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    z_dist = torch.abs(tcp_pos[:, 2] - target_z)

    at_grasp = (xy_dist < xy_threshold) & (z_dist < z_threshold)

    gripper_action = _get_last_gripper_action(env)
    open_action_score = torch.clamp(gripper_action, 0.0, 1.0)

    return at_grasp.float() * open_action_score


def close_action_at_oriented_vertical_grasp_pose(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    grasp_height_offset: float,
    target_quat: tuple[float, float, float, float],
    ori_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward negative gripper action only at correct pose and orientation."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)
    tcp_quat = get_grasp_center_quat_w(env, ee_frame_cfg)

    target_z = object_pos[:, 2] + grasp_height_offset

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    z_dist = torch.abs(tcp_pos[:, 2] - target_z)
    ori_err = _orientation_error(tcp_quat, target_quat)

    at_grasp = (xy_dist < xy_threshold) & (z_dist < z_threshold) & (ori_err < ori_threshold)

    gripper_action = _get_last_gripper_action(env)
    close_action_score = torch.clamp(-gripper_action, 0.0, 1.0)

    return at_grasp.float() * close_action_score


def open_action_at_oriented_vertical_grasp_penalty(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    grasp_height_offset: float,
    target_quat: tuple[float, float, float, float],
    ori_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty for open action at correct pose and orientation."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)
    tcp_quat = get_grasp_center_quat_w(env, ee_frame_cfg)

    target_z = object_pos[:, 2] + grasp_height_offset

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    z_dist = torch.abs(tcp_pos[:, 2] - target_z)
    ori_err = _orientation_error(tcp_quat, target_quat)

    at_grasp = (xy_dist < xy_threshold) & (z_dist < z_threshold) & (ori_err < ori_threshold)

    gripper_action = _get_last_gripper_action(env)
    open_action_score = torch.clamp(gripper_action, 0.0, 1.0)

    return at_grasp.float() * open_action_score


# ======================================================================================
# Lift-related rewards
# ======================================================================================


def object_close_to_gripper_when_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward object being lifted and staying close to virtual TCP."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    grasp_pos_w = get_grasp_center_pos_w(env, ee_frame_cfg)

    distance = torch.norm(object_pos_w - grasp_pos_w, dim=1)
    lifted = object_pos_w[:, 2] > minimal_height

    return lifted.float() * (1.0 - torch.tanh(distance / std))


def lift_gripper_after_close_reward(
    env: ManagerBasedRLEnv,
    xy_std: float,
    max_lift: float,
    close_position: float = 0.72,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closed gripper/TCP moving upward while aligned in XY."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    grasp_pos_w = get_grasp_center_pos_w(env, ee_frame_cfg)

    xy_dist = torch.norm(object_pos_w[:, :2] - grasp_pos_w[:, :2], dim=1)
    xy_score = torch.exp(-xy_dist / xy_std)

    finger_pos = _get_finger_position(env, robot_cfg)
    closed_score = torch.clamp(finger_pos / close_position, 0.0, 1.0)

    z_lift = torch.clamp((grasp_pos_w[:, 2] - object_pos_w[:, 2]) / max_lift, 0.0, 1.0)

    return xy_score * closed_score * z_lift


# ======================================================================================
# Compatibility wrappers
# ======================================================================================


def gripper_midpoint_object_distance_coarse(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    return grasp_center_object_distance_coarse(env, std, object_cfg, ee_frame_cfg)


def gripper_midpoint_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    return grasp_center_object_distance_fine(env, std, object_cfg, ee_frame_cfg)


def gripper_midpoint_xy_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    return grasp_center_xy_object_distance(env, std, object_cfg, ee_frame_cfg)


def gripper_midpoint_pregrasp_distance(
    env: ManagerBasedRLEnv,
    std: float,
    hover_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    return grasp_center_pregrasp_distance(env, std, hover_height, object_cfg, ee_frame_cfg)


def gripper_midpoint_grasp_pose_reward(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    return grasp_center_grasp_pose_reward(env, xy_std, z_std, object_cfg, ee_frame_cfg)


def soft_close_gripper_when_near_grasp_center(
    env: ManagerBasedRLEnv,
    std: float,
    close_position: float = 0.72,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Soft reward for closing gripper when virtual TCP is near object."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    grasp_pos_w = get_grasp_center_pos_w(env, ee_frame_cfg)

    distance = torch.norm(object_pos_w - grasp_pos_w, dim=1)
    near_score = torch.exp(-distance / std)

    finger_pos = _get_finger_position(env, robot_cfg)
    closed_score = torch.clamp(finger_pos / close_position, 0.0, 1.0)

    return near_score * closed_score


def close_gripper_when_near_grasp_center(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    close_threshold: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Hard reward for gripper closed while virtual TCP is near object."""
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    grasp_pos_w = get_grasp_center_pos_w(env, ee_frame_cfg)

    distance = torch.norm(object_pos_w - grasp_pos_w, dim=1)

    finger_pos = _get_finger_position(env, robot_cfg)

    near = distance < distance_threshold
    closed = finger_pos > close_threshold

    return near.float() * closed.float()


def soft_close_gripper_when_near_object(
    env: ManagerBasedRLEnv,
    std: float,
    close_position: float = 0.72,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    return soft_close_gripper_when_near_grasp_center(env, std, close_position, object_cfg, robot_cfg, ee_frame_cfg)


def close_gripper_when_near_object(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    close_threshold: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    return close_gripper_when_near_grasp_center(
        env, distance_threshold, close_threshold, object_cfg, robot_cfg, ee_frame_cfg
    )


def close_action_at_vertical_grasp_pose(
    env,
    xy_threshold: float,
    z_threshold: float,
    grasp_height_offset: float,
    close_action_threshold: float = -0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the policy for COMMANDING close when TCP is at the vertical grasp pose.

    In your logs, positive gripper action keeps the gripper open.
    So we reward negative gripper action near the grasp pose.
    """
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    tcp_pos = get_grasp_center_pos_w(env, ee_frame_cfg)

    target_z = object_pos[:, 2] + grasp_height_offset

    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=1)
    z_dist = torch.abs(tcp_pos[:, 2] - target_z)

    at_xy = xy_dist < xy_threshold
    at_z = z_dist < z_threshold
    at_grasp = at_xy & at_z

    try:
        raw_action = env.action_manager.action
        gripper_action = raw_action[:, -1]
    except Exception:
        return torch.zeros_like(xy_dist)

    close_cmd = gripper_action < close_action_threshold

    return (at_grasp & close_cmd).float()
