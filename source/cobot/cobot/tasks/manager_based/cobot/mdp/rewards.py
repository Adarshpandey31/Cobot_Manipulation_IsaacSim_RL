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
    """Return virtual TCP / grasp-center position in world frame.

    Important:
    This assumes ee_frame is configured as:

        wrist_3_link + GRASP_CENTER_OFFSET

    So all reaching / closing / grasping rewards are based on the same point that IK controls.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]


def _get_finger_position(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_joint_name: str = "finger_joint",
) -> torch.Tensor:
    """Return Robotiq main finger joint position."""
    robot = env.scene[robot_cfg.name]
    joint_names = robot.data.joint_names
    finger_idx = joint_names.index(finger_joint_name)
    return robot.data.joint_pos[:, finger_idx]


# ======================================================================================
# Basic object / EE rewards
# ======================================================================================

def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary reward: object z height is above threshold."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Fine reaching reward between object and virtual TCP/grasp center."""
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

    pregrasp_target = object_pos_w.clone()
    pregrasp_target[:, 2] += hover_height

    distance = torch.norm(pregrasp_target - grasp_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def object_xy_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """XY alignment reward between object and virtual TCP/grasp center."""
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
    """Same as object_goal_distance, kept for compatibility."""
    return object_goal_distance(
        env=env,
        std=std,
        minimal_height=minimal_height,
        command_name=command_name,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    )


# ======================================================================================
# Grasp-center rewards: these are the main pickup rewards
# ======================================================================================

def grasp_center_object_distance_coarse(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Coarse reaching reward using virtual TCP/grasp center.

    Uses exp(-d/std), so it gives non-zero signal even when robot is far away.
    """
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
    """Reward TCP for being centered at the correct grasp pose.

    This checks:
        XY close to cube center
        Z close to cube center/height

    Useful before adding lift reward.
    """
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    grasp_pos_w = get_grasp_center_pos_w(env, ee_frame_cfg)

    xy_dist = torch.norm(object_pos_w[:, :2] - grasp_pos_w[:, :2], dim=1)
    z_dist = torch.abs(object_pos_w[:, 2] - grasp_pos_w[:, 2])

    xy_score = torch.exp(-xy_dist / xy_std)
    z_score = torch.exp(-z_dist / z_std)

    return xy_score * z_score


# ======================================================================================
# Close gripper rewards based on virtual TCP/grasp center
# ======================================================================================

def soft_close_gripper_when_near_grasp_center(
    env: ManagerBasedRLEnv,
    std: float,
    close_position: float = 0.72,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Soft reward for closing gripper when virtual TCP is near object.

    Unlike hard threshold reward, this gives smooth signal.
    """
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
    """Reward closed gripper/TCP moving upward while aligned in XY.

    This does not guarantee object is lifted, but helps policy learn:
        go near object -> close -> move upward
    """
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
# These preserve old reward config names but redirect to the corrected ee_frame/TCP logic.
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


def soft_close_gripper_when_near_object(
    env: ManagerBasedRLEnv,
    std: float,
    close_position: float = 0.72,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    return soft_close_gripper_when_near_grasp_center(
        env, std, close_position, object_cfg, robot_cfg, ee_frame_cfg
    )


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


def gripper_midpoint_grasp_pose_reward(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    return grasp_center_grasp_pose_reward(env, xy_std, z_std, object_cfg, ee_frame_cfg)