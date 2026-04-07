import math
import torch

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms


ik_cfg = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=False,
    ik_method="dls",
    ik_params={"lambda_val": 0.04, "joint_weight": 0.8},
)

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

TCP_OFFSET_LOCAL = torch.tensor([0.0, 0.0, 0.17], dtype=torch.float32)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def get_downward_grasp_quat(device: torch.device, jaw_yaw_rad: float = 0.0) -> torch.Tensor:
    q_down = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    half = jaw_yaw_rad / 2.0
    q_yaw = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]], dtype=torch.float32, device=device)
    return _quat_mul(q_yaw, q_down)


def _get_arm_joint_ids(robot):
    joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    return joint_ids


def _get_body_indices(robot, body_name: str):
    body_idx_pose = robot.data.body_names.index(body_name)
    body_idx_jac = body_idx_pose - 1
    if body_idx_jac < 0:
        raise ValueError(f"Invalid jacobian body index for {body_name}")
    return body_idx_pose, body_idx_jac


def get_or_create_ik_controller(env):
    if not hasattr(env, "_ik_controller"):
        env._ik_controller = DifferentialIKController(
            cfg=ik_cfg,
            num_envs=env.num_envs,
            device=env.device,
        )
    return env._ik_controller


def differential_ik_step(env, asset_cfg, body_name, target_tcp_pos_w, target_tcp_quat_w=None):
    robot = env.scene[asset_cfg.name]
    controller = get_or_create_ik_controller(env)

    arm_joint_ids = _get_arm_joint_ids(robot)
    body_idx_pose, body_idx_jac = _get_body_indices(robot, body_name)

    device = env.device
    tcp_offset_local = TCP_OFFSET_LOCAL.to(device).unsqueeze(0).repeat(env.num_envs, 1)

    root_pos_w = robot.data.root_pos_w.clone()
    root_quat_w = robot.data.root_quat_w.clone()

    body_pos_w = robot.data.body_pos_w[:, body_idx_pose].clone()
    body_quat_w = robot.data.body_quat_w[:, body_idx_pose].clone()

    tcp_offset_w = quat_apply(body_quat_w, tcp_offset_local)
    tcp_pos_w = body_pos_w + tcp_offset_w
    tcp_quat_w = body_quat_w if target_tcp_quat_w is None else target_tcp_quat_w

    tcp_pos_b, tcp_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, tcp_pos_w, tcp_quat_w)
    target_tcp_pos_b, target_tcp_quat_b = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        target_tcp_pos_w,
        tcp_quat_w if target_tcp_quat_w is None else target_tcp_quat_w,
    )

    target_pose_b = torch.cat((target_tcp_pos_b, target_tcp_quat_b), dim=1)
    controller.set_command(target_pose_b, ee_pos=tcp_pos_b, ee_quat=tcp_quat_b)

    jacobians = robot.root_physx_view.get_jacobians()
    jacobian = jacobians[:, body_idx_jac, :, arm_joint_ids]

    current_arm_q = robot.data.joint_pos[:, arm_joint_ids].clone()
    next_arm_q = controller.compute(tcp_pos_b, tcp_quat_b, jacobian, current_arm_q)

    next_arm_q[:, 0] = torch.clamp(next_arm_q[:, 0], -3.14, 3.14)
    next_arm_q[:, 1] = torch.clamp(next_arm_q[:, 1], -2.40, 0.50)
    next_arm_q[:, 2] = torch.clamp(next_arm_q[:, 2], -2.50, 2.50)
    next_arm_q[:, 3] = torch.clamp(next_arm_q[:, 3], -3.14, 3.14)
    next_arm_q[:, 4] = torch.clamp(next_arm_q[:, 4], -3.14, 3.14)
    next_arm_q[:, 5] = torch.clamp(next_arm_q[:, 5], -3.14, 3.14)

    return next_arm_q