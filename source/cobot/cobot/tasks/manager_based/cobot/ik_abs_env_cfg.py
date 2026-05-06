from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg, mdp
from .joint_pos_env_cfg import GRASP_CENTER_OFFSET


@configclass
class CobotCubeLiftEnvCfg(joint_pos_env_cfg.CobotCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = joint_pos_env_cfg.UR10E_TASK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=GRASP_CENTER_OFFSET
            ),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": 0.72},
        )