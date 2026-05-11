from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,  # True for no RL
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg | DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        ee_to_object = ObsTerm(func=mdp.ee_to_object_position)
        gripper_opening = ObsTerm(func=mdp.gripper_opening)
        gripper_midpoint_position = ObsTerm(func=mdp.gripper_midpoint_position)
        gripper_midpoint_to_object = ObsTerm(func=mdp.gripper_midpoint_to_object)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        ee_to_object = ObsTerm(func=mdp.ee_to_object_position)
        gripper_opening = ObsTerm(func=mdp.gripper_opening)
        gripper_midpoint_position = ObsTerm(func=mdp.gripper_midpoint_position)
        gripper_midpoint_to_object = ObsTerm(func=mdp.gripper_midpoint_to_object)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    # -------------------------------------------------------------------------
    # 1. Move above cube, gripper open
    # -------------------------------------------------------------------------
    hover_above_object = RewTerm(
        func=mdp.tcp_hover_above_object,
        params={
            "std_xy": 0.20,
            "std_z": 0.10,
            "hover_height": 0.18,
        },
        weight=2.0,
    )

    open_above_object = RewTerm(
        func=mdp.open_gripper_above_object,
        params={
            "std_xy": 0.20,
            "std_z": 0.10,
            "hover_height": 0.18,
            "close_position": 0.72,
        },
        weight=2.0,
    )

    # -------------------------------------------------------------------------
    # 2. XY alignment
    # -------------------------------------------------------------------------
    xy_align_object_coarse = RewTerm(
        func=mdp.grasp_center_xy_object_distance,
        params={"std": 0.30},
        weight=4.0,
    )

    xy_align_object_fine = RewTerm(
        func=mdp.grasp_center_xy_object_distance,
        params={"std": 0.06},
        weight=4.0,
    )

    # -------------------------------------------------------------------------
    # 3. Orientation shaping only, not hard gate
    # -------------------------------------------------------------------------
    tcp_orientation = RewTerm(
        func=mdp.tcp_orientation_alignment,
        params={
            "target_quat": (0.0, 1.0, 0.0, 0.0),
            "std": 1.20,
        },
        weight=2.0,
    )

    # -------------------------------------------------------------------------
    # 4. Vertical descent
    # IMPORTANT FIX:
    # Your debug log shows correct visual grasp at tcp-cube z around 0.070 m.
    # So target grasp height must be 0.070, not 0.035.
    # -------------------------------------------------------------------------
    vertical_descent_progress = RewTerm(
        func=mdp.tcp_vertical_descent_progress,
        params={
            "std_xy": 0.07,
            "hover_height": 0.18,
            "grasp_height_offset": 0.070,
        },
        weight=8.0,
    )

    vertical_grasp_pose = RewTerm(
        func=mdp.tcp_vertical_grasp_pose,
        params={
            "std_xy": 0.05,
            "std_z": 0.045,
            "grasp_height_offset": 0.070,
        },
        weight=24.0,
    )

    # -------------------------------------------------------------------------
    # 5. Prevent side / wrong close
    # -------------------------------------------------------------------------
    low_side_approach_penalty = RewTerm(
        func=mdp.low_side_approach_penalty,
        params={
            "xy_threshold": 0.08,
            "safe_height": 0.12,
        },
        weight=-6.0,
    )

    close_before_grasp_penalty = RewTerm(
        func=mdp.close_before_vertical_grasp_penalty,
        params={
            "xy_threshold": 0.050,
            "z_threshold": 0.045,
            "grasp_height_offset": 0.070,
            "close_threshold": 0.10,
        },
        weight=-12.0,
    )

    close_too_high_penalty = RewTerm(
        func=mdp.close_too_high_penalty,
        params={
            "grasp_height_offset": 0.070,
            "height_margin": 0.035,
            "close_threshold": 0.10,
        },
        weight=-12.0,
    )

    # -------------------------------------------------------------------------
    # 6. Teach close command first
    # -------------------------------------------------------------------------
    close_action_at_grasp_pose = RewTerm(
        func=mdp.close_action_at_vertical_grasp_pose,
        params={
            "xy_threshold": 0.050,
            "z_threshold": 0.045,
            "grasp_height_offset": 0.070,
            "close_action_threshold": -0.05,
        },
        weight=18.0,
    )

    soft_close_at_vertical_grasp_pose = RewTerm(
        func=mdp.soft_close_gripper_at_vertical_grasp_pose,
        params={
            "std_xy": 0.05,
            "std_z": 0.045,
            "grasp_height_offset": 0.070,
            "close_position": 0.72,
        },
        weight=8.0,
    )

    close_at_vertical_grasp_pose = RewTerm(
        func=mdp.close_gripper_at_vertical_grasp_pose,
        params={
            "xy_threshold": 0.050,
            "z_threshold": 0.045,
            "grasp_height_offset": 0.070,
            "close_threshold": 0.10,
        },
        weight=30.0,
    )

    # -------------------------------------------------------------------------
    # 7. Smoothness
    # -------------------------------------------------------------------------
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-1e-4,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=32, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # ---------------------------------------------------------------------
        # Simulation timing
        # ---------------------------------------------------------------------
        self.decimation = 2
        self.episode_length_s = 10.0

        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        # ---------------------------------------------------------------------
        # Physics settings
        # ---------------------------------------------------------------------
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # ---------------------------------------------------------------------
        # Franka-style object lift command, adapted for UR10e
        # Target is close to cube and only moderately above table.
        # This makes pickup/lift easier before we widen the task.
        # ---------------------------------------------------------------------
        self.commands.object_pose.resampling_time_range = (10.0, 10.0)

        self.commands.object_pose.ranges.pos_x = (0.45, 0.55)
        self.commands.object_pose.ranges.pos_y = (-0.10, 0.10)
        self.commands.object_pose.ranges.pos_z = (0.18, 0.28)

        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)
