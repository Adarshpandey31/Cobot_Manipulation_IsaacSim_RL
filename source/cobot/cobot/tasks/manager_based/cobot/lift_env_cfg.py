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
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
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
        debug_vis=False,     #True for no RL
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
    # 1. Reach above cube using virtual TCP / grasp center
    # -------------------------------------------------------------------------
    pregrasp_above_object = RewTerm(
        func=mdp.grasp_center_pregrasp_distance,
        params={
            "std": 0.40,
            "hover_height": 0.10,
        },
        weight=2.0,
    )

    # -------------------------------------------------------------------------
    # 2. Coarse TCP-to-object reaching
    # -------------------------------------------------------------------------
    grasp_center_reaching_coarse = RewTerm(
        func=mdp.grasp_center_object_distance_coarse,
        params={
            "std": 0.50,
        },
        weight=8.0,
    )

    # -------------------------------------------------------------------------
    # 3. Fine TCP-to-object reaching
    # -------------------------------------------------------------------------
    grasp_center_reaching_fine = RewTerm(
        func=mdp.grasp_center_object_distance_fine,
        params={
            "std": 0.08,
        },
        weight=2.0,
    )

    # -------------------------------------------------------------------------
    # 4. XY alignment using virtual TCP / grasp center
    # -------------------------------------------------------------------------
    xy_align_object_coarse = RewTerm(
        func=mdp.grasp_center_xy_object_distance,
        params={
            "std": 0.30,
        },
        weight=4.0,
    )

    xy_align_object_fine = RewTerm(
        func=mdp.grasp_center_xy_object_distance,
        params={
            "std": 0.06,
        },
        weight=1.0,
    )

    # -------------------------------------------------------------------------
    # 5. Correct grasp pose: TCP centered in XY and correct in Z
    # -------------------------------------------------------------------------
    grasp_pose = RewTerm(
        func=mdp.grasp_center_grasp_pose_reward,
        params={
            "xy_std": 0.08,
            "z_std": 0.04,
        },
        weight=6.0,
    )

    # -------------------------------------------------------------------------
    # 6. Close gripper near cube
    # -------------------------------------------------------------------------
    soft_close_when_near = RewTerm(
        func=mdp.soft_close_gripper_when_near_grasp_center,
        params={
            "std": 0.15,
            "close_position": 0.72,
        },
        weight=3.0,
    )

    close_when_near = RewTerm(
        func=mdp.close_gripper_when_near_grasp_center,
        params={
            "distance_threshold": 0.15,
            "close_threshold": 0.10,
        },
        weight=5.0,
    )

    # -------------------------------------------------------------------------
    # 7. Lift shaping: closed gripper + aligned TCP + upward TCP
    # -------------------------------------------------------------------------
    lift_after_close = RewTerm(
        func=mdp.lift_gripper_after_close_reward,
        params={
            "xy_std": 0.08,
            "max_lift": 0.15,
            "close_position": 0.72,
        },
        weight=8.0,
    )

    # -------------------------------------------------------------------------
    # 8. Franka-style lift rewards
    # -------------------------------------------------------------------------
    small_lift_object = RewTerm(
        func=mdp.object_is_lifted,
        params={
            "minimal_height": 0.065,
        },
        weight=10.0,
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={
            "minimal_height": 0.10,
        },
        weight=30.0,
    )

    # -------------------------------------------------------------------------
    # 9. Franka-style object target tracking after lift
    # This activates only after object is lifted above minimal_height.
    # -------------------------------------------------------------------------
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.30,
            "minimal_height": 0.065,
            "command_name": "object_pose",
        },
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.065,
            "command_name": "object_pose",
        },
        weight=5.0,
    )

    # -------------------------------------------------------------------------
    # 10. Keep lifted object close to TCP
    # -------------------------------------------------------------------------
    hold_object_after_lift = RewTerm(
        func=mdp.object_close_to_gripper_when_lifted,
        params={
            "std": 0.12,
            "minimal_height": 0.08,
        },
        weight=10.0,
    )

    # -------------------------------------------------------------------------
    # 11. Smoothness penalty
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