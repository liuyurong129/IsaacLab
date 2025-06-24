# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg,RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from . import mdp
import math
##
# Scene definition
##


@configclass
class DualArmReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with dual robotic arms."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # robots - 双臂机械臂
    robot1: ArticulationCfg = MISSING  # 第一个机械臂
    robot2: ArticulationCfg = MISSING  # 第二个机械臂
    ee_frame1: FrameTransformerCfg = MISSING
    ee_frame2: FrameTransformerCfg = MISSING  # 末端执行器传感器
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


    Box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg( 
            size=(0.07,0.21,0.22),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                static_friction=5.0,
                dynamic_friction=5.0,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.2129),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.345, 0.0, -0.75),
            rot=(0.0, 0.0, 0.0, 1),
        ),
    )

    Box2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box2",
        spawn=sim_utils.CuboidCfg(  
            size=(0.21,0.22,0.14),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # 设置为运动学体，固定不动
                disable_gravity=True,    # 禁用重力影响
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0),  # 改为黑色 (R=0, G=0, B=0)
                metallic=0.0,     # 可选：设置金属度
                roughness=0.5,    # 可选：设置粗糙度
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.345, 0.0, 0.07),
            rot=(0.0, 0.0, 0.0, 1),
        ),
)



##
# MDP settings
##


@configclass
class DualArmCommandsCfg:
    """Command terms for the dual-arm MDP."""
    object_pose1 = mdp.UniformPoseCommandCfg(
        asset_name="robot1",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0,0), pos_y=(-0.345, -0.345), pos_z=(0.5, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(-math.pi/2, -math.pi/2)
        ),
    )
    object_pose2 = mdp.UniformPoseCommandCfg(
        asset_name="robot2",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0,0), pos_y=(0.345, 0.345), pos_z=(0.5, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(-math.pi/2, -math.pi/2)
        ),
    )


@configclass
class DualArmActionsCfg:
    """Action specifications for the dual-arm MDP."""

    # 第一个机械臂动作
    arm1_action: ActionTerm = MISSING

    # 第二个机械臂动作
    arm2_action: ActionTerm = MISSING

@configclass
class DualArmObservationsCfg:
    """Simplified observation specifications for the dual-arm box lifting MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 机械臂关节状态
        joint_pos_robot1 = ObsTerm(
            func=mdp.joint_pos_rel, 
            params={"asset_cfg": SceneEntityCfg("robot1")},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_robot1 = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("robot1")},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_pos_robot2 = ObsTerm(
            func=mdp.joint_pos_rel, 
            params={"asset_cfg": SceneEntityCfg("robot2")},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_robot2 = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("robot2")},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        # 末端执行器相对姿态
        ee1_relative_pose = ObsTerm(
            func=mdp.ee1_orientation_in_world_frame,
            params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame1"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        ee2_relative_pose = ObsTerm(
            func=mdp.ee2_orientation_in_world_frame,
            params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame2"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        # 箱子状态（核心观察）
        object_position = ObsTerm(func=mdp.object_pose_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose1"})
        last_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class DualArmEventCfg:
    """Configuration for dual-arm events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.235,0.235), "y": (0.0, 0.0), "z": (0.15, 0.15),"w": (0.0, 0.0),"x": (0.0, 0.0), "y": (0,0), "z": (1, 1)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("Box", body_names="Box"),
        },
    )
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("Box", body_names="Box"),
            "static_friction_range": (4.5, 5.5),
            "dynamic_friction_range": (4.5, 5.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    # robot1_joint_stiffness_and_damping = EventTerm(
    #   func=mdp.randomize_actuator_gains,
    #   mode="reset",
    #   params={
    #       "asset_cfg": SceneEntityCfg("robot1", joint_names=".*"),
    #       "stiffness_distribution_params": (0.75, 1.5),
    #       "damping_distribution_params": (0.3, 3.0),
    #       "operation": "scale",
    #       "distribution": "log_uniform",
    #   },
    # )
    # robot2_joint_stiffness_and_damping = EventTerm(
    #   func=mdp.randomize_actuator_gains,
    #   mode="reset",
    #   params={
    #       "asset_cfg": SceneEntityCfg("robot2", joint_names=".*"),
    #       "stiffness_distribution_params": (0.75, 1.5),
    #       "damping_distribution_params": (0.3, 3.0),
    #       "operation": "scale",
    #       "distribution": "log_uniform",
    #   },
    # )

@configclass
class DualArmRewardsCfg:
    """Reward terms for the dual-arm MDP."""

    reaching_object1 = RewTerm(func=mdp.object_ee1_distance, params={"std": 0.1}, weight=1.0)
    reaching_object2 = RewTerm(func=mdp.object_ee2_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.25}, weight=2)

    # object_goal_tracking1_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance1,
    #     params={"std": 0.05, "minimal_height": 0.11, "command_name": "object_pose1"},
    #     weight=1.2,
    # )
    # object_goal_tracking2_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance2,
    #     params={"std": 0.05, "minimal_height": 0.11, "command_name": "object_pose2"},
    #     weight=1.2,
    # )

    object_goal_tracking1 = RewTerm(
        func=mdp.object_goal_distance1,
        params={"std": 0.3, "minimal_height": 0.25, "command_name": "object_pose1"},
        weight=10,
    )
    object_goal_tracking2 = RewTerm(
        func=mdp.object_goal_distance2,
        params={"std": 0.3, "minimal_height": 0.25, "command_name": "object_pose2"},
        weight=10,
    )

    arm_symmetry = RewTerm(
        func=mdp.arm_mirror_symmetry,
        params={
            "std": 0.1,
            "ee1_frame_cfg": SceneEntityCfg("ee_frame1"),
            "ee2_frame_cfg": SceneEntityCfg("ee_frame2"),
            "object_cfg": SceneEntityCfg("Box"),
        },
        weight=0.001,
    )
    # end_effector_velocity = RewTerm(
    #     func=mdp.end_effector_velocity_reward,
    #     params={
    #         "ee1_frame_cfg": SceneEntityCfg("ee_frame1"),
    #         "ee2_frame_cfg": SceneEntityCfg("ee_frame2"),
    #     },
    #     weight=1000,
    # )
    joint_velocity = RewTerm(
        func=mdp.joint_velocity_reward,
        params={
            "robot1_cfg": SceneEntityCfg("robot1"),
            "robot2_cfg": SceneEntityCfg("robot2"),
        },
        weight=2,
    )
    large_motion = RewTerm(
        func=mdp.large_motion_reward,
        params={
            "robot1_cfg": SceneEntityCfg("robot1"),
            "robot2_cfg": SceneEntityCfg("robot2"),
        },
        weight=2,
    )
    action_magnitude = RewTerm(
        func=mdp.action_magnitude_reward,
        weight=0.1,
    )
    # ee1_orientation_stability = RewTerm(
    #     func=mdp.ee1_orientation_stability_reward,
    #     params={
    #         "ee1_frame_cfg": SceneEntityCfg("ee_frame1"),
    #         "object_cfg": SceneEntityCfg("Box"),
    #     },
    #     weight=0.001,
    # )
    # ee2_orientation_stability = RewTerm(
    #     func=mdp.ee2_orientation_stability_reward,
    #     params={
    #         "ee2_frame_cfg": SceneEntityCfg("ee_frame2"),
    #         "object_cfg": SceneEntityCfg("Box"),
    #     },
    #     weight=0.0001,
    # )
    
    target_orientation_stability = RewTerm(
        func=mdp.target_orientation_stability_reward,
        params={
            "object_cfg": SceneEntityCfg("Box"),
        },
        weight=0.001,
    )
    
    action_rate_robot = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.0001,
    )
    joint_vel_robot1 = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot1")}
    )
    joint_vel_robot2 = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot2")}
    )

@configclass
class DualArmTerminationsCfg:
    """Termination terms for the dual-arm MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class DualArmCurriculumCfg:
    """Curriculum terms for the dual-arm MDP."""

    # 第一个机械臂的课程学习
    action_rate_robot = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "action_rate_robot", "weight": -0.005, "num_steps": 4500}
    )
    joint_vel_robot1 = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "joint_vel_robot1", "weight": -0.001, "num_steps": 4500}
    )

    joint_vel_robot2 = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "joint_vel_robot2", "weight": -0.001, "num_steps": 4500}
    )


@configclass
class DualArmReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the dual-arm reach end-effector pose tracking environment."""

    # Scene settings
    scene: DualArmReachSceneCfg = DualArmReachSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: DualArmObservationsCfg = DualArmObservationsCfg()
    actions: DualArmActionsCfg = DualArmActionsCfg()
    commands: DualArmCommandsCfg = DualArmCommandsCfg()
    
    # MDP settings
    rewards: DualArmRewardsCfg = DualArmRewardsCfg()
    terminations: DualArmTerminationsCfg = DualArmTerminationsCfg()
    events: DualArmEventCfg = DualArmEventCfg()
    curriculum: DualArmCurriculumCfg = DualArmCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 40 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
