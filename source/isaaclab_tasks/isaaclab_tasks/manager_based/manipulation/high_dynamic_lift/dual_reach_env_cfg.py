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

import isaaclab_tasks.manager_based.manipulation.high_dynamic_lift.mdp as mdp
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
        spawn=sim_utils.CuboidCfg(  # 仍然用 CuboidCfg 定义几何和物理属性
            size=(0.15, 0.11, 0.22),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.65361),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.235, 0.0, -0.85),
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
            pos_x=(0.235, 0.235), pos_y=(0.0, 0.0), pos_z=(0.5, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )
    object_pose2 = mdp.UniformPoseCommandCfg(
        asset_name="robot2",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.235, 0.235), pos_y=(0.0, 0.0), pos_z=(0.5, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(math.pi, math.pi)
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
    # robot_action_delay1= EventTerm(
    #     func=mdp.randomize_action_delay,
    #     mode="reset", 
    #     params={
    #         "delay_range": (0.005, 0.03),  
    #         "asset_cfg": SceneEntityCfg("robot1"),     
    #         "filter_alpha": 0.5,     
    #     }
    # )
    # robot_action_delay2= EventTerm(
    #     func=mdp.randomize_action_delay,
    #     mode="reset", 
    #     params={
    #         "delay_range": (0.005, 0.03),  
    #         "asset_cfg": SceneEntityCfg("robot2"),     
    #         "filter_alpha": 0.5,     
    #     }
    # )

@configclass
class DualArmRewardsCfg:
    """Reward terms for the dual-arm MDP."""

    reaching_object1 = RewTerm(func=mdp.object_ee1_distance, params={"std": 0.1}, weight=1.0)
    reaching_object2 = RewTerm(func=mdp.object_ee2_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.11}, weight=2)

    object_goal_tracking1 = RewTerm(
        func=mdp.object_goal_distance1,
        params={"std": 0.3, "minimal_height": 0.11, "command_name": "object_pose1"},
        weight=10,
    )
    object_goal_tracking2 = RewTerm(
        func=mdp.object_goal_distance2,
        params={"std": 0.3, "minimal_height": 0.11, "command_name": "object_pose2"},
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
        weight=0.1,
    )

    action_rate_robot = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-1e-6,
    )
    joint_vel_robot1 = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot1")}
    )
    joint_vel_robot2 = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot2")}
    )

    # action_magnitude = RewTerm(
    #     func=mdp.action_magnitude_reward,
    #     weight=1.0,
    # )

    target_orientation_stability = RewTerm(
        func=mdp.target_orientation_stability_reward,
        params={
            "object_cfg": SceneEntityCfg("Box"),
        },
        weight=1,
    )
    # ee1_orientation_stability = RewTerm(
    #     func=mdp.ee1_orientation_stability_reward,
    #     params={
    #         "ee1_frame_cfg": SceneEntityCfg("ee_frame1"),
    #         "object_cfg": SceneEntityCfg("Box"),
    #     },
    #     weight=0.01,
    # )
    # ee2_orientation_stability = RewTerm(
    #     func=mdp.ee2_orientation_stability_reward,
    #     params={
    #         "ee2_frame_cfg": SceneEntityCfg("ee_frame2"),
    #         "object_cfg": SceneEntityCfg("Box"),
    #     },
    #     weight=0.01,
    # )
    # dual_arm_lift_efficiency = RewTerm(
    #     func=mdp.dual_arm_lift_efficiency,
    #     params={
    #         "ee1_frame_cfg": SceneEntityCfg("ee_frame1"),
    #         "ee2_frame_cfg": SceneEntityCfg("ee_frame2"),
    #         "object_cfg": SceneEntityCfg("Box"),
    #     },
    #     weight=1,
    # )
    # ee1_linear_velocity = RewTerm(
    #     func=mdp.ee1_linear_velocity_reward,
    #     weight=1,
    # )
    # ee2_linear_velocity = RewTerm(
    #     func=mdp.ee2_linear_velocity_reward,
    #     weight=1,
    # )



@configclass
class DualArmTerminationsCfg:
    """Termination terms for the dual-arm MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("Box")}
    )


@configclass
class DualArmCurriculumCfg:
    """Curriculum terms for the dual-arm MDP."""

    # 第一个机械臂的课程学习
    action_rate_robot = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "action_rate_robot", "weight": -1e-1, "num_steps": 10000}
    )
    joint_vel_robot1 = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "joint_vel_robot1", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel_robot2 = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "joint_vel_robot2", "weight": -1e-1, "num_steps": 10000}
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
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625