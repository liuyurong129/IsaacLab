# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from . import joint_pos_env_cfg
from isaaclab_assets.robots.dual_airbot import  DUAL_AIRBOT_CFG # isort: skip
##
# Pre-defined configs
##
from isaaclab_assets.robots.airbot import AIRBOT_CFG  # isort: skip


@configclass
class HighDynamicDualAirbotEnvCfg(joint_pos_env_cfg.DualArmReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot1 = DUAL_AIRBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/robot1")
        
        # 配置第二个机械臂 (Robot2)
        robot2_cfg = DUAL_AIRBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/robot2")
        robot2_cfg.init_state.pos = (0.47, 0.0, 0.0)  # 设置第二个机械臂的位置
        robot2_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        self.scene.robot2 = robot2_cfg

        # Set actions for the specific robot type (franka)
        self.actions.arm1_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot1",
            joint_names=["joint.*"],
            body_name="link6",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        # Set actions for the specific robot type (franka)
        self.actions.arm2_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot2",
            joint_names=["joint.*"],
            body_name="link6",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        self.commands.object_pose1.body_name = "link6" 
        self.commands.object_pose2.body_name = "link6" 

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame1 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/robot1/link6",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/robot1/link6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )
        self.scene.ee_frame2 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/robot2/link6",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/robot2/link6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )


@configclass
class HighDynamicDualAirbotEnvCfg_PLAY(HighDynamicDualAirbotEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
