# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import AIRBOT_BOX_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class AirbotBoxEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = AIRBOT_BOX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ee_link"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint.*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "ee_link"
        # self.commands.ee_pose.ranges.roll = (0,0)
        # self.commands.ee_pose.ranges.pitch = (math.pi/2,math.pi/2)
        # self.commands.ee_pose.ranges.yaw = (-math.pi/2, math.pi/2)
        self.commands.ee_pose.ranges.roll = (math.pi/2, math.pi/2)
        self.commands.ee_pose.ranges.pitch = (-math.pi/2, math.pi/2)
        self.commands.ee_pose.ranges.yaw = (math.pi/2, math.pi/2)
        # self.commands.ee_pose.ranges.pos_x = (0.75, 0.8)
        self.commands.ee_pose.ranges.pos_z = (0.2, 0.5)


@configclass
class AirbotBoxEnvCfg_PLAY(AirbotBoxEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
