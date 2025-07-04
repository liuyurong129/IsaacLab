# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        self.actions.joint_pos= mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint",
            "torso_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_elbow_pitch_joint",
            "right_elbow_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            "left_elbow_roll_joint",
            "right_elbow_roll_joint",
        ], scale=0.5, use_default_offset=True
        )
            

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # 清除原先用于运动的奖励
        # self.rewards.track_ang_vel_z_exp.weight = 0.0
        # self.rewards.lin_vel_z_l2.weight = 0.0
        # self.rewards.feet_air_time.weight = 0.0
        # self.rewards.action_rate_l2.weight = -0.005  # 可以保留，限制动作剧烈程度
        # self.rewards.dof_acc_l2.weight = -1.0e-7      # 可以保留
        # self.rewards.dof_torques_l2.weight = -2.0e-6  # 可以保留

        # 添加用于站立的奖励
        self.rewards.base_height_target.weight = 1.0
        self.rewards.base_height_target.params = {"target_height": 0.8}
        self.rewards.base_ori_penalty.weight = 0.5

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
