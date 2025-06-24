# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject,Articulation
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot1"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot1"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
) -> torch.Tensor:
    """The pose (position and orientation) of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get object position and orientation in world frame
    object_pos_w = object.data.root_pos_w[:, :3]
    object_quat_w = object.data.root_state_w[:, 3:7]
    
    # Transform object pose to robot's root frame
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], 
        robot.data.root_state_w[:, 3:7], 
        object_pos_w, 
        object_quat_w
    )
    
    # Concatenate position and orientation
    object_pose_b = torch.cat([object_pos_b, object_quat_b], dim=-1)
    return object_pose_b

def ee_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    ee1_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
    ee2_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
) -> torch.Tensor:
    ee1_frame: FrameTransformer = env.scene[ee1_frame_cfg.name]
    ee2_frame: FrameTransformer = env.scene[ee2_frame_cfg.name]
    
    # 获取当前末端执行器的四元数姿态 (num_envs, 4)
    ee1_quat_current = ee1_frame.data.target_quat_w[..., 0, :]
    ee2_quat_current = ee2_frame.data.target_quat_w[..., 0, :]

    return torch.cat([ee1_quat_current, ee2_quat_current], dim=-1)


def ee1_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
) -> torch.Tensor:
    """获取第一个末端执行器在世界坐标系中的四元数姿态"""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # print("first env1 result:", ee_frame.data.target_quat_w[0, 0, :])
    return ee_frame.data.target_quat_w[..., 0, :]  # (num_envs, 4)

def ee2_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
) -> torch.Tensor:
    """获取第二个末端执行器在世界坐标系中的四元数姿态"""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # print("second env result:", ee_frame.data.target_quat_w[0, 0, :])
    return ee_frame.data.target_quat_w[..., 0, :]  # (num_envs, 4)

