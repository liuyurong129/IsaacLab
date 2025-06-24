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
    return ee_frame.data.target_quat_w[..., 0, :]  # (num_envs, 4)

def ee2_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
) -> torch.Tensor:
    """获取第二个末端执行器在世界坐标系中的四元数姿态"""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[..., 0, :]  # (num_envs, 4)

# def ee_pose_in_robot_root_frame(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot1"),
#     ee_frame_name: str = ".*link6",
# ) -> torch.Tensor:
#     """The pose (position and orientation) of the end-effector in the robot's root frame."""
#     # 获取机械臂
#     robot: Articulation = env.scene[asset_cfg.name]
    
#     # 获取末端执行器的世界坐标位置和姿态
#     ee_body_idx = robot.find_bodies(ee_frame_name)[0]
    
#     # 获取原始数据
#     ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]  
#     ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]  
#     root_pos_w = robot.data.root_state_w[:, :3]
#     root_quat_w = robot.data.root_state_w[:, 3:7]
    
#     # 确保所有张量都是正确的形状
#     batch_size = root_pos_w.shape[0]
    
#     # 处理末端执行器位置 - 确保是 [N, 3]
#     if ee_pos_w.dim() == 1:
#         ee_pos_w = ee_pos_w.unsqueeze(0).expand(batch_size, -1)
#     elif ee_pos_w.dim() == 3:
#         ee_pos_w = ee_pos_w.squeeze(1)
    
#     # 处理末端执行器四元数 - 确保是 [N, 4]
#     if ee_quat_w.dim() == 1:
#         ee_quat_w = ee_quat_w.unsqueeze(0).expand(batch_size, -1)
#     elif ee_quat_w.dim() == 3:
#         ee_quat_w = ee_quat_w.squeeze(1)
    
#     # 最终形状验证
#     assert ee_pos_w.shape == (batch_size, 3), f"ee_pos_w shape: {ee_pos_w.shape}, expected: ({batch_size}, 3)"
#     assert ee_quat_w.shape == (batch_size, 4), f"ee_quat_w shape: {ee_quat_w.shape}, expected: ({batch_size}, 4)"
#     assert root_pos_w.shape == (batch_size, 3), f"root_pos_w shape: {root_pos_w.shape}, expected: ({batch_size}, 3)"
#     assert root_quat_w.shape == (batch_size, 4), f"root_quat_w shape: {root_quat_w.shape}, expected: ({batch_size}, 4)"
    
#     # 将末端执行器姿态转换到机器人根坐标系
#     try:
#         ee_pos_b, ee_quat_b = subtract_frame_transforms(
#             root_pos_w,    # [N, 3]
#             root_quat_w,   # [N, 4]
#             ee_pos_w,      # [N, 3]
#             ee_quat_w      # [N, 4]
#         )
#     except RuntimeError as e:
#         print(f"Error in subtract_frame_transforms:")
#         print(f"root_pos_w: {root_pos_w.shape}")
#         print(f"root_quat_w: {root_quat_w.shape}")
#         print(f"ee_pos_w: {ee_pos_w.shape}")
#         print(f"ee_quat_w: {ee_quat_w.shape}")
#         raise e
    
#     # 拼接位置和姿态
#     ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
    
#     return ee_pose_b


# def ee_relative_pose(
#     env: ManagerBasedRLEnv,
#     robot1_cfg: SceneEntityCfg = SceneEntityCfg("robot1"),
#     robot2_cfg: SceneEntityCfg = SceneEntityCfg("robot2"),
#     ee_frame_name: str = ".*link6",
# ) -> torch.Tensor:
#     """The relative pose between two robot end-effectors."""
#     robot1: Articulation = env.scene[robot1_cfg.name]
#     robot2: Articulation = env.scene[robot2_cfg.name]
    
#     # 获取两个末端执行器的世界坐标位置和姿态
#     ee1_body_idx = robot1.find_bodies(ee_frame_name)[0]
#     ee1_pos_w = robot1.data.body_pos_w[:, ee1_body_idx]
#     ee1_quat_w = robot1.data.body_quat_w[:, ee1_body_idx]
    
#     ee2_body_idx = robot2.find_bodies(ee_frame_name)[0]
#     ee2_pos_w = robot2.data.body_pos_w[:, ee2_body_idx]
#     ee2_quat_w = robot2.data.body_quat_w[:, ee2_body_idx]
    
#     # 确保四元数形状正确
#     if ee1_quat_w.dim() == 3:
#         ee1_quat_w = ee1_quat_w.squeeze(1)
#     if ee2_quat_w.dim() == 3:
#         ee2_quat_w = ee2_quat_w.squeeze(1)
    
#     # 计算相对位置和姿态 (ee2相对于ee1)
#     relative_pos, relative_quat = subtract_frame_transforms(
#         ee1_pos_w, ee1_quat_w, ee2_pos_w, ee2_quat_w
#     )
    
#     # 拼接相对位置和姿态
#     relative_pose = torch.cat([relative_pos, relative_quat], dim=-1)
    
#     return relative_pose


# def object_ee_distance(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot1"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
#     ee_frame_name: str = ".*link6",
# ) -> torch.Tensor:
#     """The distance between object and robot end-effector."""
#     robot: Articulation = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
    
#     # 获取末端执行器位置
#     ee_body_idx = robot.find_bodies(ee_frame_name)[0]
#     ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]  # shape: [N, 3]
    
#     # 获取物体位置
#     object_pos_w = object.data.root_pos_w[:, :3]  # shape: [N, 3]
    
#     # 计算距离
#     distance = torch.norm(ee_pos_w - object_pos_w, dim=-1, keepdim=True)
    
#     return distance