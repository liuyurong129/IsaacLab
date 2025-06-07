# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("Box")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee1_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel with gradient clipping."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_ee2_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel with gradient clipping."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance1(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot1"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel with stability improvements."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_goal_distance2(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot2"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel with stability improvements."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def arm_mirror_symmetry(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    ee1_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
    ee2_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
) -> torch.Tensor:
    """Reward the agent for keeping end-effectors mirror-symmetric relative to the object center."""
    ee1_frame: FrameTransformer = env.scene[ee1_frame_cfg.name]
    ee2_frame: FrameTransformer = env.scene[ee2_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    ee1_pos = ee1_frame.data.target_pos_w[..., 0, :]
    ee2_pos = ee2_frame.data.target_pos_w[..., 0, :]
    obj_center = object.data.root_pos_w

    # Expected to be close to 0 for symmetry
    symmetry_error = ((ee1_pos + ee2_pos) / 2 - obj_center)
    symmetry_dist = torch.norm(symmetry_error, dim=1)

    # Use tanh kernel for stability
    return 1.0 - torch.tanh(symmetry_dist / std)


# def object_upright_reward(
#     env: ManagerBasedRLEnv,
#     minimal_height: float = 0.05,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
#     up_axis: torch.Tensor = None,
# ) -> torch.Tensor:
#     """Reward the agent for keeping the object upright (z-axis aligned with world z-axis) after lifting."""
#     object: RigidObject = env.scene[object_cfg.name]
    
#     # 设置默认的上轴向量
#     if up_axis is None:
#         up_axis = torch.tensor([0.0, 0.0, 1.0], device=object.data.root_pos_w.device)

#     # 使用连续的lifted mask而非硬阈值
#     height_diff = object.data.root_pos_w[:, 2] - minimal_height
#     lifted_mask = torch.sigmoid(height_diff * 10.0)  # 平滑过渡

#     # 从四元数中获取物体的朝上方向（即局部 z 轴）
#     obj_quat = object.data.root_quat_w  # shape (num_envs, 4)
    
#     # 修复：使用正确的四元数到方向向量转换
#     obj_up = math_utils.quat_rotate(obj_quat, up_axis.unsqueeze(0).expand(obj_quat.shape[0], -1))

#     # 与世界 z 轴的夹角越小越好
#     alignment = torch.sum(obj_up * up_axis, dim=1)  # dot product
    
#     # 裁剪alignment值，避免数值不稳定
#     alignment = torch.clamp(alignment, min=-1.0, max=1.0)

#     # 归一化奖励（越接近 1 越好），乘 lifted_mask 避免抬不起来也给分
#     return lifted_mask * (alignment + 1.0) / 2.0  # 将[-1,1]映射到[0,1]


# def cooperative_lifting_reward(
#     env: ManagerBasedRLEnv,
#     std: float = 0.1,
#     minimal_height: float = 0.05,
#     ee1_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
#     ee2_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
# ) -> torch.Tensor:
#     """新增：奖励两个末端执行器协作抬升物体的行为"""
#     ee1_frame: FrameTransformer = env.scene[ee1_frame_cfg.name]
#     ee2_frame: FrameTransformer = env.scene[ee2_frame_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]

#     ee1_pos = ee1_frame.data.target_pos_w[..., 0, :]
#     ee2_pos = ee2_frame.data.target_pos_w[..., 0, :]
#     obj_pos = object.data.root_pos_w

#     # 计算两个末端执行器是否都在物体附近（协作条件）
#     ee1_dist = torch.norm(ee1_pos - obj_pos, dim=1)
#     ee2_dist = torch.norm(ee2_pos - obj_pos, dim=1)
    
#     # 裁剪距离
#     ee1_dist = torch.clamp(ee1_dist, min=1e-6, max=5.0)
#     ee2_dist = torch.clamp(ee2_dist, min=1e-6, max=5.0)
    
#     # 两个末端执行器都接近时才给奖励
#     ee1_close = 1.0 - torch.tanh(ee1_dist / std)
#     ee2_close = 1.0 - torch.tanh(ee2_dist / std)
#     cooperation_factor = ee1_close * ee2_close  # 乘积确保两者都接近
    
#     # 物体被抬起的奖励
#     height_diff = obj_pos[:, 2] - minimal_height
#     lift_reward = torch.sigmoid(height_diff * 10.0)
    
#     return cooperation_factor * lift_reward


# def balanced_force_reward(
#     env: ManagerBasedRLEnv,
#     std: float = 0.1,
#     ee1_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
#     ee2_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
# ) -> torch.Tensor:
#     """新增：奖励两个末端执行器施加平衡力的行为"""
#     ee1_frame: FrameTransformer = env.scene[ee1_frame_cfg.name]
#     ee2_frame: FrameTransformer = env.scene[ee2_frame_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]

#     ee1_pos = ee1_frame.data.target_pos_w[..., 0, :]
#     ee2_pos = ee2_frame.data.target_pos_w[..., 0, :]
#     obj_pos = object.data.root_pos_w

#     # 计算力的方向（从末端执行器指向物体）
#     force_dir1 = obj_pos - ee1_pos
#     force_dir2 = obj_pos - ee2_pos
    
#     # 归一化力的方向
#     force_dir1_norm = torch.norm(force_dir1, dim=1, keepdim=True)
#     force_dir2_norm = torch.norm(force_dir2, dim=1, keepdim=True)
    
#     # 避免除零
#     force_dir1_norm = torch.clamp(force_dir1_norm, min=1e-6)
#     force_dir2_norm = torch.clamp(force_dir2_norm, min=1e-6)
    
#     force_dir1_unit = force_dir1 / force_dir1_norm
#     force_dir2_unit = force_dir2 / force_dir2_norm
    
#     # 理想情况下，两个力应该相反（平衡）
#     force_balance = torch.sum(force_dir1_unit * (-force_dir2_unit), dim=1)
    
#     # 裁剪值避免数值不稳定
#     force_balance = torch.clamp(force_balance, min=-1.0, max=1.0)
    
#     # 将[-1,1]映射到[0,1]，越接近-1（相反方向）奖励越高
#     return (1.0 - force_balance) / 2.0

def ee1_orientation_stability_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.2,
    ee1_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
) -> torch.Tensor:
    ee1_frame: FrameTransformer = env.scene[ee1_frame_cfg.name]
    ee1_quat_current = ee1_frame.data.target_quat_w[..., 0, :]

    if hasattr(env, 'initial_ee1_quat'):
        ee1_quat_target = env.initial_ee1_quat
    else:
        ee1_quat_target = torch.tensor([-1.0, 0.0, 0.0, 0.0], device=ee1_quat_current.device).unsqueeze(0).expand(ee1_quat_current.shape[0], -1)

    ee1_dot_product = torch.sum(ee1_quat_current * ee1_quat_target, dim=1).clamp(-1.0, 1.0)
    ee1_angle_diff = 2.0 * torch.acos(ee1_dot_product)
    return 1.0 - torch.tanh(ee1_angle_diff / std)

def ee2_orientation_stability_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.2,
    ee2_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
) -> torch.Tensor:
    ee2_frame: FrameTransformer = env.scene[ee2_frame_cfg.name]
    ee2_quat_current = ee2_frame.data.target_quat_w[..., 0, :]

    if hasattr(env, 'initial_ee2_quat'):
        ee2_quat_target = env.initial_ee2_quat
    else:
        ee2_quat_target = torch.tensor([0.0, 0.0, 0.0, -1.0], device=ee2_quat_current.device).unsqueeze(0).expand(ee2_quat_current.shape[0], -1)

    ee2_dot_product = torch.sum(ee2_quat_current * ee2_quat_target, dim=1).clamp(-1.0, 1.0)
    ee2_angle_diff = 2.0 * torch.acos(ee2_dot_product)
    return 1.0 - torch.tanh(ee2_angle_diff / std)

# def ee_orientation_stability_reward(
#     env: ManagerBasedRLEnv,
#     std: float = 0.2,
#     ee1_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
#     ee2_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
# ) -> torch.Tensor:
#     """新增：奖励两个末端执行器保持初始orientation不变的行为"""
#     ee1_frame: FrameTransformer = env.scene[ee1_frame_cfg.name]
#     ee2_frame: FrameTransformer = env.scene[ee2_frame_cfg.name]
    
#     # 获取当前末端执行器的四元数姿态 (num_envs, 4)
#     ee1_quat_current = ee1_frame.data.target_quat_w[..., 0, :]
#     ee2_quat_current = ee2_frame.data.target_quat_w[..., 0, :]
    
#     # print(f"ee1_quat_current: {ee1_quat_current}, ee2_quat_current: {ee2_quat_current}")
#     # 获取初始目标姿态（假设在环境初始化时存储）
#     if hasattr(env, 'initial_ee1_quat'):
#         ee1_quat_target = env.initial_ee1_quat
#     else:
#         # 默认使用单位四元数 [0, 0, 0, 1] (w, x, y, z format)
#         ee1_quat_target = torch.tensor([-1.0, 0.0, 0.0, 0.0], device=ee1_quat_current.device).unsqueeze(0).expand(ee1_quat_current.shape[0], -1)
    
#     if hasattr(env, 'initial_ee2_quat'):
#         ee2_quat_target = env.initial_ee2_quat
#     else:
#         # 默认使用单位四元数
#         ee2_quat_target = torch.tensor([0.0, 0.0, 0.0, -1.0], device=ee2_quat_current.device).unsqueeze(0).expand(ee2_quat_current.shape[0], -1)
    
    
#     # 计算四元数之间的角度差异
#     # 使用四元数点积来计算角度差
#     ee1_dot_product = torch.sum(ee1_quat_current * ee1_quat_target, dim=1)
#     ee2_dot_product = torch.sum(ee2_quat_current * ee2_quat_target, dim=1)

#     # 计算角度差异（弧度）
#     ee1_angle_diff = 2.0 * torch.acos(ee1_dot_product)
#     ee2_angle_diff = 2.0 * torch.acos(ee2_dot_product)

#     # 使用tanh核函数计算奖励，角度差异越小奖励越高
#     ee1_orientation_reward = 1.0 - torch.tanh(ee1_angle_diff / std)
#     ee2_orientation_reward = 1.0 - torch.tanh(ee2_angle_diff / std)
    
#     # 返回两个末端执行器orientation稳定性的平均奖励
#     return (ee1_orientation_reward + ee2_orientation_reward) / 2.0

def target_orientation_stability_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.2,
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
) -> torch.Tensor:
    """新增：奖励目标物体保持特定orientation的行为"""
    object: RigidObject = env.scene[object_cfg.name]
    
    # 获取当前物体的四元数姿态 (num_envs, 4)
    obj_quat_current = object.data.root_quat_w
    # print(f"obj_quat_current: {obj_quat_current}")
    if hasattr(env, 'target_object_quat'):
        obj_quat_target = env.target_object_quat
    else:
        # 默认目标：保持物体直立（单位四元数）[x, y, z, w] format in Isaac
        obj_quat_target = torch.tensor([0.0, 0.0, 0.0, 1.0], device=obj_quat_current.device).unsqueeze(0).expand(obj_quat_current.shape[0], -1)
    
    # 计算四元数之间的角度差异
    # 使用四元数点积来计算相似度
    dot_product = torch.sum(obj_quat_current * obj_quat_target, dim=1)
    
    # 计算角度差异（弧度）
    angle_diff = 2.0 * torch.acos(dot_product)
    # 使用tanh核函数计算奖励，角度差异越小奖励越高
    orientation_reward = 1.0 - torch.tanh(angle_diff / std)
    
    return orientation_reward


# def combined_orientation_stability_reward(
#     env: ManagerBasedRLEnv,
#     std: float = 0.2,
#     ee_weight: float = 0.5,
#     target_weight: float = 0.5,
#     ee1_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
#     ee2_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
# ) -> torch.Tensor:
#     """新增：结合末端执行器和目标物体的orientation稳定性奖励"""
#     # 获取末端执行器orientation稳定性奖励
#     ee_stability = ee_orientation_stability_reward(
#         env, std, ee1_frame_cfg, ee2_frame_cfg
#     )
    
#     # 获取目标物体orientation稳定性奖励
#     target_stability = target_orientation_stability_reward(
#         env, std, object_cfg
#     )
    
#     # 加权组合两个奖励
#     return ee_weight * ee_stability + target_weight * target_stability