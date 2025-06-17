
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def box_height_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float = 1.0) -> torch.Tensor:
    """奖励箱子达到目标高度"""
    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_state_w[:, 2]  # Z坐标
    
    # 使用指数函数奖励接近目标高度
    height_diff = torch.abs(current_height - target_height)
    return torch.exp(-2.0 * height_diff)


def box_stability_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励箱子保持稳定（低角速度和线速度）"""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 线速度稳定性
    linear_vel = torch.norm(asset.data.root_state_w[:, 7:10], dim=1)
    linear_stability = torch.exp(-linear_vel)
    
    # 角速度稳定性  
    angular_vel = torch.norm(asset.data.root_state_w[:, 10:13], dim=1)
    angular_stability = torch.exp(-angular_vel)
    
    return 0.5 * (linear_stability + angular_stability)


def dual_arm_coordination_reward(
    env: ManagerBasedRLEnv, 
    arm1_cfg: SceneEntityCfg, 
    arm2_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg
) -> torch.Tensor:
    """奖励两个机械臂的协调性"""
    arm1 = env.scene[arm1_cfg.name]
    arm2 = env.scene[arm2_cfg.name] 
    box = env.scene[box_cfg.name]
    
    # 获取末端执行器位置

    arm1_ee_pos = arm1.data.body_state_w[:, 0, :3]
    arm2_ee_pos = arm2.data.body_state_w[:, 0, :3]
    box_pos = box.data.root_state_w[:, :3]
    
    # 两个末端执行器应该在箱子两侧对称位置
    arm1_to_box = torch.norm(arm1_ee_pos - box_pos, dim=1)
    arm2_to_box = torch.norm(arm2_ee_pos - box_pos, dim=1)
    
    # 距离相似性奖励
    distance_symmetry = torch.exp(-torch.abs(arm1_to_box - arm2_to_box))
    
    # 高度协调奖励（两个末端执行器应该在相似高度）
    height_diff = torch.abs(arm1_ee_pos[:, 2] - arm2_ee_pos[:, 2])
    height_coordination = torch.exp(-2.0 * height_diff)
    
    return 0.5 * (distance_symmetry + height_coordination)


def lifting_progress_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励箱子向上运动的进展"""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 垂直速度奖励（向上为正）
    vertical_velocity = asset.data.root_state_w[:, 9]  # Z方向速度
    
    # 只奖励向上的速度，限制最大奖励
    upward_reward = torch.clamp(vertical_velocity, min=0.0, max=2.0) / 2.0
    
    return upward_reward


def contact_force_penalty(
    env: ManagerBasedRLEnv,
    arm1_cfg: SceneEntityCfg,
    arm2_cfg: SceneEntityCfg,
    max_force: float = 100.0
) -> torch.Tensor:
    """惩罚过大的接触力"""
    # 这需要根据你的contact sensor设置来获取力的信息
    # 假设你有接触力传感器数据
    # contact_forces_arm1 = env.scene.sensors["contact_forces"].data[:, arm1_sensor_idx]
    # contact_forces_arm2 = env.scene.sensors["contact_forces"].data[:, arm2_sensor_idx]
    
    # 临时返回零惩罚，需要根据实际传感器配置修改
    return torch.zeros(env.num_envs, device=env.device)


def orientation_alignment_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励箱子保持水平方向"""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 目标四元数（水平方向）
    target_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    current_quat = asset.data.root_state_w[:, 3:7]
    
    # 计算方向误差
    quat_error = quat_error_magnitude(current_quat, target_quat)
    
    return torch.exp(-2.0 * quat_error)
