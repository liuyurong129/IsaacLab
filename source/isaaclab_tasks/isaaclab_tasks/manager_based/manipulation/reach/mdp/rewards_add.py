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
import math
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

def object_goal_linear_reward1(
    env: ManagerBasedRLEnv,
    max_distance: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot1"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
) -> torch.Tensor:
    """
    Linearly scaled reward: closer to goal → higher reward, up to max_distance.
    No reward if object not lifted above minimal_height.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # world frame goal position from relative command
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b
    )

    obj_pos_w = obj.data.root_pos_w[:, :3]
    distance = torch.norm(des_pos_w - obj_pos_w, dim=1)

    # Linearly decay reward: 1 at distance=0, 0 at distance=max_distance or beyond
    linear_reward = 1.0 - (distance / max_distance)
    linear_reward = torch.clamp(linear_reward, min=0.0)

    # Only give reward if object is lifted above threshold
    lifted = (obj.data.root_pos_w[:, 2] > minimal_height)
    return lifted.float() * linear_reward

def object_goal_linear_reward2(
    env: ManagerBasedRLEnv,
    max_distance: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot2"),   
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
) -> torch.Tensor:  
    """
    Linearly scaled reward: closer to goal → higher reward, up to max_distance.
    No reward if object not lifted above minimal_height.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # world frame goal position from relative command
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b
    )

    obj_pos_w = obj.data.root_pos_w[:, :3]
    distance = torch.norm(des_pos_w - obj_pos_w, dim=1)

    # Linearly decay reward: 1 at distance=0, 0 at distance=max_distance or beyond
    linear_reward = 1.0 - (distance / max_distance)
    linear_reward = torch.clamp(linear_reward, min=0.0)

    # Only give reward if object is lifted above threshold
    lifted = (obj.data.root_pos_w[:, 2] > minimal_height)
    return lifted.float() * linear_reward


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
    """Reward the agent for keeping end-effectors mirror-symmetric relative to the object center in z-axis."""
    ee1_frame: FrameTransformer = env.scene[ee1_frame_cfg.name]
    ee2_frame: FrameTransformer = env.scene[ee2_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    ee1_pos = ee1_frame.data.target_pos_w[..., 0, :]
    ee2_pos = ee2_frame.data.target_pos_w[..., 0, :]
    obj_center = object.data.root_pos_w
    # Only consider z-axis symmetry (index 0 for x-axis)
    symmetry_error_z = (ee1_pos[...,0] + ee2_pos[..., 0]) / 2 - obj_center[..., 0]
    symmetry_dist = torch.abs(symmetry_error_z)
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
import math
from isaaclab.utils.math import wrap_to_pi
def quat_to_euler_xyz(quat):
    """
    将四元数转换为欧拉角 (roll, pitch, yaw)
    四元数格式: [w, x, y, z] 或 [x, y, z, w]，这里假设是 [w, x, y, z]
    
    Args:
        quat: [..., 4] 四元数张量
    
    Returns:
        euler: [..., 3] 欧拉角张量 [roll, pitch, yaw]
    """
    # 确保四元数是归一化的
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # 防止数值溢出
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.stack([roll, pitch, yaw], dim=-1)
def ee1_orientation_stability_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.2,
    ee1_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    obj_quat_current = object.data.root_quat_w
    
    ee1_frame: FrameTransformer = env.scene[ee1_frame_cfg.name]
    ee1_quat_current = ee1_frame.data.target_quat_w[..., 0, :]
    
    # 将四元数转换为欧拉角 (roll, pitch, yaw)
    obj_euler = quat_to_euler_xyz(obj_quat_current)  # [batch_size, 3]
    ee1_euler = quat_to_euler_xyz(ee1_quat_current)  # [batch_size, 3]
    
    # 计算当前的欧拉角差异
    current_euler_diff = ee1_euler - obj_euler
    current_euler_diff = wrap_to_pi(current_euler_diff)
    yaw_deviation = torch.abs(current_euler_diff[:, 2])  # yaw is the 3rd component
    # print(f"yaw_deviation: {yaw_deviation}")
    reward = 1.0 - torch.tanh(yaw_deviation / std)
    
    return reward


def ee2_orientation_stability_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.2,
    ee2_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]   
    obj_quat_current = object.data.root_quat_w
    
    ee2_frame: FrameTransformer = env.scene[ee2_frame_cfg.name]
    ee2_quat_current = ee2_frame.data.target_quat_w[..., 0, :]

    obj_euler = quat_to_euler_xyz(obj_quat_current)  # [batch_size, 3]
    ee2_euler = quat_to_euler_xyz(ee2_quat_current)  # [batch_size, 3]
    current_euler_diff = ee2_euler - obj_euler
    current_euler_diff = wrap_to_pi(current_euler_diff)

    yaw_deviation = torch.abs(current_euler_diff[:, 2]-math.pi)  # yaw is the 3rd component
    # print(f"yaw_deviation: {yaw_deviation}")
    reward = 1.0 - torch.tanh(yaw_deviation / std)
    
    return reward


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


def end_effector_velocity_reward(
    env: ManagerBasedRLEnv,
    target_velocity: float = 5,
    std: float = 0.2,
    ee1_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
    ee2_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
) -> torch.Tensor:
    """奖励末端执行器保持适当的运动速度 - 使用环境内置状态管理"""
    ee1_frame: FrameTransformer = env.scene[ee1_frame_cfg.name]
    ee2_frame: FrameTransformer = env.scene[ee2_frame_cfg.name]
    
    # 获取当前位置
    ee1_pos = ee1_frame.data.target_pos_w[..., 0, :]
    ee2_pos = ee2_frame.data.target_pos_w[..., 0, :]
    
    # 使用环境的历史缓存（如果可用）
    if hasattr(env, 'episode_length_buf') and hasattr(env, 'reset_buf'):
        # 检查是否是第一步或刚重置
        is_first_step = (env.episode_length_buf == 0) | env.reset_buf.bool()
        if is_first_step.any():
            return torch.zeros(ee1_pos.shape[0], device=ee1_pos.device)
    
    # 方法1: 使用有限差分近似速度（基于dt）
    # 这种方法不需要存储历史状态，但精度较低
    dt = env.step_dt if hasattr(env, 'step_dt') else 0.02  # 默认50Hz
    
    # 从FrameTransformer或RigidObject获取速度（如果可用）
    if hasattr(ee1_frame.data, 'target_vel_w'):
        ee1_velocity_mag = torch.norm(ee1_frame.data.target_vel_w[..., 0, :], dim=1)
        ee2_velocity_mag = torch.norm(ee2_frame.data.target_vel_w[..., 0, :], dim=1)
    else:
        # 备选方案：估算速度（不太准确但避免状态存储）
        return torch.zeros(ee1_pos.shape[0], device=ee1_pos.device)
    # 奖励接近目标速度的运动
    ee1_reward = 1.0 - torch.tanh(torch.abs(ee1_velocity_mag - target_velocity) / std)
    ee2_reward = 1.0 - torch.tanh(torch.abs(ee2_velocity_mag - target_velocity) / std)
    
    return (ee1_reward + ee2_reward) / 2.0


def action_magnitude_reward(
    env: ManagerBasedRLEnv,
    scale: float = 0.1,
    max_action: float = 10,
) -> torch.Tensor:
    """奖励较大的动作幅度（鼓励快速运动）- 最简单可靠的方法"""
    # 获取当前动作
    actions = env.action_manager.action

    # 计算动作的L2范数并归一化
    action_magnitude = torch.norm(actions, dim=1) / max_action
    action_magnitude = torch.clamp(action_magnitude, 0.0, 1.0)
    
    # 线性奖励：动作幅度越大奖励越高
    return action_magnitude * scale


def joint_velocity_reward(
    env: ManagerBasedRLEnv,
    target_velocity: float = 3.14,
    std: float = 0.5,
    robot1_cfg: SceneEntityCfg = SceneEntityCfg("robot1"),
    robot2_cfg: SceneEntityCfg = SceneEntityCfg("robot2"),
) -> torch.Tensor:
    """奖励关节保持适当的运动速度 - 使用内置关节速度"""
    robot1 = env.scene[robot1_cfg.name]
    robot2 = env.scene[robot2_cfg.name]
    
    # 获取关节速度（这是内置的，不需要手动计算）
    joint_vel1 = robot1.data.joint_vel
    joint_vel2 = robot2.data.joint_vel
    
    # 计算平均关节速度大小
    vel_magnitude1 = torch.mean(torch.abs(joint_vel1), dim=1)
    vel_magnitude2 = torch.mean(torch.abs(joint_vel2), dim=1)
    # 使用tanh核函数奖励接近目标速度的运动
    reward1 = 1.0 - torch.tanh(torch.abs(vel_magnitude1 - target_velocity) / std)
    reward2 = 1.0 - torch.tanh(torch.abs(vel_magnitude2 - target_velocity) / std)
    
    return (reward1 + reward2) / 2.0

def large_motion_reward(
    env: ManagerBasedRLEnv,
    min_velocity_threshold: float = 1.0,  # 最小速度阈值，低于此值不给奖励
    scale: float = 1.0,
    power: float = 2.0,  # 幂次，越大越奖励高速运动
    robot1_cfg: SceneEntityCfg = SceneEntityCfg("robot1"),
    robot2_cfg: SceneEntityCfg = SceneEntityCfg("robot2"),
) -> torch.Tensor:
    """只奖励大幅运动，避免抖动 - 使用阈值和幂函数"""
    robot1 = env.scene[robot1_cfg.name]
    robot2 = env.scene[robot2_cfg.name]
    
    joint_vel1 = robot1.data.joint_vel
    joint_vel2 = robot2.data.joint_vel
    
    # 计算RMS速度（比平均绝对值更能反映整体运动强度）
    vel_rms1 = torch.sqrt(torch.mean(joint_vel1 ** 2, dim=1))
    vel_rms2 = torch.sqrt(torch.mean(joint_vel2 ** 2, dim=1))
    
    # 只有超过阈值的运动才给奖励
    reward1 = torch.where(
        vel_rms1 > min_velocity_threshold,
        torch.pow((vel_rms1 - min_velocity_threshold) / min_velocity_threshold, power),
        torch.zeros_like(vel_rms1)
    )

    reward2 = torch.where(
        vel_rms2 > min_velocity_threshold,
        torch.pow((vel_rms2 - min_velocity_threshold) / min_velocity_threshold, power),
        torch.zeros_like(vel_rms2)
    )
    
    return scale * (reward1 + reward2) / 2.0
