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
import math
from isaaclab.utils.math import wrap_to_pi
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

def collaborative_lift_speed(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    height_weight: float = 2.0,
    velocity_weight: float = 1.0,
    coordination_weight: float = 0.5,
    ee1_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
    ee2_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
) -> torch.Tensor:
    """
    Reward function to encourage fast collaborative lifting of the box by two arms.
    
    This reward combines multiple factors:
    1. Vertical velocity of the object (encourages fast lifting)
    2. Height achievement (encourages lifting the object higher)
    3. Coordination between arms (encourages synchronized movement)
    4. Proximity to object (encourages both arms to stay close to the box)
    
    Args:
        env: The RL environment
        std: Standard deviation for tanh normalization
        height_weight: Weight for height component of reward
        velocity_weight: Weight for velocity component of reward
        coordination_weight: Weight for coordination component of reward
        ee1_frame_cfg: Configuration for first end-effector frame
        ee2_frame_cfg: Configuration for second end-effector frame
        object_cfg: Configuration for the object (box)
    
    Returns:
        Reward tensor for each environment
    """
    # Extract scene entities
    ee1_frame: FrameTransformer = env.scene[ee1_frame_cfg.name]
    ee2_frame: FrameTransformer = env.scene[ee2_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get positions and velocities
    ee1_pos = ee1_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    ee2_pos = ee2_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    obj_pos = object.data.root_pos_w  # (num_envs, 3)
    obj_vel = object.data.root_lin_vel_w  # (num_envs, 3)
    
    # 1. Adaptive vertical velocity reward - speed depends on distance to target
    vertical_velocity = obj_vel[:, 2]  # z-component of velocity
    # Only reward positive (upward) velocities
    upward_velocity = torch.clamp(vertical_velocity, min=0.0)
    
    # Distance-based velocity scaling
    target_height = 0.3  # Adjust this to your target height
    current_height = obj_pos[:, 2]
    height_distance = torch.clamp(target_height - current_height, min=0.0)
    
    # Adaptive velocity multiplier based on distance to target
    # Far from target: high multiplier (encourage fast movement)
    # Close to target: low multiplier (encourage slow, precise movement)
    max_velocity_multiplier = 5.0  # Super high reward when far
    min_velocity_multiplier = 0.2  # Low reward when close
    distance_threshold = 0.2  # Distance range for scaling
    
    # Exponential scaling: far = high reward, close = low reward
    velocity_multiplier = min_velocity_multiplier + (max_velocity_multiplier - min_velocity_multiplier) * torch.exp(-3.0 * (distance_threshold - height_distance) / distance_threshold)
    
    # Clamp to avoid negative multipliers
    velocity_multiplier = torch.clamp(velocity_multiplier, min=min_velocity_multiplier, max=max_velocity_multiplier)
    
    velocity_reward = upward_velocity * velocity_multiplier * velocity_weight
    
    # # 2. Height reward - encourage lifting the object higher
    # # Use a progressive reward that increases with height
    # height_above_ground = obj_pos[:, 2] - 0.0  # Assuming ground is at z=0
    # height_reward = torch.tanh(height_above_ground / std) * height_weight
    
    # # 3. Coordination reward - encourage arms to move in sync
    # # Distance between the two end-effectors (they should maintain reasonable distance)
    # ee_distance = torch.norm(ee1_pos - ee2_pos, dim=1)
    # # Ideal distance could be around the width of the box + some margin
    # ideal_ee_distance = 0.3  # Adjust based on your box size
    # coordination_error = torch.abs(ee_distance - ideal_ee_distance)
    # coordination_reward = (1.0 - torch.tanh(coordination_error / std)) * coordination_weight
    
    # # 4. Proximity rewards - both arms should stay close to the object
    # ee1_obj_distance = torch.norm(ee1_pos - obj_pos, dim=1)
    # ee2_obj_distance = torch.norm(ee2_pos - obj_pos, dim=1)
    
    # # Average proximity reward
    # avg_proximity = (ee1_obj_distance + ee2_obj_distance) / 2.0
    # proximity_reward = 1.0 - torch.tanh(avg_proximity / std)
    
    # # 5. Bonus for simultaneous contact/close proximity
    # # When both arms are very close to the object, give extra reward
    # contact_threshold = 0.05  # Very close distance threshold
    # both_close = (ee1_obj_distance < contact_threshold) & (ee2_obj_distance < contact_threshold)
    # contact_bonus = both_close.float() * 0.5
    
    # # Combine all reward components
    # total_reward = (
    #     velocity_reward +
    #     height_reward +
    #     coordination_reward +
    #     proximity_reward +
    #     contact_bonus
    # )
    
    return velocity_reward


def dual_arm_lift_efficiency(
    env: ManagerBasedRLEnv,
    target_height: float = 0.2,
    max_time_bonus: float = 1.0,
    ee1_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
    ee2_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
) -> torch.Tensor:
    """
    Reward function that gives higher rewards for lifting the object faster.
    
    This reward decreases over time to encourage quick lifting actions.
    
    Args:
        env: The RL environment
        target_height: The height threshold for successful lift
        max_time_bonus: Maximum bonus for quick lifting
        ee1_frame_cfg: Configuration for first end-effector frame
        ee2_frame_cfg: Configuration for second end-effector frame
        object_cfg: Configuration for the object (box)
    
    Returns:
        Reward tensor for each environment
    """
    object: RigidObject = env.scene[object_cfg.name]
    
    # Check if object is lifted above target height
    is_lifted = object.data.root_pos_w[:, 2] > target_height
    
    # Time-based bonus (assuming env has a step counter)
    # You may need to adjust this based on your environment's time tracking
    if hasattr(env, 'episode_length_buf'):
        # Normalize time (0 to 1, where 0 is start of episode)
        time_factor = env.episode_length_buf.float() / env.max_episode_length
        # Bonus decreases with time (more reward for lifting quickly)
        time_bonus = max_time_bonus * (1.0 - time_factor)
    else:
        # Fallback if no time tracking available
        time_bonus = max_time_bonus * 0.5
    
    # Only give time bonus when object is successfully lifted
    efficiency_reward = is_lifted.float() * time_bonus
    
    return efficiency_reward


def progressive_lift_stages(
    env: ManagerBasedRLEnv,
    stage_heights: list = [0.05, 0.1, 0.15, 0.2],
    stage_rewards: list = [0.25, 0.5, 0.75, 1.0],
    object_cfg: SceneEntityCfg = SceneEntityCfg("Box"),
) -> torch.Tensor:
    """
    Progressive reward that gives incremental bonuses as the object reaches different height stages.
    
    This encourages consistent upward progress rather than just final achievement.
    
    Args:
        env: The RL environment
        stage_heights: List of height thresholds for each stage
        stage_rewards: List of rewards for reaching each stage
        object_cfg: Configuration for the object (box)
    
    Returns:
        Reward tensor for each environment
    """
    object: RigidObject = env.scene[object_cfg.name]
    obj_height = object.data.root_pos_w[:, 2]
    
    # Initialize reward tensor
    reward = torch.zeros_like(obj_height)
    
    # Check each stage and accumulate rewards
    for height_threshold, stage_reward in zip(stage_heights, stage_rewards):
        stage_achieved = obj_height > height_threshold
        reward += stage_achieved.float() * stage_reward
    
    return reward

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


def ee1_linear_velocity_reward(
    env: ManagerBasedRLEnv,
    speed_threshold: float = 0.05,
    speed_scale: float = 5.0,
    max_reward: float = 1.0,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame1"),
) -> torch.Tensor:
    """Reward EE1 for fast linear movement based on end-effector velocity."""
    
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # 获取当前end-effector位置
    current_ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    
    # 初始化或获取上一帧位置
    if not hasattr(env, '_prev_ee1_pos'):
        env._prev_ee1_pos = current_ee_pos.clone()
        return torch.zeros(current_ee_pos.shape[0], device=current_ee_pos.device)
    
    # 计算时间步长（通常从环境配置中获取）
    dt = getattr(env, 'dt', 1.0/60.0)  # 默认60Hz
    
    # 计算end-effector线速度
    ee_velocity_vec = (current_ee_pos - env._prev_ee1_pos) / dt
    ee_speed = torch.norm(ee_velocity_vec, dim=1)  # 线速度大小
    
    # 更新上一帧位置
    env._prev_ee1_pos = current_ee_pos.clone()
    
    # 速度奖励计算
    # 超过阈值才给奖励，避免微小抖动
    threshold_mask = (ee_speed > speed_threshold).float()
    speed_reward = torch.tanh(ee_speed * speed_scale)
    
    # final_reward = threshold_mask * speed_reward
    final_reward = threshold_mask * torch.clamp((torch.exp(ee_speed * speed_scale) - 1.0), 0.0, max_reward)

    return torch.clamp(final_reward, 0.0, max_reward)

def ee2_linear_velocity_reward(
    env: ManagerBasedRLEnv,
    speed_threshold: float = 0.05,
    speed_scale: float = 5.0,
    max_reward: float = 1.0,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame2"),
) -> torch.Tensor:
    """Reward EE2 for fast linear movement based on end-effector velocity."""
    
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # 获取当前end-effector位置
    current_ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    
    # 初始化或获取上一帧位置
    if not hasattr(env, '_prev_ee2_pos'):
        env._prev_ee2_pos = current_ee_pos.clone()
        return torch.zeros(current_ee_pos.shape[0], device=current_ee_pos.device)
    
    # 计算时间步长
    dt = getattr(env, 'dt', 1.0/60.0)  # 默认60Hz
    
    # 计算end-effector线速度
    ee_velocity_vec = (current_ee_pos - env._prev_ee2_pos) / dt
    ee_speed = torch.norm(ee_velocity_vec, dim=1)  # 线速度大小
    
    # 更新上一帧位置
    env._prev_ee2_pos = current_ee_pos.clone()
    
    # 速度奖励计算
    threshold_mask = (ee_speed > speed_threshold).float()
    speed_reward = torch.tanh(ee_speed * speed_scale)
    
    # final_reward = threshold_mask * speed_reward
    final_reward = threshold_mask * torch.clamp((torch.exp(ee_speed * speed_scale) - 1.0), 0.0, max_reward)

    return torch.clamp(final_reward, 0.0, max_reward)