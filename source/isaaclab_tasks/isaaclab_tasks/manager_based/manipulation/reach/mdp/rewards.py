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


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    std = max(std, 1e-6) 
    return torch.exp(-0.5 * (distance / std) ** 2)
    # return 1 - torch.tanh(distance / std)

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

# def stay_still_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_pos: torch.Tensor) -> torch.Tensor:
#     """Reward the right_base_link for staying still at a target position."""
#     # 提取 asset
#     asset: RigidObject = env.scene[asset_cfg.name]
    
#     # 当前坐标（world frame）
#     curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # shape: (num_envs, 3)

#     target_pos = torch.tensor(target_pos, device=curr_pos_w.device, dtype=curr_pos_w.dtype)

#     # 与目标位置的差异（L2 距离）
#     distance = torch.norm(curr_pos_w - target_pos, dim=1)
    
#     # 使用一个指数型函数惩罚偏移，鼓励 stay still
#     return torch.exp(-1.0 * distance)  


# def gated_ee_velocity_penalty(
#     env: ManagerBasedRLEnv,command_name: str, asset_cfg: SceneEntityCfg, threshold: float = 0.05) -> torch.Tensor:
#     asset: RigidObject = env.scene[asset_cfg.name]
#     ee_pos = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
#     cmd = env.command_manager.get_command(command_name)
#     des_pos_b = cmd[:, :3]
#     des_pos_w, _ = combine_frame_transforms(
#         asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
#     )
#     dist = torch.norm(ee_pos - des_pos_w, dim=1)

#     joint_vel = asset.data.joint_vel
#     joint_speed = torch.norm(joint_vel, dim=1)

#     penalty = torch.where(dist < threshold, -joint_speed, torch.zeros_like(joint_speed))

#     return penalty
