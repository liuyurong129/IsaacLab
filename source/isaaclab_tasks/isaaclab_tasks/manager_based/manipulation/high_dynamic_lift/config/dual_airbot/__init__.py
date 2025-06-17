# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments for Airbot
##

##
# Joint Position Control
##

gym.register(
    id="High-Dynamic-Dual-Airbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:HighDynamicDualAirbotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighDynamicDualAirbotPPORunnerCfg",
    },
)

gym.register(
    id="High-Dynamic-Dual-Airbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:HighDynamicDualAirbotEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighDynamicDualAirbotPPORunnerCfg",
    },
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="High-Dynamic-Dual-Airbot-IK-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:HighDynamicDualAirbotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighDynamicDualAirbotPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="High-Dynamic-Dual-Airbot-IK-Abs-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:HighDynamicDualAirbotEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighDynamicDualAirbotPPORunnerCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="High-Dynamic-Dual-Airbot-IK-Rel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:HighDynamicDualAirbotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighDynamicDualAirbotPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="High-Dynamic-Dual-Airbot-IK-Rel-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:HighDynamicDualAirbotEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighDynamicDualAirbotPPORunnerCfg",
    },
    disable_env_checker=True,
)

##
# Operational Space Control
##

gym.register(
    id="High-Dynamic-Dual-Airbot-OSC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:HighDynamicDualAirbotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighDynamicDualAirbotPPORunnerCfg",
    },
)

gym.register(
    id="High-Dynamic-Dual-Airbot-OSC-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:HighDynamicDualAirbotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighDynamicDualAirbotPPORunnerCfg",
    },
)
