# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##
DUAL_AIRBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/LYR/airbot_play_v3_1_with_G2_UC_calibration/airbot_play_v3_1/airbot_play_v3_1.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,solver_position_iteration_count=8, solver_velocity_iteration_count=0, 
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),


    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
        },
    ),
    actuators={
        "airbot_arm_box_1": ImplicitActuatorCfg(
            joint_names_expr=[".*1"],
            effort_limit=18.0,
            velocity_limit=3.14,
            stiffness=150.0,
            damping=1.5,
        ),
        "airbot_arm_box_2": ImplicitActuatorCfg(
            joint_names_expr=[".*2"],
            effort_limit=18.0,
            velocity_limit=3.14,
            stiffness=150.0,
            damping=1.5,
        ),
        "airbot_arm_box_3": ImplicitActuatorCfg(
            joint_names_expr=[".*3"],
            effort_limit=18.0,
            velocity_limit=3.14,
            stiffness=150.0,
            damping=1.5,
        ),
        "airbot_arm_box_4": ImplicitActuatorCfg(
            joint_names_expr=[".*4"],
            effort_limit=3.0,
            velocity_limit=6.28,
            stiffness=25.0,
            damping=0.5,
        ),
        "airbot_arm_box_5": ImplicitActuatorCfg(
            joint_names_expr=[".*5"],
            effort_limit=3.0,
            velocity_limit=6.28,
            stiffness=25.0,
            damping=1.5,
        ),
        "airbot_arm_box_6": ImplicitActuatorCfg(
            joint_names_expr=[".*6"],
            effort_limit=3.0,
            velocity_limit=6.28,
            stiffness=25.0,
            damping=0.5,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""

# AIRBOT_BOX_HIGH_PD_CFG = AIRBOT_BOX_CFG.copy()
# AIRBOT_BOX_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# AIRBOT_BOX_HIGH_PD_CFG.actuators["airbot_arm_box_1"].stiffness = 400.0
# AIRBOT_BOX_HIGH_PD_CFG.actuators["airbot_arm_box_1"].damping = 80.0
# AIRBOT_BOX_HIGH_PD_CFG.actuators["airbot_arm_box_2"].stiffness = 400.0
# AIRBOT_BOX_HIGH_PD_CFG.actuators["airbot_arm_box_2"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.


# This configuration is useful for task-space control using differential IK.
# """
