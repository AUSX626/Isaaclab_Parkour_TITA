"""
DDT TITA Wheeled-Legged Robot asset for Isaac Sim

TITA is a bipedal-wheeled robot by Direct Drive Technology (DDT).
Structure: 2 legs, each with 3 joints (hip_yaw, hip_pitch, knee) + 1 wheel
Total: 8 DOF (6 leg joints + 2 wheels)

Joint name order (from URDF):
[
    'joint_left_leg_1',   # left hip yaw
    'joint_left_leg_2',   # left hip pitch
    'joint_left_leg_3',   # left knee
    'joint_left_leg_4',   # left wheel (continuous)
    'joint_right_leg_1',  # right hip yaw
    'joint_right_leg_2',  # right hip pitch
    'joint_right_leg_3',  # right knee
    'joint_right_leg_4',  # right wheel (continuous)
]
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))


# ========================
# TITA Body Links (for camera mesh rendering)
# ========================
TITA_LINKS = [
    "base_link",
    "left_leg_1",
    "left_leg_2",
    "left_leg_3",
    "left_leg_4",
    "right_leg_1",
    "right_leg_2",
    "right_leg_3",
    "right_leg_4",
]


# ========================
# Action scale for TITA
# For leg joints: 0.25 (position control)
# For wheels: 1.0 (velocity-like control via large effort/stiffness ratio)
# ========================
tita_action_scale = {
    "joint_left_leg_1": 0.25,
    "joint_right_leg_1": 0.25,
    "joint_left_leg_2": 0.25,
    "joint_right_leg_2": 0.25,
    "joint_left_leg_3": 0.25,
    "joint_right_leg_3": 0.25,
    "joint_left_leg_4": 10.0,   # wheels: large scale for velocity-like control
    "joint_right_leg_4": 10.0,
}


# ========================
# Symmetric augmentation mappings (left <-> right swap)
# Joint order: L1, L2, L3, L4, R1, R2, R3, R4
# Swap: R1, R2, R3, R4, L1, L2, L3, L4
# ========================
TITA_symmetric_augmentation_joint_mapping = [4, 5, 6, 7, 0, 1, 2, 3]

# For symmetric augmentation, which joints need sign reversal
# hip_yaw needs reversal (left/right mirror), others keep sign
TITA_symmetric_augmentation_joint_reverse_buf = [
    -1,  # joint_left_leg_1 (hip yaw) -> reversed
    1,   # joint_left_leg_2 (hip pitch) -> same
    1,   # joint_left_leg_3 (knee) -> same
    1,   # joint_left_leg_4 (wheel) -> same
    -1,  # joint_right_leg_1 (hip yaw) -> reversed
    1,   # joint_right_leg_2 (hip pitch) -> same
    1,   # joint_right_leg_3 (knee) -> same
    1,   # joint_right_leg_4 (wheel) -> same
]


# ========================
# Main TITA Configuration
# ========================
TITA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,  # Keep cylinders for wheels
        asset_path=os.path.join(__file_dir__, "resources/tita/urdf/tita_simplified.urdf"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),  # TITA stands ~0.3-0.4m tall
        joint_pos={
            "joint_left_leg_1": 0.0,    # hip yaw
            "joint_right_leg_1": 0.0,
            "joint_left_leg_2": 0.8,    # hip pitch
            "joint_right_leg_2": 0.8,
            "joint_left_leg_3": -1.5,   # knee
            "joint_right_leg_3": -1.5,
            "joint_left_leg_4": 0.0,    # wheel
            "joint_right_leg_4": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        # Leg joints: PD position control with delay
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                "joint_.*_leg_1",   # hip yaw
                "joint_.*_leg_2",   # hip pitch
                "joint_.*_leg_3",   # knee
            ],
            effort_limit={
                "joint_.*_leg_1": 100.0,
                "joint_.*_leg_2": 100.0,
                "joint_.*_leg_3": 100.0,
            },
            velocity_limit=100.0,
            stiffness={
                "joint_.*_leg_1": 40.0,
                "joint_.*_leg_2": 40.0,
                "joint_.*_leg_3": 40.0,
            },
            damping={
                "joint_.*_leg_1": 1.0,
                "joint_.*_leg_2": 1.0,
                "joint_.*_leg_3": 1.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=1,
        ),
        # Wheels: lower stiffness, acts more like velocity control
        # The wheel PD target = action * scale + default_pos
        # With low stiffness and high scale, this effectively becomes velocity control
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=[
                "joint_.*_leg_4",   # wheels
            ],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=10.0,
            damping=0.5,
            armature=0.01,
            min_delay=0,
            max_delay=1,
        ),
    },
)
