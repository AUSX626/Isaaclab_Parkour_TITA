"""
TITA wheeled-legged robot flat-ground velocity tracking environment config.

Adapted from G1 parkour config with the following key changes:
- Removed AMP / motion reference (pure PPO)
- Removed feet-specific rewards (feet_air_time, feet_flat_ori, feet_at_plane, feet_close_xy)
- Removed height scanners, volume points (no feet)
- Removed depth camera (Phase 1: proprioception only)
- Added wheel-specific rewards
- Flat terrain only (no rough/stairs/gaps)
- 8 DOF instead of 29 DOF
"""

import copy
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinctlab.tasks.parkour.mdp as mdp
from instinctlab.assets.tita import TITA_CFG, tita_action_scale
from instinctlab.managers import MultiRewardCfg
from instinctlab.terrains import TerrainImporterCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

##
# Flat terrain config (simple flat plane)
##
FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3,
    num_rows=5,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=False,
    sub_terrains={},  # empty = flat plane
)


##
# Scene definition
##
@configclass
class TitaSceneCfg(InteractiveSceneCfg):
    """Scene config for TITA flat ground task."""

    # ground terrain - flat
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = TITA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # contact sensor on all bodies
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# Observations
##
@configclass
class TitaObservationsCfg:
    """Observation config for TITA.

    Phase 1: proprioception only (no depth camera).
    Observations: base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=8,
            flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            history_length=8,
            flatten_history_dim=True,
            params={"command_name": "base_velocity"},
            noise=None,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=8,
            flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.05,
            history_length=8,
            flatten_history_dim=True,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            history_length=8,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group (same as policy but with base_lin_vel and no corruption)."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            history_length=8,
            flatten_history_dim=True,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            history_length=8,
            flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            history_length=8,
            flatten_history_dim=True,
            params={"command_name": "base_velocity"},
            noise=None,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            history_length=8,
            flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            history_length=8,
            flatten_history_dim=True,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            history_length=8,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


##
# Actions
##
@configclass
class TitaActionsCfg:
    """Action config: joint position targets with TITA-specific action scales."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=tita_action_scale,
        use_default_offset=True,
    )


##
# Commands
##
@configclass
class TitaCommandsCfg:
    """Command config: flat ground velocity tracking.

    lin_vel_x: [-1.0, 1.0] m/s (forward and backward)
    lin_vel_y: [0.0, 0.0] m/s (no lateral for wheeled robot)
    ang_vel_z: [-1.0, 1.0] rad/s (turning)
    """

    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        debug_vis=False,
        heading_command=False,
        heading_control_stiffness=2.0,
        rel_standing_envs=0.05,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


##
# Rewards
##
@configclass
class TitaRewards:
    """Reward terms for TITA velocity tracking task."""

    # ===== Task rewards =====
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    heading_error = RewTerm(
        func=mdp.heading_error,
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )
    dont_wait = RewTerm(
        func=mdp.dont_wait,
        weight=-0.5,
        params={"command_name": "base_velocity"},
    )
    is_alive = RewTerm(func=mdp.is_alive, weight=3.0)
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.3,
        params={"command_name": "base_velocity", "offset": 4.0},
    )

    # ===== Regularization rewards =====
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)

    # Joint deviation for hip yaw (keep legs aligned)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_square,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_leg_1"])},
    )

    # Energy penalty
    energy = RewTerm(
        func=mdp.motors_power_square,
        weight=-5e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "normalize_by_stiffness": True,
        },
    )

    # Wheel contact sliding penalty (wheels should roll, not slide)
    wheel_slide = RewTerm(
        func=mdp.contact_slide,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_leg_4"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_leg_4"),
            "threshold": 1.0,
        },
    )

    # ===== Safety rewards =====
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits_by_ratio,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "limit_ratio": 0.8,
        },
    )
    # Penalize contacts on body (not wheels)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*_leg_4).*"),
            "threshold": 1.0,
        },
    )


@configclass
class TitaRewardsCfg(MultiRewardCfg):
    rewards: TitaRewards = TitaRewards()


##
# Terminations
##
@configclass
class TitaTerminationsCfg:
    """Termination terms for TITA."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Terminate if base_link contacts ground
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    # TITA stands ~0.3-0.4m, terminate if falls below 0.15m
    root_height = DoneTerm(func=mdp.root_height_below_env_origin_minimum, params={"minimum_height": 0.15})


##
# Events
##
@configclass
class TitaEventCfg:
    """Event configuration for TITA."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.6),
            "restitution_range": (0.05, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.15, 0.15),
            "velocity_range": (0.0, 0.0),
        },
    )


##
# Curriculum (flat terrain, no levels needed)
##
@configclass
class TitaCurriculumCfg:
    pass


##
# Monitor
##
@configclass
class TitaMonitorCfg:
    pass


##
# Environment configuration
##
@configclass
class TitaFlatEnvCfg(ManagerBasedRLEnvCfg):
    """TITA flat ground velocity tracking environment config."""

    # Scene settings
    scene: TitaSceneCfg = TitaSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: TitaObservationsCfg = TitaObservationsCfg()
    actions: TitaActionsCfg = TitaActionsCfg()
    commands: TitaCommandsCfg = TitaCommandsCfg()
    # MDP settings
    rewards: TitaRewardsCfg = TitaRewardsCfg()
    terminations: TitaTerminationsCfg = TitaTerminationsCfg()
    events: TitaEventCfg = TitaEventCfg()
    curriculum: TitaCurriculumCfg = TitaCurriculumCfg()
    monitors: TitaMonitorCfg = TitaMonitorCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**29
        # update sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class TitaFlatEnvCfg_PLAY(TitaFlatEnvCfg):
    """TITA flat ground environment config for play/evaluation."""

    def __post_init__(self):
        super().__post_init__()
        # smaller scene for play
        self.scene.num_envs = 10
        self.viewer = ViewerCfg(
            eye=[3.0, 0.5, 1.0],
            lookat=[0.0, 0.5, 0.0],
            origin_type="asset_root",
            asset_name="robot",
        )
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        self.terminations.root_height = None
        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
        self.commands.base_velocity.debug_vis = True
