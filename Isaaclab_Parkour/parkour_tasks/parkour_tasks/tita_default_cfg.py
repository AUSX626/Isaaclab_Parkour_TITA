from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from parkour_tasks.tita_asset_cfg import TITA_CFG
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from parkour_isaaclab.terrains.parkour_terrain_importer import ParkourTerrainImporter
from parkour_tasks.extreme_parkour_task.config.tita import agents
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.envs import ViewerCfg
import os, torch
from parkour_isaaclab.actuators.parkour_actuator_cfg import ParkourDCMotorCfg


def quat_from_euler_xyz_tuple(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> tuple:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    convert = torch.stack([qw, qx, qy, qz], dim=-1) * torch.tensor([1.0, 1.0, 1.0, -1.0])
    return tuple(convert.numpy().tolist())


@configclass
class TitaParkourDefaultSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = TITA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    terrain = TerrainImporterCfg(
        class_type=ParkourTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=None,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    def __post_init__(self):
        self.robot.spawn.articulation_props.enabled_self_collisions = False
        self.robot.actuators = {
            "base_legs": ParkourDCMotorCfg(
            joint_names_expr=["joint_left_leg_1", "joint_right_leg_1", "joint_left_leg_2", "joint_right_leg_2", "joint_left_leg_3", "joint_right_leg_3", "joint_left_leg_4", "joint_right_leg_4"],
            effort_limit={
                "joint_.*_leg_1": 60.0,
                "joint_.*_leg_2": 60.0,
                "joint_.*_leg_3": 60.0,
                "joint_.*_leg_4": 15.0,
            },
            saturation_effort={
                "joint_.*_leg_1": 60.0,
                "joint_.*_leg_2": 60.0,
                "joint_.*_leg_3": 60.0,
                "joint_.*_leg_4": 15.0,
            },
            velocity_limit={
                "joint_.*_leg_1": 25.0,
                "joint_.*_leg_2": 25.0,
                "joint_.*_leg_3": 25.0,
                "joint_.*_leg_4": 20.0,
            },
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
            )
        }


CAMERA_CFG = RayCasterCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",
    data_types=["distance_to_camera"],
    offset=RayCasterCameraCfg.OffsetCfg(
        pos=(0.15, 0.0, 0.20),
        # Match the upstream Go2 depth-camera convention/orientation.
        # The previous quaternion had no ROS convention and all rays returned max_distance.
        rot=quat_from_euler_xyz_tuple(*tuple(torch.deg2rad(torch.tensor([180, 70, -90])))),
        convention="ros",
    ),
    depth_clipping_behavior="max",
    pattern_cfg=PinholeCameraPatternCfg(
        focal_length=11.041,
        horizontal_aperture=20.955,
        vertical_aperture=12.240,
        height=60,
        width=106,
    ),
    mesh_prim_paths=["/World/ground"],
    max_distance=2.0,
)

CAMERA_USD_CFG = None

VIEWER = ViewerCfg(
    eye=(-0.0, 2.6, 1.6),
    asset_name="robot",
    origin_type="asset_root",
)
