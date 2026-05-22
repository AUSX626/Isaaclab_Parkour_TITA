import copy
import os

from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass

import instinctlab.tasks.parkour.mdp as mdp
from instinctlab.assets.ddtrobot import DDTROBOT_TITA_CFG, TITA_LINKS
from instinctlab.sensors import get_link_prim_targets
from instinctlab.tasks.parkour.config.tita_parkour_multi_cam_env_cfg import ROUGH_TERRAINS_CFG, ParkourMultiCamEnvCfg

ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class TitaParkourRoughMultiCamEnvCfg(ParkourMultiCamEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.robot = DDTROBOT_TITA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot.actuators = beyondmimic_Tita_29dof_delayed_actuators
        self.scene.camera1.mesh_prim_paths.extend(get_link_prim_targets(TITA_LINKS))
        self.scene.camera2.mesh_prim_paths.extend(get_link_prim_targets(TITA_LINKS))


@configclass
class TitaParkourRoughMultiCamEnvCfg_PLAY(TitaParkourRoughMultiCamEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        # make a smaller scene for play
        self.scene.num_envs = 10
        self.viewer = ViewerCfg(
            eye=[4.0, 0.75, 1.0],
            lookat=[0.0, 0.75, 0.0],
            origin_type="asset_root",
            asset_name="robot",
        )

        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        self.terminations.root_height = None
        # spawn the robot randomly in the grid (instead of their terrain levels)
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 10

        self.scene.leg_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }


@configclass
class TitaParkourMultiCamEnvCfg(TitaParkourRoughMultiCamEnvCfg):
    def __post_init__(self):
        super().__post_init__()


@configclass
class TitaParkourMultiCamEnvCfg_PLAY(TitaParkourRoughMultiCamEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
