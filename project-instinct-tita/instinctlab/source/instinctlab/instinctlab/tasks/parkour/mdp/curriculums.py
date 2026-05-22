from __future__ import annotations

import torch
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def tracking_exp_vel(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int],
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     lin_vel_threshold: tuple = (0.3, 0.6),
#     ang_vel_threshold: tuple = (0.3, 0.5),
# ) -> torch.Tensor:
#     """Curriculum based on the velocity tracking performance (exponential score) of the robot.

#     This term is used to increase the difficulty of the terrain when the robot tracks its commanded velocity well
#     (high score). It decreases the difficulty when the robot tracks its commanded velocity poorly (low score).

#     Args:
#         env: The learning environment.
#         env_ids: The environment ids for which the curriculum should be computed.
#         asset_cfg: The configuration of the robot articulation in the scene.
#         lin_vel_threshold: A tuple specifying the lower and upper threshold for the linear velocity tracking
#             score (exponential kernel).
#             If the score is below the lower threshold (poor tracking), the terrain difficulty is decreased.
#             If the score is above the upper threshold (good tracking), the terrain difficulty is increased.
#         ang_vel_threshold: A tuple specifying the lower and upper threshold for the angular velocity tracking
#             score (exponential kernel).
#             Similar logic applies as lin_vel_threshold.
#     Returns:
#         The mean terrain level for each terrain type.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     terrain: TerrainImporter = env.scene.terrain
#     command = env.command_manager.get_term("base_velocity")
#     tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_ids]
#     tracking_exp_vel_yaw = command.metrics["tracking_exp_vel_yaw"][env_ids]
#     move_up = (tracking_exp_vel_xy > lin_vel_threshold[1]) * (tracking_exp_vel_yaw > ang_vel_threshold[1])
#     move_down = tracking_exp_vel_xy < lin_vel_threshold[0]
#     move_down *= ~move_up
#     # update terrain levels
#     terrain.update_env_origins(env_ids, move_up, move_down)
    
#     # Calculate mean terrain level per terrain type
#     unique_terrain_types = torch.unique(terrain.terrain_types)
#     mean_levels = []
#     for terrain_type in unique_terrain_types:
#         mask = terrain.terrain_types == terrain_type
#         if mask.any():
#             mean_level = torch.mean(terrain.terrain_levels[mask].float())
#             mean_levels.append(mean_level)
#         else:
#             mean_levels.append(torch.tensor(0.0, device=terrain.terrain_levels.device))
    
#     if mean_levels:
#         return torch.mean(torch.stack(mean_levels))
#     else:
#         return torch.mean(terrain.terrain_levels.float())


def tracking_exp_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_threshold: tuple = (0.3, 0.6),
    ang_vel_threshold: tuple = (0.3, 0.5),
) -> dict:
    """Curriculum based on the velocity tracking performance (exponential score) of the robot.

    This term is used to increase the difficulty of the terrain when the robot tracks its commanded velocity well
    (high score). It decreases the difficulty when the robot tracks its commanded velocity poorly (low score).

    Args:
        env: The learning environment.
        env_ids: The environment ids for which the curriculum should be computed.
        asset_cfg: The configuration of the robot articulation in the scene.
        lin_vel_threshold: A tuple specifying the lower and upper threshold for the linear velocity tracking
            score (exponential kernel).
            If the score is below the lower threshold (poor tracking), the terrain difficulty is decreased.
            If the score is above the upper threshold (good tracking), the terrain difficulty is increased.
        ang_vel_threshold: A tuple specifying the lower and upper threshold for the angular velocity tracking
            score (exponential kernel).
            Similar logic applies as lin_vel_threshold.
    Returns:
        A dictionary containing the mean terrain level for each terrain type.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_term("base_velocity")
    tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_ids]
    tracking_exp_vel_yaw = command.metrics["tracking_exp_vel_yaw"][env_ids]
    move_up = (tracking_exp_vel_xy > lin_vel_threshold[1]) * (tracking_exp_vel_yaw > ang_vel_threshold[1])
    move_down = tracking_exp_vel_xy < lin_vel_threshold[0]
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    # Calculate mean terrain level per terrain type
    unique_terrain_types = torch.unique(terrain.terrain_types)
    mean_levels_dict = {}
    
    # Hardcoded terrain type mapping for Tita Parkour (20 columns)
    column_to_terrain_name = {
        0: "perlin_rough",
        1: "perlin_rough_stand",
        2: "square_gaps",
        3: "square_gaps",
        4: "pyramid_stairs",
        5: "pyramid_stairs",
        6: "pyramid_stairs",
        7: "pyramid_stairs_high",
        8: "pyramid_stairs_high",
        9: "pyramid_stairs_inv",
        10: "pyramid_stairs_inv",
        11: "pyramid_stairs_inv",
        12: "pyramid_stairs_inv_high",
        13: "pyramid_stairs_inv_high",
        14: "boxes",
        15: "boxes",
        16: "mesh_boxes",
        17: "mesh_boxes",
        18: "hf_pyramid_slope_inv",
        19: "hf_pyramid_slope_inv"
    }
    
    for terrain_type in unique_terrain_types:
        mask = terrain.terrain_types == terrain_type
        if mask.any():
            mean_level = torch.mean(terrain.terrain_levels[mask].float())
            # Map column index to terrain name using hardcoded mapping
            terrain_key = column_to_terrain_name.get(terrain_type.item(), f"type_{terrain_type.item()}")
            mean_levels_dict[terrain_key] = mean_level
    
    # Also include the overall mean
    mean_levels_dict["overall"] = torch.mean(terrain.terrain_levels.float())
    
    return mean_levels_dict
