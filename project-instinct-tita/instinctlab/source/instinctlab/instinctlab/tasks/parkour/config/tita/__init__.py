# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.parkour.config.tita"


gym.register(
    id="Instinct-Parkour-Target-Tita-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.tita_parkour_target_cfg:TitaParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_cfg:TitaParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Tita-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.tita_parkour_target_cfg:TitaParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_cfg:TitaParkourPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Parkour-Target-Tita-Multi-Cam-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.tita_parkour_target_multi_cam_cfg:TitaParkourMultiCamEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_multi_cam_cfg:TitaParkourMultiCamPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Tita-Multi-Cam-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.tita_parkour_target_multi_cam_cfg:TitaParkourMultiCamEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_multi_cam_cfg:TitaParkourMultiCamPPORunnerCfg",
    },
)