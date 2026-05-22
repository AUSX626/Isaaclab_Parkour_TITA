# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.parkour.config.tita"


gym.register(
    id="Instinct-Velocity-Flat-Tita-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.tita_flat_env_cfg:TitaFlatEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:TitaFlatPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Velocity-Flat-Tita-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.tita_flat_env_cfg:TitaFlatEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:TitaFlatPPORunnerCfg",
    },
)
