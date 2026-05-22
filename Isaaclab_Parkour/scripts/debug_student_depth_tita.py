# Quick visual-only runner for TITA Student depth camera.
# It creates the Student Play environment and steps zero actions so ObservationTerm debug_vis can show depth.
import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize TITA Student depth camera without loading an RL checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-Extreme-Parkour-Student-TITA-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100000)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--real-time", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import parkour_tasks  # noqa: F401
import parkour_tasks.extreme_parkour_task.config.tita  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    obs, info = env.reset()
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    try:
        action_dim = env.unwrapped.action_manager.total_action_dim
    except Exception:
        action_dim = env.action_space.shape[-1]
    actions = torch.zeros((num_envs, action_dim), device=device)

    print(f"[INFO] Running {args_cli.task} for depth visualization: num_envs={num_envs}, action_dim={action_dim}")
    print("[INFO] Look for the Tk window named: isaaclab_parkour_student_depth")

    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        env.step(actions)
        if args_cli.real_time:
            time.sleep(env.unwrapped.step_dt)
        if step % 100 == 0:
            print(f"[INFO] visual step={step}")
        step += 1

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
