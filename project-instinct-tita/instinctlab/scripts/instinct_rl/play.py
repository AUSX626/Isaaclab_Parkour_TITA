"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import subprocess
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--exportonnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
# custom play arguments
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--sample", action="store_true", default=False, help="Sample actions instead of using the policy.")
parser.add_argument("--zero_act_until", type=int, default=0, help="Zero actions until this timestep.")
parser.add_argument("--camera_env_id", type=int, default=0, help="Index of the robot environment to track with camera.")
parser.add_argument("--video_subdir", type=str, default="play", help="Subdirectory under videos/ for saving recorded videos.")
parser.add_argument("--difficulty_sweep", action="store_true", default=False,
    help="Put each env on a different difficulty level (0 to num_envs-1), all same terrain type.")
parser.add_argument("--sweep_terrain_type", type=int, default=3,
    help="Which terrain column to sweep difficulty on (0=rough,2=gaps,3=stairs,4=stairs_high,...). Default=3.")
parser.add_argument(
    "--no_terminate", action="store_true", default=False, help="Do not remove termination conditions in simulation."
)
parser.add_argument(
    "--aux_reward",
    action="store_true",
    default=False,
    help="Whether to assign auxiliary rewards to each of the env's reward term.",
)
# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import math
import torch

from instinct_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import instinctlab.tasks  # noqa: F401
from instinctlab.managers.reward_manager import MultiRewardManager
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def main():
    """Play with Instinct-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif not args_cli.no_resume:
        raise RuntimeError(
            f"\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
            f" a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
        )
    else:
        print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    if args_cli.env_cfg:
        env_cfg = load_yaml(os.path.join(log_dir, "params", "env.yaml"))
    if args_cli.agent_cfg:
        agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", args_cli.video_subdir),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{resume_path.split('_')[-1].split('.')[0]}_env{args_cli.camera_env_id}",
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # react to custom play arguments
    if args_cli.no_terminate:
        # NOTE: This is only applicable with shadowing task
        env.unwrapped.termination_manager._term_cfgs = [
            env.unwrapped.termination_manager._term_cfgs[
                env.unwrapped.termination_manager._term_names.index("dataset_exhausted")
            ]
        ]
        env.unwrapped.termination_manager._term_names = ["dataset_exhausted"]

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    if args_cli.sample:
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    if agent_cfg.load_run is not None:
        export_model_dir = os.path.join(log_dir, "exported")
        if args_cli.exportonnx:
            assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
            if not os.path.exists(export_model_dir):
                os.makedirs(export_model_dir)
            obs, _ = env.get_observations()
            ppo_runner.export_as_onnx(obs, export_model_dir)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # 难度全扫模式：reset 之后再强制设置地形，确保不被覆盖
    if args_cli.difficulty_sweep:
        import torch as _torch
        terrain = env.unwrapped.scene.terrain
        num_envs_actual = env.unwrapped.num_envs
        num_rows = terrain.terrain_origins.shape[0]
        col = min(args_cli.sweep_terrain_type, terrain.terrain_origins.shape[1] - 1)
        levels = _torch.arange(num_envs_actual, device=terrain.terrain_levels.device) % num_rows
        terrain.terrain_levels[:] = levels
        terrain.terrain_types[:] = col
        terrain.env_origins[:] = terrain.terrain_origins[levels, col]
        # 强制把机器人 root 位置也移过去
        robot = env.unwrapped.scene["robot"]
        new_origins = terrain.terrain_origins[levels, col].clone()
        # 把 spawn 点沿 x 方向后退 3m，避免直接落入缝隙正中
        new_origins[:, 0] -= 3.0
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += new_origins  # 局部站立高度 + 世界原点
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        # 同步更新 scene.env_origins 防止后续 reset 把机器人拉回去
        env.unwrapped.scene.env_origins[:] = new_origins
        # 打印实际世界坐标方便检查
        print(f"[INFO] Robot world positions (x,y,z):")
        for i, p in enumerate(root_state[:, :3].cpu().tolist()):
            print(f"  env {i} (level {levels[i].item()}): x={p[0]:.2f}, y={p[1]:.2f}, z={p[2]:.2f}")
        print(f"[INFO] Difficulty sweep: terrain_type col={col}, levels={levels.tolist()}")
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if timestep < args_cli.zero_act_until:
                actions[:] = 0.0
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
        timestep += 1

        # 相机跟踪机器人（仅录像模式）
        if args_cli.video:
            try:
                robot_asset = env.unwrapped.scene["robot"]
                cam_idx = min(args_cli.camera_env_id, robot_asset.data.root_pos_w.shape[0] - 1)
                root_pos = robot_asset.data.root_pos_w[cam_idx].cpu()
                root_quat = robot_asset.data.root_quat_w[cam_idx].cpu()
                # 从四元数提取 yaw 角（w,x,y,z 格式）
                w, x, y, z = root_quat.tolist()
                yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                # 相机位置：在机器人正后方 3m、高 1.5m
                cam_eye = (
                    root_pos[0].item() - 3.0 * math.cos(yaw),
                    root_pos[1].item() - 3.0 * math.sin(yaw),
                    root_pos[2].item() + 1.5,
                )
                cam_target = (root_pos[0].item(), root_pos[1].item(), root_pos[2].item() + 0.3)
                env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)
            except Exception as e:
                pass  # 相机更新失败不影响主流程

        # override reward terms if auxiliary reward is enabled
        if args_cli.aux_reward:
            # NOTE: This is only applicable when reward_term has `.reward` to be overridden
            aux_rewards = ppo_runner.alg.compute_auxiliary_reward(infos["observations"])
            for aux_reward_name, aux_reward in aux_rewards.items():
                aux_term_cfg = env.unwrapped.reward_manager.get_term_cfg(aux_reward_name)  # type: ignore
                aux_term_cfg.func.reward[:] = aux_reward * getattr(ppo_runner.alg, aux_reward_name + "_coef", 1.0)

        # exit the loop if video_length is meet
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()

    if args_cli.video:
        subprocess.run(
            [
                "code",
                "-r",
                os.path.join(log_dir, "videos", "play", f"model_{resume_path.split('_')[-1].split('.')[0]}-step-0.mp4"),
            ]
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
