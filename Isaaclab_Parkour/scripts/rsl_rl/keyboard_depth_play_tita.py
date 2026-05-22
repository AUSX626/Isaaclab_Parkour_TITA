# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Keyboard-controlled TITA parkour play with a synchronized depth window.

First version: uses the TITA Teacher policy for motion and attaches the TITA
RayCasterCamera only for visualization. This lets us verify camera/terrain
alignment while the robot is actually running in Isaac Sim.

Controls, when the depth window or Isaac window has focus:
    W/S      increase/decrease forward velocity command
    A/D      adjust yaw command for the command visualizer (policy mainly uses forward command)
    Space    stop
    R        reset envs
    G        toggle gray/turbo depth colormap
    Q/Esc    quit
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import weakref
import traceback

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Keyboard TITA play with synchronized depth view.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Extreme-Parkour-Teacher-TITA-Play-v0")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use published pretrained checkpoint if available.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--depth_colormap", choices=["gray", "turbo"], default="gray")
parser.add_argument("--depth_scale", type=int, default=5)
parser.add_argument("--max_vx", type=float, default=0.8)
parser.add_argument("--vx_step", type=float, default=0.1)
parser.add_argument("--yaw_step", type=float, default=0.2)
parser.add_argument("--print_env_cfg", action="store_true", help="Print full environment config before running.")
parser.add_argument("--no_depth_window", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import numpy as np
import omni
import torch
from isaaclab.utils import configclass
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    def get_published_pretrained_checkpoint(*args, **kwargs):
        return None
from isaaclab_tasks.utils import get_checkpoint_path
from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg
from parkour_tasks.extreme_parkour_task.config.tita.parkour_teacher_cfg import (
    TitaParkourTeacherSceneCfg,
    TitaTeacherParkourEnvCfg_PLAY,
)
from parkour_tasks.tita_default_cfg import CAMERA_CFG
from scripts.rsl_rl.depth_visualizer import DepthVisualizer
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper


@configclass
class TitaTeacherDepthSceneCfg(TitaParkourTeacherSceneCfg):
    """Teacher scene plus a depth camera used only for visualization."""

    depth_camera = CAMERA_CFG


@configclass
class TitaKeyboardDepthTeacherEnvCfg(TitaTeacherParkourEnvCfg_PLAY):
    """TITA Teacher play env with an attached depth camera."""

    scene: TitaTeacherDepthSceneCfg = TitaTeacherDepthSceneCfg(num_envs=1, env_spacing=1.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = args_cli.num_envs
        self.scene.depth_camera.update_period = self.sim.dt * self.decimation
        self.episode_length_s = 1000000.0
        self.curriculum = None
        self.events.push_by_setting_velocity = None
        self.commands.base_velocity.resampling_time_range = (1000000.0, 1000000.0)
        self.commands.base_velocity.debug_vis = True
        self.parkours.base_parkour.debug_vis = True


class KeyboardCommand:
    def __init__(self, device: torch.device, num_envs: int):
        self.device = device
        self.num_envs = num_envs
        self.vx = 0.0
        self.yaw = 0.0
        self.quit = False
        self.reset_requested = False
        self.colormap_toggle_requested = False
        self._pressed: set[str] = set()

    def on_key(self, key: str, pressed: bool):
        key = key.lower()
        if pressed:
            self._pressed.add(key)
        else:
            self._pressed.discard(key)
        if pressed and key in {"q", "escape"}:
            self.quit = True
        if pressed and key == "r":
            self.reset_requested = True
        if pressed and key == "g":
            self.colormap_toggle_requested = True
        if pressed and key in {"space", " "}:
            self.vx = 0.0
            self.yaw = 0.0

    def update(self):
        if "w" in self._pressed:
            self.vx = min(args_cli.max_vx, self.vx + args_cli.vx_step)
        if "s" in self._pressed:
            self.vx = max(-args_cli.max_vx, self.vx - args_cli.vx_step)
        if "a" in self._pressed:
            self.yaw = min(1.0, self.yaw + args_cli.yaw_step)
        if "d" in self._pressed:
            self.yaw = max(-1.0, self.yaw - args_cli.yaw_step)
        return self.as_tensor()

    def as_tensor(self):
        command = torch.zeros((self.num_envs, 3), device=self.device)
        command[:, 0] = self.vx
        command[:, 2] = self.yaw
        return command


class OmniKeyboardBridge:
    """Optional keyboard listener for Isaac/Kit window focus."""

    def __init__(self, command: KeyboardCommand):
        self.command = command
        self._sub = None
        try:
            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
            )
            print("[INFO] Isaac/Kit keyboard listener enabled.")
        except Exception as exc:
            print(f"[WARN] Isaac/Kit keyboard listener unavailable: {exc}")

    def _on_keyboard_event(self, event, *args):
        key = str(event.input).split(".")[-1].lower()
        event_type = str(event.type).lower()
        if "press" in event_type or "repeat" in event_type:
            self.command.on_key(key, True)
        elif "release" in event_type:
            self.command.on_key(key, False)
        return True


def _get_checkpoint(agent_cfg: ParkourRslRlOnPolicyRunnerCfg) -> str:
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if getattr(args_cli, "use_pretrained_checkpoint", False):
        checkpoint = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not checkpoint:
            raise RuntimeError("No published pretrained checkpoint for this task.")
        return checkpoint
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _set_command(env: ParkourRslRlVecEnvWrapper, command_tensor: torch.Tensor):
    term = env.unwrapped.command_manager.get_term("base_velocity")
    if hasattr(term, "vel_command_b"):
        term.vel_command_b[:, :] = command_tensor
    if hasattr(term, "heading_target"):
        # Keep heading target close to current heading so the command generator does not fight keyboard yaw too much.
        term.heading_target[:] = env.unwrapped.scene["robot"].data.heading_w + command_tensor[:, 2]


def _policy_step(agent_cfg, policy, estimator, obs, num_prop, num_scan, num_priv_explicit):
    with torch.inference_mode():
        obs[:, num_prop + num_scan : num_prop + num_scan + num_priv_explicit] = estimator.inference(obs[:, :num_prop])
        return policy(obs, hist_encoding=True)


def main():
    if "Student" in args_cli.task:
        print("[WARN] This first keyboard-depth version uses the Teacher policy for motion.")
        print("[WARN] Use task Isaac-Extreme-Parkour-Teacher-TITA-Play-v0 for now.")

    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    checkpoint = _get_checkpoint(agent_cfg)

    env_cfg = TitaKeyboardDepthTeacherEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.terrain.max_init_terrain_level = None

    if args_cli.print_env_cfg:
        print("[INFO] Environment configuration:")
        print_dict(env_cfg.to_dict(), nesting=2)
    else:
        print("[INFO] Environment configuration ready. Use --print_env_cfg to dump details.", flush=True)

    env = ParkourRslRlVecEnvWrapper(ParkourManagerBasedRLEnv(cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device
    ppo_runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=device)
    print(f"[INFO] Loading model checkpoint from: {checkpoint}")
    ppo_runner.load(checkpoint)
    policy = ppo_runner.get_inference_policy(device=device)
    estimator = ppo_runner.get_estimator_inference_policy(device=device)

    estimator_paras = agent_cfg.to_dict()["estimator"]
    num_prop = estimator_paras["num_prop"]
    num_scan = estimator_paras["num_scan"]
    num_priv_explicit = estimator_paras["num_priv_explicit"]

    obs, extras = env.reset()
    command = KeyboardCommand(device, env.unwrapped.num_envs)
    omni_keyboard = OmniKeyboardBridge(command)
    depth_vis = None
    if not args_cli.no_depth_window:
        depth_vis = DepthVisualizer(
            title="tita_depth_view_keyboard",
            colormap=args_cli.depth_colormap,
            scale=args_cli.depth_scale,
            invert_gray=False,
            key_callback=command.on_key,
        )

    camera = env.unwrapped.scene["depth_camera"]
    dt = env.unwrapped.step_dt
    step = 0
    print("[INFO] Keyboard depth play is running.")
    print("[INFO] Controls: W/S forward-back, A/D yaw hint, Space stop, R reset, G colormap, Q quit.")
    print("[INFO] Focus either the Isaac window or the tita_depth_view_keyboard window for keyboard input.")

    while simulation_app.is_running() and not command.quit:
        start_time = time.time()
        command_tensor = command.update()
        _set_command(env, command_tensor)
        # The current Teacher observation uses obs[:, 9] as forward velocity command.
        obs[:, 9] = command_tensor[:, 0]

        actions = _policy_step(agent_cfg, policy, estimator, obs, num_prop, num_scan, num_priv_explicit)
        obs, _, _, extras = env.step(actions)

        if command.reset_requested:
            obs, extras = env.reset()
            command.reset_requested = False

        if depth_vis is not None and not depth_vis.closed:
            if command.colormap_toggle_requested:
                depth_vis.colormap = "turbo" if depth_vis.colormap == "gray" else "gray"
                command.colormap_toggle_requested = False
            raw = camera.data.output["distance_to_camera"].squeeze(-1).detach().cpu().numpy()
            depth_vis.update(raw[0], step=step, mode_name=f"raw_m vx={command.vx:.2f} yaw={command.yaw:.2f}")

        if step % 100 == 0:
            raw = camera.data.output["distance_to_camera"].squeeze(-1).detach().cpu().numpy()[0]
            finite = raw[np.isfinite(raw)]
            print(
                f"[INFO] step={step} vx={command.vx:.2f} yaw={command.yaw:.2f} "
                f"depth min={float(finite.min()):.3f} max={float(finite.max()):.3f} mean={float(finite.mean()):.3f}",
                flush=True,
            )
        step += 1

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("[ERROR] keyboard_depth_play_tita crashed:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
