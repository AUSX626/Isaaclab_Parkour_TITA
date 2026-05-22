# Manual TITA Student depth-camera visualizer.
# This disables ObservationTerm debug_vis and displays the actual RayCasterCamera output directly.
import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Manual visualizer for TITA Student RayCasterCamera depth.")
parser.add_argument("--task", type=str, default="Isaac-Extreme-Parkour-Student-TITA-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=100000)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import tkinter as tk

import isaaclab_tasks  # noqa: F401
import parkour_tasks  # noqa: F401
import parkour_tasks.extreme_parkour_task.config.tita  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg


def colorize_depth(depth: np.ndarray, scale: int = 6) -> bytes:
    depth = np.asarray(depth, dtype=np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=np.float32)
    lo = float(np.percentile(finite, 2.0))
    hi = float(np.percentile(finite, 98.0))
    if hi <= lo + 1e-6:
        lo = float(finite.min())
        hi = float(finite.max() + 1e-6)
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)
    rgb = (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)
    rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    h, w, _ = rgb.shape
    return f"P6 {w} {h} 255\n".encode("ascii") + rgb.tobytes()


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # Avoid the ObservationTerm opening its own Tk window. This script visualizes the sensor directly.
    try:
        env_cfg.observations.depth_camera.depth_cam.params["debug_vis"] = False
    except Exception as exc:
        print(f"[WARN] could not disable observation debug_vis: {exc}", flush=True)

    print("[DEBUG] before gym.make", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[DEBUG] after gym.make", flush=True)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    print("[DEBUG] before env.reset", flush=True)
    obs, info = env.reset()
    print("[DEBUG] after env.reset", flush=True)

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    action_dim = env.unwrapped.action_manager.total_action_dim
    actions = torch.zeros((num_envs, action_dim), device=device)
    camera = env.unwrapped.scene["depth_camera"]

    root = tk.Tk()
    root.title("isaaclab_parkour_student_depth_manual")
    label = tk.Label(root)
    label.pack()
    photo = None

    print(f"[INFO] manual depth visualizer running: task={args_cli.task}, num_envs={num_envs}, action_dim={action_dim}", flush=True)
    print("[INFO] window: isaaclab_parkour_student_depth_manual", flush=True)

    for step in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        env.step(actions)
        raw = camera.data.output["distance_to_camera"].squeeze(-1).detach().cpu().numpy()
        img = raw[0]
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            finite = np.array([0.0], dtype=np.float32)
        ppm = colorize_depth(img)
        photo = tk.PhotoImage(data=ppm, format="PPM")
        label.configure(image=photo)
        title = (
            "isaaclab_parkour_student_depth_manual | ONE ENV raw distance_to_camera | "
            f"min={float(finite.min()):.3f} max={float(finite.max()):.3f} mean={float(finite.mean()):.3f} step={step}"
        )
        root.title(title)
        root.update_idletasks()
        root.update()
        if step % 50 == 0:
            print("[INFO] " + title, flush=True)
        if args_cli.real_time:
            time.sleep(env.unwrapped.step_dt)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
