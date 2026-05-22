# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""
from __future__ import annotations
import torchvision
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera
from isaaclab.assets import Articulation
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent 
from collections.abc import Sequence
import numpy as np 
import cv2
if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


class ExtremeParkourObservations(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors['contact_forces']
        self.ray_sensor: RayCaster = env.scene.sensors['height_scanner']
        self.parkour_event: ParkourEvent =  env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.history_length = cfg.params['history_length']
        self.num_prop = 3 + 2 + 3 + 2 + 1 + 2 + 3 * self.asset.num_joints + len(self.sensor_cfg.body_ids)
        self._obs_history_buffer = torch.zeros(self.num_envs, self.history_length, self.num_prop, device=self.device)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
        self.delta_next_yaw = torch.zeros(self.num_envs, device=self.device)
        self.measured_heights = torch.zeros(self.num_envs, 132, device=self.device)
        self.env = env
        try:
            self.body_id = self.asset.find_bodies('base')[0]
        except ValueError:
            self.body_id = self.asset.find_bodies('base_link')[0]
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._obs_history_buffer[env_ids, :, :] = 0. 

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        history_length: int,
        ) -> torch.Tensor:
        
        terrain_names = self.parkour_event.env_per_terrain_name
        env_idx_tensor = torch.tensor((terrain_names != 'parkour_flat')).to(dtype = torch.bool, device=self.device)
        invert_env_idx_tensor = torch.tensor((terrain_names == 'parkour_flat')).to(dtype = torch.bool, device=self.device)
        roll, pitch, yaw = euler_xyz_from_quat(self.asset.data.root_quat_w)
        imu_obs = torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch)), dim=1).to(self.device)
        if env.common_step_counter % 5 == 0:
            self.delta_yaw = self.parkour_event.target_yaw - wrap_to_pi(yaw)
            self.delta_next_yaw = self.parkour_event.next_target_yaw - wrap_to_pi(yaw)
            self.measured_heights = self._get_heights()
        commands = env.command_manager.get_command('base_velocity')
        obs_buf = torch.cat((
                            self.asset.data.root_ang_vel_b * 0.25,   #[1,3] 0~2
                            imu_obs,    #[1,2] 3~4
                            0*self.delta_yaw[:, None],   #[1,1] 5
                            self.delta_yaw[:, None], #[1,1] 6
                            self.delta_next_yaw[:, None], #[1,1] 7 
                            0*commands[:, 0:2], #[1,2] 8 
                            commands[:, 0:1],  #[1,1] 9
                            env_idx_tensor,
                            invert_env_idx_tensor,
                            self.asset.data.joint_pos - self.asset.data.default_joint_pos,
                            self.asset.data.joint_vel * 0.05 ,
                            env.action_manager.get_term('joint_pos').action_history_buf[:, -1],
                            self._get_contact_fill(),
                            ),dim=-1)
        priv_explicit = self._get_priv_explicit()
        priv_latent = self._get_priv_latent()
        observations = torch.cat([obs_buf, #53
                                  self.measured_heights, #132
                                  priv_explicit, # 9
                                  priv_latent, # 29
                                  self._obs_history_buffer.view(self.num_envs, -1)
                                  ],dim=-1)
        obs_buf[:, 6:8] = 0
        self._obs_history_buffer = torch.where(
            (env.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.history_length, dim=1),
            torch.cat([
                self._obs_history_buffer[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        return observations 

    def _get_contact_fill(
        self,
        ):
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
        contact = torch.norm(contact_forces, dim=-1) > 2.
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = torch.logical_or(contact, last_contacts) 
        return (contact_filt.float()-0.5).to(self.device)
    
    def _get_priv_explicit(
        self,
        ):
        base_lin_vel = self.asset.data.root_lin_vel_b 
        return torch.cat((base_lin_vel * 2.0,
                        0 * base_lin_vel,
                        0 * base_lin_vel), dim=-1).to(self.device)
    
    def _get_priv_latent(
        self,
        ):
        body_mass = self.asset.root_physx_view.get_masses()[:,self.body_id].to(self.device)
        body_com = self.asset.data.com_pos_b[:,self.body_id,:].to(self.device).squeeze(1)
        mass_params_tensor = torch.cat([body_mass, body_com],dim=-1).to(self.device)
        friction_coeffs_tensor = self.asset.root_physx_view.get_material_properties()[:, 0, 0]
        joint_stiffness = self.asset.data.joint_stiffness.to(self.device)
        default_joint_stiffness = self.asset.data.default_joint_stiffness.to(self.device)
        joint_damping = self.asset.data.joint_damping.to(self.device)
        default_joint_damping = self.asset.data.default_joint_damping.to(self.device)
        return torch.cat((
            mass_params_tensor,
            friction_coeffs_tensor.unsqueeze(1).to(self.device),
            (joint_stiffness/ default_joint_stiffness) - 1, 
            (joint_damping/ default_joint_damping) - 1
        ), dim=-1).to(self.device)
    
    def _get_heights(self):
        return torch.clip(self.ray_sensor.data.pos_w[:, 2].unsqueeze(1) - self.ray_sensor.data.ray_hits_w[..., 2] - 0.3, -1, 1).to(self.device)

class image_features(ManagerTermBase):
    
    def __init__(self, cfg: ObservationTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.camera_sensor: RayCasterCamera = env.scene[cfg.params["sensor_cfg"].name]
        self.clipping_range = self.camera_sensor.cfg.max_distance
        resized = cfg.params["resize"]
        self.buffer_len = cfg.params['buffer_len']
        self.debug_vis = cfg.params['debug_vis']
        self.resize_transform = torchvision.transforms.Resize(
                                    (resized[0], resized[1]), 
                                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC).to(env.device)
        self.depth_buffer = torch.zeros(self.num_envs,  
                                        self.buffer_len, 
                                        resized[0], 
                                        resized[1]).to(self.device)
        self._debug_root = None
        self._debug_label = None
        self._debug_photo = None
        self._debug_error_printed = False

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(0, self.num_envs)
        depth_images = self.camera_sensor.data.output["distance_to_camera"].squeeze(-1)[env_ids]
        for depth_image, env_id in zip(depth_images, env_ids):
            processed_image = self._process_depth_image(depth_image)
            self.depth_buffer[env_id] = torch.stack([processed_image]* 2, dim=0)

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        sensor_cfg: SceneEntityCfg,
        resize: tuple(int,int), 
        buffer_len: int,
        debug_vis:bool
        ):
        if env.common_step_counter % 5 == 0:
            depth_images = self.camera_sensor.data.output["distance_to_camera"].squeeze(-1)
            for env_id, depth_image in enumerate(depth_images):
                processed_image = self._process_depth_image(depth_image)
                self.depth_buffer[env_id] = torch.cat([self.depth_buffer[env_id, 1:], 
                                                    processed_image.to(self.device).unsqueeze(0)], dim=0)
        if self.debug_vis:
            # Show the raw distance camera image, not the normalized policy tensor, so the popup
            # directly verifies what the RayCasterCamera sees before crop/resize/normalization.
            depth_images_np = self.camera_sensor.data.output["distance_to_camera"].squeeze(-1).detach().cpu().numpy()
            self._debug_visualize_depth_images(depth_images_np, value_name="raw_m")
        return self.depth_buffer[:, -2].to(env.device)

    def _debug_visualize_depth_images(self, depth_images_np, value_name="value"):
        """Show Student depth observations in a real X11/Tkinter popup."""
        try:
            import tkinter as tk

            if depth_images_np.ndim == 2:
                depth_images_np = depth_images_np[None, ...]
            depth_images_np = depth_images_np[: min(len(depth_images_np), 16)]

            tiles = []
            for img in depth_images_np:
                tiles.append(np.asarray(img, dtype=np.float32))
            if not tiles:
                return

            ncols = min(4, len(tiles))
            nrows = int(np.ceil(len(tiles) / ncols))
            pad_h, pad_w = tiles[0].shape
            while len(tiles) < nrows * ncols:
                tiles.append(np.zeros((pad_h, pad_w), dtype=np.float32))
            rows = [np.hstack(tiles[i:i + ncols]) for i in range(0, len(tiles), ncols)]
            grid = np.vstack(rows)

            finite = grid[np.isfinite(grid)]
            if finite.size == 0:
                finite = np.array([0.0], dtype=np.float32)
            lo = float(np.percentile(finite, 2.0))
            hi = float(np.percentile(finite, 98.0))
            if hi <= lo + 1e-6:
                lo = float(finite.min())
                hi = float(finite.max() + 1e-6)

            norm = np.clip((grid - lo) / (hi - lo), 0.0, 1.0)
            # Simple blue -> cyan/green -> yellow/red map. Nearer normalized depths are blue, farther are red.
            red = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
            green = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
            blue = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)
            rgb = (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)
            rgb = np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1)

            height, width, _ = rgb.shape
            ppm = f"P6 {width} {height} 255\n".encode("ascii") + rgb.tobytes()

            title = (
                "isaaclab_parkour_student_depth | blue=near red=far | "
                f"{value_name} min={float(finite.min()):.3f} max={float(finite.max()):.3f} mean={float(finite.mean()):.3f}"
            )
            if self._debug_root is None:
                self._debug_root = tk.Tk()
                self._debug_root.title(title)
                self._debug_label = tk.Label(self._debug_root)
                self._debug_label.pack()

                def _disable_debug_vis():
                    self.debug_vis = False
                    try:
                        self._debug_root.destroy()
                    except Exception:
                        pass
                    self._debug_root = None

                self._debug_root.protocol("WM_DELETE_WINDOW", _disable_debug_vis)

            self._debug_photo = tk.PhotoImage(data=ppm, format="PPM")
            self._debug_label.configure(image=self._debug_photo)
            self._debug_root.title(title)
            self._debug_root.update_idletasks()
            self._debug_root.update()
        except Exception as exc:
            if not self._debug_error_printed:
                print(f"[image_features] Tk depth visualizer failed: {exc}")
                self._debug_error_printed = True
            self.debug_vis = False

    def _process_depth_image(self, depth_image):
        depth_image = self._crop_depth_image(depth_image)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self._normalize_depth_image(depth_image)
        return depth_image

    def _crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def _normalize_depth_image(self, depth_image):
        depth_image = depth_image  # make similiar to scandot 
        depth_image = (depth_image) / (self.clipping_range)  - 0.5
        return depth_image
    
class obervation_delta_yaw_ok(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,    
        parkour_name: str,
        threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        if env.common_step_counter % 5 == 0:
            parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
            asset: Articulation = env.scene[asset_cfg.name]
            _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
            self.delta_yaw = parkour_event.target_yaw - wrap_to_pi(yaw)
        return self.delta_yaw < threshold
