"""Small Tkinter depth image visualizer for IsaacLab debug scripts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class DepthStats:
    min_value: float
    max_value: float
    mean_value: float


class DepthVisualizer:
    """Non-blocking Tkinter viewer for one depth image.

    The viewer intentionally avoids OpenCV because OpenCV highgui is unreliable in
    the current Isaac Sim/conda/VNC environment.
    """

    def __init__(
        self,
        title: str = "tita_depth_view",
        colormap: str = "gray",
        scale: int = 4,
        invert_gray: bool = False,
        key_callback: Callable[[str, bool], None] | None = None,
    ):
        import tkinter as tk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root)
        self.label.pack()
        self.photo = None
        self.title = title
        self.colormap = colormap
        self.scale = scale
        self.invert_gray = invert_gray
        self.key_callback = key_callback
        self.closed = False

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)

    def close(self):
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def _on_key_press(self, event):
        if self.key_callback is not None:
            self.key_callback(str(event.keysym).lower(), True)

    def _on_key_release(self, event):
        if self.key_callback is not None:
            self.key_callback(str(event.keysym).lower(), False)

    def update(self, depth_image, step: int = 0, mode_name: str = "raw_m") -> DepthStats:
        depth = np.asarray(depth_image, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        finite = depth[np.isfinite(depth)]
        if finite.size == 0:
            finite = np.array([0.0], dtype=np.float32)
        stats = DepthStats(float(finite.min()), float(finite.max()), float(finite.mean()))

        lo = float(np.percentile(finite, 2.0))
        hi = float(np.percentile(finite, 98.0))
        if hi <= lo + 1e-6:
            lo = stats.min_value
            hi = stats.max_value + 1e-6
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        rgb = self._to_rgb(norm)
        if self.scale > 1:
            rgb = np.repeat(np.repeat(rgb, self.scale, axis=0), self.scale, axis=1)
        h, w, _ = rgb.shape
        ppm = f"P6 {w} {h} 255\n".encode("ascii") + rgb.tobytes()
        self.photo = self._tk.PhotoImage(data=ppm, format="PPM")
        self.label.configure(image=self.photo)
        self.root.title(
            f"{self.title} | {mode_name} | min={stats.min_value:.3f} "
            f"max={stats.max_value:.3f} mean={stats.mean_value:.3f} step={step}"
        )
        self.root.update_idletasks()
        self.root.update()
        return stats

    def _to_rgb(self, norm: np.ndarray) -> np.ndarray:
        if self.colormap == "turbo":
            red = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
            green = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
            blue = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)
            return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)

        gray = 1.0 - norm if self.invert_gray else norm
        gray_u8 = (gray * 255.0).astype(np.uint8)
        return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
