"""
Video exporter for primitive skill decomposition.

Produces per-episode and per-segment MP4 videos showing:
  - 2D arm stick figure (planar FK from joints 1-4)
  - Gripper aperture bar
  - Scrolling timeline coloured by primitive type
  - Large colour-coded primitive label in the header

Output structure:
  <output_dir>/videos/
    episode_0000/
      overview.mp4
      left_reach_00000-00058.mp4
      left_grasp_00058-00077.mp4
      ...
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from tqdm import tqdm

from ..decomposition.primitives import PrimitiveType, PRIMITIVE_COLORS

# Approximate ALOHA link lengths (m); used for 2D stick-figure
_LINK_LENGTHS = [0.28, 0.22, 0.16, 0.10]

RENDER_FPS = 15   # output video fps (downsampled from 50)
W, H = 960, 540   # output resolution


# ---------------------------------------------------------------------------
# 2D forward kinematics (simplified planar projection)
# ---------------------------------------------------------------------------

def _fk_2d(joints6: np.ndarray) -> np.ndarray:
    """
    2D planar FK from a 6-DOF arm joint vector.
    Returns an (N+1, 2) array of (x, y) points (base → end-effector).
    """
    pts = [(0.0, 0.0)]
    angle = np.pi / 2          # arm points upward from base
    for q, l in zip(joints6[1:5], _LINK_LENGTHS):
        angle += q * 0.45      # dampen joint range for cleaner visuals
        pts.append((pts[-1][0] + l * np.cos(angle),
                    pts[-1][1] + l * np.sin(angle)))
    return np.array(pts)


# ---------------------------------------------------------------------------
# Figure factory  (creates once; reused for all frames)
# ---------------------------------------------------------------------------

class _EpisodeFigure:
    """Persistent matplotlib figure that is updated per frame."""

    def __init__(self, ep_result, ep_idx, states, cfg):
        self.ep_result = ep_result
        self.ep_idx    = ep_idx
        self.states    = states
        self.cfg       = cfg
        self.T         = states.shape[0]
        self.t_arr     = np.arange(self.T) / cfg.fps

        self._build()

    def _build(self):
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="#1a1a2e")
        gs  = gridspec.GridSpec(
            3, 2, figure=fig,
            height_ratios=[0.10, 0.57, 0.33],
            hspace=0.06, wspace=0.04,
            left=0.04, right=0.96, top=0.97, bottom=0.03,
        )

        # ── Header ─────────────────────────────────────────────────────
        ax_hdr = fig.add_subplot(gs[0, :])
        ax_hdr.set_xlim(0, 1); ax_hdr.set_ylim(0, 1); ax_hdr.axis("off")
        self.hdr_text = ax_hdr.text(
            0.5, 0.5, "", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white",
            transform=ax_hdr.transAxes,
        )
        self.ax_hdr = ax_hdr

        # ── Arm panels ─────────────────────────────────────────────────
        self.arm_artists = {}
        for col, arm in enumerate(("left", "right")):
            ax = fig.add_subplot(gs[1, col])
            ax.set_facecolor("#16213e")
            ax.set_xlim(-0.70, 0.70)
            ax.set_ylim(-0.18, 0.90)
            ax.set_aspect("equal")
            ax.axis("off")

            # draw static base square
            ax.plot(0, 0, "s", color="#aaaaaa", ms=10, zorder=4)

            arm_line, = ax.plot([], [], "-o", lw=3, ms=7,
                                markerfacecolor="white", zorder=3)
            jaw_l,    = ax.plot([], [], lw=2.5, zorder=4)
            jaw_r,    = ax.plot([], [], lw=2.5, zorder=4)

            # Gripper bar track
            ax.add_patch(plt.Rectangle(
                (-0.40, -0.10), 0.80, 0.028,
                color="#2d2d2d", zorder=2, transform=ax.transData
            ))
            g_bar = plt.Rectangle(
                (-0.40, -0.10), 0.0, 0.028,
                color="white", alpha=0.9, zorder=3, transform=ax.transData
            )
            ax.add_patch(g_bar)

            g_text = ax.text(-0.40, -0.14, "", color="#cccccc",
                             fontsize=8, va="top")
            prim_text = ax.text(
                0, 0.87, arm.upper(), ha="center", va="top",
                fontsize=10, fontweight="bold", color="white",
            )

            self.arm_artists[arm] = dict(
                ax=ax, arm_line=arm_line, jaw_l=jaw_l, jaw_r=jaw_r,
                g_bar=g_bar, g_text=g_text, prim_text=prim_text,
            )

        # ── Timeline ───────────────────────────────────────────────────
        ax_tl = fig.add_subplot(gs[2, :])
        ax_tl.set_facecolor("#16213e")
        ax_tl.set_xlim(self.t_arr[0], self.t_arr[-1])
        ax_tl.set_ylim(-0.25, 2.50)
        ax_tl.axis("off")

        # Pre-draw segment colour bands (static)
        for arm_row, arm in enumerate(("left", "right")):
            labels = self.ep_result[arm]["frame_labels"]
            row_y  = arm_row * 1.15
            i = 0
            while i < self.T:
                p = labels[i]
                j = i + 1
                while j < self.T and labels[j] == p:
                    j += 1
                t0 = self.t_arr[i]
                t1 = self.t_arr[min(j, self.T - 1)]
                ax_tl.axhspan(row_y - 0.08, row_y + 0.55,
                              xmin=(t0 - self.t_arr[0]) / (self.t_arr[-1] - self.t_arr[0]),
                              xmax=(t1 - self.t_arr[0]) / (self.t_arr[-1] - self.t_arr[0]),
                              color=PRIMITIVE_COLORS[p], alpha=0.30)
                i = j

            # Gripper curve (static)
            g = self.ep_result[arm]["gripper_norm"]
            ax_tl.plot(self.t_arr, g * 0.50 + row_y,
                       color="#ffffff", lw=1.0, alpha=0.75)
            ax_tl.text(self.t_arr[0] - 0.05, row_y + 0.25,
                       arm[0].upper(), ha="right", va="center",
                       color="#cccccc", fontsize=9, fontweight="bold")

        ax_tl.set_xlabel("time (s)", color="#666666", fontsize=8)

        # Dynamic: cursor and segment highlight
        self.tl_cursor    = ax_tl.axvline(0, color="white", lw=2, zorder=10)
        self.tl_highlight = mpatches.Rectangle(
            (0, -0.25), 0, 2.75,
            color="yellow", alpha=0.0, zorder=5
        )
        ax_tl.add_patch(self.tl_highlight)

        self.fig = fig
        self.ax_tl = ax_tl

    # ------------------------------------------------------------------

    def render(self, fi: int,
               focus_arm: str = None,
               focus_seg=None) -> np.ndarray:
        """Update all artists to frame fi and return an RGB uint8 array."""
        fi = min(fi, self.T - 1)
        t  = self.t_arr[fi]

        left_labels  = self.ep_result["left"]["frame_labels"]
        right_labels = self.ep_result["right"]["frame_labels"]

        # Determine header primitive
        if focus_seg is not None:
            hdr_prim = focus_seg.primitive_type
            hdr_arm  = focus_seg.arm
        else:
            hdr_prim = left_labels[fi]
            hdr_arm  = "left"

        # ── Header ──
        hdr_color = PRIMITIVE_COLORS[hdr_prim]
        self.ax_hdr.set_facecolor(hdr_color)
        self.hdr_text.set_text(
            f"Episode {self.ep_idx}  │  t = {t:.2f}s / {self.t_arr[-1]:.2f}s"
            f"  │  {hdr_arm.upper()} ARM  ●  {hdr_prim.value.upper()}"
        )

        # ── Arms ──
        for arm, joint_off in (("left", 0), ("right", 7)):
            art    = self.arm_artists[arm]
            labels = left_labels if arm == "left" else right_labels
            g_arr  = self.ep_result[arm]["gripper_norm"]

            prim    = labels[fi]
            color   = PRIMITIVE_COLORS[prim]
            g_val   = float(g_arr[fi])
            joints6 = self.states[fi, joint_off:joint_off + 6]
            pts     = _fk_2d(joints6)

            # Arm line
            art["arm_line"].set_data(pts[:, 0], pts[:, 1])
            art["arm_line"].set_color(color)

            # Gripper jaws at end-effector
            ee = pts[-1]
            prev = pts[-2]
            angle = np.arctan2(ee[1] - prev[1], ee[0] - prev[0])
            perp  = angle + np.pi / 2
            spread  = g_val * 0.07
            jaw_len = 0.06
            for side, artist in ((+1, art["jaw_l"]), (-1, art["jaw_r"])):
                ox = ee[0] + side * spread * np.cos(perp)
                oy = ee[1] + side * spread * np.sin(perp)
                art["jaw_l"].set_color(color) if side == +1 else art["jaw_r"].set_color(color)
                artist.set_data(
                    [ee[0], ox + jaw_len * np.cos(angle)],
                    [ee[1], oy + jaw_len * np.sin(angle)],
                )

            # Gripper bar
            art["g_bar"].set_width(0.80 * g_val)
            art["g_bar"].set_facecolor(color)
            art["g_text"].set_text(f"Gripper {g_val:.2f}  ({prim.value})")

            # Primitive label
            art["prim_text"].set_text(f"{arm.upper()}  {prim.value.upper()}")
            art["prim_text"].set_color(color)

        # ── Timeline cursor ──
        self.tl_cursor.set_xdata([t, t])

        # ── Focus segment highlight ──
        if focus_seg is not None:
            t0 = focus_seg.start_frame / self.cfg.fps
            t1 = focus_seg.end_frame   / self.cfg.fps
            self.tl_highlight.set_x(t0)
            self.tl_highlight.set_width(t1 - t0)
            self.tl_highlight.set_alpha(0.20)
        else:
            self.tl_highlight.set_alpha(0.0)

        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf.reshape(H, W, 4)[..., :3].copy()

    def close(self):
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class VideoExporter:

    def __init__(self, config, output_dir: str):
        self.cfg        = config
        self.output_dir = os.path.join(output_dir, "videos")

    def export_episode(self, ep_result: dict, ep_idx: int,
                       states: np.ndarray) -> list[str]:
        """
        Render overview + per-segment videos for one episode.
        Returns list of written file paths.
        """
        ep_dir = os.path.join(self.output_dir, f"episode_{ep_idx:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        stride = max(1, self.cfg.fps // RENDER_FPS)
        T      = states.shape[0]

        # Count total frames to render for progress bar
        all_segs = (
            ep_result["left"]["segments"] + ep_result["right"]["segments"]
        )
        total_frames = (
            len(range(0, T, stride))
            + sum(len(range(s.start_frame, s.end_frame, stride)) for s in all_segs)
        )

        efig   = _EpisodeFigure(ep_result, ep_idx, states, self.cfg)
        paths  = []

        with tqdm(total=total_frames, desc=f"  Ep {ep_idx} videos",
                  unit="frame", leave=False) as pbar:

            # ── Overview ──────────────────────────────────────────────
            ov_path = os.path.join(ep_dir, "overview.mp4")
            writer  = _make_writer(ov_path)
            for fi in range(0, T, stride):
                frame = efig.render(fi)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                pbar.update(1)
            writer.release()
            paths.append(ov_path)

            # ── Per-segment ───────────────────────────────────────────
            for arm in ("left", "right"):
                for seg in ep_result[arm]["segments"]:
                    fname = (
                        f"{arm}_{seg.primitive_type.value}"
                        f"_{seg.start_frame:05d}-{seg.end_frame:05d}.mp4"
                    )
                    seg_path = os.path.join(ep_dir, fname)
                    writer   = _make_writer(seg_path)
                    for fi in range(seg.start_frame, seg.end_frame, stride):
                        frame = efig.render(fi, focus_arm=arm, focus_seg=seg)
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        pbar.update(1)
                    writer.release()
                    paths.append(seg_path)

        efig.close()
        return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_writer(path: str) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, RENDER_FPS, (W, H))
