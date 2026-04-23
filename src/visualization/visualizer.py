"""
Visualisation utilities for primitive skill decomposition results.

Produces one figure per episode with:
  - Row 1: left-arm gripper signal + velocity, colour-coded by primitive
  - Row 2: right-arm gripper signal + velocity, colour-coded by primitive
  - Row 3: stacked primitive-type bar (left arm)
  - Row 4: stacked primitive-type bar (right arm)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from ..decomposition.primitives import PrimitiveType, PRIMITIVE_COLORS
from ..kinematics.fk import fk_trajectory


def _make_colored_line(x, y, colors, ax, lw=1.5):
    """Draw a line where each segment can have a different colour."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, colors=colors[:-1], linewidths=lw)
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min() - 0.05, y.max() + 0.05)


def _frame_colors(labels: np.ndarray) -> list:
    return [PRIMITIVE_COLORS[p] for p in labels]


def visualize_episode(
    episode_result: dict,
    episode_index: int,
    fps: float,
    output_dir: str,
) -> str:
    """
    Draw and save a decomposition figure for one episode.

    Returns the path to the saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Episode {episode_index} — Primitive Skill Decomposition", fontsize=13)

    for row, arm in enumerate(["left", "right"]):
        data   = episode_result[arm]
        g_norm = data["gripper_norm"]
        g_vel  = data["gripper_vel"]
        labels = data["frame_labels"]
        t      = np.arange(len(g_norm)) / fps

        colors = _frame_colors(labels)

        # ---- Gripper position ----
        ax_g = axes[row * 2]
        _make_colored_line(t, g_norm, colors, ax_g)
        ax_g.set_ylabel("Gripper\n(norm)", fontsize=9)
        ax_g.set_title(f"{arm.capitalize()} arm", fontsize=10, loc="left")
        ax_g.axhline(0.35, color="grey", lw=0.7, ls="--", alpha=0.6)
        ax_g.axhline(0.65, color="grey", lw=0.7, ls="--", alpha=0.6)

        # Shade primitive segments
        for seg in data["segments"]:
            x0 = seg.start_frame / fps
            x1 = seg.end_frame   / fps
            c  = PRIMITIVE_COLORS[seg.primitive_type]
            ax_g.axvspan(x0, x1, alpha=0.12, color=c)

        # ---- Gripper velocity ----
        ax_v = axes[row * 2 + 1]
        _make_colored_line(t, g_vel, colors, ax_v)
        ax_v.set_ylabel("Velocity\n(norm/s)", fontsize=9)
        ax_v.axhline(0, color="grey", lw=0.7, alpha=0.6)

    axes[-1].set_xlabel("Time (s)", fontsize=10)

    # ---- Legend ----
    legend_patches = [
        mpatches.Patch(color=PRIMITIVE_COLORS[p], label=p.value.capitalize())
        for p in [PrimitiveType.REACH, PrimitiveType.GRASP,
                  PrimitiveType.MOVE,  PrimitiveType.RELEASE,
                  PrimitiveType.UNKNOWN]
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(output_dir, f"episode_{episode_index:04d}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def visualize_summary(all_episodes: list, output_dir: str) -> str:
    """
    Bar chart: distribution of primitive durations across all episodes.
    """
    os.makedirs(output_dir, exist_ok=True)

    from collections import defaultdict
    durations: dict[str, list] = defaultdict(list)

    for ep in all_episodes:
        for arm in ("left", "right"):
            for seg in ep[arm]["segments"]:
                key = f"{arm}-{seg.primitive_type.value}"
                durations[key].append(seg.duration_seconds)

    if not durations:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Primitive Skill Duration Statistics", fontsize=12)

    for col, arm in enumerate(("left", "right")):
        ax = axes[col]
        for ptype in [PrimitiveType.REACH, PrimitiveType.GRASP,
                      PrimitiveType.MOVE,  PrimitiveType.RELEASE]:
            key = f"{arm}-{ptype.value}"
            if key not in durations:
                continue
            d = durations[key]
            ax.bar(
                ptype.value, np.mean(d),
                yerr=np.std(d), capsize=4,
                color=PRIMITIVE_COLORS[ptype], alpha=0.85,
                label=f"n={len(d)}"
            )
        ax.set_title(f"{arm.capitalize()} arm")
        ax.set_ylabel("Mean duration (s)")
        handles, lbls = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "summary_durations.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def visualize_ee_pose(
    episode_result: dict,
    episode_index: int,
    states: np.ndarray,
    config,
    fps: float,
    output_dir: str,
) -> str:
    """
    Plot end-effector X / Y / Z position over time for both arms,
    colour-coded by primitive label.

    Returns the path to the saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Episode {episode_index} — End-Effector Pose over Time", fontsize=13
    )

    axis_labels = ["X (m)", "Y (m)", "Z (m)"]
    axis_colors = ["#e74c3c", "#2ecc71", "#3498db"]  # red, green, blue

    for col, (arm, arm_indices) in enumerate([
        ("left",  config.left_arm_indices),
        ("right", config.right_arm_indices),
    ]):
        labels = episode_result[arm]["frame_labels"]
        t      = np.arange(len(labels)) / fps
        colors = _frame_colors(labels)

        # Compute EE trajectory via FK
        traj = fk_trajectory(states, arm_indices)   # [T, 3]

        for row, (axis_val, ylabel, ac) in enumerate(
            zip(traj.T, axis_labels, axis_colors)
        ):
            ax = axes[row, col]
            ax.set_facecolor("#f8f9fa")

            # Colour-coded line
            _make_colored_line(t, axis_val, colors, ax, lw=1.8)

            # Thin axis-colour underlay so axis identity is clear
            ax.plot(t, axis_val, color=ac, lw=0.6, alpha=0.35)

            # Shade primitive segments
            for seg in episode_result[arm]["segments"]:
                ax.axvspan(
                    seg.start_frame / fps,
                    seg.end_frame   / fps,
                    alpha=0.10,
                    color=PRIMITIVE_COLORS[seg.primitive_type],
                )

            ax.set_ylabel(ylabel, fontsize=9)
            ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.5)

            if row == 0:
                ax.set_title(f"{arm.capitalize()} arm", fontsize=10)

    for col in range(2):
        axes[-1, col].set_xlabel("Time (s)", fontsize=9)

    # Legend
    legend_patches = [
        mpatches.Patch(color=PRIMITIVE_COLORS[p], label=p.value.capitalize())
        for p in [PrimitiveType.REACH, PrimitiveType.GRASP,
                  PrimitiveType.MOVE,  PrimitiveType.RELEASE,
                  PrimitiveType.UNKNOWN]
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center", ncol=5, fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(output_dir, f"episode_{episode_index:04d}_ee_pose.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
