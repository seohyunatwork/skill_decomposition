"""
Visualisation utilities for primitive skill decomposition results.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from ..decomposition.primitives import PrimitiveType, PRIMITIVE_COLORS
from ..kinematics.fk import fk_trajectory


def _make_colored_line(x, y, colors, ax, lw=1.5):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, colors=colors[:-1], linewidths=lw)
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min() - 0.05, y.max() + 0.05)


def _frame_colors(labels):
    return [PRIMITIVE_COLORS[p] for p in labels]


def _shade_segments(ax, segments, fps):
    for seg in segments:
        ax.axvspan(
            seg.start_frame / fps,
            seg.end_frame   / fps,
            alpha=0.12,
            color=PRIMITIVE_COLORS[seg.primitive_type],
        )


# ---------------------------------------------------------------------------
# Combined per-episode figure
# ---------------------------------------------------------------------------

def visualize_episode(
    episode_result: dict,
    episode_index: int,
    states: np.ndarray,
    config,
    fps: float,
    output_dir: str,
) -> str:
    """
    One figure per episode, two columns (left arm | right arm).

    Rows (shared time axis):
      0 - EE X (m)
      1 - EE Y (m)
      2 - EE Z (m)
      3 - Gripper (normalised)
      4 - Gripper velocity (norm/s)
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(
        5, 2, figsize=(14, 12),
        sharex=True, sharey="row",
    )
    fig.suptitle(
        f"Episode {episode_index} — Primitive Skill Decomposition",
        fontsize=13, y=0.995,
    )

    row_labels = ["EE X (m)", "EE Y (m)", "EE Z (m)",
                  "Gripper\n(norm)", "Gripper velocity\n(norm/s)"]
    ee_colors  = ["#e74c3c", "#2ecc71", "#3498db"]

    for col, (arm, arm_indices) in enumerate([
        ("left",  config.left_arm_indices),
        ("right", config.right_arm_indices),
    ]):
        data    = episode_result[arm]
        labels  = data["frame_labels"]
        g_norm  = data["gripper_norm"]
        g_vel   = data["gripper_vel"]
        t       = np.arange(len(labels)) / fps
        colors  = _frame_colors(labels)
        segs    = data["segments"]

        traj = fk_trajectory(states, arm_indices)

        for row, (axis_val, ec) in enumerate(zip(traj.T, ee_colors)):
            ax = axes[row, col]
            ax.set_facecolor("#f8f9fa")
            _make_colored_line(t, axis_val, colors, ax, lw=1.8)
            ax.plot(t, axis_val, color=ec, lw=0.5, alpha=0.30)
            ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.4)
            _shade_segments(ax, segs, fps)

        ax = axes[3, col]
        ax.set_facecolor("#f8f9fa")
        _make_colored_line(t, g_norm, colors, ax, lw=1.8)
        ax.axhline(0.35, color="grey", lw=0.7, ls="--", alpha=0.6)
        ax.axhline(0.65, color="grey", lw=0.7, ls="--", alpha=0.6)
        _shade_segments(ax, segs, fps)

        ax = axes[4, col]
        ax.set_facecolor("#f8f9fa")
        _make_colored_line(t, g_vel, colors, ax, lw=1.8)
        ax.axhline(0, color="grey", lw=0.7, alpha=0.6)
        _shade_segments(ax, segs, fps)
        ax.set_xlabel("Time (s)", fontsize=9)

        axes[0, col].set_title(f"{arm.capitalize()} Arm", fontsize=11,
                                fontweight="bold", pad=6)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)

    legend_patches = [
        mpatches.Patch(color=PRIMITIVE_COLORS[p], label=p.value.capitalize())
        for p in [PrimitiveType.REACH, PrimitiveType.GRASP,
                  PrimitiveType.MOVE,  PrimitiveType.RELEASE,
                  PrimitiveType.UNKNOWN]
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center", ncol=5, fontsize=9,
        bbox_to_anchor=(0.5, -0.005),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = os.path.join(output_dir, f"episode_{episode_index:04d}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------

def visualize_summary(all_episodes: list, output_dir: str) -> str:
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
                label=f"n={len(d)}",
            )
        ax.set_title(f"{arm.capitalize()} arm")
        ax.set_ylabel("Mean duration (s)")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "summary_durations.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
