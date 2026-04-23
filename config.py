from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ALOHAConfig:
    """Configuration for ALOHA dataset loading and primitive skill decomposition."""

    # --- Dataset ---
    dataset_repo_id: str = "lerobot/aloha_sim_transfer_cube_human"
    local_root: Optional[str] = None  # None → use HF cache (~/.cache/huggingface/lerobot)
    n_episodes: int = 5               # number of episodes to decompose

    # --- ALOHA joint layout (14-dim state / action) ---
    # Standard ALOHA: [left_q0..5, left_gripper, right_q0..5, right_gripper]
    state_dim: int = 14
    left_arm_indices: List[int] = field(default_factory=lambda: list(range(6)))
    left_gripper_idx: int = 6
    right_arm_indices: List[int] = field(default_factory=lambda: list(range(7, 13)))
    right_gripper_idx: int = 13
    fps: int = 50  # overridden from dataset metadata at runtime

    # --- Gripper signal processing ---
    smooth_window: int = 7        # Gaussian kernel half-width (frames)
    open_threshold: float = 0.65  # normalised gripper ≥ this → "open"
    closed_threshold: float = 0.35  # normalised gripper ≤ this → "closed"
    # Velocity thresholds are in normalised-units / second
    closing_vel_threshold: float = -0.25   # dG/dt < this → gripper closing
    opening_vel_threshold: float = 0.25    # dG/dt > this → gripper opening
    min_segment_frames: int = 3            # discard segments shorter than this

    # --- Output ---
    output_dir: str = "outputs"
    save_json: bool = True
    visualize: bool = True
