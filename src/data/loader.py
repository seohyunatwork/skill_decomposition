"""
Loads ALOHA episodes and yields (episode_index, states) pairs.

Priority order:
  1. lerobot.datasets.LeRobotDataset  (lerobot ≥ 0.4)
  2. lerobot.common.datasets.LeRobotDataset  (lerobot < 0.4, legacy path)
  3. HuggingFace `datasets` library
  4. Synthetic data generator (offline fallback)
"""

import numpy as np
from typing import Iterator, Tuple

# --- Try lerobot (new path, ≥0.4) ---
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LRDataset
    _LEROBOT_AVAILABLE = True
except ImportError:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as _LRDataset
        _LEROBOT_AVAILABLE = True
    except ImportError:
        _LEROBOT_AVAILABLE = False

# --- Try plain HF datasets ---
try:
    from datasets import load_dataset as _hf_load
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


class EpisodeLoader:
    """
    Loads ALOHA episodes and yields (episode_index, states) pairs.
    Falls back to a realistic synthetic generator when HF Hub is unreachable.

    states : np.ndarray, shape [T, state_dim]
    """

    def __init__(self, config):
        self.cfg = config
        self._dataset = None
        self._episode_boundaries = None   # list of (from_idx, to_idx)
        self._fps = config.fps
        self._synthetic = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Load dataset or fall back to synthetic generation."""
        loaded = False
        if _LEROBOT_AVAILABLE:
            loaded = self._try_setup_lerobot()
        if not loaded and _HF_AVAILABLE:
            loaded = self._try_setup_hf_datasets()
        if not loaded:
            print(
                "  [INFO] Hub unreachable or lerobot not found — "
                "using synthetic ALOHA transfer-cube data."
            )
            self._setup_synthetic()

    # --- lerobot path ---
    def _try_setup_lerobot(self) -> bool:
        try:
            cfg = self.cfg
            kwargs = {"repo_id": cfg.dataset_repo_id}
            if cfg.local_root:
                kwargs["root"] = cfg.local_root

            self._dataset = _LRDataset(**kwargs)

            if hasattr(self._dataset, "fps"):
                self._fps = self._dataset.fps
                self.cfg.fps = self._fps

            n_ep = min(cfg.n_episodes, self._dataset.num_episodes)
            ei   = self._dataset.episode_data_index
            self._episode_boundaries = [
                (int(ei["from"][i].item()), int(ei["to"][i].item()))
                for i in range(n_ep)
            ]
            return True
        except Exception as exc:
            print(f"  [WARN] lerobot loader failed: {exc}")
            return False

    # --- HF datasets path ---
    def _try_setup_hf_datasets(self) -> bool:
        try:
            cfg = self.cfg
            raw = _hf_load(cfg.dataset_repo_id, split="train")
            self._dataset = raw
            ep_indices = np.array(raw["episode_index"])
            n_ep = min(cfg.n_episodes, int(ep_indices.max()) + 1)
            self._episode_boundaries = []
            for ep in range(n_ep):
                mask = np.where(ep_indices == ep)[0]
                if len(mask):
                    self._episode_boundaries.append(
                        (int(mask[0]), int(mask[-1]) + 1)
                    )
            return True
        except Exception as exc:
            print(f"  [WARN] HF datasets loader failed: {exc}")
            return False

    # --- synthetic fallback ---
    def _setup_synthetic(self) -> None:
        self._synthetic = True
        n_ep = self.cfg.n_episodes
        # Each synthetic episode is (None, None) — generated lazily in iter_episodes
        self._episode_boundaries = [(None, None)] * n_ep

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    @property
    def num_episodes(self) -> int:
        return len(self._episode_boundaries) if self._episode_boundaries else 0

    @property
    def fps(self) -> float:
        return self._fps

    def iter_episodes(self) -> Iterator[Tuple[int, np.ndarray]]:
        if self._episode_boundaries is None:
            raise RuntimeError("Call setup() before iterating.")

        for ep_idx, (from_i, to_i) in enumerate(self._episode_boundaries):
            if self._synthetic:
                states = _synthetic_transfer_cube_episode(
                    fps=self._fps, rng=np.random.default_rng(seed=ep_idx)
                )
            else:
                states = self._load_states(from_i, to_i)
            if states is not None and states.shape[0] > 0:
                yield ep_idx, states

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_states(self, from_i: int, to_i: int) -> np.ndarray:
        if _LEROBOT_AVAILABLE and isinstance(self._dataset, _LRDataset):
            raw = self._dataset.hf_dataset["observation.state"][from_i:to_i]
            return np.array(raw, dtype=np.float32)
        raw = self._dataset["observation.state"][from_i:to_i]
        return np.array(raw, dtype=np.float32)


# ---------------------------------------------------------------------------
# Synthetic ALOHA transfer-cube episode generator
# ---------------------------------------------------------------------------

def _synthetic_transfer_cube_episode(
    fps: int = 50,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate a single synthetic ALOHA episode mimicking the transfer-cube task.

    State layout (14-dim, matching real ALOHA):
      [left_q0..5, left_gripper, right_q0..5, right_gripper]

    Task sequence per arm:
      REACH  → GRASP → MOVE → RELEASE  (one cube-transfer cycle)

    The left arm is the "active" arm (picks up and transfers the cube).
    The right arm performs its own mirrored transfer at a slight offset.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Randomise timing (seconds) with small noise
    def jitter(base, std=0.05):
        return max(base * 0.6, base + rng.normal(0, std))

    reach_dur   = jitter(1.2)
    grasp_dur   = jitter(0.5)
    move_dur    = jitter(1.5)
    release_dur = jitter(0.5)
    post_dur    = jitter(0.8)

    total_s  = reach_dur + grasp_dur + move_dur + release_dur + post_dur
    T        = int(total_s * fps)
    t        = np.arange(T) / fps

    states = np.zeros((T, 14), dtype=np.float32)

    # ---- Build gripper signal for one arm ----
    def make_gripper(offset_s: float = 0.0) -> np.ndarray:
        """Piecewise linear gripper: open → close → open."""
        g = np.ones(T)  # start open
        t_shifted = t - offset_s

        t_grasp_start  = reach_dur + offset_s
        t_grasp_end    = t_grasp_start + grasp_dur
        t_release_start= t_grasp_end + move_dur
        t_release_end  = t_release_start + release_dur

        for i, ti in enumerate(t):
            if ti < t_grasp_start:
                g[i] = 1.0
            elif ti < t_grasp_end:
                alpha = (ti - t_grasp_start) / grasp_dur
                g[i] = 1.0 - alpha        # closing
            elif ti < t_release_start:
                g[i] = 0.0               # closed (holding)
            elif ti < t_release_end:
                alpha = (ti - t_release_start) / release_dur
                g[i] = alpha             # opening
            else:
                g[i] = 1.0               # open again

        # Add a touch of sensor noise
        g += rng.normal(0, 0.01, size=T)
        return g.astype(np.float32)

    # Left arm: active primary mover
    left_gripper  = make_gripper(offset_s=0.0)

    # Right arm: mirrored with a small time offset (assists / mirrors)
    right_offset  = jitter(0.2, std=0.03)
    right_gripper = make_gripper(offset_s=right_offset)

    # ---- Arm joints: smooth sinusoidal trajectories ----
    # 6 joints per arm; each joint sweeps a small range
    for j in range(6):
        freq      = 0.3 + j * 0.05          # Hz
        amplitude = 0.2 + rng.uniform(0, 0.1)
        phase     = rng.uniform(0, np.pi)
        base      = rng.uniform(-0.3, 0.3)
        states[:, j]     = (base + amplitude * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
        states[:, j + 7] = (base + amplitude * np.sin(2 * np.pi * freq * t + phase + np.pi)).astype(np.float32)

    states[:, 6]  = left_gripper
    states[:, 13] = right_gripper

    return states
