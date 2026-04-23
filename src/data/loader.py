"""
Thin wrapper around LeRobotDataset that:
  - loads a dataset from HuggingFace Hub (or a local cache)
  - exposes per-episode state arrays as numpy arrays
  - auto-detects FPS and state dimension from dataset metadata
"""

import numpy as np
from typing import Iterator, Tuple

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as _LRDataset
    _LEROBOT_AVAILABLE = True
except ImportError:
    _LEROBOT_AVAILABLE = False

try:
    from datasets import load_dataset as _hf_load
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


class EpisodeLoader:
    """
    Loads ALOHA episodes and yields (episode_index, states) pairs.

    states : np.ndarray, shape [T, state_dim]
    """

    def __init__(self, config):
        self.cfg = config
        self._dataset = None
        self._episode_boundaries = None  # list of (from_idx, to_idx)
        self._fps = config.fps

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Download / verify the dataset and build episode index."""
        if _LEROBOT_AVAILABLE:
            self._setup_lerobot()
        elif _HF_AVAILABLE:
            self._setup_hf_datasets()
        else:
            raise ImportError(
                "Neither 'lerobot' nor 'datasets' package found. "
                "Install with: pip install lerobot"
            )

    def _setup_lerobot(self) -> None:
        cfg = self.cfg
        kwargs = {"repo_id": cfg.dataset_repo_id}
        if cfg.local_root:
            kwargs["root"] = cfg.local_root

        self._dataset = _LRDataset(**kwargs)

        # Override fps from dataset metadata
        if hasattr(self._dataset, "fps"):
            self._fps = self._dataset.fps
            self.cfg.fps = self._fps

        # Build episode boundaries
        n_ep = min(cfg.n_episodes, self._dataset.num_episodes)
        ei   = self._dataset.episode_data_index
        self._episode_boundaries = [
            (int(ei["from"][i].item()), int(ei["to"][i].item()))
            for i in range(n_ep)
        ]

    def _setup_hf_datasets(self) -> None:
        cfg = self.cfg
        raw = _hf_load(cfg.dataset_repo_id, split="train")
        self._dataset = raw

        # Build episode boundaries by scanning episode_index column
        ep_indices = np.array(raw["episode_index"])
        n_ep = min(cfg.n_episodes, int(ep_indices.max()) + 1)
        self._episode_boundaries = []
        for ep in range(n_ep):
            mask = np.where(ep_indices == ep)[0]
            if len(mask):
                self._episode_boundaries.append((int(mask[0]), int(mask[-1]) + 1))

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
        """
        Yields (episode_index, states) for each episode.
        states has shape [T, state_dim].
        """
        if self._episode_boundaries is None:
            raise RuntimeError("Call setup() before iterating.")

        for ep_idx, (from_i, to_i) in enumerate(self._episode_boundaries):
            states = self._load_states(from_i, to_i)
            if states is not None and states.shape[0] > 0:
                yield ep_idx, states

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_states(self, from_i: int, to_i: int) -> np.ndarray:
        if _LEROBOT_AVAILABLE and isinstance(self._dataset, _LRDataset):
            # Fast path: use the underlying HuggingFace arrow dataset
            raw = self._dataset.hf_dataset["observation.state"][from_i:to_i]
            return np.array(raw, dtype=np.float32)

        # Fallback: HF datasets
        raw = self._dataset["observation.state"][from_i:to_i]
        return np.array(raw, dtype=np.float32)
