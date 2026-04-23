"""
Gripper-state-machine based primitive skill segmenter.

Algorithm
---------
1. Extract normalised gripper signal G[t] ∈ [0,1]  (1 = fully open, 0 = fully closed)
2. Smooth with a Gaussian filter to suppress sensor noise
3. Compute per-frame velocity  V[t] = dG/dt  (normalised units / second)
4. Run a finite-state machine with hysteresis:

     REACH  ──(V < closing_vel)──►  GRASP
     GRASP  ──(G < closed_thresh)──► MOVE
     GRASP  ──(V > opening_vel)───► REACH   (abort: never got contact)
     MOVE   ──(V > opening_vel)───► RELEASE
     RELEASE──(G > open_thresh)───► REACH   (cycle complete)
     RELEASE──(V < closing_vel)───► MOVE    (abort: gripper re-closed)

5. Merge consecutive frames with the same label into PrimitiveSegment objects
6. Drop segments shorter than min_segment_frames (noise artefacts)
7. If the gripper barely moves across the whole episode (arm inactive),
   label all frames UNKNOWN.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .primitives import PrimitiveType, PrimitiveSegment


class PrimitiveSegmenter:
    def __init__(self, config):
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_episode(self, states: np.ndarray, episode_index: int) -> dict:
        """
        Decompose a single episode into primitive segments for both arms.

        Parameters
        ----------
        states : np.ndarray, shape [T, state_dim]
        episode_index : int

        Returns
        -------
        dict with keys 'left' and 'right', each containing:
          - 'segments': list[PrimitiveSegment]
          - 'frame_labels': np.ndarray[PrimitiveType], length T
          - 'gripper_norm': np.ndarray, length T
          - 'gripper_vel': np.ndarray, length T
        """
        cfg = self.cfg
        results = {}

        for arm, gripper_idx, arm_indices in [
            ("left",  cfg.left_gripper_idx,  cfg.left_arm_indices),
            ("right", cfg.right_gripper_idx, cfg.right_arm_indices),
        ]:
            gripper_raw = states[:, gripper_idx]
            arm_joints  = states[:, arm_indices]

            g_norm, g_vel = self._preprocess_gripper(gripper_raw)
            frame_labels  = self._run_state_machine(g_norm, g_vel)

            segments = self._labels_to_segments(
                frame_labels, g_norm, g_vel, arm_joints,
                arm, episode_index
            )

            results[arm] = {
                "segments":     segments,
                "frame_labels": frame_labels,
                "gripper_norm": g_norm,
                "gripper_vel":  g_vel,
            }

        return results

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def _preprocess_gripper(self, raw: np.ndarray):
        """Smooth and normalise raw gripper signal; compute velocity."""
        smoothed = gaussian_filter1d(raw.astype(float), sigma=self.cfg.smooth_window / 3)

        g_min, g_max = smoothed.min(), smoothed.max()
        spread = g_max - g_min

        # Arm considered inactive when gripper barely moves
        if spread < 1e-4:
            norm = np.full_like(smoothed, 0.5)
            vel  = np.zeros_like(smoothed)
            return norm, vel

        norm = (smoothed - g_min) / spread
        vel  = np.gradient(norm) * self.cfg.fps
        return norm, vel

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _run_state_machine(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        n   = len(g)

        # If gripper doesn't move enough, everything is UNKNOWN
        if g.std() < 0.02:
            return np.array([PrimitiveType.UNKNOWN] * n)

        # Determine initial state from first gripper value
        state = PrimitiveType.REACH if g[0] >= 0.5 else PrimitiveType.MOVE

        labels = []
        for i in range(n):
            gi, vi = g[i], v[i]

            if state == PrimitiveType.REACH:
                if vi < cfg.closing_vel_threshold:
                    state = PrimitiveType.GRASP

            elif state == PrimitiveType.GRASP:
                if gi < cfg.closed_threshold:
                    state = PrimitiveType.MOVE
                elif vi > cfg.opening_vel_threshold:
                    # gripper re-opened without achieving grasp → back to REACH
                    state = PrimitiveType.REACH

            elif state == PrimitiveType.MOVE:
                if vi > cfg.opening_vel_threshold:
                    state = PrimitiveType.RELEASE

            elif state == PrimitiveType.RELEASE:
                if gi > cfg.open_threshold:
                    state = PrimitiveType.REACH          # completed a full cycle
                elif vi < cfg.closing_vel_threshold:
                    state = PrimitiveType.MOVE            # gripper closed again

            labels.append(state)

        return np.array(labels)

    # ------------------------------------------------------------------
    # Segment extraction
    # ------------------------------------------------------------------

    def _labels_to_segments(
        self,
        labels:       np.ndarray,
        g_norm:       np.ndarray,
        g_vel:        np.ndarray,
        arm_joints:   np.ndarray,
        arm:          str,
        episode_index: int,
    ) -> list[PrimitiveSegment]:

        cfg = self.cfg
        segments: list[PrimitiveSegment] = []
        n = len(labels)

        # Joint velocity magnitude (L2 norm across joints, per frame)
        joint_vel = np.linalg.norm(np.gradient(arm_joints, axis=0) * cfg.fps, axis=1)

        i = 0
        while i < n:
            current_type = labels[i]
            j = i + 1
            while j < n and labels[j] == current_type:
                j += 1

            if j - i >= cfg.min_segment_frames:
                segments.append(PrimitiveSegment(
                    primitive_type  = current_type,
                    arm             = arm,
                    episode_index   = episode_index,
                    start_frame     = i,
                    end_frame       = j,
                    fps             = float(cfg.fps),
                    gripper_start   = float(g_norm[i]),
                    gripper_end     = float(g_norm[j - 1]),
                    gripper_mean    = float(g_norm[i:j].mean()),
                    joint_vel_mean  = float(joint_vel[i:j].mean()),
                ))

            i = j

        return segments
