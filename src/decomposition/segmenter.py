"""
EE-speed + gripper-velocity finite-state machine for primitive skill segmentation.

State machine
-------------
  IDLE    ──(ee_speed > thr)───────────────────► REACH
  REACH   ──(ee_speed ≤ thr)──────────────────► IDLE
  REACH   ──(gripper_vel < closing_thr)───────► GRASP
  GRASP   ──(ee_speed > thr)──────────────────► MOVE
  GRASP   ──(gripper_vel > opening_thr)───────► REACH   (aborted grasp)
  MOVE    ──(gripper_vel > opening_thr)───────► RELEASE
  RELEASE ──(gripper_norm > open_thr, slow EE)──► IDLE
  RELEASE ──(gripper_norm > open_thr, fast EE)──► REACH
  RELEASE ──(gripper_vel < closing_thr)───────► MOVE    (re-grasped)

RELEASE can only be reached from MOVE (i.e., after a GRASP cycle).
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .primitives import PrimitiveType, PrimitiveSegment
from ..kinematics.fk import fk_trajectory


class PrimitiveSegmenter:
    def __init__(self, config):
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_episode(self, states: np.ndarray, episode_index: int) -> dict:
        cfg = self.cfg
        results = {}

        for arm, gripper_idx, arm_indices in [
            ("left",  cfg.left_gripper_idx,  cfg.left_arm_indices),
            ("right", cfg.right_gripper_idx, cfg.right_arm_indices),
        ]:
            gripper_raw = states[:, gripper_idx]

            g_norm, g_vel = self._preprocess_gripper(gripper_raw)
            ee_speed      = self._compute_ee_speed(states, arm_indices)
            frame_labels  = self._run_state_machine(g_norm, g_vel, ee_speed)

            arm_joints = states[:, arm_indices]
            segments   = self._labels_to_segments(
                frame_labels, g_norm, g_vel, arm_joints,
                arm, episode_index
            )

            results[arm] = {
                "segments":     segments,
                "frame_labels": frame_labels,
                "gripper_norm": g_norm,
                "gripper_vel":  g_vel,
                "ee_speed":     ee_speed,
            }

        return results

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def _preprocess_gripper(self, raw: np.ndarray):
        sigma    = self.cfg.smooth_window / 3
        smoothed = gaussian_filter1d(raw.astype(float), sigma=sigma)

        g_min, g_max = smoothed.min(), smoothed.max()
        spread = g_max - g_min

        if spread < 1e-4:
            norm = np.full_like(smoothed, 0.5)
            vel  = np.zeros_like(smoothed)
            return norm, vel

        norm = (smoothed - g_min) / spread
        vel  = np.gradient(norm) * self.cfg.fps
        return norm, vel

    def _compute_ee_speed(self, states: np.ndarray, arm_indices: list) -> np.ndarray:
        traj     = fk_trajectory(states, arm_indices)           # [T, 3]  metres
        ee_vel   = np.gradient(traj, axis=0) * self.cfg.fps     # [T, 3]  m/s
        speed    = np.linalg.norm(ee_vel, axis=1)               # [T]
        # Smooth to suppress FK noise
        speed    = gaussian_filter1d(speed, sigma=self.cfg.smooth_window / 3)
        return speed

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _run_state_machine(
        self,
        g:        np.ndarray,
        v:        np.ndarray,
        ee_speed: np.ndarray,
    ) -> np.ndarray:
        cfg = self.cfg
        n   = len(g)

        state  = PrimitiveType.IDLE
        labels = []

        for i in range(n):
            gi = g[i]
            vi = v[i]
            si = ee_speed[i]

            if state == PrimitiveType.IDLE:
                if si > cfg.ee_speed_threshold:
                    state = PrimitiveType.REACH

            elif state == PrimitiveType.REACH:
                if vi < cfg.closing_vel_threshold:
                    state = PrimitiveType.GRASP
                elif si <= cfg.ee_speed_threshold:
                    state = PrimitiveType.IDLE

            elif state == PrimitiveType.GRASP:
                if si > cfg.ee_speed_threshold:
                    state = PrimitiveType.MOVE
                elif vi > cfg.opening_vel_threshold:
                    # gripper re-opened before contact → aborted
                    state = PrimitiveType.REACH

            elif state == PrimitiveType.MOVE:
                if vi > cfg.opening_vel_threshold:
                    state = PrimitiveType.RELEASE

            elif state == PrimitiveType.RELEASE:
                if gi > cfg.open_threshold:
                    if si > cfg.ee_speed_threshold:
                        state = PrimitiveType.REACH
                    else:
                        state = PrimitiveType.IDLE
                elif vi < cfg.closing_vel_threshold:
                    state = PrimitiveType.MOVE

            labels.append(state)

        return np.array(labels)

    # ------------------------------------------------------------------
    # Segment extraction
    # ------------------------------------------------------------------

    def _labels_to_segments(
        self,
        labels:        np.ndarray,
        g_norm:        np.ndarray,
        g_vel:         np.ndarray,
        arm_joints:    np.ndarray,
        arm:           str,
        episode_index: int,
    ) -> list[PrimitiveSegment]:

        cfg = self.cfg
        segments: list[PrimitiveSegment] = []
        n = len(labels)

        joint_vel = np.linalg.norm(
            np.gradient(arm_joints, axis=0) * cfg.fps, axis=1
        )

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
