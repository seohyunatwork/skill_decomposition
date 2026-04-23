"""
EE-speed + gripper-velocity finite-state machine for primitive skill segmentation.

State definitions
-----------------
  IDLE    : arm stationary, gripper not closing
  REACH   : arm moving (ee_speed > thr), gripper not yet closing
  GRASP   : gripper closing (vel < 0); arm may still be moving
  MOVE    : arm moving (ee_speed > thr) after a grasp cycle
  RELEASE : gripper opening (vel > 0) after MOVE

Transitions
-----------
  IDLE    -> REACH   : ee_speed > thr
  IDLE    -> GRASP   : g_vel < closing_thr AND ee_speed <= thr

  REACH   -> IDLE    : ee_speed <= thr (arm stops, gripper not closing)
  REACH   -> GRASP   : g_vel < closing_thr  (gripper closing is the strong signal)

  GRASP   -> MOVE    : ee_speed > thr
  GRASP   -> REACH   : g_vel > opening_thr  (aborted grasp)

  MOVE    -> RELEASE : g_vel > opening_thr

  RELEASE -> IDLE    : g_norm > open_thr AND ee_speed <= thr
  RELEASE -> REACH   : g_norm > open_thr AND ee_speed > thr
  RELEASE -> MOVE    : g_vel < closing_thr  (re-grasped)

RELEASE can only be reached via MOVE, never at episode start.
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
        traj   = fk_trajectory(states, arm_indices)          # [T, 3]  metres
        ee_vel = np.gradient(traj, axis=0) * self.cfg.fps    # [T, 3]  m/s
        speed  = np.linalg.norm(ee_vel, axis=1)              # [T]
        speed  = gaussian_filter1d(speed, sigma=self.cfg.smooth_window / 3)
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
        cfg    = self.cfg
        n      = len(g)
        state  = PrimitiveType.IDLE
        labels = []

        moving  = lambda s: s > cfg.ee_speed_threshold
        closing = lambda vel: vel < cfg.closing_vel_threshold
        opening = lambda vel: vel > cfg.opening_vel_threshold

        for i in range(n):
            gi, vi, si = g[i], v[i], ee_speed[i]

            if state == PrimitiveType.IDLE:
                if moving(si):
                    state = PrimitiveType.REACH
                elif closing(vi):           # arm stopped, gripper starts closing
                    state = PrimitiveType.GRASP

            elif state == PrimitiveType.REACH:
                if closing(vi):        # gripper closing is the definitive grasp signal
                    state = PrimitiveType.GRASP
                elif not moving(si):
                    state = PrimitiveType.IDLE

            elif state == PrimitiveType.GRASP:
                if moving(si):
                    state = PrimitiveType.MOVE
                elif opening(vi):           # aborted: gripper re-opened without lift
                    state = PrimitiveType.REACH

            elif state == PrimitiveType.MOVE:
                if opening(vi):
                    state = PrimitiveType.RELEASE

            elif state == PrimitiveType.RELEASE:
                if gi > cfg.open_threshold:
                    state = PrimitiveType.REACH if moving(si) else PrimitiveType.IDLE
                elif closing(vi):
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
