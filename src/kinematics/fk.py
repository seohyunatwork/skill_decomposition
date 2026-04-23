"""
Simplified forward kinematics for ALOHA / WidowX 250s arm.

Model: planar 4-link arm rotated by the waist joint.
Joint order: [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]

Approximate link lengths (meters) from Interbotix WidowX 250s URDF:
  D1   = 0.11065   base → shoulder (height)
  A2   = 0.09455   shoulder horizontal offset
  A3   = 0.20000   upper arm
  A4   = 0.20000   forearm
  D5   = 0.10425   wrist
  DEE  = 0.11500   wrist → end-effector tip

This is an approximation sufficient for visualising relative motion.
"""

import numpy as np

_D1  = 0.11065
_A2  = 0.09455
_A3  = 0.20000
_A4  = 0.20000
_DEE = 0.11500


def fk_position(q: np.ndarray) -> np.ndarray:
    """
    Compute approximate EE position for one frame.

    Parameters
    ----------
    q : array-like, length 6
        [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]

    Returns
    -------
    np.ndarray, shape (3,)  —  (x, y, z) in metres, base frame (Z up)
    """
    waist = float(q[0])
    # shoulder zero = arm pointing straight up → add π/2 offset
    sh    = float(q[1]) + np.pi / 2
    el    = float(q[2])
    wr    = float(q[4])   # skip forearm_roll (q[3]) — pure rotation, no translation

    # Radial reach and height in the sagittal plane
    r = (_A2
         + _A3  * np.cos(sh)
         + _A4  * np.cos(sh + el)
         + _DEE * np.cos(sh + el + wr))

    z = (_D1
         + _A3  * np.sin(sh)
         + _A4  * np.sin(sh + el)
         + _DEE * np.sin(sh + el + wr))

    # Rotate sagittal plane by waist angle
    x = r * np.cos(waist)
    y = r * np.sin(waist)

    return np.array([x, y, z], dtype=np.float32)


def fk_trajectory(states: np.ndarray, arm_indices: list) -> np.ndarray:
    """
    Compute EE position for every frame of an episode.

    Parameters
    ----------
    states      : np.ndarray [T, state_dim]
    arm_indices : list of 6 ints — joint column indices in states

    Returns
    -------
    np.ndarray [T, 3]  —  (x, y, z) per frame
    """
    T = states.shape[0]
    traj = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        traj[t] = fk_position(states[t, arm_indices])
    return traj
