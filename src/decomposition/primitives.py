from enum import Enum
from dataclasses import dataclass


class PrimitiveType(Enum):
    IDLE    = "idle"     # arm stationary, no significant EE motion
    REACH   = "reach"    # arm moving toward object, gripper open
    GRASP   = "grasp"    # gripper closing around object
    MOVE    = "move"     # gripper closed, transporting object
    RELEASE = "release"  # gripper opening to place object


# Colour map for visualisation
PRIMITIVE_COLORS = {
    PrimitiveType.IDLE:    "#9E9E9E",  # grey
    PrimitiveType.REACH:   "#4CAF50",  # green
    PrimitiveType.GRASP:   "#FF9800",  # orange
    PrimitiveType.MOVE:    "#2196F3",  # blue
    PrimitiveType.RELEASE: "#E91E63",  # pink
}


@dataclass
class PrimitiveSegment:
    primitive_type: PrimitiveType
    arm: str              # "left" | "right"
    episode_index: int
    start_frame: int
    end_frame: int        # exclusive
    fps: float

    gripper_start: float  # normalised gripper value at segment start
    gripper_end: float    # normalised gripper value at segment end
    gripper_mean: float   # mean normalised gripper value
    joint_vel_mean: float # mean L2 joint velocity (rad/s) during segment

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def duration_seconds(self) -> float:
        return self.duration_frames / self.fps

    def to_dict(self) -> dict:
        return {
            "primitive": self.primitive_type.value,
            "arm": self.arm,
            "episode_index": self.episode_index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "duration_seconds": round(self.duration_seconds, 4),
            "gripper_start": round(float(self.gripper_start), 4),
            "gripper_end": round(float(self.gripper_end), 4),
            "gripper_mean": round(float(self.gripper_mean), 4),
            "joint_vel_mean": round(float(self.joint_vel_mean), 4),
        }
