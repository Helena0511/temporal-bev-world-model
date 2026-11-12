"""Minimal raw nuScenes reader for synchronized camera keyframes."""

import json
from pathlib import Path
import math
import torch


CAMERAS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)


class NuScenesTwoHistory:
    """Return one current sample and two previous samples.

    The returned order is chronological from oldest to newest:
    ``[t-2, t-1, t]``.  Each item contains six synchronized camera paths.
    """

    def __init__(self, dataset_root):
        self.root = Path(dataset_root)
        version_root = self.root / "v1.0-mini"

        self.samples = self._load_table(version_root / "sample.json")
        sample_data = self._load_table(version_root / "sample_data.json")

        ego_poses=self._load_table(version_root/"ego_pose.json")
        self.ego_pose_by_token={
            row["token"]:row for row in ego_poses
        }

        sensors = self._load_table(version_root / "sensor.json")
        calibrated = self._load_table(version_root / "calibrated_sensor.json")

        sensor_channel = {row["token"]: row["channel"] for row in sensors}
        calib_channel = {
            row["token"]: sensor_channel[row["sensor_token"]]
            for row in calibrated
        }

        self.sample_by_token = {row["token"]: row for row in self.samples}
        self.camera_paths = {}
        self.camera_poses={}
        for row in sample_data:
            channel = calib_channel.get(row["calibrated_sensor_token"])
            if channel not in CAMERAS:
                continue
            if row["fileformat"] != "jpg":
                continue
            # Keep keyframe images only. Their sample_token is synchronized.
            if not row["is_key_frame"]:
                continue
            self.camera_paths.setdefault(row["sample_token"], {})[channel] = (
                self.root / row["filename"]
            )
            self.camera_poses.setdefault(row["sample_token"],{})[channel]=(self.ego_pose_by_token[row["ego_pose_token"]])

    @staticmethod
    def _load_table(path):
        with path.open("r") as file:
            return json.load(file)

    def _has_all_cameras(self, sample_token):
        paths = self.camera_paths.get(sample_token, {})
        return all(camera in paths for camera in CAMERAS)

    def find_sample_with_two_history(self):
        """Find the first sample with valid t, t-1 and t-2 camera data."""
        for sample in self.samples:
            token_t = sample["token"]
            token_t1 = sample["prev"]
            if not token_t1:
                continue
            token_t2 = self.sample_by_token[token_t1]["prev"]
            if not token_t2:
                continue
            if all(self._has_all_cameras(token) for token in (token_t2, token_t1, token_t)):
                return token_t
        raise RuntimeError("could not find a sample with two history frames")

    def get_three_frame_paths(self, current_token):
        """Return paths in chronological order: t-2, t-1, t."""
        current = self.sample_by_token[current_token]
        token_t1 = current["prev"]
        token_t2 = self.sample_by_token[token_t1]["prev"]
        tokens = (token_t2, token_t1, current_token)

        result = []
        for token in tokens:
            if not self._has_all_cameras(token):
                raise RuntimeError(f"sample {token} does not have six camera keyframes")
            result.append([self.camera_paths[token][camera] for camera in CAMERAS])
        return result
    
    def get_three_frame_poses(self,current_token):
        current=self.sample_by_token[current_token]
        token_t1=current["prev"]
        token_t2=self.sample_by_token[token_t1]["prev"]

        tokens=(token_t2,token_t1,current_token)
        return [
            self.camera_poses[token]["CAM_FRONT"] for token in tokens
        ]


def ego_pose_to_planar_matrix(ego_pose):
    tx, ty, _ = ego_pose["translation"]
    w, x, y, z = ego_pose["rotation"]

    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )

    c = math.cos(yaw)
    s = math.sin(yaw)

    return torch.tensor(
        [
            [c, -s, tx],
            [s, c, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def get_current2history(current_ego_pose, history_ego_pose):
    current_pose = ego_pose_to_planar_matrix(current_ego_pose)
    history_pose = ego_pose_to_planar_matrix(history_ego_pose)
    current2history = torch.linalg.solve(history_pose, current_pose).float()
    return {"current2history": current2history}
