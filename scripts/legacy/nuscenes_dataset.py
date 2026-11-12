# REFERENCE_DISABLED: """Minimal raw nuScenes reader for the ResWorld reproduction.
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: This is intentionally independent of mmdet3d/MMCV. It validates the data
# REFERENCE_DISABLED: contract before connecting images to a BEV encoder.
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: Returned trajectory convention:
# REFERENCE_DISABLED:   x = forward, y = left, fixed current-ego frame, units = meters.
# REFERENCE_DISABLED:   gt_positions: [T, 2] absolute positions relative to the current ego pose.
# REFERENCE_DISABLED:   gt_deltas:    [T, 2] per-step displacement deltas.
# REFERENCE_DISABLED: """
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: from __future__ import annotations
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: import json
# REFERENCE_DISABLED: import os
# REFERENCE_DISABLED: from pathlib import Path
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: import numpy as np
# REFERENCE_DISABLED: from torch.utils.data import Dataset
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: CAMERAS = (
# REFERENCE_DISABLED:     "CAM_FRONT_LEFT",
# REFERENCE_DISABLED:     "CAM_FRONT",
# REFERENCE_DISABLED:     "CAM_FRONT_RIGHT",
# REFERENCE_DISABLED:     "CAM_BACK_LEFT",
# REFERENCE_DISABLED:     "CAM_BACK",
# REFERENCE_DISABLED:     "CAM_BACK_RIGHT",
# REFERENCE_DISABLED: )
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: COMMAND_LEFT = 0
# REFERENCE_DISABLED: COMMAND_STRAIGHT = 1
# REFERENCE_DISABLED: COMMAND_RIGHT = 2
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: def load_table(root: Path, version: str, name: str):
# REFERENCE_DISABLED:     with (root / version / f"{name}.json").open() as f:
# REFERENCE_DISABLED:         return json.load(f)
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: def quaternion_to_rotation(q):
# REFERENCE_DISABLED:     """Convert nuScenes [w, x, y, z] quaternion to a 3x3 matrix."""
# REFERENCE_DISABLED:     w, x, y, z = [float(v) for v in q]
# REFERENCE_DISABLED:     return np.array([
# REFERENCE_DISABLED:         [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
# REFERENCE_DISABLED:         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
# REFERENCE_DISABLED:         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
# REFERENCE_DISABLED:     ], dtype=np.float32)
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: class RawNuScenesTrajectoryDataset(Dataset):
# REFERENCE_DISABLED:     """Read camera paths and ego future trajectories from raw nuScenes JSON."""
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:     def __init__(
# REFERENCE_DISABLED:         self,
# REFERENCE_DISABLED:         data_root="/mnt/ssd0/datasets/nuscenes",
# REFERENCE_DISABLED:         version="v1.0-mini",
# REFERENCE_DISABLED:         future_steps=6,
# REFERENCE_DISABLED:         cameras=CAMERAS,
# REFERENCE_DISABLED:         command_lateral_threshold=2.0,
# REFERENCE_DISABLED:     ):
# REFERENCE_DISABLED:         self.root = Path(data_root)
# REFERENCE_DISABLED:         self.version = version
# REFERENCE_DISABLED:         self.future_steps = future_steps
# REFERENCE_DISABLED:         self.cameras = tuple(cameras)
# REFERENCE_DISABLED:         self.command_lateral_threshold = command_lateral_threshold
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:         samples = load_table(self.root, version, "sample")
# REFERENCE_DISABLED:         sample_data = load_table(self.root, version, "sample_data")
# REFERENCE_DISABLED:         ego_poses = load_table(self.root, version, "ego_pose")
# REFERENCE_DISABLED:         scenes = load_table(self.root, version, "scene")
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:         self.samples = {item["token"]: item for item in samples}
# REFERENCE_DISABLED:         self.poses = {item["token"]: item for item in ego_poses}
# REFERENCE_DISABLED:         self.scene_by_token = {item["token"]: item for item in scenes}
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:         # sample_token -> camera channel -> sample_data record
# REFERENCE_DISABLED:         self.camera_data = {}
# REFERENCE_DISABLED:         for item in sample_data:
# REFERENCE_DISABLED:             if not item["is_key_frame"] or item["fileformat"] not in {"jpg", "jpeg"}:
# REFERENCE_DISABLED:                 continue
# REFERENCE_DISABLED:             channel = item.get("channel")
# REFERENCE_DISABLED:             # Raw nuScenes sample_data does not always carry channel. The
# REFERENCE_DISABLED:             # calibrated sensor table is loaded below for that fallback.
# REFERENCE_DISABLED:             self.camera_data.setdefault(item["sample_token"], {})[channel] = item
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:         calibrated = load_table(self.root, version, "calibrated_sensor")
# REFERENCE_DISABLED:         sensors = load_table(self.root, version, "sensor")
# REFERENCE_DISABLED:         sensor_channel = {item["token"]: item["channel"] for item in sensors}
# REFERENCE_DISABLED:         calib_channel = {
# REFERENCE_DISABLED:             item["token"]: sensor_channel[item["sensor_token"]]
# REFERENCE_DISABLED:             for item in calibrated
# REFERENCE_DISABLED:         }
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:         # Fill channel names when sample_data itself does not contain them.
# REFERENCE_DISABLED:         for item in sample_data:
# REFERENCE_DISABLED:             if not item["is_key_frame"] or item["fileformat"] not in {"jpg", "jpeg"}:
# REFERENCE_DISABLED:                 continue
# REFERENCE_DISABLED:             channel = calib_channel[item["calibrated_sensor_token"]]
# REFERENCE_DISABLED:             self.camera_data.setdefault(item["sample_token"], {})[channel] = item
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:         self.entries = []
# REFERENCE_DISABLED:         for sample in samples:
# REFERENCE_DISABLED:             if self._is_valid_entry(sample["token"]):
# REFERENCE_DISABLED:                 self.entries.append(sample["token"])
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:     def _walk_prev(self, token, steps):
# REFERENCE_DISABLED:         for _ in range(steps):
# REFERENCE_DISABLED:             token = self.samples[token]["prev"]
# REFERENCE_DISABLED:             if not token:
# REFERENCE_DISABLED:                 return None
# REFERENCE_DISABLED:         return token
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:     def _walk_next(self, token, steps):
# REFERENCE_DISABLED:         for _ in range(steps):
# REFERENCE_DISABLED:             token = self.samples[token]["next"]
# REFERENCE_DISABLED:             if not token:
# REFERENCE_DISABLED:                 return None
# REFERENCE_DISABLED:         return token
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:     def _is_valid_entry(self, token):
# REFERENCE_DISABLED:         current = self.samples[token]
# REFERENCE_DISABLED:         history = self._walk_prev(token, 1)
# REFERENCE_DISABLED:         future = self._walk_next(token, self.future_steps)
# REFERENCE_DISABLED:         if history is None or future is None:
# REFERENCE_DISABLED:             return False
# REFERENCE_DISABLED:         if self.samples[history]["scene_token"] != current["scene_token"]:
# REFERENCE_DISABLED:             return False
# REFERENCE_DISABLED:         if self.samples[future]["scene_token"] != current["scene_token"]:
# REFERENCE_DISABLED:             return False
# REFERENCE_DISABLED:         for sample_token in (token, history):
# REFERENCE_DISABLED:             channels = self.camera_data.get(sample_token, {})
# REFERENCE_DISABLED:             if any(camera not in channels for camera in self.cameras):
# REFERENCE_DISABLED:                 return False
# REFERENCE_DISABLED:         return True
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:     def _pose_for_sample(self, sample_token):
# REFERENCE_DISABLED:         channels = self.camera_data[sample_token]
# REFERENCE_DISABLED:         pose_token = channels[self.cameras[0]]["ego_pose_token"]
# REFERENCE_DISABLED:         return self.poses[pose_token]
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:     def _future_tokens(self, token):
# REFERENCE_DISABLED:         result = []
# REFERENCE_DISABLED:         for _ in range(self.future_steps):
# REFERENCE_DISABLED:             token = self.samples[token]["next"]
# REFERENCE_DISABLED:             if not token:
# REFERENCE_DISABLED:                 return None
# REFERENCE_DISABLED:             result.append(token)
# REFERENCE_DISABLED:         return result
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:     def _positions_in_current_ego(self, current_token, future_tokens):
# REFERENCE_DISABLED:         current_pose = self._pose_for_sample(current_token)
# REFERENCE_DISABLED:         current_rotation = quaternion_to_rotation(current_pose["rotation"])
# REFERENCE_DISABLED:         current_translation = np.asarray(
# REFERENCE_DISABLED:             current_pose["translation"], dtype=np.float32
# REFERENCE_DISABLED:         )
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:         positions = []
# REFERENCE_DISABLED:         for token in future_tokens:
# REFERENCE_DISABLED:             pose = self._pose_for_sample(token)
# REFERENCE_DISABLED:             future_translation = np.asarray(
# REFERENCE_DISABLED:                 pose["translation"], dtype=np.float32
# REFERENCE_DISABLED:             )
# REFERENCE_DISABLED:             # R is ego->global, so R.T maps global displacement to current ego.
# REFERENCE_DISABLED:             relative = current_rotation.T @ (future_translation - current_translation)
# REFERENCE_DISABLED:             positions.append(relative[:2])
# REFERENCE_DISABLED:         return np.stack(positions, axis=0).astype(np.float32)
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:     def _command_from_future(self, positions):
# REFERENCE_DISABLED:         """A transparent fallback because raw nuScenes has no VAD command label."""
# REFERENCE_DISABLED:         final_y = float(positions[-1, 1])
# REFERENCE_DISABLED:         if final_y > self.command_lateral_threshold:
# REFERENCE_DISABLED:             return COMMAND_LEFT
# REFERENCE_DISABLED:         if final_y < -self.command_lateral_threshold:
# REFERENCE_DISABLED:             return COMMAND_RIGHT
# REFERENCE_DISABLED:         return COMMAND_STRAIGHT
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:     def __len__(self):
# REFERENCE_DISABLED:         return len(self.entries)
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:     def __getitem__(self, index):
# REFERENCE_DISABLED:         token = self.entries[index]
# REFERENCE_DISABLED:         history_token = self._walk_prev(token, 1)
# REFERENCE_DISABLED:         future_tokens = self._future_tokens(token)
# REFERENCE_DISABLED:         current_channels = self.camera_data[token]
# REFERENCE_DISABLED:         history_channels = self.camera_data[history_token]
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:         positions = self._positions_in_current_ego(token, future_tokens)
# REFERENCE_DISABLED:         origin = np.zeros((1, 2), dtype=np.float32)
# REFERENCE_DISABLED:         full_positions = np.concatenate([origin, positions], axis=0)
# REFERENCE_DISABLED:         deltas = np.diff(full_positions, axis=0).astype(np.float32)
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED:         return {
# REFERENCE_DISABLED:             "token": token,
# REFERENCE_DISABLED:             "scene_token": self.samples[token]["scene_token"],
# REFERENCE_DISABLED:             "current_image_paths": [
# REFERENCE_DISABLED:                 str(self.root / current_channels[c]["filename"])
# REFERENCE_DISABLED:                 for c in self.cameras
# REFERENCE_DISABLED:             ],
# REFERENCE_DISABLED:             "history_image_paths": [
# REFERENCE_DISABLED:                 str(self.root / history_channels[c]["filename"])
# REFERENCE_DISABLED:                 for c in self.cameras
# REFERENCE_DISABLED:             ],
# REFERENCE_DISABLED:             "gt_positions": positions,
# REFERENCE_DISABLED:             "gt_deltas": deltas,
# REFERENCE_DISABLED:             "command": self._command_from_future(positions),
# REFERENCE_DISABLED:             "timestamp": self.samples[token]["timestamp"],
# REFERENCE_DISABLED:         }
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: def inspect_dataset():
# REFERENCE_DISABLED:     dataset = RawNuScenesTrajectoryDataset()
# REFERENCE_DISABLED:     print(f"dataset size: {len(dataset)}")
# REFERENCE_DISABLED:     item = dataset[0]
# REFERENCE_DISABLED:     print(f"token: {item['token']}")
# REFERENCE_DISABLED:     print(f"current cameras: {len(item['current_image_paths'])}")
# REFERENCE_DISABLED:     print(f"history cameras: {len(item['history_image_paths'])}")
# REFERENCE_DISABLED:     print(f"first image exists: {os.path.exists(item['current_image_paths'][0])}")
# REFERENCE_DISABLED:     print(f"gt_positions shape: {item['gt_positions'].shape}")
# REFERENCE_DISABLED:     print(f"gt_deltas shape: {item['gt_deltas'].shape}")
# REFERENCE_DISABLED:     print(f"command: {item['command']}")
# REFERENCE_DISABLED:     print(f"gt_positions:\n{np.array2string(item['gt_positions'], precision=3)}")
# REFERENCE_DISABLED:     print(f"gt_deltas:\n{np.array2string(item['gt_deltas'], precision=3)}")
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: 
# REFERENCE_DISABLED: if __name__ == "__main__":
# REFERENCE_DISABLED:     inspect_dataset()
