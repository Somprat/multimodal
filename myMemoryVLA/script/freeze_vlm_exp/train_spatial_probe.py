"""Frame dataset for the frozen-VLM spatial probe experiment."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ProbeFrameDataset(Dataset):
    def __init__(self, data_dir, cache_dir, episode_names):
        data_dir = Path(data_dir)
        cache_dir = Path(cache_dir)

        rgb_features = []
        depths = []
        intrinsics = []
        extrinsics = []
        object_relative = []
        target_relative = []
        object_distance = []
        valid_grasp = []
        timesteps = []
        instructions = []
        positions = []
        episode_ids = []

        #maybe this is episode id?
        for episode_id, episode_name in enumerate(episode_names):
            episode_path = data_dir / episode_name
            cache_path = cache_dir / f"{episode_path.stem}.pt"
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Missing RGB cache for {episode_name}: {cache_path}"
                )
        
            cache = torch.load(cache_path, map_location="cpu")
            cached_rgb = cache["rgb_features"].float()
            

            with np.load(episode_path, allow_pickle=False) as data:
                frame_count = len(data["rgb"])

                # .long() changes the elements from float to integer
                timesteps = data.get("timestep", np.arange(frame_count)).long()

                depth = torch.from_numpy(data["depth"].copy()).float()
                camera_intrinsics = torch.from_numpy(
                    data["camera_intrinsics"].copy()
                ).float()
                camera_extrinsics = torch.from_numpy(
                    data["camera_extrinsics"].copy()
                ).float()
                object_rel = torch.from_numpy(
                    (data["object_xyz"] - data["gripper_xyz"]).astype(np.float32)
                )
                target_rel = torch.from_numpy(
                    (data["target_xyz"] - data["gripper_xyz"]).astype(np.float32)
                )
                distance = torch.linalg.vector_norm(object_rel, dim=1)
                grasp = torch.from_numpy(
                    data["is_grasping"].astype(np.float32)
                )
                raw_instructions = data["instructions"]
                if raw_instructions.ndim == 0:
                    instruction = [str(raw_instructions.item())] * frame_count
                else:
                    instruction = [str(value) for value in raw_instructions]
                gripper_xyz = torch.from_numpy(
                    data["gripper_xyz"].astype(np.float32)
                )
                episode_id_tensor = torch.full(
                    (frame_count,), episode_id, dtype=torch.long
                )

            frame_count = len(depth)
            tensors = {
                "rgb cache": cached_rgb,
                "intrinsics": camera_intrinsics,
                "extrinsics": camera_extrinsics,
                "object target": object_rel,
                "placement target": target_rel,
                "distance target": distance,
                "grasp target": grasp,
                "timestep": timesteps,
                "instruction": instruction,
                "gripper_xyz": gripper_xyz,
                "episode_ids": episode_id
            }
            for name, tensor in tensors.items():
                if len(tensor) != frame_count:
                    raise ValueError(
                        f"{episode_name}: {name} has {len(tensor)} frames, expected {frame_count}"
                    )

            rgb_features.append(cached_rgb)
            depths.append(depth)
            intrinsics.append(camera_intrinsics)
            extrinsics.append(camera_extrinsics)
            object_relative.append(object_rel)
            target_relative.append(target_rel)
            object_distance.append(distance)
            valid_grasp.append(grasp)
            timesteps.append(timesteps)
            instructions.extend(instruction)
            positions.append(gripper_xyz)
            episode_ids.append(episode_id)

        if not rgb_features:
            raise ValueError("The split contains no episodes")

        self.rgb_features = torch.cat(rgb_features)
        self.depth = torch.cat(depths)
        self.intrinsics = torch.cat(intrinsics)
        self.extrinsics = torch.cat(extrinsics)
        self.object_relative = torch.cat(object_relative)
        self.target_relative = torch.cat(target_relative)
        self.object_distance = torch.cat(object_distance)
        self.valid_grasp = torch.cat(valid_grasp)
        self.timesteps = torch.cat(timesteps)
        self.instructions = instructions
        self.positions = torch.cat(positions)


    @property
    def rgb_feature_dim(self):
        return self.rgb_features.shape[1]

    def __len__(self):
        return len(self.depth)

    def __getitem__(self, index):
        return {
            "rgb_features": self.rgb_features[index],
            "depth": self.depth[index],
            "intrinsics": self.intrinsics[index],
            "extrinsics": self.extrinsics[index],
            "object_relative": self.object_relative[index],
            "target_relative": self.target_relative[index],
            "object_distance": self.object_distance[index],
            "valid_grasp": self.valid_grasp[index],
            "timestep": self.timesteps[index],
            "instructions": self.instructions[index],
            "self.positions": self.positions[index]
        }
    
