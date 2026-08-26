"""Convert successful WidowX NPZ trajectories to an RLDS/TFDS dataset."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tensorflow_datasets as tfds


NPZ_DIR = os.environ.get("WIDOWX_NPZ_DIR", "datasets/widowx_simpler_npz")
RLDS_DIR = os.environ.get("WIDOWX_RLDS_DIR", "datasets/widowx_simpler_rlds")


@dataclass
class RLDS:
    image: torch.Tensor
    depth: torch.Tensor
    camera_intrinsics: torch.Tensor
    camera_extrinsics: torch.Tensor
    proprio: torch.Tensor
    action: torch.Tensor
    language_instruction: str
    is_first: bool
    is_last: bool
    is_terminal: bool


def find_npz_paths(npz_dir: str = NPZ_DIR) -> list[Path]:
    npz_paths = sorted(Path(npz_dir).glob("*/success/*.npz"))
    if not npz_paths:
        raise FileNotFoundError(
            f"No files found under {npz_dir}/*/success/*.npz"
        )
    return npz_paths



def load_episode(npz_path: Path) -> list[RLDS]:
    """Load one trajectory; it can be released after TFDS writes the example."""
    with np.load(npz_path, allow_pickle=False) as episode:
        action = episode["expert_action"]
        episode_length = action.shape[0]
        instruction = str(episode["instruction"].item())

        each_episode = []
        for t in range(episode_length):
            proprio = np.concatenate(
                (
                    episode["proprio"][t],
                    np.zeros(1, dtype=np.float32),
                    action[t, -1:],
                )
            ).astype(np.float32)

            depth = episode["depth"][t]
            if depth.ndim == 2:
                depth = depth[..., None]

            rlds = RLDS(
                image=torch.from_numpy(episode["rgb"][t].copy()),
                depth=torch.from_numpy(depth.astype(np.float32, copy=True)),
                camera_intrinsics=torch.from_numpy(
                    episode["camera_intrinsics"][t].astype(np.float32, copy=True)
                ),
                camera_extrinsics=torch.from_numpy(
                    episode["camera_extrinsics"][t].astype(np.float32, copy=True)
                ),
                proprio=torch.from_numpy(proprio),
                action=torch.from_numpy(action[t].astype(np.float32, copy=True)),
                language_instruction=instruction,
                is_first=t == 0,
                is_last=t == episode_length - 1,
                is_terminal=t == episode_length - 1,
            )
            each_episode.append(rlds)

    return each_episode


def load_episodes(npz_dir: str = NPZ_DIR) -> list[list[RLDS]]:
    """Convenience helper for debugging; the TFDS writer does not use this."""
    return [load_episode(path) for path in find_npz_paths(npz_dir)]


class WidowxSimplerRgbd(tfds.core.GeneratorBasedBuilder):
    """Write the in-memory RLDS records using TensorFlow Datasets."""

    VERSION = tfds.core.Version("1.0.0")

    def __init__(self, npz_dir: str = NPZ_DIR, **kwargs):
        self.npz_paths = find_npz_paths(npz_dir)
        first_episode = load_episode(self.npz_paths[0])
        self.height, self.width = first_episode[0].image.shape[:2]
        super().__init__(**kwargs)

    def _info(self):
        step = {
            "observation": {
                "image": tfds.features.Tensor(
                    shape=(self.height, self.width, 3), dtype=np.uint8
                ),
                "depth": tfds.features.Tensor(
                    shape=(self.height, self.width, 1), dtype=np.float32
                ),
                "camera_intrinsics": tfds.features.Tensor(
                    shape=(3, 3), dtype=np.float32
                ),
                "camera_extrinsics": tfds.features.Tensor(
                    shape=(4, 4), dtype=np.float32
                ),
                "proprio": tfds.features.Tensor(shape=(8,), dtype=np.float32),
            },
            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
            "language_instruction": tfds.features.Text(),
            "is_first": np.bool_,
            "is_last": np.bool_,
            "is_terminal": np.bool_,
        }

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(step),
                    "episode_metadata": {
                        "episode_id": tfds.features.Text(),
                        "source": tfds.features.Text(),
                    },
                }
            )
        )

    def _split_generators(self, dl_manager):
        del dl_manager
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        for npz_path in self.npz_paths:
            episode = load_episode(npz_path)
            task = npz_path.parent.parent.name
            episode_id = f"{task}_{npz_path.stem}"
            steps = []

            for step in episode:
                steps.append(
                    {
                        "observation": {
                            "image": step.image.numpy(),
                            "depth": step.depth.numpy(),
                            "camera_intrinsics": step.camera_intrinsics.numpy(),
                            "camera_extrinsics": step.camera_extrinsics.numpy(),
                            "proprio": step.proprio.numpy(),
                        },
                        "action": step.action.numpy(),
                        "language_instruction": step.language_instruction,
                        "is_first": step.is_first,
                        "is_last": step.is_last,
                        "is_terminal": step.is_terminal,
                    }
                )

            yield episode_id, {
                "steps": steps,
                "episode_metadata": {
                    "episode_id": episode_id,
                    "source": str(npz_path),
                },
            }


def write_rlds(
    npz_dir: str = NPZ_DIR, output_dir: str = RLDS_DIR
) -> WidowxSimplerRgbd:
    builder = WidowxSimplerRgbd(
        npz_dir=npz_dir,
        data_dir=str(Path(output_dir).resolve()),
    )
    builder.download_and_prepare()
    return builder


if __name__ == "__main__":
    npz_paths = find_npz_paths()
    print(f"Found {len(npz_paths)} successful episodes")

    builder = write_rlds()
    print(f"Created {builder.info.full_name}")
    print(f"Dataset directory: {builder.data_dir}")
