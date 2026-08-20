from pathlib import Path
import numpy as np

root = Path(
    "/Users/macbooka/Documents/VsCode/multimodal/"
    "artifacts/maniskill_depth_intrinsics_first100"
)

files = sorted(root.glob("episode_*.npz"))

for key in [
    "main_camera_extrinsic_cv",
    "wrist_camera_extrinsic_cv",
]:
    changed_episodes = 0
    maximum_difference = 0.0

    for path in files:
        with np.load(path) as episode:
            matrices = episode[key]  # [steps, 4, 4]

        difference = np.max(np.abs(matrices - matrices[0]))
        maximum_difference = max(maximum_difference, float(difference))

        if not np.allclose(matrices, matrices[0], atol=1e-6):
            changed_episodes += 1

    print(key, changed_episodes, maximum_difference)