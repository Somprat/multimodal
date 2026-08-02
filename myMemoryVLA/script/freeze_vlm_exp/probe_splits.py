from pathlib import Path
from collections import defaultdict
import random

def train_test_val_split(path, random_seed=42):
    data_path = Path(path)

    file_by_task = defaultdict(list)
    splitting_dict = defaultdict(list)
    for episode_path in sorted(data_path.glob("*.npz")):
        if "_episode_" not in episode_path.stem:
            raise ValueError(
                f"Unexpected filename {episode_path.name!r}; expected '<task>_episode_<id>.npz'"
            )
        task, _ = episode_path.stem.rsplit("_episode_", 1)
        file_by_task[task].append(episode_path.name)

    if not file_by_task:
        raise ValueError(f"No NPZ episodes found in {data_path}")

    rng = random.Random(random_seed)
    for task, files in file_by_task.items():
        if len(files) < 3:
            raise ValueError(f"Task {task!r} needs at least 3 episodes, found {len(files)}")
        rng.shuffle(files)

        sixty_percent_data = int(len(files) * 0.6)
        twenty_percent_data = max(1, int(len(files) * 0.2))

        train_data = files[:sixty_percent_data]
        val_data = files[sixty_percent_data:(sixty_percent_data+twenty_percent_data)]
        test_data = files[(sixty_percent_data+twenty_percent_data):]

        splitting_dict["train"].extend(train_data)
        splitting_dict["validation"].extend(val_data)
        splitting_dict["test"].extend(test_data)

    return dict(splitting_dict)
