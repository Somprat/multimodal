"""Training-only WidowX task manifest."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrainTask:
    source_asset: str
    target_asset: str
    relation: Literal["in", "on"]
    instruction: str

    source_x_range: tuple[float, float] = (-0.245, -0.075)
    source_y_range: tuple[float, float] = (-0.085, 0.085)
    target_x_range: tuple[float, float] = (-0.245, -0.075)
    target_y_range: tuple[float, float] = (-0.085, 0.085)

    min_object_separation: float = 0.10
    grasp_z_offset: float = 0.01
    placement_clearance: float = 0.008


TRAINING_TASKS = {
    # Smoke-test entry. Replace with a reviewed Bridge V2 instruction before
    # collecting the real training dataset.
    "apple_on_sponge": TrainTask(
        source_asset="apple",
        target_asset="sponge",
        relation="on",
        instruction="put the apple on the sponge",
    ),
}
