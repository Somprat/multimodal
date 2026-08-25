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
    # Stable Bridge-asset task used for calibrated collection.
    "green_cube_in_sink": TrainTask(
        source_asset="green_cube_3cm",
        target_asset="sink",
        relation="in",
        instruction="put the green cube in the sink",
        source_x_range=(-0.080, -0.070),
        source_y_range=(-0.060, 0.060),
        target_x_range=(-0.245, -0.235),
        target_y_range=(-0.060, 0.060),
        min_object_separation=0.15,
        grasp_z_offset=0.0,
    ),
}
