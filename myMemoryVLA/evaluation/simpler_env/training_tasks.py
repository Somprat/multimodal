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
    "carrot_in_sink": TrainTask(
        source_asset="bridge_carrot_generated_modified",
        target_asset="sink",
        relation="in",
        instruction="put the carrot in the sink",

        # Keep the long carrot outside the sink collision footprint.
        # The previous y range placed it inside the 39.3 cm-wide sink.
        source_x_range=(-0.065, -0.055),
        source_y_range=(-0.200, -0.160),

        target_x_range=(-0.245, -0.235),
        target_y_range=(-0.040, 0.040),

        min_object_separation=0.22,
        grasp_z_offset=0.0,
        placement_clearance=0.015,
    ),
    "yellow_cube_on_green_cube": TrainTask(
        source_asset="yellow_cube_3cm",
        target_asset="green_cube_3cm",
        relation="on",
        instruction="stack the yellow cube on the green cube",

        source_x_range=(-0.245, -0.075),
        source_y_range=(-0.085, 0.085),

        target_x_range=(-0.245, -0.075),
        target_y_range=(-0.085, 0.085),

        min_object_separation=0.10,
        grasp_z_offset=0.0,
        placement_clearance=0.008,
    ),
    "put_spoon_on_plate": TrainTask(
        source_asset="bridge_spoon_generated_modified",
        target_asset="bridge_plate_objaverse_larger",
        instruction="put the spoon on the plate",
        relation="on",

        source_x_range=(-0.245, -0.075),
        source_y_range=(-0.085, 0.085),

        target_x_range=(-0.245, -0.075),
        target_y_range=(-0.085, 0.085),

        min_object_separation=0.10,
        grasp_z_offset=0.005,
        placement_clearance=0.008,
    )
    }
