"""Collect calibrated WidowX pick-and-place demonstrations in SimplerEnv.

This collector is intentionally separate from the official 24-episode-per-task
evaluation. It resets an official task, moves the task objects to continuously
sampled training positions, executes a privileged waypoint expert, and keeps
successful trajectories with exact simulator RGB-D calibration.

The stored ``action`` is the Bridge-style learning target: reached TCP delta in
robot-base coordinates plus an absolute gripper label (0 closed, 1 open). The
exact action passed to SimplerEnv is also stored as ``commanded_action``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill2_real2sim.envs  # noqa: F401 - registers SimplerEnv environments
import numpy as np
from scipy.spatial.transform import Rotation

from simpler_env.utils.env.observation_utils import (
    get_image_depth_intrinsics_from_maniskill2_obs_dict,
)
from simpler_env.utils.visualization import write_video


CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
CAMERA_NAME = "3rd_view_camera"


@dataclass(frozen=True)
class TaskSpec:
    env_name: str
    robot: str
    scene_name: str
    robot_xy: tuple[float, float]
    max_episode_steps: int
    source_x_range: tuple[float, float]
    source_y_range: tuple[float, float]
    target_x_range: tuple[float, float] | None
    target_y_range: tuple[float, float] | None
    min_object_separation: float
    grasp_z_offset: float
    placement_clearance: float


TASKS: dict[str, TaskSpec] = {
    "stack_cube": TaskSpec(
        env_name="StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        robot="widowx",
        scene_name="bridge_table_1_v1",
        robot_xy=(0.147, 0.028),
        max_episode_steps=60,
        source_x_range=(-0.245, -0.075),
        source_y_range=(-0.085, 0.085),
        target_x_range=(-0.245, -0.075),
        target_y_range=(-0.085, 0.085),
        min_object_separation=0.10,
        grasp_z_offset=0.005,
        placement_clearance=0.006,
    ),
    "carrot_on_plate": TaskSpec(
        env_name="PutCarrotOnPlateInScene-v0",
        robot="widowx",
        scene_name="bridge_table_1_v1",
        robot_xy=(0.147, 0.028),
        max_episode_steps=60,
        source_x_range=(-0.245, -0.075),
        source_y_range=(-0.085, 0.085),
        target_x_range=(-0.245, -0.075),
        target_y_range=(-0.085, 0.085),
        min_object_separation=0.10,
        grasp_z_offset=0.012,
        placement_clearance=0.008,
    ),
    "spoon_on_towel": TaskSpec(
        env_name="PutSpoonOnTableClothInScene-v0",
        robot="widowx",
        scene_name="bridge_table_1_v1",
        robot_xy=(0.147, 0.028),
        max_episode_steps=60,
        source_x_range=(-0.245, -0.075),
        source_y_range=(-0.085, 0.085),
        target_x_range=(-0.245, -0.075),
        target_y_range=(-0.085, 0.085),
        min_object_separation=0.10,
        grasp_z_offset=0.010,
        placement_clearance=0.008,
    ),
    "eggplant_in_basket": TaskSpec(
        env_name="PutEggplantInBasketScene-v0",
        robot="widowx_sink_camera_setup",
        scene_name="bridge_table_1_v2",
        robot_xy=(0.127, 0.060),
        max_episode_steps=120,
        source_x_range=(-0.125, -0.085),
        source_y_range=(0.170, 0.240),
        target_x_range=None,
        target_y_range=None,
        min_object_separation=0.0,
        grasp_z_offset=0.018,
        placement_clearance=0.015,
    ),
}


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).copy()


def _pose_in_robot_base(base_env, world_pose) -> np.ndarray:
    pose_at_base = base_env.agent.robot.pose.inv().transform(world_pose)
    euler = Rotation.from_quat(np.asarray(pose_at_base.q)[[1, 2, 3, 0]]).as_euler("xyz")
    return np.concatenate([np.asarray(pose_at_base.p), euler]).astype(np.float32)


def _wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    delta = delta.copy()
    delta[3:6] = (delta[3:6] + np.pi) % (2 * np.pi) - np.pi
    return delta


def _sample_xy(rng: np.random.Generator, x_range, y_range) -> np.ndarray:
    return np.array(
        [rng.uniform(*x_range), rng.uniform(*y_range)],
        dtype=np.float64,
    )


def _randomize_training_layout(base_env, spec: TaskSpec, rng: np.random.Generator) -> dict[str, list[float]]:
    """Move objects off the finite official evaluation grid.

    The registered task reset selects an official configuration first because
    that is how these environments are implemented. We then continuously sample
    new positions and update the task's post-settle reference state.
    """

    source_xy = _sample_xy(rng, spec.source_x_range, spec.source_y_range)
    target_xy = np.asarray(base_env.target_obj_pose.p[:2], dtype=np.float64)

    if spec.target_x_range is not None and spec.target_y_range is not None:
        for _ in range(100):
            candidate = _sample_xy(rng, spec.target_x_range, spec.target_y_range)
            if np.linalg.norm(candidate - source_xy) >= spec.min_object_separation:
                target_xy = candidate
                break
        else:
            raise RuntimeError("Could not sample separated source and target positions")

    actors_to_move = [(base_env.episode_source_obj, source_xy)]
    if spec.target_x_range is not None:
        actors_to_move.append((base_env.episode_target_obj, target_xy))

    for actor, desired_xy in actors_to_move:
        current_com_xy = np.asarray(
            actor.pose.transform(actor.cmass_local_pose).p[:2], dtype=np.float64
        )
        pose = actor.pose
        pose.set_p(np.asarray(pose.p) + np.r_[desired_xy - current_com_xy, 0.025])
        actor.set_pose(pose)
        actor.set_velocity(np.zeros(3))
        actor.set_angular_velocity(np.zeros(3))

    base_env._settle(0.75)

    base_env.episode_obj_xyzs_after_settle = [
        np.asarray(obj.pose.p).copy() for obj in base_env.episode_objs
    ]
    source_idx = base_env.episode_objs.index(base_env.episode_source_obj)
    target_idx = base_env.episode_objs.index(base_env.episode_target_obj)
    base_env.episode_source_obj_xyz_after_settle = (
        base_env.episode_obj_xyzs_after_settle[source_idx].copy()
    )
    base_env.episode_target_obj_xyz_after_settle = (
        base_env.episode_obj_xyzs_after_settle[target_idx].copy()
    )
    base_env._initialize_episode_stats()

    return {
        "source_xy": np.asarray(base_env.source_obj_pose.p[:2]).tolist(),
        "target_xy": np.asarray(base_env.target_obj_pose.p[:2]).tolist(),
    }


class EpisodeRecorder:
    def __init__(self, env, obs, instruction: str):
        self.env = env
        self.base_env = env.unwrapped
        self.obs = obs
        self.instruction = instruction
        self.terminated = False
        self.truncated = False
        self.last_info: dict[str, Any] = {}
        self.frames: dict[str, list[Any]] = {
            "rgb": [],
            "depth": [],
            "camera_intrinsics": [],
            "camera_extrinsics": [],
            "proprio": [],
            "tcp_pose_world": [],
            "source_pose_world": [],
            "target_pose_world": [],
            "action": [],
            "commanded_action": [],
            "is_grasping": [],
        }

    @property
    def stopped(self) -> bool:
        return self.terminated or self.truncated

    def step(self, commanded_action: np.ndarray, gripper_open: float) -> None:
        if self.stopped:
            return

        image, depth, intrinsic, extrinsic = (
            get_image_depth_intrinsics_from_maniskill2_obs_dict(
                self.env, self.obs, camera_name=CAMERA_NAME
            )
        )
        state_before = _pose_in_robot_base(self.base_env, self.base_env.tcp.pose)

        self.frames["rgb"].append(_to_numpy(image).astype(np.uint8))
        self.frames["depth"].append(_to_numpy(depth).astype(np.float32))
        self.frames["camera_intrinsics"].append(_to_numpy(intrinsic).astype(np.float32))
        self.frames["camera_extrinsics"].append(_to_numpy(extrinsic).astype(np.float32))
        self.frames["proprio"].append(state_before)
        self.frames["tcp_pose_world"].append(
            np.concatenate([self.base_env.tcp.pose.p, self.base_env.tcp.pose.q]).astype(np.float32)
        )
        self.frames["source_pose_world"].append(
            np.concatenate([self.base_env.source_obj_pose.p, self.base_env.source_obj_pose.q]).astype(np.float32)
        )
        self.frames["target_pose_world"].append(
            np.concatenate([self.base_env.target_obj_pose.p, self.base_env.target_obj_pose.q]).astype(np.float32)
        )
        self.frames["commanded_action"].append(commanded_action.astype(np.float32))
        self.frames["is_grasping"].append(
            bool(self.base_env.agent.check_grasp(self.base_env.episode_source_obj))
        )

        self.obs, _, self.terminated, self.truncated, self.last_info = self.env.step(
            commanded_action.astype(np.float32)
        )

        state_after = _pose_in_robot_base(self.base_env, self.base_env.tcp.pose)
        reached_delta = _wrap_angle_delta(state_after - state_before)
        self.frames["action"].append(
            np.concatenate([reached_delta, [gripper_open]]).astype(np.float32)
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            key: np.stack(values)
            for key, values in self.frames.items()
        }


class WidowXWaypointExpert:
    def __init__(self, recorder: EpisodeRecorder, max_translation: float = 0.025):
        self.recorder = recorder
        self.base_env = recorder.base_env
        self.max_translation = max_translation

    def _arm_target_world(self) -> np.ndarray:
        arm = self.base_env.agent.controller.controllers["arm"]
        return np.asarray(
            self.base_env.agent.robot.pose.transform(arm._target_pose).p,
            dtype=np.float64,
        )

    def _command(self, goal_world: np.ndarray, gripper_open: float) -> np.ndarray:
        delta = np.asarray(goal_world, dtype=np.float64) - self._arm_target_world()
        norm = np.linalg.norm(delta)
        if norm > self.max_translation:
            delta *= self.max_translation / norm
        gripper_env = 1.0 if gripper_open >= 0.5 else -1.0
        return np.concatenate([delta, np.zeros(3), [gripper_env]])

    def move_to(
        self,
        goal_world: np.ndarray,
        gripper_open: float,
        tolerance: float = 0.012,
        max_steps: int = 16,
    ) -> bool:
        for _ in range(max_steps):
            if self.recorder.stopped:
                break
            error = np.linalg.norm(np.asarray(goal_world) - self.base_env.tcp.pose.p)
            if error <= tolerance:
                return True
            action = self._command(goal_world, gripper_open)
            self.recorder.step(action, gripper_open)
        return np.linalg.norm(np.asarray(goal_world) - self.base_env.tcp.pose.p) <= tolerance

    def hold(self, gripper_open: float, steps: int) -> None:
        for _ in range(steps):
            if self.recorder.stopped:
                return
            self.recorder.step(
                self._command(self._arm_target_world(), gripper_open),
                gripper_open,
            )

    def run(self, spec: TaskSpec) -> bool:
        source = np.asarray(self.base_env.source_obj_pose.p, dtype=np.float64)
        approach_source = source + np.array([0.0, 0.0, 0.12])
        grasp_source = source + np.array([0.0, 0.0, spec.grasp_z_offset])

        if not self.move_to(approach_source, gripper_open=1.0):
            return False
        if not self.move_to(grasp_source, gripper_open=1.0, tolerance=0.009):
            return False
        self.hold(gripper_open=0.0, steps=7)

        if not self.base_env.agent.check_grasp(self.base_env.episode_source_obj):
            return False

        source_to_tcp = (
            np.asarray(self.base_env.source_obj_pose.p)
            - np.asarray(self.base_env.tcp.pose.p)
        )
        lift_tcp = np.asarray(self.base_env.tcp.pose.p).copy()
        lift_tcp[2] += 0.14
        if not self.move_to(lift_tcp, gripper_open=0.0):
            return False

        target = np.asarray(self.base_env.target_obj_pose.p, dtype=np.float64)
        source_half_z = abs(float(self.base_env.episode_source_obj_bbox_world[2])) / 2
        target_half_z = abs(float(self.base_env.episode_target_obj_bbox_world[2])) / 2
        desired_source = target.copy()
        desired_source[2] += source_half_z + target_half_z + spec.placement_clearance
        desired_tcp = desired_source - source_to_tcp

        above_target = desired_tcp.copy()
        above_target[2] = max(lift_tcp[2], desired_tcp[2] + 0.10)
        if not self.move_to(above_target, gripper_open=0.0):
            return False
        if not self.move_to(desired_tcp, gripper_open=0.0, tolerance=0.012):
            return False

        self.hold(gripper_open=1.0, steps=7)
        if not self.recorder.stopped:
            retreat = np.asarray(self.base_env.tcp.pose.p).copy()
            retreat[2] += 0.10
            self.move_to(retreat, gripper_open=1.0, max_steps=8)

        return bool(self.base_env.evaluate().get("success", False))


def _validate_episode(arrays: dict[str, np.ndarray]) -> None:
    lengths = {key: value.shape[0] for key, value in arrays.items()}
    if len(set(lengths.values())) != 1 or not lengths:
        raise ValueError(f"Unaligned trajectory arrays: {lengths}")
    if arrays["rgb"].dtype != np.uint8:
        raise ValueError(f"RGB must be uint8, got {arrays['rgb'].dtype}")
    if arrays["depth"].dtype != np.float32 or not np.all(np.isfinite(arrays["depth"])):
        raise ValueError("Depth must be finite float32 metric depth")
    if arrays["camera_intrinsics"].shape[-2:] != (3, 3):
        raise ValueError(f"Bad intrinsic shape: {arrays['camera_intrinsics'].shape}")
    if arrays["camera_extrinsics"].shape[-2:] != (4, 4):
        raise ValueError(f"Bad extrinsic shape: {arrays['camera_extrinsics'].shape}")
    if arrays["proprio"].shape[-1] != 6 or arrays["action"].shape[-1] != 7:
        raise ValueError("Expected 6D POS_EULER proprio and 7D EEF_POS action")
    if arrays["commanded_action"].shape[-1] != 7:
        raise ValueError("Expected a 7D SimplerEnv command")


def _make_env(spec: TaskSpec):
    return gym.make(
        spec.env_name,
        obs_mode="rgbd",
        robot=spec.robot,
        control_mode=CONTROL_MODE,
        scene_name=spec.scene_name,
        control_freq=5,
        sim_freq=500,
        max_episode_steps=spec.max_episode_steps,
        camera_cfgs={"add_segmentation": True},
        # Overlay RGB is intentionally disabled: RGB and depth must describe
        # the same simulated geometry in a calibrated training example.
        rgb_overlay_path=None,
    )


def collect_attempt(
    task_name: str,
    seed: int,
    output_root: Path,
    save_failures: bool,
) -> tuple[bool, Path | None]:
    spec = TASKS[task_name]
    rng = np.random.default_rng(seed)
    env = _make_env(spec)
    try:
        obs, reset_info = env.reset(
            seed=seed,
            options={
                # This seed configuration is only a loading scaffold. The
                # actors are moved to continuous training positions below.
                "obj_init_options": {"episode_id": 0},
                "robot_init_options": {
                    "init_xy": np.asarray(spec.robot_xy),
                    "init_rot_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                },
            },
        )
        base_env = env.unwrapped
        layout = _randomize_training_layout(base_env, spec, rng)
        obs = env.get_obs()
        instruction = base_env.get_language_instruction()

        recorder = EpisodeRecorder(env, obs, instruction)
        expert = WidowXWaypointExpert(recorder)
        expert_success = expert.run(spec)
        success = bool(base_env.evaluate().get("success", False)) and expert_success
        arrays = recorder.arrays()
        _validate_episode(arrays)

        if not success and not save_failures:
            return False, None

        status = "success" if success else "failure"
        output_dir = output_root / task_name / status
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"seed_{seed:08d}.npz"
        metadata = {
            "schema_version": 1,
            "task": task_name,
            "env_name": spec.env_name,
            "instruction": instruction,
            "seed": seed,
            "success": success,
            "training_layout": layout,
            "reset_info": {key: str(value) for key, value in reset_info.items()},
            "robot": spec.robot,
            "control_mode": CONTROL_MODE,
            "control_freq_hz": 5,
            "sim_freq_hz": 500,
            "camera_name": CAMERA_NAME,
            "depth_unit": "meter",
            "extrinsic_field": "Simulator intrinsic_cv/extrinsic_cv stored without modification",
            "action_field": "Reached TCP delta POS_EULER in robot base + absolute gripper (0 closed, 1 open)",
            "commanded_action_field": "World-aligned target delta rotvec + SimplerEnv gripper (-1 closed, +1 open)",
            "official_eval_layout_used_for_training": False,
        }
        np.savez_compressed(
            output_path,
            **arrays,
            instruction=np.asarray(instruction),
            success=np.asarray(success),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        write_video(str(output_path.with_suffix(".mp4")), list(arrays["rgb"]), fps=5)
        return success, output_path
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), default="stack_cube")
    parser.add_argument("--num-successes", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/widowx_simpler_npz"),
    )
    parser.add_argument("--save-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("DISPLAY", "")
    args = parse_args()
    if args.num_successes < 1 or args.max_attempts < 1:
        raise ValueError("--num-successes and --max-attempts must be positive")

    successes = 0
    for attempt in range(args.max_attempts):
        seed = args.seed + attempt
        success, path = collect_attempt(
            task_name=args.task,
            seed=seed,
            output_root=args.output_dir,
            save_failures=args.save_failures,
        )
        print(
            f"attempt={attempt + 1}/{args.max_attempts} seed={seed} "
            f"success={success} path={path}"
        )
        successes += int(success)
        if successes >= args.num_successes:
            break

    if successes < args.num_successes:
        raise RuntimeError(
            f"Collected {successes}/{args.num_successes} successes in "
            f"{args.max_attempts} attempts. Inspect saved failures before scaling."
        )


if __name__ == "__main__":
    main()
