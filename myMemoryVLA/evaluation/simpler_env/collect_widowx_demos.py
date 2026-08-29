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

# Keep Vulkan device selection deterministic in headless GPU containers.
# SAPIEN will still select the CUDA-visible GPU explicitly in _make_env().
os.environ.setdefault("NODEVICE_SELECT", "1")
os.environ.setdefault("DISPLAY", "")
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

from training_tasks import TRAINING_TASKS, TrainTask

# Importing registers the Gym environment.
import training_pick_place_env  # noqa: F401


CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
CAMERA_NAME = "3rd_view_camera"

ROBOT_XY = (0.147, 0.028)
TRAINING_ENV_NAME = "WidowXTrainingPickPlace-v0"
ROBOT_NAME = "widowx"


class UnstableLayoutError(RuntimeError):
    """Raised when randomized actors do not settle near sampled poses."""


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


def _settle_with_robot_held(base_env, seconds: float) -> None:
    """Advance physics while holding the robot at its current joint pose."""

    controllers = base_env.agent.controller.controllers
    for controller in controllers.values():
        controller.set_drive_targets(controller.qpos.copy())

    sim_steps = int(base_env.sim_freq * seconds)
    for _ in range(sim_steps):
        base_env.agent.before_simulation_step()
        base_env._scene.step()

    # Resetting the controller synchronizes its Cartesian target with the TCP's
    # post-settle pose, preventing a large correction on the first env.step().
    base_env.agent.set_control_mode(CONTROL_MODE)


def _randomize_training_layout(
    base_env, spec: TrainTask, rng: np.random.Generator
) -> dict[str, list[float]]:
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

    carrot_actor = None
    sink_actor = None
    sink_desired_xy = None
    for actor, desired_xy in actors_to_move:
        current_pose_xy = np.asarray(actor.pose.p[:2], dtype=np.float64)

        pose = actor.pose
        if (
            spec.source_asset == "bridge_carrot_generated_modified"
            and actor is base_env.episode_source_obj
        ):
            # The carrot collision mesh can intersect the table and receive a
            # large impulse when teleported only millimetres above it. Drop it
            # from a safe height in a stable, yaw-only grasp orientation.
            yaw = rng.uniform(-np.deg2rad(12.0), np.deg2rad(12.0))
            pose.set_q(
                np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            )
            pose.set_p(
                np.asarray(pose.p)
                + np.r_[desired_xy - current_pose_xy, 0.06]
            )
            actor.lock_motion(0, 0, 0, 1, 1, 0)
            carrot_actor = actor
        elif spec.target_asset == "sink" and actor is base_env.episode_target_obj:
            # Leave the already-settled sink asleep while the carrot drops.
            # Moving it first wakes its non-convex body during the long settle.
            sink_actor = actor
            sink_desired_xy = desired_xy
            continue
        else:
            pose.set_p(
                np.asarray(pose.p)
                + np.r_[desired_xy - current_pose_xy, 0.025]
            )

        actor.set_pose(pose)
        actor.set_velocity(np.zeros(3))
        actor.set_angular_velocity(np.zeros(3))

    _settle_with_robot_held(base_env, 0.75)
    if carrot_actor is not None:
        carrot_actor.lock_motion(0, 0, 0, 0, 0, 0)
        carrot_actor.set_pose(carrot_actor.pose)
        carrot_actor.set_velocity(np.zeros(3))
        carrot_actor.set_angular_velocity(np.zeros(3))
    if sink_actor is not None:
        sink_pose = sink_actor.pose
        sink_xy = np.asarray(sink_pose.p[:2], dtype=np.float64)
        sink_pose.set_p(
            np.asarray(sink_pose.p)
            + np.r_[sink_desired_xy - sink_xy, 0.002]
        )
        sink_actor.set_pose(sink_pose)
        sink_actor.set_velocity(np.zeros(3))
        sink_actor.set_angular_velocity(np.zeros(3))
    if carrot_actor is not None or sink_actor is not None:
        _settle_with_robot_held(base_env, 0.25)

    actual_source = np.asarray(base_env.source_obj_pose.p)
    actual_target = np.asarray(base_env.target_obj_pose.p)
    source_drift = np.linalg.norm(actual_source[:2] - source_xy)
    target_drift = np.linalg.norm(actual_target[:2] - target_xy)
    source_speed = np.linalg.norm(base_env.episode_source_obj.velocity)
    source_angular_speed = np.linalg.norm(
        base_env.episode_source_obj.angular_velocity
    )
    if (
        source_drift > 0.025
        or target_drift > 0.025
        or source_speed > 1e-3
        or source_angular_speed > 1e-2
    ):
        raise UnstableLayoutError(
            "Unstable randomized layout: "
            f"source_drift={source_drift:.4f}, "
            f"target_drift={target_drift:.4f}, "
            f"source_speed={source_speed:.4f}, "
            f"source_angular_speed={source_angular_speed:.4f}"
        )


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
            "gripper_open_state": [],
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
        gripper_open_state = np.float32(
            1.0
            - np.clip(
                float(self.base_env.agent.get_gripper_closedness()),
                0.0,
                1.0,
            )
        )

        self.frames["rgb"].append(_to_numpy(image).astype(np.uint8))
        self.frames["depth"].append(_to_numpy(depth).astype(np.float32))
        self.frames["camera_intrinsics"].append(_to_numpy(intrinsic).astype(np.float32))
        self.frames["camera_extrinsics"].append(_to_numpy(extrinsic).astype(np.float32))
        self.frames["proprio"].append(state_before)
        self.frames["gripper_open_state"].append(gripper_open_state)
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

        translation_delta = state_after[:3] - state_before[:3]
        rotation_delta = np.zeros(3, dtype=np.float32)

        reached_delta = np.concatenate(
            [translation_delta, rotation_delta]
        )
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
        self.failure_reason: str | None = None

    def _fail(self, reason: str) -> bool:
        self.failure_reason = reason
        return False

    def _arm_target_world(self) -> np.ndarray:
        arm = self.base_env.agent.controller.controllers["arm"]
        return np.asarray(
            self.base_env.agent.robot.pose.transform(arm._target_pose).p,
            dtype=np.float64,
        )

    def _command(self, goal_world: np.ndarray, gripper_open: float) -> np.ndarray:
        delta_world = np.asarray(goal_world, dtype=np.float64) - self._arm_target_world()
        norm = np.linalg.norm(delta_world)
        if norm > self.max_translation:
            delta_world *= self.max_translation / norm
        base_quat_wxyz = np.asarray(self.base_env.agent.robot.pose.q)
        delta_base = Rotation.from_quat(
            base_quat_wxyz[[1, 2, 3, 0]]
        ).inv().apply(delta_world)
        gripper_env = 1.0 if gripper_open >= 0.5 else -1.0
        return np.concatenate([delta_base, np.zeros(3), [gripper_env]])

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

    def _contact_reached(
        self, goal_world: np.ndarray, xy_tol: float, z_tol: float
    ) -> bool:
        tcp = np.asarray(self.base_env.tcp.pose.p)
        goal = np.asarray(goal_world)
        return bool(
            np.linalg.norm(tcp[:2] - goal[:2]) <= xy_tol
            and goal[2] - 0.01 <= tcp[2] <= goal[2] + z_tol
        )

    def _source_centered_over_target(self) -> bool:
        source = np.asarray(self.base_env.source_obj_pose.p)
        target = np.asarray(self.base_env.target_obj_pose.p)
        source_half = np.abs(self.base_env.episode_source_obj_bbox_world) / 2
        target_half = np.abs(self.base_env.episode_target_obj_bbox_world) / 2
        clearance = target_half[:2] - source_half[:2] + 0.005
        return bool(
            np.all(clearance >= 0)
            and np.all(np.abs(source[:2] - target[:2]) <= clearance)
            and source[2] >= target[2]
        )

    def _source_inside_target(self) -> bool:
        source = np.asarray(self.base_env.source_obj_pose.p)
        target = np.asarray(self.base_env.target_obj_pose.p)
        source_half = np.abs(self.base_env.episode_source_obj_bbox_world) / 2
        target_half = np.abs(self.base_env.episode_target_obj_bbox_world) / 2
        offset = np.abs(source - target)
        clearance = target_half[:2] - source_half[:2] + 0.005
        return bool(
            np.all(clearance >= 0)
            and np.all(offset[:2] <= clearance)
            and offset[2] + source_half[2] <= target_half[2] + 0.02
        )

    def _grasp_source(self, spec: TrainTask, max_attempts: int = 2) -> bool:
        """Acquire the source, retrying once from its post-contact pose."""

        for attempt in range(max_attempts):
            source = np.asarray(self.base_env.source_obj_pose.p, dtype=np.float64)
            approach_source = source + np.array([0.0, 0.0, 0.12])
            grasp_source = source + np.array([0.0, 0.0, spec.grasp_z_offset])

            if not self.move_to(
                approach_source, gripper_open=1.0, max_steps=20
            ):
                self.failure_reason = "approach_source"
            else:
                reached_grasp = self.move_to(
                    grasp_source,
                    gripper_open=1.0,
                    tolerance=0.009,
                    max_steps=20,
                )
                if not reached_grasp and not self._contact_reached(
                    grasp_source, xy_tol=0.012, z_tol=0.04
                ):
                    self.failure_reason = "reach_grasp"
                else:
                    self.hold(gripper_open=0.0, steps=12)
                    if self.base_env.agent.check_grasp(
                        self.base_env.episode_source_obj
                    ):
                        self.failure_reason = None
                        return True
                    self.failure_reason = "close_gripper"

            if attempt + 1 < max_attempts and not self.recorder.stopped:
                self.hold(gripper_open=1.0, steps=4)
                retreat = np.asarray(self.base_env.tcp.pose.p).copy()
                retreat[2] += 0.10
                self.move_to(
                    retreat, gripper_open=1.0, max_steps=10
                )

        return False

    def run(self, spec: TrainTask) -> bool:
        if not self._grasp_source(spec):
            return False

        source_to_tcp = (
            np.asarray(self.base_env.source_obj_pose.p)
            - np.asarray(self.base_env.tcp.pose.p)
        )
        lift_tcp = np.asarray(self.base_env.tcp.pose.p).copy()
        lift_tcp[2] += 0.14
        if not self.move_to(lift_tcp, gripper_open=0.0):
            return self._fail("lift")
        if not self.base_env.agent.check_grasp(self.base_env.episode_source_obj):
            return self._fail("dropped_during_lift")

        target = np.asarray(self.base_env.target_obj_pose.p, dtype=np.float64)
        source_half_z = abs(float(self.base_env.episode_source_obj_bbox_world[2])) / 2
        target_half_z = abs(float(self.base_env.episode_target_obj_bbox_world[2])) / 2
        desired_source = target.copy()
        if spec.relation == "in":
            desired_source[2] += spec.placement_clearance
        else:
            desired_source[2] += (
                source_half_z + target_half_z + spec.placement_clearance
            )
        desired_tcp = desired_source - source_to_tcp

        above_target = desired_tcp.copy()
        above_target[2] = max(lift_tcp[2], desired_tcp[2] + 0.10)
        if not self.move_to(above_target, gripper_open=0.0):
            return self._fail("move_above_target")
        if not self.base_env.agent.check_grasp(self.base_env.episode_source_obj):
            return self._fail("dropped_during_transport")
        reached_place = self.move_to(
            desired_tcp, gripper_open=0.0, tolerance=0.012
        )
        if not reached_place:
            safely_centered = (
                spec.relation == "in"
                and self._source_centered_over_target()
            )
            if not safely_centered and not self._source_inside_target():
                return self._fail("reach_place")

        self.hold(gripper_open=1.0, steps=7)
        if not self.recorder.stopped:
            retreat = np.asarray(self.base_env.tcp.pose.p).copy()
            retreat[2] += 0.10
            self.move_to(retreat, gripper_open=1.0, max_steps=8)

        success = bool(self.base_env.evaluate().get("success", False))
        if not success:
            return self._fail("final_evaluation")
        return True


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
    if arrays["gripper_open_state"].shape != (len(arrays["action"]),):
        raise ValueError("gripper_open_state must have shape [T]")
    if not np.all(
        (arrays["gripper_open_state"] >= 0.0)
        & (arrays["gripper_open_state"] <= 1.0)
    ):
        raise ValueError("gripper_open_state must be in [0, 1]")
    if not np.all(np.isfinite(arrays["action"])):
        raise ValueError("Actions must be finite")
    if not np.allclose(arrays["action"][:, 3:6], 0.0):
        raise ValueError("Fixed-wrist expert must have zero rotation labels")

    translation_norms = np.linalg.norm(arrays["action"][:, :3], axis=1)
    invalid_steps = np.flatnonzero(translation_norms > 0.05)
    if invalid_steps.size:
        step = int(invalid_steps[0])
        raise ValueError(
            f"Implausible reached translation at step {step}: "
            f"{arrays['action'][step, :3]} (norm={translation_norms[step]:.6f} m)"
        )


def _make_env(task):
    return gym.make(
        TRAINING_ENV_NAME,
        source_obj_name=task.source_asset,
        target_obj_name=task.target_asset,
        relation=task.relation,
        instruction=task.instruction,
        obs_mode="rgbd",
        robot=ROBOT_NAME,
        control_mode=CONTROL_MODE,
        scene_name="bridge_table_1_v1",
        control_freq=5,
        sim_freq=500,
        renderer_kwargs={
            "offscreen_only": True,
            "device": os.environ.get("SAPIEN_RENDER_DEVICE", "cuda"),
        },
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=None,
    )


def collect_attempt(
    task_name: str,
    seed: int,
    output_root: Path,
    save_failures: bool,
) -> tuple[bool, Path | None]:
    spec = TRAINING_TASKS[task_name]
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
                    "init_xy": np.asarray(ROBOT_XY),
                    "init_rot_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                },
            },
        )
        base_env = env.unwrapped
        layout = _randomize_training_layout(base_env, spec, rng)
        # Refresh after moving the actors, then apply the same RGB-D conversion
        # that the observation wrapper applies to reset() and step().
        obs = env.get_wrapper_attr("observation")(env.unwrapped.get_obs())
        instruction = base_env.get_language_instruction()

        recorder = EpisodeRecorder(env, obs, instruction)
        max_translation = (
            0.020
            if spec.source_asset == "bridge_carrot_generated_modified"
            else 0.025
        )
        expert = WidowXWaypointExpert(
            recorder, max_translation=max_translation
        )
        expert_success = expert.run(spec)
        success = bool(base_env.evaluate().get("success", False)) and expert_success
        if not success:
            print(f"task={task_name} seed={seed} expert_failure={expert.failure_reason}")
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
            "env_name": TRAINING_ENV_NAME,
            "instruction": instruction,
            "seed": seed,
            "success": success,
            "training_layout": layout,
            "reset_info": {key: str(value) for key, value in reset_info.items()},
            "robot": ROBOT_NAME,
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
            K=arrays["camera_intrinsics"],
            E=arrays["camera_extrinsics"],
            expert_action=arrays["action"],
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
    parser.add_argument("--num-successes", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/widowx_simpler_npz"),
    )
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument(
        "--task",
        choices=("all", *TRAINING_TASKS),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("DISPLAY", "")
    args = parse_args()
    if args.num_successes < 1 or args.max_attempts < 1:
        raise ValueError("--num-successes and --max-attempts must be positive")

    if not TRAINING_TASKS:
        raise ValueError("TRAINING_TASKS is empty")

    task_names = list(TRAINING_TASKS) if args.task == "all" else [args.task]
    incomplete_tasks: list[tuple[str, int]] = []
    for task_name in task_names:
        successes = 0

        for attempt in range(args.max_attempts):
            try:
                success, path = collect_attempt(
                    task_name=task_name,
                    seed=args.seed + attempt,
                    output_root=args.output_dir,
                    save_failures=args.save_failures,
                )
            except UnstableLayoutError as exc:
                success, path = False, None
                print(
                    f"task={task_name} attempt={attempt + 1}/"
                    f"{args.max_attempts} rejected_layout={exc}"
                )

            print(
                f"task={task_name} attempt={attempt + 1}/{args.max_attempts} "
                f"success={success} path={path}"
            )
            successes += int(success)

            if successes >= args.num_successes:
                break

        if successes < args.num_successes:
            incomplete_tasks.append((task_name, successes))

    if incomplete_tasks:
        raise RuntimeError(
            f"Incomplete tasks: {incomplete_tasks}; requested "
            f"{args.num_successes} successes per task in {args.max_attempts} attempts"
        )


if __name__ == "__main__":
    main()
