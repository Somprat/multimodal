import gymnasium as gym
import mani_skill2.envs
import numpy as np

from simpler_env.utils.env.observation_utils import (
    get_image_depth_intrinsics_from_maniskill2_obs_dict,
)


def evaluate_pickcube(
    model,
    seed: int = 0,
    max_steps: int = 200,
    instruction: str = "pick up the cube and move it to the goal position",
):
    # Panda is the default PickCube-v0 robot, so robot_uids can be omitted
    # if your installed ManiSkill2 version does not accept it.
    env = gym.make(
        "PickCube-v0",
        robot_uids="panda",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        reward_mode="sparse",
        max_episode_steps=max_steps,
    )

    obs, info = env.reset(seed=seed)

    print("Action space:", env.action_space)
    print("Observation keys:", obs.keys())
    print("Available cameras:", obs["image"].keys())

    assert env.action_space.shape == (7,)

    model.reset(instruction)
    success = False

    try:
        for timestep in range(max_steps):
            # This helper already supports arbitrary camera names when the
            # name is supplied explicitly.
            image, depth, intrinsic, extrinsic = (
                get_image_depth_intrinsics_from_maniskill2_obs_dict(
                    env,
                    obs,
                    camera_name="base_camera",
                )
            )

            tcp_position = np.asarray(
                env.unwrapped.tcp.pose.p,
                dtype=np.float32,
            )

            raw_action, action = model.step(
                image=image,
                depth=depth,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                task_description=instruction,
                current_position=tcp_position,
                episode_first_frame="True" if timestep == 0 else "False",
            )

            env_action = np.concatenate(
                [
                    action["world_vector"],
                    action["rot_axangle"],
                    action["gripper"],
                ]
            ).astype(np.float32)

            if env_action.shape != (7,):
                raise ValueError(
                    f"Expected environment action (7,), got {env_action.shape}"
                )

            if not np.all(np.isfinite(env_action)):
                raise ValueError(f"Non-finite environment action: {env_action}")

            env_action = np.clip(
                env_action,
                env.action_space.low,
                env.action_space.high,
            )

            obs, reward, terminated, truncated, info = env.step(env_action)

            success = bool(info.get("success", terminated))

            print(
                f"step={timestep} "
                f"reward={float(reward):.3f} "
                f"gripper={env_action[6]:.1f} "
                f"success={success}"
            )

            if terminated or truncated:
                break

    finally:
        model.finish_episode(success)
        env.close()

    return success