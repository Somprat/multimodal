from collections import deque
from typing import Optional, Sequence
import os
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import torch
from transforms3d.euler import euler2axangle

from vla import load_vla
from evaluation.simpler_env.adaptive_ensemble import AdaptiveEnsembler

class VLAInference:
    def __init__(
        self,
        saved_model_path: str = '',
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 0,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] = [224, 224],
        future_action_window_size: int = 15,
        action_dim: int = 7,
        action_model_type: str = "DiT-L",
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        use_bf16: bool = False,
        action_ensemble = True,
        adaptive_ensemble_alpha = 0.1,
        experiment_mode: str = "full",
        **kwargs,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                # Set 7 for widowx_bridge to fix the window size of motion scale between each frame. see appendix in our paper for details
                action_ensemble_horizon = 7
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                # Set 2 for google_robot to fix the window size of motion scale between each frame. see appendix in our paper for details
                action_ensemble_horizon = 2
            self.sticky_gripper_num_repeat = 10
        elif policy_setup == "maniskill_panda":
            unnorm_key = (
                "maniskill_dataset_converted_externally_to_rlds"
                if unnorm_key is None
                else unnorm_key
            )
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 1
            self.sticky_gripper_num_repeat = 1

        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        if experiment_mode not in {"baseline", "full"}:
            raise ValueError(
                f"Unsupported experiment_mode={experiment_mode!r}; expected 'baseline' or 'full'."
            )
        self.experiment_mode = experiment_mode

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"experiment_mode: {experiment_mode} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.vla = load_vla(
          saved_model_path,
          load_for_training=False, 
          action_model_type=action_model_type,
          future_action_window_size=future_action_window_size,
          action_dim=action_dim,
          use_bf16=use_bf16,
          experiment_mode=experiment_mode,
          **kwargs,
        )

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        device = torch.device("cuda")
        self.vla = self.vla.to(device).to(dtype).eval()
        print(f"Model loaded to {device} with dtype {dtype}.")

        torch.cuda.empty_cache()

        self.cfg_scale = cfg_scale

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = 1.0 if self.policy_setup == "maniskill_panda" else None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.depth_history = deque(maxlen=self.horizon)
        self.intrinsic_history = deque(maxlen=self.horizon)
        self.extrinsic_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        self.num_depth_history = 0
        self.num_intrinsic_history = 0
        self.num_extrinsic_history = 0

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _add_depth_to_history(self, depth: np.float64) -> None:
        self.depth_history.append(depth)
        self.num_depth_history = min(self.num_depth_history+1, self.horizon)

    def _add_intrinsic_to_history(self, intrinsic: np.float64) -> None:
        self.intrinsic_history.append(intrinsic)
        self.num_intrinsic_history = min(self.num_intrinsic_history+1, self.horizon)

    def _add_extrinsic_to_history(self, extrinsic: np.float64) -> None:
        self.extrinsic_history.append(extrinsic)
        self.num_extrinsic_history = min(self.num_extrinsic_history+1, self.horizon)

    def finish_episode(self, success):
        self.vla.finish_episode(success)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.depth_history.clear()
        self.intrinsic_history.clear()
        self.extrinsic_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.num_depth_history = 0
        self.num_intrinsic_history = 0
        self.num_extrinsic_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = 1.0 if self.policy_setup == "maniskill_panda" else None


    def step(
        self, image: np.ndarray, depth: np.float64, intrinsic:np.float64,
        extrinsic:np.float64,
            task_description: Optional[str] = None,
            current_position: Optional[list] = None,
            episode_first_frame: str = 'False',
            *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            assert task_description is not None, "task_description should not be None"
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        if self.experiment_mode == "full":
            if depth is None:
                raise ValueError("pointcloud mode requires depth")
            if intrinsic is None:
                raise ValueError("pointcloud mode requires camera intrinsics")
            if extrinsic is None:
                raise ValueError("pointcloud mode requires camera extrinsics")
        self._add_image_to_history(self._resize_image(image))
        self._add_depth_to_history(depth)
        self._add_intrinsic_to_history(intrinsic)
        self._add_extrinsic_to_history(extrinsic)
        image: Image.Image = Image.fromarray(image)
        spatial_inputs = (
            {"depth": depth, "intrinsics": intrinsic, "extrinsics": extrinsic}
            if self.experiment_mode == "full"
            else {}
        )
        raw_actions, normalized_actions = self.vla.predict_action(
            image=image,
            instruction=self.task_description,
            unnorm_key=self.unnorm_key,
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
            episode_first_frame=episode_first_frame,
            current_position=current_position,
            **spatial_inputs,
            )

        # binarize the gripper action
        # to remove
        # raw_actions[:, 6] = np.clip(raw_actions[:, 6], -1, 1)
        # raw_actions[:, 6] = np.where(raw_actions[:, 6] < 0.5, 0, 1)



        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]
        
        a = np.asarray(raw_actions[0], dtype=np.float32)

        if a.shape != (7,):
            raise ValueError(f"Expected action shape (7,) got {a.shape}")

        if not np.all(np.isfinite(a)):
            raise ValueError(f"Action contains non-finite values: {a}")

        a = np.clip(a, -1.0, 1.0)

            
        raw_action = {
            "world_vector": np.array(a[:3]),
            "rotation_delta": np.array(a[3:6]),
            "open_gripper": np.array(a[6:7]),
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)

        if self.policy_setup == "maniskill_panda":
            action["rot_axangle"] = (action_rotation_delta * self.action_scale).astype(np.float32)
        else:
            roll, pitch, yaw = action_rotation_delta
            axes, angles = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = axes * angles
            action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        elif self.policy_setup == "maniskill_panda":
            score = float(a[6])

            if score > 0.5:
                gripper_command = 1.0
            elif score < -0.5:
                gripper_command = -1.0
            else:
                # Predictions around zero correspond to uncertainty/the terminal
                # placeholder, so retain the preceding absolute command.
                gripper_command = float(self.previous_gripper_action)

            self.previous_gripper_action = gripper_command
            action["gripper"] = np.array(
                [gripper_command],
                dtype=np.float32,
            )
        action["terminate_episode"] = np.array([0.0], dtype=np.float32)
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
