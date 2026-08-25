import numpy as np
from mani_skill2_real2sim.envs.custom_scenes.base_env import (
    CustomBridgeObjectsInSceneEnv,
)
from mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import PutOnInSceneEnv
from mani_skill2_real2sim.utils.registration import register_env


@register_env("WidowXTrainingPickPlace-v0", max_episode_steps=120)
class WidowXTrainingPickPlaceEnv(PutOnInSceneEnv, CustomBridgeObjectsInSceneEnv):
    """One reusable environment for manifest-selected source/target objects."""

    def __init__(
        self,
        source_obj_name: str,
        target_obj_name: str,
        relation: str,
        instruction: str,
        **kwargs,
    ):
        if relation not in {"in", "on"}:
            raise ValueError(f"Unsupported relation: {relation}")
        self.source_obj_name = source_obj_name
        self.target_obj_name = target_obj_name
        self.relation = relation
        self.instruction = instruction
        super().__init__(
            model_ids=[source_obj_name, target_obj_name],
            **kwargs,
        )

    def reset(self, seed=None, options=None):
        options = {} if options is None else options.copy()
        options["model_ids"] = [self.source_obj_name, self.target_obj_name]

        obj_options = options.get("obj_init_options", {}).copy()
        obj_options["source_obj_id"] = 0
        obj_options["target_obj_id"] = 1
        obj_options.setdefault(
            "init_xys",
            np.array([[-0.18, -0.06], [-0.18, 0.06]], dtype=np.float64),
        )
        obj_options.setdefault(
            "init_rot_quats",
            np.array([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float64),
        )
        options["obj_init_options"] = obj_options
        return super().reset(seed=seed, options=options)

    def _load_model(self):
        self.episode_objs = []
        for (model_id, model_scale) in zip(
            self.episode_model_ids, self.episode_model_scales
        ):
            density = self.model_db[model_id].get("density", 1000)

            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = model_id
            self.episode_objs.append(obj)

    def get_language_instruction(self, **kwargs):
        return self.instruction

    def evaluate(self, **kwargs):
        if self.relation == "on":
            return super().evaluate(**kwargs)

        source_pos = np.asarray(self.source_obj_pose.p)
        target_pos = np.asarray(self.target_obj_pose.p)
        source_half = np.abs(self.episode_source_obj_bbox_world) / 2
        target_half = np.abs(self.episode_target_obj_bbox_world) / 2
        offset = np.abs(source_pos - target_pos)

        xy_clearance = target_half[:2] - source_half[:2] + 0.005
        fits_target = bool(np.all(xy_clearance >= 0))
        inside_xy = fits_target and bool(np.all(offset[:2] <= xy_clearance))
        inside_z = bool(offset[2] + source_half[2] <= target_half[2] + 0.02)
        is_grasped = bool(self.agent.check_grasp(self.episode_source_obj))
        success = inside_xy and inside_z and not is_grasped

        moved_correct_obj = bool(
            np.linalg.norm(
                self.episode_source_obj_xyz_after_settle[:2] - source_pos[:2]
            )
            > 0.03
        )
        moved_wrong_obj = bool(
            np.linalg.norm(
                self.episode_target_obj_xyz_after_settle[:2] - target_pos[:2]
            )
            > 0.03
        )
        success = success and moved_correct_obj
        self.episode_stats["moved_correct_obj"] = moved_correct_obj
        self.episode_stats["moved_wrong_obj"] = moved_wrong_obj
        self.episode_stats["src_on_target"] = success
        self.episode_stats["is_src_obj_grasped"] |= is_grasped

        return dict(
            moved_correct_obj=moved_correct_obj,
            moved_wrong_obj=moved_wrong_obj,
            is_src_obj_grasped=is_grasped,
            consecutive_grasp=self.episode_stats["consecutive_grasp"],
            src_on_target=success,
            episode_stats=self.episode_stats,
            success=success,
        )
