import torch

def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]

def get_imageanddepth_from_maniskill2_obs_dict(env, obs, camera_name=None):

    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()

    image = obs["image"][camera_name]["rgb"]
    depth = obs["image"][camera_name]["depth"]
    return image, torch.tensor(depth)

def get_image_depth_intrinsics_from_maniskill2_obs_dict(env, obs, camera_name=None):

    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()

    image = obs["image"][camera_name]["rgb"]
    depth = obs["image"][camera_name]["depth"]

    if "sensor_param" in obs and camera_name in obs["sensor_param"]:
        intrinsics = obs["sensor_param"][camera_name]["intrinsic_cv"]
        extrinsics = obs["sensor_param"][camera_name]["extrinsic_cv"]
    elif "camera_param" in obs and camera_name in obs["camera_param"]:
        intrinsics = obs["camera_param"][camera_name]["intrinsic_cv"]
        extrinsics = obs["camera_param"][camera_name]["extrinsic_cv"]

    else:
        sensor = env.unwrapped._sensors[camera_name]
        intrinsics = sensor.get_params()["intrinsic_cv"]
        extrinsics = sensor.get_params()["extrinsic_cv"]

    

    return image, torch.tensor(depth), torch.tensor(intrinsics), torch.tensor(extrinsics)

