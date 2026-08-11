import numpy as np
import os

dir_path = "../../models/model_b/probe_data"


with np.load(os.path.joint(dir_path, os.listdir(dir_path)[0])) as f:
    for key in f:
        print(key)
for npz_file in os.listdir(dir_path):
    npz_path = os.path.join(dir_path, npz_file)
    with np.load(npz_file) as file:
        action = file["action"]
        T = action.shape[0]

        if action.shape[1] != 7:
            print("DOF action is not equal to 7")


        