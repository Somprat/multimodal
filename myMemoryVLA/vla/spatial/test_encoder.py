from encoder import PointCloudSpatialEncoder
from memory import SpatialMemBank


import torch


batch_size = 3

points = torch.rand(size=[batch_size, 3, 3])
proprio = torch.rand(size=[batch_size, 3])
camera = torch.rand(size=[batch_size, 3])


encoder = PointCloudSpatialEncoder(spatial_token_size=5,
                                   num_spatial_tokens=10,
                                   point_dim=3,
                                   # proprio = Internal states like joint and imu
                                   proprio_dim=3,
                                   camera_dim=3,
                                   hidden_dim=128,
                                   num_heads=4,
                                   max_points=10)

features = encoder(points=points, proprio=proprio, camera=camera)

mem_bank = SpatialMemBank(max_steps=5,
                          num_spatial_tokens=features.shape[1],
                          spatial_token_size=features.shape[2],
                          novelty_threshold=0.15,
                          merge_rate=0.25,
                          update_mode = "novelty")
num_iter = 2
for i in range(num_iter):
    memory = mem_bank(features)
    assert memory.shape[0] == batch_size
    assert memory.shape[1] == i+1
    assert memory.shape[2] == 10
    assert memory.shape[3] == 5
    print(memory.shape)


    # print(memory)

def input_without_propcam(points):


    features = encoder(points=points)
    memory = mem_bank(features)
    assert memory.shape[0] == batch_size
    assert memory.shape[1] == 3
    assert memory.shape[2] == 10
    assert memory.shape[3] == 5
    print(memory.shape)

input_without_propcam(points)