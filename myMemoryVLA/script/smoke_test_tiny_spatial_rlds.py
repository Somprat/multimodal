#!/usr/bin/env python3
"""Run one tiny RLDS step through geometry, encoder, memory, and retrieval."""
import argparse
from pathlib import Path
import tensorflow_datasets as tfds
import torch
from vla.spatial.encoder import PointCloudSpatialEncoder
from vla.spatial.geometry import depth_to_points
from vla.spatial.memory import SpatialMemBank
from vla.spatial.retrieval import MemoryRecord, MemoryRetriever, RetrievalQuery

def main():
    p=argparse.ArgumentParser(); p.add_argument("dataset_dir",type=Path); a=p.parse_args()
    b=tfds.builder_from_directory(str(a.dataset_dir.resolve())); episode=next(iter(b.as_dataset(split="train").take(1))); step=next(iter(episode["steps"].take(1))); o=step["observation"]
    depth=torch.as_tensor(o["depth"].numpy()).unsqueeze(0); intrinsic=torch.as_tensor(o["camera_intrinsics"].numpy()).unsqueeze(0); proprio=torch.as_tensor(o["proprio"].numpy()).unsqueeze(0)
    points,mask=depth_to_points(depth,intrinsic); encoder=PointCloudSpatialEncoder(32,4,proprio_dim=8,hidden_dim=32,num_heads=4,max_points=512).eval()
    with torch.no_grad(): tokens=encoder(points,proprio=proprio,point_mask=mask)
    memory=SpatialMemBank(2,4,32)(tokens); embedding=tokens.mean(1).squeeze(0)
    result=MemoryRetriever().retrieve(RetrievalQuery("where was the green object?",embedding,torch.zeros(3),1.0,"semantic_spatial_recent",object_ids=("green_object",)),[MemoryRecord("step_0",embedding=embedding,position=torch.zeros(3),timestamp=0,object_ids=("green_object",),task_tags=("semantic_spatial_recent",),modality="spatial")],1)
    assert tokens.shape==(1,4,32) and memory.shape==(1,1,4,32) and result[0].memory.id=="step_0"
    print(f"dataset={b.info.full_name}\ndepth={tuple(depth.shape)} intrinsics={tuple(intrinsic.shape)}\npoints={tuple(points.shape)} valid_points={int(mask.sum())}\nspatial_tokens={tuple(tokens.shape)} memory={tuple(memory.shape)}\nretrieval={result[0].memory.id} score={result[0].score:.4f}\nTiny spatial RLDS workflow: OK")
if __name__ == "__main__": main()
