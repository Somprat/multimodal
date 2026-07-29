#!/usr/bin/env python3
"""Create a tiny renderer-independent ManiSkill-shaped RLDS spatial fixture."""
import argparse
from pathlib import Path
import numpy as np
import tensorflow_datasets as tfds

class TinyManiskillSpatial(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    def __init__(self, *, episodes=2, steps=4, height=64, width=64, **kwargs):
        self.episodes, self.steps, self.height, self.width = episodes, steps, height, width
        super().__init__(**kwargs)
    def _info(self):
        # tfds = tensorflow dataset
        step = {
            
            "observation": {
                "image": tfds.features.Tensor(shape=(self.height, self.width, 3), dtype=np.uint8),
                "depth": tfds.features.Tensor(shape=(self.height, self.width, 1), dtype=np.float32),
                "camera_intrinsics": tfds.features.Tensor(shape=(3, 3), dtype=np.float32),
                "proprio": tfds.features.Tensor(shape=(8,), dtype=np.float32),
            },
            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
            "language_instruction": tfds.features.Text(),
            "is_first": np.bool_, "is_last": np.bool_, "is_terminal": np.bool_,
        }
        return self.dataset_info_from_configs(features=tfds.features.FeaturesDict({
            "steps": tfds.features.Dataset(step),
            "episode_metadata": {"episode_id": tfds.features.Text(), "source": tfds.features.Text()},
        }))
    def _split_generators(self, dl_manager):
        del dl_manager
        return {"train": self._generate_examples()}
    def _generate_examples(self):
        for episode in range(self.episodes):
            episode_id = f"episode_{episode:03d}"
            yield episode_id, {"steps": [self._step(episode, i) for i in range(self.steps)],
                "episode_metadata": {"episode_id": episode_id, "source": "synthetic_maniskill_camera_fixture"}}
    def _step(self, episode, index):
        yy, xx = np.mgrid[:self.height, :self.width]
        mask = (xx-self.width*(.35+.08*index))**2 + (yy-self.height*(.45+.03*episode))**2 <= (min(self.height,self.width)*.12)**2
        rgb = np.empty((self.height,self.width,3), np.uint8)
        rgb[...,0] = np.clip(30+xx*120/self.width,0,255); rgb[...,1] = np.clip(45+yy*100/self.height,0,255); rgb[...,2] = 80
        rgb[mask] = (40,210,90)
        depth = .75+.25*yy.astype(np.float32)/max(self.height-1,1); depth[mask] = .48+.01*index; depth = depth[...,None]
        sx, sy = self.width/640, self.height/480
        intrinsic = np.array([[623.588*sx,0,319.501*sx],[0,623.588*sy,239.545*sy],[0,0,1]],np.float32)
        proprio=np.zeros(8,np.float32); proprio[:3]=(.30+.01*index,-.1,.2); proprio[-1]=1
        action=np.zeros(7,np.float32); action[0]=.01; action[-1]=1
        return {"observation":{"image":rgb,"depth":depth,"camera_intrinsics":intrinsic,"proprio":proprio},
            "action":action,"language_instruction":"move the green object to the right",
            "is_first":index==0,"is_last":index==self.steps-1,"is_terminal":index==self.steps-1}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-root",type=Path,default=Path("artifacts/tiny_maniskill_rlds"))
    p.add_argument("--episodes",type=int,default=2); p.add_argument("--steps",type=int,default=4); p.add_argument("--height",type=int,default=64); p.add_argument("--width",type=int,default=64); a=p.parse_args()
    if min(a.episodes,a.steps,a.height,a.width)<=0: p.error("sizes must be positive")
    b=TinyManiskillSpatial(data_dir=a.output_root.resolve(),episodes=a.episodes,steps=a.steps,height=a.height,width=a.width)
    b.download_and_prepare(); print(f"Created {b.info.full_name}\nDataset directory: {b.data_dir}")
if __name__ == "__main__": main()
