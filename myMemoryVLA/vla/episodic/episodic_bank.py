import torch
from transformers import CLIPModel, CLIPProcessor
from dataclasses import dataclass
import torch.nn.functional as F
from typing import Optional
import torch.nn as nn
from typing import Any

# Main memoryVLA integration
# 1. Initialize. at the beginning. Never reset
# 2. The begining of each episode, do the start ep (append the instruction and scene embedding)
# 3. In the beginning, get top k most similar. 
# 4. For each timestep in the episode do the cross attention with the current observation
# 5. At the end, do the summarize + append the success, cog and per


@dataclass
class MemoryUnit:
    instruction_embedding: str
    scene_embedding: torch.tensor
    success: bool
    cog_mem_bank: Any
    per_mem_bank: Any


@dataclass
class BankEntry:
    timestep: Optional[torch.Tensor]
    feat: torch.tensor
    image_embedding: Optional[torch.Tensor]
    task_tags: tuple[str, ...]
    position: Optional[torch.Tensor] = None




class EpisodicMemBank(nn.Module):
    def __init__(self, max_steps: int = 5,
                 kick_method: str = "fifo",
                 top_k: int = 4,
                 novelty_threshold = 0.8,
                 dataloader_type: str = "stream",
                 group_size: int = 16,
                 token_size: int = 256,
                 mem_length: int = 16,
                 retrieval_layers: int = 2,
                 use_timestep_pe: bool = True,
                 fusion_type: str = 'gate',
                 consolidate_type: str = 'tome',
                 update_fused: bool = False,
                 query_retrieval_mode: str = "off",
                 query_retrieval_top_k: int = 4
                 
                 ):
        super().__init__()
        self.max_steps = max_steps
        self.kick_method = kick_method
        self.top_k = top_k
        self.novelty_threshold = novelty_threshold

        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.token_size = token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused
        self.query_retrieval_mode = query_retrieval_mode
        self.query_retrieval_top_k = query_retrieval_top_k
        # 1. make self.bank become a dictionary
        # 2. At the end of each episode, self.bank[episode_id].cog_summary = pool(memoryvla's cogmembank[episode_id])
        # design like a function that takes in that episodes' cog mem bank


        
        self.bank = {}

        self.episode_id = 1

    def start_episode(self, 
                      # should be the initial scene
                      image: torch.Tensor,
                      instruction: torch.Tensor):
        image_inputs = self.clip_processor(
            images=list(image),
            return_tensors="pt",
    
        )
        text_inputs = self.clip_processor(
            text=list(instruction),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        image_outputs = self.clip_model.get_image_features(**image_inputs)
        text_outputs = self.clip_model.get_text_features(**text_inputs)

        memory_unit = MemoryUnit(
            instruction_embedding=text_outputs,
            scene_embedding=image_outputs,
            success=True,
            cog_mem_bank=None,
            per_mem_bank=None
        )
        self.bank[self.episode_id] = memory_unit
        self.episode_id += 1
        
        return self.episode_id -1
    

    def summarize_mem_bank(self, episode_banks):
        if not episode_banks:
            return None
        timestep = torch.tensor(0)
        mean_feat = torch.stack([entry.feat for entry in episode_banks], dim=0).mean(dim=0)


        first_scene_embedding = episode_banks[0].image_embedding
        task_tags = episode_banks[0].task_tags

        summary = BankEntry(
            timestep=timestep,
            feat=mean_feat,
            image_embedding=first_scene_embedding,
            task_tags=task_tags
        )
        return summary
        # each feat = [1, N, D]

    
        

    def end_episode(self,
                    success: bool,
                    episode_cog_banks: list,
                    episode_per_banks: list):
        summarized_cog = self.summarize_mem_bank(episode_cog_banks)
        summarized_per = self.summarize_mem_bank(episode_per_banks)

        self.bank[self.episode_id-1].success = success
        self.bank[self.episode_id-1].cog_mem_bank = summarized_cog
        self.bank[self.episode_id-1].per_mem_bank = summarized_per


    

    def novelty_update(self):
        scores = []
        T = len(self.bank)
        for i in range(T-2):
            bank1 = self.bank[i+1].instruction_embedding.flatten()
            bank2 = self.bank[i+2].instruction_embedding.flatten()
            score = F.cosine_similarity(bank1, bank2, dim=0).item()


            scores.append(score)

         
        index_max_score = int(torch.tensor(scores).argmax().item())
        fuesed_instruction = (self.bank[index_max_score+1].instruction_embedding + self.bank[index_max_score+2].instruction_embedding)/2
        fused_image =(self.bank[index_max_score+1].scene_embedding + self.bank[index_max_score+2].scene_embedding)/2

        self.bank[index_max_score] = MemoryUnit(
            instruction_embedding=fuesed_instruction,
            scene_embedding=fused_image
        )

        

    def kick_memory(self):
        if self.kick_method == "fifo":
            min_eid = min(self.bank.keys())
            self.bank.pop(min_eid)
        elif self.kick_method == "novelty":
            self.novelty_update()
            # will be something like compare cosine similarity of the adjacent memory
        else:
            raise ValueError("Can only chose between fifo and novelty")

    # retrieval mechanism
    # only use the success episodes for now. might have 2 separate routes of cross attention. 
    # One for success and one for failure. Fuse with learned gate
    # context = current + success_gate * success_context \
    #              - failure_gate * failure_context

    # only run at the start of the episode. the same set of memories would be applied across 1 episode 
    def instruction_score(self, memory_unit, current_instruction):
        text_inputs = self.clip_processor(
            text=list(current_instruction),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )        
        current_embedding = self.clip_model.get_text_features(**text_inputs).flatten()

        episode_embedding = memory_unit.instruction_embedding.flatten()

        score = F.cosine_similarity(current_embedding, episode_embedding, dim=0)
        return _clamp01(score)


    def image_score(self, memory_unit, initial_frame):
        image_inputs = self.clip_processor(
            images=list(initial_frame),
            return_tensors="pt",
        )
        current_embedding = self.clip_model.get_image_features(**image_inputs).float().detach().flatten()

        episode_embedding = memory_unit.scene_embedding.float().detach().flatten()
        score = F.cosine_similarity(current_embedding, episode_embedding, dim=0)

        return _clamp01(score)



    def retrieve(self, 
                      current_instruction: str,
                      initial_frame: torch.Tensor):
        # comparing the episode instruction
        # comparing the initial scene embedding.

        scores = []
        for episode_id, memory_unit in self.bank.items():
            semantic_score = self.instruction_score(memory_unit, current_instruction)
            image_score = self.image_score(memory_unit, initial_frame)

            total_score = (semantic_score + image_score)/2
            scores.append((total_score, episode_id))
        scores.sort(key=lambda item: item[0], reverse=True)

        return [self.bank[eid] for _, eid in scores[:self.top_k]]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))