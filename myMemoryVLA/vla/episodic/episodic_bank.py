import torch
from transformers import CLIPModel, CLIPProcessor
from dataclasses import dataclass
import torch.nn.functional as F
# only use clip features for comparing cosine similarity.
# feature for storing the summaries. DO the mean pooling from each memory we store


# Initial CLIP instruction/image embeddings only for episode retrieval.
# cog_summary, per_summary, and spatial_summary for cross-attention.
# The summaries can still preserve the initial state by including it during pooling. 
# If the start state is especially important, add a learned “initial timestep” embedding 
# or reserve one summary token for the first timestep rather than storing separate initial features.



@dataclass
class MemoryUnit:
    instruction_embedding: str
    scene_embedding: torch.tensor
    success: bool


class EpisodicMemBank:
    def __init__(self, max_steps: int = 5,
                 kick_method: str = "fifo",
                 top_k: int = 4,
                 novelty_threshold = 0.8):
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

        self.bank = []
        

    def start_episode(self, 
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
            success=True
        )
        self.bank.append(memory_unit)


    def end_episode(self,
                    success: bool):
        self.bank.success = success

    def novelty_update(self):
        scores = []
        T = len(self.bank)
        for i in range(T-1):
            bank1 = self.bank[i].instruction_embedding.float().flatten()
            bank2 = self.bank[i+1].instruction_embedding.float().flatten()
            score = F.cosine_similarity(bank1, bank2)

            scores.append(score)

        
        index_max_score = int(torch.tensor(scores).argmax().item())
        fuesed_instruction = (self.bank[index_max_score].instruction_embedding + self.bank[index_max_score+1].instruction_embedding)/2
        fused_image =(self.bank[index_max_score].scene_embedding + self.bank[index_max_score+1].scene_embedding)/2

        self.bank[index_max_score] = MemoryUnit(
            instruction_embedding=fuesed_instruction,
            scene_embedding=fused_image
        )

        

    def kick_memory(self):
        if self.kick_method == "fifo":
            for key in self.bank:
                self.bank[key] = self.bank[key][-(len(self.bank.instructions_embedding) -1):]
        elif self.kick_method == "novelty":
            self.novelty_update()
            # will be something like compare cosine similarity of the adjacent memory
        else:
            raise ValueError("Can only chosse between fifo and novelty")

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
        current_embedding = self.clip_model.get_text_features(**text_inputs).float().flatten()

        episode_embedding = memory_unit.instructions_embedding.float().flatten()

        score = F.cosine_similarity(current_embedding, episode_embedding)
        return _clamp01(score)

    def image_score(self, memory_unit, initial_frame):
        image_inputs = self.clip_processor(
            images=list(initial_frame),
            return_tensors="pt",
        )
        current_embedding = self.clip_model.get_image_features(**image_inputs).float().flatten()

        episode_embedding = memory_unit.scene_embedding.float().detach()
        score = F.cosine_similarity(current_embedding, episode_embedding)

        return _clamp01(score)


    def process_batch(self, 
                      current_instruction: str,
                      initial_frame: torch.Tensor):
        # comparing the episode instruction
        # comparing the initial scene embedding.

        score = []
        for memory_unit in self.bank:
            semantic_score = self.instruction_score(memory_unit, current_instruction)
            image_score = self.image_score(memory_unit, initial_frame)

            total_score = (semantic_score + image_score)/2
            score.append(total_score)
        # implement cross_attentio 

    



def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))