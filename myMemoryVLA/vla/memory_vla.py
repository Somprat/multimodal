from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
import torch.nn.functional as F
from transformers import LlamaTokenizerFast
from transformers import CLIPModel, CLIPProcessor

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector
from .spatial import geometry
from .spatial.encoder import PointCloudSpatialEncoder as SpatialPointCloudEncoder
from .spatial import retrieval
from .episodic.episodic_bank import EpisodicMemBank
from .spatial.modal_retrieval import ModalRetrievalQuery, ModalMemoryRecord, ModalMemoryRetriever

from action_model.action_model import ActionModel
from action_model.models import DiT
from dataclasses import dataclass

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

@dataclass
class BankEntry:
    timestep: Optional[torch.Tensor]
    feat: torch.tensor
    image_embedding: Optional[torch.Tensor]
    task_tags: tuple[str, ...]
    position: Optional[torch.Tensor] = None



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.to(next(self.mlp.parameters()).device)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.mlp.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class CrossTransformerBlock(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)

        # Feed‑Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(self,
                query: torch.Tensor, # (B, N, D)
                k: torch.Tensor, # (B, M, D)
                v: torch.Tensor, # (B, M, D)
                ) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(k)
        v = self.v_proj(v)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        # residual + LN
        x = self.attn_norm(query + attn_out)

        # FFN + LN
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + ffn_out)

# compress visual tokens to the size of 256 i.e. Preprocessing
class BottleneckSE(nn.Module):
    def __init__(self, C_in, C_mid, C_out):
        super().__init__()
        self.C_in = C_in
        self.C_mid = C_mid
        self.C_out = C_out

        # takes in whatever dimension -> convert it to c_mid = 512.
        self.reduce = nn.Conv2d(C_in, C_mid, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

        self.excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_mid, C_mid//16, 1),
            nn.ReLU(),
            nn.Conv2d(C_mid//16, C_mid, 1),
            nn.Sigmoid()
        )

        self.expand = nn.Conv2d(C_mid, C_out, 1, bias=False)


    def forward(self, x):
        _b, _n, _c = x.shape
        _h = _w = int(math.sqrt(_n))
        assert _h * _h == _n, "Input feature has no spatial structure"

        x = x.reshape(_b, _h, _w, _c).permute(0, 3, 1, 2)  # (B, C_in, H, W)
        # main reduce mechanism: reduce whatever c_in is to c_mid = 512 and c_out = 256
        z = self.act(self.reduce(x))
        w = self.excite(z)

        # c_mid -> c_out
        final = self.expand(z * w)
        final = final.reshape(_b, self.C_out, _n).permute(0, 2, 1)
        return final


class GateFusion(nn.Module):
    def __init__(self, dim: int, preserve_first_input: bool = False):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        if preserve_first_input:
            # Newly added branches must not corrupt a pretrained policy before
            # they have learned useful features. sigmoid(6) ~= 0.9975, so the
            # initial result is effectively x1 while gradients still flow.
            nn.init.zeros_(self.proj.weight)
            nn.init.constant_(self.proj.bias, 6.0)
        else:
            nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
            nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(
            self.proj(
                torch.cat([x1, x2],
                dim=-1)
            )
        )

        fused = scale * x1 + (1 - scale) * x2
        return fused
    

MODALITY_SCORES_SWEEP = [{
    "cog": {
        "token": 0.8,
        "task": 0.10,
        "time": 0.10,
        "position": 0.00,
    },
    "per": {
        "token": 0.8,
        "task": 0.10,
        "time": 0.10,
        "position": 0.00,
    },
    "spatial": {
        "token": 0.65,
        "task": 0.10,
        "time": 0.05,
        "position": 0.20,
    }},
    {
    "cog": {
        "token": 0.75,
        "task": 0.15,
        "time": 0.10,
        "position": 0.00,
    },
    "per": {
        "token": 0.75,
        "task": 0.15,
        "time": 0.10,
        "position": 0.00,
    },
    "spatial": {
        "token": 0.65,
        "task": 0.10,
        "time": 0.05,
        "position": 0.20,
    }},
    {
    "cog": {
        "token": 0.75,
        "task": 0.10,
        "time": 0.15,
        "position": 0.00,
    },
    "per": {
        "token": 0.75,
        "task": 0.10,
        "time": 0.15,
        "position": 0.00,
    },
    "spatial": {
        "token": 0.65,
        "task": 0.10,
        "time": 0.05,
        "position": 0.20,
    }},
    {
    "cog": {
        "token": 0.75,
        "task": 0.15,
        "time": 0.10,
        "position": 0.00,
    },
    "per": {
        "token": 0.75,
        "task": 0.15,
        "time": 0.10,
        "position": 0.00,
    },
    "spatial": {
        "token": 0.80,
        "task": 0.05,
        "time": 0.05,
        "position": 0.10,
    }},
]

EXPERIMENT_MODES = frozenset(
    {"baseline", "episodic", "query", "query_episodic", "full"}
)


class CogMemBank(nn.Module):
    modality = 'cog'
    def __init__(self,
                 dataloader_type: str,
                 group_size: int,
                 token_size: int,
                 mem_length: int = 16,
                 retrieval_layers: int = 2,
                 use_timestep_pe: bool = True,
                 fusion_type: str = 'gate',
                 consolidate_type: str = 'tome',
                 update_fused: bool = False,
                 query_retrieval_mode: str = "off",
                 query_retrieval_top_k: int = 4,
                 use_query_classifier: bool = False,
                 modality_weights: dict = MODALITY_SCORES_SWEEP,
                 modality_weights_index: int = 1,
                 ):
        super().__init__()
        assert dataloader_type in ('stream', 'group')
        assert fusion_type in ('gate', 'add')
        assert consolidate_type in ('fifo', 'tome')
        if query_retrieval_mode not in ("off", "query", "shuffled", "by_modal"):
            raise ValueError(
                "query_retrieval_mode must be one of: off, query, shuffled, by_modal"
            )
        if query_retrieval_top_k < 1:
            raise ValueError("query_retrieval_top_k must be at least 1")

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
        self.query_retriever = retrieval.MemoryRetriever()
        self.modal_retriever = ModalMemoryRetriever(
            use_classifier=use_query_classifier
        )

        self.retrieval_blocks = nn.ModuleList([
            CrossTransformerBlock(self.token_size)
            for _ in range(self.retrieval_layers)
        ])
        self.modality_weights = modality_weights
        self.modality_weights_index = modality_weights_index

        if self.fusion_type == 'gate':
            self.gate_fusion_blocks = GateFusion(self.token_size)

        if self.use_timestep_pe:
            self.timestep_encoder = TimestepEmbedder(
                self.token_size,
                frequency_embedding_size=self.token_size // 4)
        else:
            self.timestep_encoder = None

        self.reset()

    def reset(self):
        # bank[episode_id] = [(timestep, feat[N,D]), ...]
        self.bank = {}
        self.eid_stream = None

    def clear_episode(self, episode_id):
        self.bank.pop(episode_id, None)

    @torch.no_grad()
    def _consolidate_with_token_merge(self, episode_id):
        bank = self.bank.get(episode_id, [])
        T = len(bank)
        if T < 2:
            return

        # add the third element to make the structure consistent
        # when the capacity hits, the bank would compare the adjacent memory (collected around the same time)
        feats = [memory.feat for memory in bank]
        positions = [memory.position for memory in bank]

        # with cosine similarity. Then, try to fuse them to keep the bank size bounded
        sims = []
        for i in range(T - 1):
            f1 = feats[i].flatten(1) if feats[i].dim() > 1 else feats[i].unsqueeze(0)
            f2 = feats[i+1].flatten(1) if feats[i+1].dim() > 1 else feats[i+1].unsqueeze(0)
            sims.append(F.cosine_similarity(f1, f2, dim=1).mean().item())

        idx_max = int(torch.tensor(sims).argmax().item())

        memory_i = bank[idx_max]
        memory_j = bank[idx_max + 1]
        fused_feat = 0.5 * (memory_i.feat + memory_j.feat)
        fused_position = 0.5*(memory_i.position + memory_j.position) if memory_i.position is not None and memory_j.position is not None else None
        # normalize this because it would be used in cosine similarity in the future
        if memory_i.image_embedding is not None and memory_j.image_embedding is not None:
            fused_image_embedding = F.normalize(
                0.5 * (memory_i.image_embedding + memory_j.image_embedding),
                dim=0,
            ) # dim =0 is just the tensor's first axis

        else:
            fused_image_embedding = memory_i.image_embedding if memory_i.image_embedding is not None else memory_j.image_embedding

        max_bank = bank[idx_max]
        max_bank.timestep = memory_i.timestep
        max_bank.feat = fused_feat.detach().clone()
        max_bank.image_embedding = fused_image_embedding
        max_bank.position = fused_position

        bank.pop(idx_max + 1)

    # This fn is basically telling what to do when the bank capacity hits
    @torch.no_grad()
    def _memory_consolidate(
            self,
            episode_id,
            feat: torch.Tensor,
            timestep: Optional[torch.Tensor],
            image_embedding: Optional[torch.Tensor] = None,
            position: Optional[torch.Tensor] = None,
            task_tags: tuple[str,...] = None
            ):
        if episode_id not in self.bank:
            self.bank[episode_id] = []

        stored_embedding = (
            image_embedding.detach().clone()
            if image_embedding is not None
            else None
        )

        stored_position = (
            position.detach().clone()
            if position is not None
            else None
        )

        # detach = require_grad = False. stop back propogation.
        # clone = change this stuff later won't change the original one
        self.bank[episode_id].append(BankEntry(
            timestep=timestep,
            feat=feat.detach().clone(),
            image_embedding=image_embedding,
            position=stored_position,
            task_tags=task_tags
        ))
        
        # a dictionary keyed by episode ids are appending the memory unit
        # could be like {1: [mem1, mem2], 2: [mem3, mem4], ...}


        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == 'fifo':
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length:]
            elif self.consolidate_type == "tome":
                self._consolidate_with_token_merge(episode_id)
            else:
                raise NotImplementedError

    def process_batch(
        self,
        tokens: torch.Tensor, # [B, N, D_role]
        episode_ids: np.array,
        timesteps: np.array,
        instructions: Optional[List[str]] = None,
        retrieval_image_embeddings: Optional[torch.Tensor] = None,
        retrieval_query_embeddings: Optional[torch.Tensor] = None,
        positions: Optional[list] = None,
        task_type: Optional[str] = None
    ) -> torch.Tensor:
        assert episode_ids is not None, "episode_ids must be provided during training"

        if self.use_timestep_pe:
            assert timesteps is not None, "timesteps must be provided during training"

        B, N, D = tokens.shape
        outputs = []

        if self.training:
            if self.dataloader_type == 'group':
                self.bank.clear()
                self.eid_stream = None
            elif self.dataloader_type == 'stream':
                first_eid = episode_ids[0]
                if self.eid_stream is not None and self.eid_stream != first_eid:
                    self.clear_episode(self.eid_stream)
                self.eid_stream = first_eid


        for i in range(B):
            instruction = (
                instructions[i]
                if instructions is not None and i < len(instructions)
                else ""
            )
            mode, _ = self.query_retriever.router.route(
                retrieval.RetrievalQuery(text=instruction)
            )
            task_type=mode.value    
            position = positions[i] if positions is not None else None

            # 1) episode management
            eid = episode_ids[i] # eid = episoed id
            if self.training:
                # group is the episodes that are relevant ome together
                if self.dataloader_type == 'group':
                    if i > 0 and i % self.group_size == 0:
                        prev_group_eid = episode_ids[i - self.group_size]
                        self.clear_episode(prev_group_eid)
                # stream = chronological order. The memory persists across batches
                if self.dataloader_type == 'stream':
                    if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                        self.clear_episode(episode_ids[i - 1])
                        self.eid_stream = episode_ids[i]

            # 2) memory retrieval
            working_mem = tokens[i].unsqueeze(0)  # (1, N, D)

            
            hist = self.bank.get(eid, [])
            if len(hist) > 0:
                if self.query_retrieval_mode == "query":
                    hist = self._select_history(
                        hist=hist,
                        working_mem=working_mem,
                        instruction=instruction,
                        timestep=timesteps[i],
                        query_embedding=(
                            retrieval_query_embeddings[i]
                            if retrieval_query_embeddings is not None
                            else None
                        ),
                        current_position=position,
                        task_type=task_type
                    ) # become a list of (id, cognitive features, image_embedding)
                elif self.query_retrieval_mode == "by_modal":
                    hist = self._select_history_by_modals(
                        hist=hist,
                        instruction=instruction,
                        timestep=timesteps[i],
                        tokens = working_mem.squeeze(0),
                        current_position=position,
                        task_type=task_type
                    ) # become a list of (id, cognitive features, image_embedding)

                hist_feats = [memory.feat for memory in hist]
                episode_mem = torch.stack(hist_feats, dim=0).reshape(-1, D).unsqueeze(0)  # (1, T*N, D)

                if self.use_timestep_pe:
                    hist_timesteps = [memory.timestep for memory in hist]
                    hist_timesteps = torch.tensor(hist_timesteps).to(working_mem.device)
                    pe = self.timestep_encoder(hist_timesteps).unsqueeze(0)  # (1, T, D)
                    pe = pe.repeat_interleave(N, dim=1) # (1, T*N, D)
                else:
                    pe = torch.zeros_like(episode_mem)

                # pe = positional encoding. when did this event occur


                query = working_mem
                for block in self.retrieval_blocks:
                    query = block(query, episode_mem + pe, episode_mem)      

                retrieved_episode_mem = query

            else:
                # without history：working memory as episode memory
                retrieved_episode_mem = working_mem  # (1, N, D)

            # 3) memory adaptive fusion
            if self.fusion_type == 'add':
                fused_feats = (working_mem + retrieved_episode_mem) * 0.5
            elif self.fusion_type == 'gate':
                fused_feats = self.gate_fusion_blocks(working_mem, retrieved_episode_mem)

            outputs.append(fused_feats)


            # 4) memory consolidate
            timestep_i = timesteps[i] if self.use_timestep_pe else None
            image_embedding_i = (
                retrieval_image_embeddings[i]
                if retrieval_image_embeddings is not None
                else None
            )


            if self.update_fused:
                self._memory_consolidate(
                    eid,
                    fused_feats.squeeze(0),
                    timestep_i,
                    image_embedding=image_embedding_i,
                    position=positions[i] if positions is not None else None,
                    task_tags=(task_type,)
                )
            else:
                self._memory_consolidate(
                    eid,
                    tokens[i],
                    timestep_i,
                    image_embedding=image_embedding_i,
                    position=positions[i] if positions is not None else None,
                    task_tags=(task_type,)
                )
            

        return torch.cat(outputs, dim=0)  # [B, N, D_role]

    def _select_history(
        self,
        hist,
        working_mem: torch.Tensor,
        instruction: str,
        timestep,
        query_embedding: Optional[torch.Tensor] = None,
        current_position: Optional[list] = None,
        task_type: Optional[str] = None
    ):
        if self.query_retrieval_mode == "off" or len(hist) <= self.query_retrieval_top_k:
            return hist

        if self.query_retrieval_mode == "shuffled":
            indices = torch.randperm(len(hist))[:self.query_retrieval_top_k].tolist()
            return [hist[index] for index in indices]
# need the query position here and task type

        query = retrieval.RetrievalQuery(
            text=instruction,
            embedding=(
                query_embedding
                if query_embedding is not None
                else working_mem.detach().mean(dim=(0, 1))
            ),
            current_time=self._as_float_timestep(timestep),
            modality_hints=("visual",),
            current_position=current_position,
            task_type=task_type
        )

        # needs to find memory positions and its task_type
        records = [
            retrieval.MemoryRecord(
                id=str(index),
                embedding=(
                    memory.image_embedding
                    if memory.image_embedding is not None
                    else memory.feat.detach().mean(dim=0)
                ),
                timestamp=self._as_float_timestep(memory.timestep),
                modality="visual",
                position=memory.position,
                task_tags=memory.task_tags
            )
            
            for index, memory in enumerate(hist)
        ]

        results = self.query_retriever.retrieve(
            query=query,
            memories=records,
            top_k=self.query_retrieval_top_k,
        )

        return [hist[int(result.memory.id)] for result in results]

    def _select_history_by_modals(
            self,
            hist,
            instruction,
            timestep,
            tokens,
            current_position,
            task_type
    ):
        # 1. create a retrieval query but replace the embeddings with bank tokens
        # 2. create a memory class and replace everything
        # 3. create a function that compares the 2 query
        query = ModalRetrievalQuery(
            text=instruction,
            tokens=tokens,
            current_position=current_position,
            current_time=timestep,
            task_type=task_type,
            modality_hints=("visual",),
        )

        memories = [ModalMemoryRecord(
            id=index,
            tokens=memory.feat,
            position=memory.position,
            timestamp=memory.timestep,
            task_tags=memory.task_tags
        ) for index, memory in enumerate(hist)]


        results = self.modal_retriever.retrieve(
            query=query,
            memories=memories,
            weights=self.modality_weights[self.modality_weights_index][self.modality],
            modal = self.modality
        )
        return [hist[int(result.memory.id)] for result in results]





    @staticmethod
    def _as_float_timestep(timestep) -> Optional[float]:
        if timestep is None:
            return None
        if torch.is_tensor(timestep):
            return float(timestep.detach().cpu().item())
        return float(timestep)
    



class PerMemBank(CogMemBank):
    modality='per'
    def __init__(self,
                 dataloader_type: str,
                 group_size: int,
                 token_size: int,
                 mem_length: int = 16,
                 retrieval_layers: int = 2,
                 use_timestep_pe: bool = True,
                 fusion_type: str = 'gate',
                 consolidate_type: str = 'tome',
                 update_fused: bool = False,
                 query_retrieval_mode: str = "off",
                 query_retrieval_top_k: int = 4,
                 current_position: Optional[list] = None,
                 modality_weights_index: int = 1,
                 ):
        super().__init__(
            dataloader_type=dataloader_type,
            group_size=group_size,
            token_size=token_size,
            mem_length=mem_length,
            retrieval_layers=retrieval_layers,
            use_timestep_pe=use_timestep_pe,
            fusion_type=fusion_type,
            consolidate_type=consolidate_type,
            update_fused=update_fused,
            query_retrieval_mode=query_retrieval_mode,
            query_retrieval_top_k=query_retrieval_top_k,
            modality_weights=MODALITY_SCORES_SWEEP,
            modality_weights_index=modality_weights_index,
        )


class SpatialMemBank(CogMemBank):
    modality='spatial'
    def __init__(self,
                 dataloader_type: str,
                 group_size: int,
                 token_size: int,
                 mem_length: int = 16,
                 retrieval_layers: int = 2,
                 use_timestep_pe: bool = True,
                 fusion_type: str = 'gate',
                 consolidate_type: str = 'tome',
                 update_fused: bool = False,
                 query_retrieval_mode: str = "off",
                 query_retrieval_top_k: int = 4,
                 modality_weights_index: int = 1,
                 ):
        super().__init__(
            dataloader_type=dataloader_type,
            group_size=group_size,
            token_size=token_size,
            mem_length=mem_length,
            retrieval_layers=retrieval_layers,
            use_timestep_pe=use_timestep_pe,
            fusion_type=fusion_type,
            consolidate_type=consolidate_type,
            update_fused=update_fused,
            query_retrieval_mode=query_retrieval_mode,
            query_retrieval_top_k=query_retrieval_top_k,
            modality_weights=MODALITY_SCORES_SWEEP,
            modality_weights_index=modality_weights_index,
        )
class SpatialEncoder(nn.Module):
    def __init__(self, spatial_token_size: int, depth_patch_size: int = 16):
        super().__init__()
        self.spatial_token_size = spatial_token_size
        self.depth_patch_size = depth_patch_size

        # encoder for depth. other encders are only 1 linear
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, spatial_token_size, kernel_size=depth_patch_size, stride=depth_patch_size),
            nn.ReLU(inplace=True),
        )
        self.proprio_scalar_encoder = nn.Linear(1, spatial_token_size)
        self.camera_scalar_encoder = nn.Linear(1, spatial_token_size)
        self.depth_modality = nn.Parameter(torch.zeros(1, 1, spatial_token_size))
        self.proprio_modality = nn.Parameter(torch.zeros(1, 1, spatial_token_size))
        self.camera_modality = nn.Parameter(torch.zeros(1, 1, spatial_token_size))

    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)
        assert depth.dim() == 4 and depth.shape[1] == 1, "depth must be [B, 1, H, W], [B, H, W], or [B, H, W, 1]"
        return depth
    
    def forward(
        self,
        depth: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        camera: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        spatial_tokens = []

        # each stuff goes through its encoder and got appended to spatial tokens
        if depth is not None:
            # mske sure that they are in the order of [B, 1, H, W]
            depth = self._normalize_depth(depth)
            depth_tokens = self.depth_encoder(depth).flatten(2).transpose(1, 2)
            spatial_tokens.append(depth_tokens + self.depth_modality)

        if proprio is not None:
            proprio_tokens = self.proprio_scalar_encoder(proprio.flatten(1).unsqueeze(-1))
            spatial_tokens.append(proprio_tokens + self.proprio_modality)

        if camera is not None:
            camera_tokens = self.camera_scalar_encoder(camera.flatten(1).unsqueeze(-1))
            spatial_tokens.append(camera_tokens + self.camera_modality)

        if len(spatial_tokens) == 0:
            return None

        return torch.cat(spatial_tokens, dim=1)



class PointCloudSpatialEncoder(nn.Module):
    """Encode point-cloud observations into a fixed set of spatial tokens."""

    def __init__(
        self,
        spatial_token_size: int,
        num_spatial_tokens: int,
        point_dim: int = 3,
        proprio_dim: Optional[int] = None,
        camera_dim: Optional[int] = None,
        hidden_dim: int = 128,
        num_heads: int = 4,
        max_points: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.spatial_token_size = spatial_token_size
        self.num_spatial_tokens = num_spatial_tokens
        self.max_points = max_points

        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.proprio_proj = (
            nn.Sequential(
                nn.Linear(proprio_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if proprio_dim is not None
            else None
        )
        self.camera_proj = (
            nn.Sequential(
                nn.Linear(camera_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if camera_dim is not None
            else None
        )

        self.query_tokens = nn.Parameter(torch.randn(num_spatial_tokens, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, spatial_token_size)

    def forward(
        self,
        points: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        camera: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            points: Point cloud tensor shaped [B, N, point_dim].
            proprio: Optional proprioception vector shaped [B, proprio_dim].
            camera: Optional camera metadata vector shaped [B, camera_dim].
            point_mask: Optional boolean mask shaped [B, N], where True marks valid points.

        Returns:
            Spatial tokens shaped [B, num_spatial_tokens, spatial_token_size].
        """

        if points.ndim != 3:
            raise ValueError(f"points must be shaped [B, N, point_dim], got {tuple(points.shape)}")

        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        points, point_mask = self._maybe_subsample(points, point_mask)

        point_tokens = self.point_mlp(points)

        context = self._encode_context(
            batch_size=points.shape[0],
            device=points.device,
            dtype=point_tokens.dtype,
            proprio=proprio,
            camera=camera,
        )
        if context is not None:
            point_tokens = point_tokens + context[:, None, :]

        queries = self.query_tokens[None, :, :].expand(points.shape[0], -1, -1)
        key_padding_mask = None if point_mask is None else ~point_mask.bool()

        spatial_tokens, _ = self.cross_attn(
            query=queries,
            key=point_tokens,
            value=point_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        spatial_tokens = self.output_norm(spatial_tokens)
        return self.output_proj(spatial_tokens)

    def _maybe_subsample(
        self,
        points: torch.Tensor,
        point_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.max_points is None or points.shape[1] <= self.max_points:
            return points, point_mask

        # Deterministic stride sampling keeps this module lightweight and reproducible.
        indices = torch.linspace(
            0,
            points.shape[1] - 1,
            steps=self.max_points,
            device=points.device,
        ).long()
        points = points.index_select(dim=1, index=indices)
        if point_mask is not None:
            point_mask = point_mask.index_select(dim=1, index=indices)
        return points, point_mask


class MemoryVLA(nn.Module):
    def __init__(
        self,
        vlm: PrismaticVLM,
        # equivaletn to per_token_size
        # spatial_token_size: np.int16,
        num_spatial_tokens: int = 16,
        depth_patch_size: int = 16,
        camera_dim: Optional[np.int16] = None ,
        proprio_dim: Optional[np.int16] = None, # should be fixed but shouldn't be required
        max_points_spatial: Optional[int] = 1024, # this should be optional which is good
        action_model_type: str = 'DiT-L',
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        dataloader_type: str = "group",
        group_size: int = 16,
        per_token_size: int = 256,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        fusion_type: str = 'gate',
        consolidate_type: str = 'tome',
        update_fused: bool = False,
        query_retrieval_mode: str = "query",
        query_retrieval_top_k: int = 4,
        experiment_mode: str = "full",
        freeze_vlm: bool = True,
        freeze_action_model: bool = True,
        modality_weights_index: int = 1,
        episodic_max_steps: int = 10,
        episodic_top_k: int = 2,


        max_steps: int = 5,
    
        kick_method: str = "fifo",
        top_k: int = 4,
        novelty_threshold = 0.8,



        **kwargs,
    ) -> None:
        super().__init__()

        if experiment_mode not in EXPERIMENT_MODES:
            raise ValueError(
                "experiment_mode must be one of "
                f"{sorted(EXPERIMENT_MODES)}, got {experiment_mode!r}"
            )
        if not 0 <= modality_weights_index < len(MODALITY_SCORES_SWEEP):
            raise ValueError(
                "modality_weights_index must be between 0 and "
                f"{len(MODALITY_SCORES_SWEEP) - 1}, got {modality_weights_index}"
            )

        self.vlm = vlm
        self.experiment_mode = experiment_mode
        self.use_query = experiment_mode in {"query", "query_episodic", "full"}
        self.use_episodic = experiment_mode in {"episodic", "query_episodic", "full"}
        self.use_spatial = experiment_mode == "full"
        self.freeze_vlm = freeze_vlm
        self.freeze_action_model = freeze_action_model
        self.future_action_window_size = future_action_window_size
        self.use_ema = use_ema
        self.norm_stats = norm_stats

        self.cog_token_size = token_size

        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.per_token_size = per_token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused
        self.query_retrieval_mode = query_retrieval_mode
        self.query_retrieval_top_k = query_retrieval_top_k


        self.max_steps = max_steps
        self.kick_method = kick_method
        self.top_k = top_k
        self.novelty_threshold = novelty_threshold
        self.modality_weights_index = modality_weights_index
        if episodic_max_steps < 1:
            raise ValueError("episodic_max_steps must be at least 1")
        if episodic_top_k < 1:
            raise ValueError("episodic_top_k must be at least 1")
        if episodic_top_k > episodic_max_steps:
            raise ValueError("episodic_top_k cannot exceed episodic_max_steps")
        self.episodic_max_steps = episodic_max_steps
        self.episodic_top_k = episodic_top_k

        

        self.cur_timestep = 0

        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model.requires_grad_(False)
        self.clip_model.eval()

        self.vision_dim = self.vlm.vision_backbone.dino_featurizer.patch_embed.proj.weight.shape[0] + \
                 self.vlm.vision_backbone.siglip_featurizer.patch_embed.proj.weight.shape[0]


        self.token_size = token_size


        self.per_compr = BottleneckSE(
            C_in=self.vision_dim,
            C_mid=self.per_token_size * 2,
            C_out=self.per_token_size,
        )

        # The paper baseline uses the original PCMB over the complete
        # within-episode history. Both non-baseline modes use selective retrieval.
        pcmb_query_retrieval_mode = (
            self.query_retrieval_mode if self.use_query else "off"
        )

        self.cog_mem_bank = CogMemBank(
            dataloader_type=self.dataloader_type,
            group_size=self.group_size,
            token_size=self.cog_token_size,
            mem_length=self.mem_length,
            retrieval_layers=self.retrieval_layers,
            use_timestep_pe=self.use_timestep_pe,
            fusion_type=self.fusion_type,
            consolidate_type=self.consolidate_type,
            update_fused=self.update_fused,
            query_retrieval_mode=pcmb_query_retrieval_mode,
            query_retrieval_top_k=self.query_retrieval_top_k,
            modality_weights_index=self.modality_weights_index
        )

        self.per_mem_bank = PerMemBank(
            dataloader_type=self.dataloader_type,
            group_size=self.group_size,
            token_size=self.per_token_size,
            mem_length=self.mem_length,
            retrieval_layers=self.retrieval_layers,
            use_timestep_pe=self.use_timestep_pe,
            fusion_type=self.fusion_type,
            consolidate_type=self.consolidate_type,
            update_fused=self.update_fused,
            query_retrieval_mode=pcmb_query_retrieval_mode,
            query_retrieval_top_k=self.query_retrieval_top_k,
            modality_weights_index=self.modality_weights_index
        )
        self.episodic_bank = EpisodicMemBank(max_steps=self.episodic_max_steps,
                 top_k=self.episodic_top_k,
                                             
                 dataloader_type = "stream",
                 group_size = self.group_size,
                 token_size = self.token_size,
                 mem_length = self.mem_length,
                 retrieval_layers = self.retrieval_layers,
                 use_timestep_pe = self.use_timestep_pe,
                 fusion_type = self.fusion_type,
                 consolidate_type = self.consolidate_type,
                 update_fused = self.update_fused,
                 query_retrieval_mode = self.query_retrieval_mode,
                 query_retrieval_top_k = self.episodic_top_k)
        
        self.spatial_encoder = SpatialEncoder(
            spatial_token_size = self.per_token_size,
            depth_patch_size = depth_patch_size
        )
        # Legacy depth-patch encoder. The active full-mode path below uses
        # point_cloud_spatial_encoder exclusively, so do not advertise or
        # optimize parameters that can never receive a gradient.
        self.spatial_encoder.requires_grad_(False)

        self.point_cloud_spatial_encoder = SpatialPointCloudEncoder(
                    spatial_token_size=self.per_token_size,
                    num_spatial_tokens=num_spatial_tokens,
                    point_dim = 3,
                    proprio_dim = proprio_dim,
                    camera_dim = camera_dim,
                    hidden_dim = 128,
                    num_heads = 4,
                    max_points = max_points_spatial
        )
        self.spatial_mem_bank = SpatialMemBank(
            dataloader_type=self.dataloader_type,
            group_size=self.group_size,
            token_size=self.per_token_size,
            mem_length=self.mem_length,
            retrieval_layers=self.retrieval_layers,
            use_timestep_pe=self.use_timestep_pe,
            fusion_type=self.fusion_type,
            consolidate_type=self.consolidate_type,
            update_fused=self.update_fused,
            query_retrieval_mode=self.query_retrieval_mode,
            query_retrieval_top_k=self.query_retrieval_top_k,
            modality_weights_index=self.modality_weights_index
        )
        self.spatial_to_per_fusion = CrossTransformerBlock(self.per_token_size)
        self.per_spatial_gate = GateFusion(
            self.per_token_size, preserve_first_input=True
        )

        self.action_model = ActionModel(
            model_type=action_model_type,
            token_size=token_size,
            in_channels=action_dim,
            future_action_window_size=future_action_window_size,
            use_per_attn=True,
            per_token_size=per_token_size,
        )

        self.active_ep_id = None

        self.episodic_cog_attn = CrossTransformerBlock(self.cog_token_size)
        self.episodic_per_attn = CrossTransformerBlock(self.per_token_size)

        self.episodic_cog_gate = GateFusion(
            self.cog_token_size, preserve_first_input=True
        )
        self.episodic_per_gate = GateFusion(
            self.per_token_size, preserve_first_input=True
        )
        self.active_ep_contexts = {}
        self.episode_recordings = {}

        inactive_modules = []
        if not self.use_episodic:
            inactive_modules.extend((
                self.episodic_bank,
                self.episodic_cog_attn,
                self.episodic_per_attn,
                self.episodic_cog_gate,
                self.episodic_per_gate,
            ))
        if not self.use_spatial:
            inactive_modules.extend((
                self.spatial_encoder,
                self.point_cloud_spatial_encoder,
                self.spatial_mem_bank,
                self.spatial_to_per_fusion,
                self.per_spatial_gate,
            ))
        for module in inactive_modules:
            module.requires_grad_(False)

        self.all_module_keys = []
        self._trainable_module_keys = []

        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys.append('ema_diffusion')

        for module_keys in self.vlm.all_module_keys:
            self.all_module_keys.append("vlm." + module_keys)

        for name, module in self.named_children():
            if name != "vlm" and any(p.requires_grad for p in module.parameters()):
                self.all_module_keys.append(name)
                self._trainable_module_keys.append(name)

        self.apply_training_scope()




    @property
    def trainable_module_keys(self) -> List[str]:
        keys = []
        for module_keys in self.vlm.trainable_module_keys:
            keys.append("vlm." + module_keys)
        keys += self._trainable_module_keys
        return keys
    
    @property
    def llm_backbone(self) -> LLMBackbone:
        return self.vlm.llm_backbone
    
    @property
    def vision_backbone(self) -> VisionBackbone:
        return self.vlm.vision_backbone

    def _refresh_trainable_module_keys(self) -> None:
        self._trainable_module_keys = [
            name
            for name, module in self.named_children()
            if name != "vlm" and any(p.requires_grad for p in module.parameters())
        ]

    def apply_training_scope(self) -> None:
        if self.freeze_vlm:
            self.vlm.requires_grad_(False)
            self.vlm.eval()
            self.vlm.trainable_module_keys = []
            # Freeze the learned paper PCMB path together with its VLM source.
            # per_compr is the learned vision-feature -> perceptual-token
            # compressor and belongs to that same pretrained path.
            self.per_compr.requires_grad_(False)
            self.cog_mem_bank.requires_grad_(False)
            self.per_mem_bank.requires_grad_(False)

        self.action_model.requires_grad_(not self.freeze_action_model)
        if self.freeze_action_model:
            self.action_model.eval()
        self._refresh_trainable_module_keys()

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_model.eval()
        if self.freeze_vlm:
            self.vlm.eval()
        if self.freeze_action_model:
            self.action_model.eval()
        return self
    
    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)
        self.apply_training_scope()

    @torch.no_grad()
    def _encode_retrieval_inputs(
        self,
        images,
        instructions: Optional[List[str]],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if images is None or instructions is None:
            return None, None
        if len(images) != len(instructions):
            raise ValueError(
                "retrieval images and instructions must have the same batch size"
            )

        self.clip_model.eval()
        clip_param = next(self.clip_model.parameters())
        image_inputs = self.clip_processor(
            images=list(images),
            return_tensors="pt",
        ).to(clip_param.device)
        text_inputs = self.clip_processor(
            text=list(instructions),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(clip_param.device)

        image_outputs = self.clip_model.get_image_features(**image_inputs)
        text_outputs = self.clip_model.get_text_features(**text_inputs)
        image_embeddings = (
            image_outputs.pooler_output
            if hasattr(image_outputs, "pooler_output")
            else image_outputs
        )
        text_embeddings = (
            text_outputs.pooler_output
            if hasattr(text_outputs, "pooler_output")
            else text_outputs
        )
        return (
            F.normalize(image_embeddings.float(), dim=-1),
            F.normalize(text_embeddings.float(), dim=-1),
        )

    def _begin_episode(self, images, instruction, episode_id):
        selected = self.episodic_bank.retrieve(
            current_instruction=[instruction],
            initial_frame=images
        )
        if selected:
            active_ep_cog = torch.cat(
                [memory.cog_mem_bank.feat for memory in selected], dim=0
            ).unsqueeze(0)
            active_ep_per = torch.cat(
                [memory.per_mem_bank.feat for memory in selected], dim=0
                ).unsqueeze(0)

        else:
            active_ep_cog=None
            active_ep_per=None
        bank_episode_id = self.episodic_bank.start_episode(
            image=images, instruction=[instruction]
        )

        #write somenotes about these things
        self.active_ep_contexts[episode_id] = {
            "cog": active_ep_cog,
            "per": active_ep_per,
            "bank_episode_id": bank_episode_id,
        }
        self.episode_recordings[episode_id] = {
            "cog": [],
            "per": []
        }



    # episode_mem_ids here is the ids in the traininig collator so like (4,4,4,5) and coorresponds with timesteps
    def _fuse_episodic_tokens(self, cog_tokens, per_tokens, episode_mem_ids):
        cog_outputs = []
        per_outputs = []

        for i, raw_eid in enumerate(episode_mem_ids):
            eid = int(raw_eid)
            cog_token = cog_tokens[i:i+1]
            per_token = per_tokens[i:i+1]
            context = self.active_ep_contexts.get(eid)

            if context and context["cog"] is not None:
                
                cog_context = self.episodic_cog_attn(
                    cog_token   
                    ,context["cog"].to(cog_token.device,cog_token.dtype)
                    ,context["cog"].to(cog_token.device,cog_token.dtype)
                )
            
                cog_token = self.episodic_cog_gate(
                    cog_token,
                    cog_context
                )
            cog_outputs.append(cog_token)


            if context and context["per"] is not None:

                per_context = self.episodic_per_attn(
                    per_token,
                    context["per"].to(per_token.device, per_token.dtype),
                    context["per"].to(per_token.device, per_token.dtype)
                )

                per_token = self.episodic_per_gate(
                    per_token,
                    per_context
                )
            per_outputs.append(per_token)



        return torch.cat(cog_outputs), torch.cat(per_outputs)


    def finish_episode(self, success, episode_id = None):
        if not self.use_episodic:
            return

        if episode_id is None:
            episode_id = self.active_ep_id
        context = self.active_ep_contexts.get(episode_id)
        if context is None:
            raise KeyError(f"Cannot finish inactive episode {episode_id}")
        self.episodic_bank.end_episode(
            success=success,
            episode_cog_banks=self.cog_mem_bank.bank.get(episode_id, []),
            episode_per_banks=self.per_mem_bank.bank.get(episode_id, []),
            episode_id=context["bank_episode_id"],
        )

        finished_id = episode_id

        self.active_ep_contexts.pop(finished_id, None)
        self.episode_recordings.pop(finished_id, None)

        self.active_ep_id = None

        if len(self.episodic_bank.bank) > self.episodic_bank.max_steps:
            self.episodic_bank.kick_memory()



    def _fuse_spatial_tokens(
        self,
        per_tokens: torch.Tensor,
        spatial_tokens: Optional[torch.Tensor],
        episode_ids,
        timesteps,
        instructions: Optional[List[str]] = None,
        retrieval_image_embeddings: Optional[torch.Tensor] = None,
        retrieval_query_embeddings: Optional[torch.Tensor] = None,
        positions: Optional[torch.tensor] = None
    ) -> torch.Tensor:
        if spatial_tokens is None:
            return per_tokens

        spatial_tokens = self.spatial_mem_bank.process_batch(
            tokens=spatial_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
            instructions=instructions,
            retrieval_image_embeddings=retrieval_image_embeddings,
            retrieval_query_embeddings=retrieval_query_embeddings,
            positions=positions
        )
        spatial_context = self.spatial_to_per_fusion(
            per_tokens,
            spatial_tokens,
            spatial_tokens,
        )
        return self.per_spatial_gate(per_tokens, spatial_context)
# it literally is declared with self. What are you talking about?
    def _add_batch_depth(self, depth):
        depth = depth.squeeze(-1)
        depth = depth.unsqueeze(0)

        encoder_param = next(self.point_cloud_spatial_encoder.parameters())
        device = encoder_param.device
        dtype = encoder_param.dtype

        return depth.to(device = device, dtype = dtype)

    def _add_batch_intrinsics(self, intrinsics):
        intrinsics = intrinsics.unsqueeze(0)

        encoder_param = next(self.point_cloud_spatial_encoder.parameters())
        device = encoder_param.device
        dtype = encoder_param.dtype
        return intrinsics.to(device, dtype)

    def _add_batch_extrinsics(self, extrinsics):
        extrinsics = torch.as_tensor(extrinsics)
        if extrinsics.ndim == 2:
            extrinsics = extrinsics.unsqueeze(0)

        encoder_param = next(self.point_cloud_spatial_encoder.parameters())
        return extrinsics.to(device=encoder_param.device, dtype=torch.float32)

    def forward(
        self,
        depth: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
        camera: Optional[torch.FloatTensor] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
        extrinsics: Optional[torch.FloatTensor] = None,
        spatial_valid: Optional[torch.BoolTensor] = None,
        input_ids: torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        action_masks: Optional[torch.FloatTensor] = None,
        timesteps: np.array = None,
        episode_ids: np.array = None,
        instructions: Optional[List[str]] = None,
        images=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
        episode_ends: torch.Tensor = None,
        episode_successes: torch.Tensor = None
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        output = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        positions = proprio[:, :3]if proprio is not None else None

        # extract the visual token number
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif hasattr(self.vlm.vision_backbone, 'siglip_featurizer') and self.vlm.vision_backbone.siglip_featurizer is not None:
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")

        # extract the last hidden state and the learnable EOS token feature
        last_hidden_state = output.hidden_states[-1]
        last_hidden_state = last_hidden_state[:, num_patch :]

        # extract the cognition feature
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)  
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden_state.size(-1))

        cog_tokens = last_hidden_state.gather(
            1, expanded_indices.unsqueeze(1))  # [B, 1, D]

        vision_feats = self.vlm.vision_feats
        per_tokens = self.per_compr(vision_feats)
        

        retrieval_image_embeddings = None
        retrieval_query_embeddings = None
        if self.use_query or self.use_episodic:
            retrieval_image_embeddings, retrieval_query_embeddings = (
                self._encode_retrieval_inputs(images, instructions)
            )

        if self.use_episodic:
            for i in range(len(episode_ids)):
                episode_id = int(torch.as_tensor(episode_ids[i]).item())
                episode_timestep = int(torch.as_tensor(timesteps[i]).item())
                if episode_timestep == 0 and episode_id not in self.active_ep_contexts:
                    self._begin_episode(
                        images=[images[i]],
                        instruction=instructions[i],
                        episode_id=episode_id,
                    )

        # All ablations retain the published within-episode PCMB. Its configured
        # retrieval mode is "off" unless use_query is true.
        cog_tokens = self.cog_mem_bank.process_batch(
            tokens=cog_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
            instructions=instructions,
            retrieval_image_embeddings=retrieval_image_embeddings,
            retrieval_query_embeddings=retrieval_query_embeddings,
            positions=positions,
        )
        per_tokens = self.per_mem_bank.process_batch(
            tokens=per_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
            instructions=instructions,
            retrieval_image_embeddings=retrieval_image_embeddings,
            retrieval_query_embeddings=retrieval_query_embeddings,
            positions=positions,
        )

        if self.use_episodic:
            cog_tokens, per_tokens = self._fuse_episodic_tokens(
                cog_tokens=cog_tokens,
                per_tokens=per_tokens,
                episode_mem_ids=episode_ids,
            )

        if self.use_spatial:
            if spatial_valid is None:
                spatial_valid = torch.ones(
                    per_tokens.shape[0], device=per_tokens.device, dtype=torch.bool
                )
            else:
                spatial_valid = torch.as_tensor(
                    spatial_valid, device=per_tokens.device, dtype=torch.bool
                )
            if spatial_valid.ndim != 1 or spatial_valid.numel() != per_tokens.shape[0]:
                raise ValueError("spatial_valid must have shape [batch_size]")
            valid = spatial_valid.nonzero(as_tuple=True)[0]

            if valid.numel() > 0:
                if depth is None or intrinsics is None or extrinsics is None:
                    raise ValueError(
                        "Spatially valid samples require depth, intrinsics, and extrinsics"
                    )
                points_camera, valid_masks = geometry.depth_to_points(
                    depth=depth[valid],
                    intrinsics=intrinsics[valid],
                    mask=None,
                    flatten=True,
                )
                points_world = geometry.transform_points(
                    points_camera,
                    torch.linalg.inv(extrinsics[valid].float()),
                )
                spatial_tokens = self.point_cloud_spatial_encoder(
                    points=points_world,
                    proprio=proprio[valid] if proprio is not None else None,
                    camera=camera[valid] if camera is not None else None,
                    point_mask=valid_masks,
                )
                valid_cpu = valid.detach().cpu().tolist()
                fused_valid = self._fuse_spatial_tokens(
                    per_tokens=per_tokens[valid],
                    spatial_tokens=spatial_tokens,
                    episode_ids=[episode_ids[i] for i in valid_cpu],
                    timesteps=[timesteps[i] for i in valid_cpu],
                    instructions=[instructions[i] for i in valid_cpu],
                    retrieval_image_embeddings=retrieval_image_embeddings[valid],
                    retrieval_query_embeddings=retrieval_query_embeddings[valid],
                    positions=positions[valid] if positions is not None else None,
                )
                fused_valid = fused_valid.to(
                    device=per_tokens.device, dtype=per_tokens.dtype
                )
                per_tokens = per_tokens.index_copy(0, valid, fused_valid)

        if self.use_episodic:
            for i in range(len(episode_ids)):
                if bool(torch.as_tensor(episode_ends[i].item())):
                    self.finish_episode(
                        episode_id=int(torch.as_tensor(episode_ids[i]).item()),
                        success=bool(torch.as_tensor(episode_successes[i]).item()),
                    )
        # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
        actions_future = actions[:, -(self.future_action_window_size+1):, :]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)

        cog_tokens_repeated = cog_tokens.repeat(
            repeated_diffusion_steps, 1, 1)

        per_tokens_repeated = per_tokens.repeat(
            repeated_diffusion_steps, 1, 1)

        # Action model forward and compute loss
        loss = self.action_model.loss(
            actions_repeated,
            cog_tokens_repeated,
            per_tokens_repeated,
        )


        return loss, output

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, DiT},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        device: Optional[torch.device] = None,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        action_model_type: str = 'DiT-L',
        use_ema: bool = False,
        norm_stats = None,
        use_bf16: bool = False,
        **kwargs,
    ) -> MemoryVLA:

        # Load VLM backbone, borrowed from PrismaticVLM
        vlm = PrismaticVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # Keep inference loading below the host-memory limit. Construct the
        # destination modules in BF16 before copying the large checkpoint.
        if use_bf16:
            vlm = vlm.to(dtype=torch.bfloat16)

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(
            str(pretrained_checkpoint),
            map_location="cpu",
            mmap=True,
        )["model"]

        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        # Initialize
        memory_vla = MemoryVLA(vlm,
                        token_size = vlm.llm_backbone.llm.lm_head.in_features,
                        action_dim = action_dim,
                        future_action_window_size = future_action_window_size,
                        action_model_type = action_model_type,
                        use_ema = use_ema,
                        norm_stats = norm_stats,
                        **kwargs,
                        )

        if use_bf16:
            memory_vla = memory_vla.to(dtype=torch.bfloat16)

        # Load ActionModel from Checkpoint
        if "action_model" in model_state_dict:
            memory_vla.action_model.load_state_dict(model_state_dict["action_model"], strict=False)
            assert use_ema is False, "Does not support using EMA weights from pretrained checkpoint."
            if "ema_diffusion" in model_state_dict and use_ema:
                memory_vla.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"])
            elif use_ema:
                memory_vla.ema_diffusion.load_state_dict(model_state_dict["action_model"])
        else:
            overwatch.warning("No ActionModel found in the pretrained checkpoint. Initializing a new one.")

        spatial_checkpoint_keys = {
            "point_cloud_spatial_encoder", "spatial_mem_bank",
            "spatial_to_per_fusion", "per_spatial_gate",
        }
        missing_spatial_keys = sorted(spatial_checkpoint_keys - model_state_dict.keys())
        if missing_spatial_keys and memory_vla.experiment_mode == "full":
            overwatch.warning(
                "Checkpoint has no trained weights for spatial modules: "
                + ", ".join(missing_spatial_keys)
                + ". Those modules remain randomly initialized."
            )

        # load other weights
        for key, sub_state in model_state_dict.items():
            if key not in {"projector", "llm_backbone", "vision_backbone",
                           "action_model", "ema_diffusion"}:
                module = getattr(memory_vla, key, None)
                if module is None:
                    overwatch.warning(f"Ignoring unknown checkpoint module: {key}")
                    continue
                module.load_state_dict(
                    sub_state,
                    strict=key in spatial_checkpoint_keys,
                )
        

        del model_state_dict
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if use_bf16:
            memory_vla = memory_vla.to("cuda", dtype=torch.bfloat16)

        return memory_vla
    # [H, W, 1] -> [1, H, W]

    @torch.inference_mode()
    def predict_action(
        self, image: Image, 
        instruction: str,
        depth: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
        camera: Optional[torch.FloatTensor] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
        extrinsics: Optional[torch.FloatTensor] = None,
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        episode_first_frame: str = 'False',
        current_position: torch.Tensor = None,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer
 
        # Build VLA Prompt
        positions = None
        if current_position is not None:
            positions = torch.as_tensor(
                current_position,
                device=self.vlm.device,
                dtype=torch.float32
            ).reshape(1, 3)


        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871, 2]).long(), dim=0).to(self.vlm.device)), dim=1
            )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        model_dtype = next(self.parameters()).dtype
        
        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vlm.device, dtype=model_dtype)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.vlm.device, dtype=model_dtype) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        autocast_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float32

        with torch.autocast("cuda", dtype=autocast_dtype, enabled=(autocast_dtype == torch.bfloat16)):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
                **kwargs,
            )
            # fmt: on

        model_dtype = next(self.action_model.net.parameters()).dtype
        cog_tokens = output.hidden_states[-1][-1][:,-1,:]
        assert (cog_tokens.shape[0], cog_tokens.shape[1]) == (1,4096), "Batch size must be 1 for action prediction"

        cog_tokens = cog_tokens.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        vision_feats = self.vlm.vision_feats
        per_tokens = self.per_compr(vision_feats)

        assert episode_first_frame in ['True', 'False'], "episode_first_frame must be 'True' or 'False'"
        if episode_first_frame == 'True':
            self.cur_timestep = 0

        if episode_first_frame == 'True':
            print(" ** reset memory ** ")
            self.cog_mem_bank.reset()
            self.per_mem_bank.reset()
            if self.use_spatial:
                self.spatial_mem_bank.reset()
            if self.use_episodic:
                self.active_ep_id = self.episodic_bank.episode_id
                self._begin_episode(
                    images=[image],
                    instruction=instruction,
                    episode_id=self.active_ep_id,
                )

        episode_ids = [self.active_ep_id if self.use_episodic else 0]
        timesteps = [torch.tensor(self.cur_timestep, device=self.vlm.device)]
        retrieval_image_embeddings = None
        retrieval_query_embeddings = None
        if self.use_query or self.use_episodic:
            retrieval_image_embeddings, retrieval_query_embeddings = (
                self._encode_retrieval_inputs([image], [instruction])
            )

        cog_tokens = self.cog_mem_bank.process_batch(
            tokens=cog_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
            instructions=[instruction],
            retrieval_image_embeddings=retrieval_image_embeddings,
            retrieval_query_embeddings=retrieval_query_embeddings,
            positions=positions,
        )
        per_tokens = self.per_mem_bank.process_batch(
            tokens=per_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
            instructions=[instruction],
            retrieval_image_embeddings=retrieval_image_embeddings,
            retrieval_query_embeddings=retrieval_query_embeddings,
            positions=positions,
        )
        if self.use_episodic:
            cog_tokens, per_tokens = self._fuse_episodic_tokens(
                cog_tokens=cog_tokens,
                per_tokens=per_tokens,
                episode_mem_ids=episode_ids,
            )

        if self.use_spatial:
            if depth is None or intrinsics is None or extrinsics is None:
                raise ValueError(
                    "Full experiment mode requires depth, intrinsics, and extrinsics"
                )
            depth = self._add_batch_depth(depth)
            intrinsics = self._add_batch_intrinsics(intrinsics)
            extrinsics = self._add_batch_extrinsics(extrinsics)

            points_camera, valid_masks = geometry.depth_to_points(
                depth=depth,
                intrinsics=intrinsics,
                mask=None,
                flatten=True,
            )

            camera_to_world = torch.linalg.inv(extrinsics.float())
            points_world = geometry.transform_points(points_camera, camera_to_world)

            spatial_tokens = self.point_cloud_spatial_encoder(
                points=points_world,
                proprio=proprio,
                camera=camera,
                point_mask=valid_masks,
            )
            per_tokens = self._fuse_spatial_tokens(
                per_tokens=per_tokens,
                spatial_tokens=spatial_tokens,
                episode_ids=episode_ids,
                timesteps=timesteps,
                instructions=[instruction],
                retrieval_image_embeddings=retrieval_image_embeddings,
                retrieval_query_embeddings=retrieval_query_embeddings,
                positions=positions,
            )

        self.cur_timestep += 1



        # Sample random noise
        B = cog_tokens.shape[0]
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=cog_tokens.device).to(model_dtype)  #[B, T, D]
    
        # Setup classifier-free guidance:
        using_cfg = cfg_scale > 1.0
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[k, D]
            uncondition = uncondition.expand(B, *uncondition.shape[1:]) #[B, k, D]
            z = torch.cat([cog_tokens, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
            model_kwargs.update({'per_token': per_tokens.repeat(2, 1, 1)})  # Repeat for unconditioned and conditioned samples
        else:
            model_kwargs = dict(z=cog_tokens)
            sample_fn = self.action_model.net.forward
            model_kwargs.update({'per_token': per_tokens})

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                noise.shape, 
                                                                noise, 
                                                                clip_denoised=False,
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=cog_tokens.device,
                                                                eta=0.0
                                                                )
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=cog_tokens.device
                                                                    )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize Actions        
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions


    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
