from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialMemBank(nn.Module):
    """Episode-local spatial memory with FIFO and novelty-aware updates."""

    def __init__(
        self,
        max_steps: int,
        num_spatial_tokens: Optional[int] = None,
        spatial_token_size: Optional[int] = None,
        novelty_threshold: float = 0.15,
        merge_rate: float = 0.25,
        update_mode: str = "novelty",
    ) -> None:
        super().__init__()

        if update_mode not in ("fifo", "novelty"):
            raise ValueError(f"update_mode must be 'fifo' or 'novelty', got {update_mode}")

        self.max_steps = max_steps
        self.num_spatial_tokens = num_spatial_tokens
        self.spatial_token_size = spatial_token_size
        self.novelty_threshold = novelty_threshold
        self.merge_rate = merge_rate
        self.update_mode = update_mode

        self.reset()

    def reset(self) -> None:
        self.memory = None

    @torch.no_grad()
    def update(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """Update memory from tokens shaped [B, N, D] and return [B, T, N, D]."""

        self._validate_tokens(spatial_tokens)
        new_entry = spatial_tokens.detach().clone().unsqueeze(1)

        if self.memory is None:
            self.memory = new_entry
            return self.memory

        if self.memory.shape[0] != spatial_tokens.shape[0]:
            raise ValueError(
                f"batch size changed from {self.memory.shape[0]} to {spatial_tokens.shape[0]}; call reset() first"
            )

        if self.memory.shape[1] < self.max_steps:
            self.memory = torch.cat([self.memory, new_entry], dim=1)
            return self.memory

        if self.update_mode == "fifo":
            self.memory = torch.cat([self.memory[:, 1:], new_entry], dim=1)
        else:
            self.memory = self._novelty_update(self.memory, new_entry.squeeze(1))

        return self.memory

    def get(self, flatten: bool = False) -> Optional[torch.Tensor]:
        """Return memory as [B, T, N, D] or flattened [B, T * N, D]."""

        if self.memory is None:
            return None

        if not flatten:
            return self.memory

        B, T, N, D = self.memory.shape
        return self.memory.reshape(B, T * N, D)

    def forward(self, spatial_tokens: torch.Tensor, flatten: bool = False) -> torch.Tensor:
        memory = self.update(spatial_tokens)
        if not flatten:
            return memory

        B, T, N, D = memory.shape
        return memory.reshape(B, T * N, D)


    def _validate_tokens(self, spatial_tokens: torch.Tensor) -> None:
        if spatial_tokens.ndim != 3:
            raise ValueError(f"spatial_tokens must be shaped [B, N, D], got {tuple(spatial_tokens.shape)}")

        if self.num_spatial_tokens is not None and spatial_tokens.shape[1] != self.num_spatial_tokens:
            raise ValueError(
                f"expected {self.num_spatial_tokens} spatial tokens, got {spatial_tokens.shape[1]}"
            )

        if self.spatial_token_size is not None and spatial_tokens.shape[2] != self.spatial_token_size:
            raise ValueError(
                f"expected token size {self.spatial_token_size}, got {spatial_tokens.shape[2]}"
            )

    def _novelty_update(self, memory: torch.Tensor, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """
        Replace stale memory slots only when the new spatial state is novel.

        Novelty is measured as 1 - cosine similarity between mean-pooled spatial
        token sets. If the new tokens are close to an existing slot, softly merge
        them into that slot. If they are novel, replace the least similar slot.
        """

        B, T, N, D = memory.shape
        memory_summary = memory.mean(dim=2)
        new_summary = spatial_tokens.mean(dim=1)

        memory_norm = F.normalize(memory_summary, dim=-1)
        new_norm = F.normalize(new_summary, dim=-1)
        similarities = torch.einsum("btd,bd->bt", memory_norm, new_norm)

        best_similarity, best_idx = similarities.max(dim=1)
        worst_idx = similarities.argmin(dim=1)
        novelty = 1.0 - best_similarity

        updated = memory.clone()
        for b in range(B):
            if novelty[b] >= self.novelty_threshold:
                updated[b, worst_idx[b]] = spatial_tokens[b]
            else:
                slot = best_idx[b]
                updated[b, slot] = (1.0 - self.merge_rate) * updated[b, slot] + self.merge_rate * spatial_tokens[b]

        return updated
