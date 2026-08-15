import torch
import torch.nn as nn

# 1. Setup dummy memory tensor
# Dimensions: [Batch Size, Sequence Length (Memory Slots), Embedding Dimension]
batch_size = 2
seq_len = 5
embed_dim = 8

memory = torch.randn(batch_size, seq_len, embed_dim)
print("Original Memory Shape:", memory.shape)
print(memory)


# 2. Mean Pooling (Average across memory slots)
# We dim=1 to collapse the sequence length dimension
mean_pooled = torch.mean(memory, dim=1)
print("\nMean Pooled Shape:", mean_pooled.shape)
print(mean_pooled)


# 3. Max Pooling (Take maximum values across memory slots)
# torch.max returns both values and indices; we only need values [0]
max_pooled, _ = torch.max(memory, dim=1)
print("Max Pooled Shape: ", max_pooled.shape)


# 4. Combined Pooling (Concatenate both for richer representation)
combined_pooled = torch.cat((mean_pooled, max_pooled), dim=-1)
print("Combined Pooled Shape:", combined_pooled.shape)


# **** Find the mean along the length axis (the second one)