#!/usr/bin/env python3
"""Train and evaluate frozen-RGB/spatial geometry probes."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_spatial_probe import ProbeFrameDataset
from vla.spatial.encoder import PointCloudSpatialEncoder
from vla.spatial.geometry import depth_to_points, transform_points
import vla.memory_vla as memory_vla


class ProbeHeads(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.object_head = nn.Linear(hidden_dim, 3)
        self.target_head = nn.Linear(hidden_dim, 3)
        self.distance_head = nn.Linear(hidden_dim, 1)
        self.grasp_head = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        hidden = self.shared(features)
        return {
            "object_relative": self.object_head(hidden),
            "target_relative": self.target_head(hidden),
            "object_distance": self.distance_head(hidden).squeeze(-1),
            "valid_grasp": self.grasp_head(hidden).squeeze(-1),
        }


class ProbeModel(nn.Module):
    def __init__(self, variant, 
            rgb_feature_dim, 
            query_retrieval_mode: str = "query", # One of: off, query, shuffled
            dataloader_type: str = "group", # Type of dataloader, chose from ['group', 'stream', 'parallel_stream']
            group_size: int = 16, # Group size for 'group' dataloader
            per_token_size: int = 256, # Token size for perception compression
            mem_length: int = 16, # Memory length
            retrieval_layers: int = 2, # Number of layers of memory retrieval
            query_retrieval_top_k: int = 4, # Historical cognition records selected per query
            use_timestep_pe: bool = True, # Whether to use timestep positional encoding
            fusion_type: str = 'gate',# Memory fusion type, chose from ['gate', 'add'])
            consolidate_type: str = 'tome', # Memory consolidate type, chose from ['fifo', 'tome']
            update_fused: bool = False, # Whether to update fused memory
            load_depth: bool = True, # Whether to load depth observations from RLDS
            load_proprio: bool = True, # Whether to load proprioceptive observations from RLDS
            use_spatial_features: bool = True,
            query_retrieval: bool = True): # Whether to load/use precomputed spatial features if available
        
        
        super().__init__()
        self.variant = variant
        self.spatial_encoder = None
        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.per_token_size = per_token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.query_retrieval_mode = query_retrieval_mode
        self.query_retrieval_top_k = query_retrieval_top_k
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.updated_fused = update_fused
        self.load_depth = load_depth
        self.load_proprio = load_proprio
        self.use_spatial_features = use_spatial_features


        self.memory_vla=memory_vla.MemoryVLA()

        if variant in {"rgb_spatial", "spatial"}:
            self.spatial_encoder = PointCloudSpatialEncoder(
                spatial_token_size=256,
                num_spatial_tokens=16,
                hidden_dim=128,
                num_heads=4,
                max_points=1024,
            )

        self.per_mem_bank = memory_vla.PerMemBank(
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
        )

        self.get_clip_features = memory_vla.MemoryVLA._encode_retrieval_inputs()
        self.fuse_tokens = memory_vla.MemoryVLA._fuse_spatial_tokens()

        input_dim = {
            "rgb": rgb_feature_dim,
            "rgb_spatial": rgb_feature_dim + 256,
            "spatial": 256,
        }[variant]

        self.heads = ProbeHeads(input_dim)
        
        

    def forward(self, batch):
        rgb_features = batch.rgb_features.float().detach()
        episode_ids = batch.episode_id # right now doesn't have one
        timesteps = batch.timestep
        instructions = batch.instructions
        positions = batch.positions
        # these are arguments for process_batch
        
        if self.query_retrieval_mode == "off":
            if self.variant == "rgb":
                probe_input = rgb_features
            else:
                points_camera, point_mask = depth_to_points(
                    batch.depth, batch.intrinsics
                )
                points_world = transform_points(
                    points_camera, torch.linalg.inv(batch.extrinsics)
                )
                spatial_tokens = self.spatial_encoder(
                    points=points_world,
                    point_mask=point_mask,
                )
                spatial_features = spatial_tokens.mean(dim=1)
                probe_input = (
                    torch.cat([rgb_features, spatial_features], dim=-1)
                    if self.variant == "rgb_spatial"
                    else spatial_features
                )
            return self.heads(probe_input)

        if self.query_retrieval_mode == 'query':
            if self.variant == 'rgb':
                input_tokens = rgb_features.unsqueeze(1)
                retrieval_image_embeddings, retrieval_query_embeddings =  self.get_clip_features(rgb_features, instructions)


                result_tokens = self.per_mem_bank.process_batch(
                    tokens=input_tokens,
                    episode_ids=episode_ids,
                    timesteps=timesteps,
                    instructions=instructions,
                    retrieval_image_embeddings=retrieval_image_embeddings,
                    retrieval_query_embeddings=retrieval_query_embeddings,
                    positions=positions,
                )
                result_features = result_tokens.squeeze(1)
            elif self.variant == 'rgb_spatial':
                #rgb
                input_tokens = rgb_features.unsqueeze(1)
                retrieval_image_embeddings, retrieval_query_embeddings =  self.get_clip_features(rgb_features, instructions)


                per_tokens = self.per_mem_bank.process_batch(
                    tokens=input_tokens,
                    episode_ids=episode_ids,
                    timesteps=timesteps,
                    instructions=instructions,
                    retrieval_image_embeddings=retrieval_image_embeddings,
                    retrieval_query_embeddings=retrieval_query_embeddings,
                    positions=positions,
                )

                # spatial
                points_camera, point_mask = depth_to_points(
                    batch["depth"], batch["intrinsics"]
                )
                points_world = transform_points(
                    points_camera, torch.linalg.inv(batch["extrinsics"])
                )
                spatial_tokens = self.spatial_encoder(
                    points=points_world,
                    point_mask=point_mask,
                )
                fused_tokens = self.fuse_tokens(
                    per_tokens=per_tokens,
                    spatial_tokens=spatial_tokens,
                    episode_ids=episode_ids,
                    spatial_tokens=spatial_tokens,
                    timesteps=timesteps,
                    instructions=instructions,
                    retrieval_image_embeddings=retrieval_image_embeddings,
                    retrieval_query_embeddings=retrieval_query_embeddings,
                    positions=positions
                )
                result_features = fused_tokens.squeeze(1)
            elif self.variant == 'spatial':
                points_camera, point_mask = depth_to_points(
                    batch["depth"], batch["intrinsics"]
                )
                points_world = transform_points(
                    points_camera, torch.linalg.inv(batch["extrinsics"])
                )
                spatial_tokens = self.spatial_encoder(
                    points=points_world,
                    point_mask=point_mask,
                )
                result_features = spatial_tokens.squeeze(1)

            return self.heads(result_features)
        else:
            raise ValueError("Please put only either off or query")


def move_batch(batch, device):
    return {name: tensor.to(device) for name, tensor in batch.items()}


def probe_loss(predictions, batch, grasp_loss):
    return (
        F.mse_loss(predictions["object_relative"], batch["object_relative"])
        + F.mse_loss(predictions["target_relative"], batch["target_relative"])
        + F.mse_loss(predictions["object_distance"], batch["object_distance"])
        + grasp_loss(predictions["valid_grasp"], batch["valid_grasp"])
    )


def balanced_accuracy(logits, targets):
    predicted = torch.sigmoid(logits) >= 0.5
    positive = targets >= 0.5
    negative = ~positive
    rates = []
    if positive.any():
        rates.append((predicted[positive] == positive[positive]).float().mean())
    if negative.any():
        rates.append((predicted[negative] == positive[negative]).float().mean())
    return torch.stack(rates).mean().item() if rates else float("nan")


@torch.no_grad()
def evaluate(model, loader, device, grasp_loss):
    model.eval()
    total_loss = 0.0
    total_frames = 0
    predictions = {name: [] for name in (
        "object_relative", "target_relative", "object_distance", "valid_grasp"
    )}
    targets = {name: [] for name in predictions}

    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        count = len(batch["depth"])
        total_loss += probe_loss(output, batch, grasp_loss).item() * count
        total_frames += count
        for name in predictions:
            predictions[name].append(output[name].cpu())
            targets[name].append(batch[name].cpu())

    predictions = {name: torch.cat(values) for name, values in predictions.items()}
    targets = {name: torch.cat(values) for name, values in targets.items()}
    return {
        "loss": total_loss / total_frames,
        "object_mae_cm": (
            predictions["object_relative"] - targets["object_relative"]
        ).abs().mean().item() * 100.0,
        "target_mae_cm": (
            predictions["target_relative"] - targets["target_relative"]
        ).abs().mean().item() * 100.0,
        "distance_mae_cm": (
            predictions["object_distance"] - targets["object_distance"]
        ).abs().mean().item() * 100.0,
        "grasp_balanced_accuracy": balanced_accuracy(
            predictions["valid_grasp"], targets["valid_grasp"]
        ),
    }
# figure out how to include the query stuff here

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--variant", choices=("rgb", "rgb_spatial", "spatial"), required=True
    )
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query_retrieval_mode", type=Path, default="query")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = args.data_dir.resolve()
    splits_path = (args.splits or data_dir / "splits.json").resolve()
    cache_dir = (args.cache_dir or data_dir / "feature_cache").resolve()
    output_dir = (args.output_dir or data_dir / "probe_runs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = json.loads(splits_path.read_text())

    train_dataset = ProbeFrameDataset(data_dir, cache_dir, splits["train"])
    validation_dataset = ProbeFrameDataset(
        data_dir, cache_dir, splits["validation"]
    )
    test_dataset = ProbeFrameDataset(data_dir, cache_dir, splits["test"])

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProbeModel(args.variant, train_dataset.rgb_feature_dim, query_retrieval_mode=args.query_retrieval_mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    positives = train_dataset.valid_grasp.sum().item()
    negatives = len(train_dataset) - positives
    pos_weight = negatives / positives if positives > 0 else 1.0
    grasp_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)
    )

    checkpoint_path = output_dir / f"{args.variant}_seed{args.seed}.pt"
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_frames = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            output = model(batch)
            loss = probe_loss(output, batch, grasp_loss)
            loss.backward()
            optimizer.step()
            count = len(batch["depth"])
            train_loss += loss.item() * count
            train_frames += count

        validation = evaluate(model, validation_loader, device, grasp_loss)
        print(
            f"epoch={epoch} train_loss={train_loss / train_frames:.6f} "
            f"validation_loss={validation['loss']:.6f}"
        )
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, grasp_loss)
    results = {
        "variant": args.variant,
        "seed": args.seed,
        "train_frames": len(train_dataset),
        "validation_frames": len(validation_dataset),
        "test_frames": len(test_dataset),
        "best_validation_loss": best_validation_loss,
        "test": test_metrics,
    }
    results_path = output_dir / f"{args.variant}_seed{args.seed}.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
