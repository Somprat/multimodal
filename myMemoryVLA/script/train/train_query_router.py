import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

LABELS = ['navigation', "object_state", "default", "temporal"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

device = torch.device(
    "cuda" if torch.cuda.is_available()
    # else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model = model.to(device)
model.requires_grad_(False)
model.eval()


@torch.inference_mode()
def encode_texts(texts: list[str]) -> torch.Tensor:
    inputs = processor(
        text=texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs = model.get_text_features(**inputs)
    embeddings = (
        outputs.pooler_output
        if hasattr(outputs, "pooler_output")
        else outputs
    )
    return F.normalize(embeddings.float(), dim=-1).cpu()

class CustomDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

        self.features = self.data.loc[:, "instruction"]
        self.features = [encode_texts([text]) for text in self.features]

        self.labels = self.data.loc[:, "primary_mode"]
        self.labels = [LABEL_TO_ID[label] for label in self.labels]

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]

        feature_tensor = feature.squeeze(0).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return (feature_tensor, label_tensor)


data_path = Path(__file__).with_name("maniskill_augmented_instructions_300.csv")
train_dataset = CustomDataset(csv_path=data_path)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32
)
classifier = nn.Linear(512, len(LABELS)).to(device)

class_weights = torch.tensor(
    [0.35714286, 2.5, 2.5, 2.5],
    dtype=torch.float32, 
    device=device)

optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)


epochs = 100
for epoch in range(epochs):
    for instruction, label in train_loader:
        instruction = instruction.to(device)
        label = label.to(device)

        predicted = classifier(instruction)
        loss = F.cross_entropy(
            predicted,
            label,
            weight=class_weights
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        accuracy = (predicted.argmax(dim=-1) == label).float().mean().item()
        print(accuracy)

checkpoint_path = Path(__file__).resolve().parents[2] / "checkpoints/query_router/clip_linear_4class.pt"
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "state_dict": classifier.state_dict(),
        "labels": LABELS,
        "encoder": "openai/clip-vit-base-patch32",
        "embedding_dim": 512, 
        "normalized": True,
    },
    checkpoint_path,
)
