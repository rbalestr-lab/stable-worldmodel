from datasets import load_from_disk, Dataset

import stable_worldmodel as swm
from stable_worldmodel.data import StepsDataset
from stable_worldmodel.policy import AutoCostModel
from stable_worldmodel.wm.dinowm import DINOWM

import stable_pretraining as spt

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import math

cache_dir = swm.data.get_cache_dir()  

train_dir = "pusht_expert_dataset_train"
val_dir = "pusht_expert_dataset_val"

NUM_STEPS = 1
BATCH_SIZE = 256
NUM_WORKERS = 6

def step_transform():
    transforms = []
    for t in range(NUM_STEPS):
        key = f"pixels.{t}"
        transforms.append(
            spt.data.transforms.Compose(
                spt.data.transforms.ToImage(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    source=key,
                    target=key,
                ),
                spt.data.transforms.Resize(224, source=key, target=key),
                spt.data.transforms.CenterCrop(224, source=key, target=key),
            )
        )
    return spt.data.transforms.Compose(*transforms)

transform = step_transform()
train_data = StepsDataset("pusht_expert_dataset_train", num_steps=NUM_STEPS, transform=transform)
val_data   = StepsDataset("pusht_expert_dataset_val",   num_steps=NUM_STEPS, transform=transform)

for df in (train_data, val_data):
    df.data_dir = df.data_dir.parent

# optionally pin_memory on CUDA, not Mac
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    persistent_workers=True
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    persistent_workers=True
)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("device:", device)

def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()  

# torch load better?
checkpoint_name = "dinowm_pusht_object.ckpt"

checkpoint = cache_dir / checkpoint_name
dinowm = torch.load(checkpoint, map_location=device, weights_only=False)
dinowm = dinowm.to(device).eval()

for p in dinowm.parameters():
    p.requires_grad_(False)

# no proprio encoder for now?
def encode(batch):
    info_d = {"pixels": batch["pixels"], "proprio": batch["proprio"]} # debug -> .to(device)
    with torch.no_grad():
        info_d = dinowm.encode(
            info_d,
            target="embed",
            pixels_key= "pixels",
            proprio_key="proprio")
    # Bx(d_pixels + d_proprio)?
    return info_d["embed"][:,-1].mean(dim=1) # last step, mean across patches


LATENT_DIM = dinowm.backbone.config.hidden_size
ACTION_DIM = len(train_data.dataset['action'][0])
EXPANSION_FACTOR = 2

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


action_head = MLP(LATENT_DIM, ACTION_DIM)
optimizer = torch.optim.AdamW(action_head.parameters(), lr=3e-4, weight_decay=1e-4)

EPOCHS = 25
for epoch in range(1, EPOCHS + 1):
    # train
    action_head.train()
    for batch in train_loader:
        latent = encode(batch)
        action = batch['action'][:,0].to(device)
        pred = action_head(latent)
        loss = F.mse_loss(pred, action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # eval
    action_head.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            latent = encode(batch)
            action = batch['action'][:,0].to(device)
            pred = action_head(latent)
            val_loss += F.mse_loss(pred, action)
    val_rmse = math.sqrt(val_loss / len(val_data))
    print(f'epoch {epoch}: RMSE: {val_rmse}')