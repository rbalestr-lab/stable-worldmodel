#!/usr/bin/env python3
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

import math, time, contextlib

NUM_STEPS = 1
BATCH_SIZE = 256
NUM_WORKERS = 6

def run():
    cache_dir = swm.data.get_cache_dir()
    train_dir = "pusht_expert_dataset_train"
    val_dir = "pusht_expert_dataset_val"

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
    train_data = StepsDataset(train_dir, num_steps=NUM_STEPS, transform=transform)
    val_data   = StepsDataset(val_dir, num_steps=NUM_STEPS, transform=transform)

    for df in (train_data, val_data):
        df.data_dir = df.data_dir.parent


    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device:", device)

    def sync():
        if device.type=='cuda':
            torch.cuda.synchronize()
        if device.type=='mps':
            torch.mps.synchronize()

    # optionally pin_memory on CUDA, not Mac
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=(device.type=='cuda')
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=(device.type=='cuda')
    )

    # torch load better?
    checkpoint_name = "dinowm_pusht_object.ckpt"

    checkpoint = cache_dir / checkpoint_name
    dinowm = torch.load(checkpoint, map_location=device, weights_only=False)
    dinowm = dinowm.to(device).eval()

    for p in dinowm.parameters():
        p.requires_grad_(False)
    
    @torch.inference_mode()
    def encode(batch):
        if device.type in ("cuda", "mps"):
            context = torch.autocast(device_type=device.type, dtype=torch.float16)
        else:
            context = contextlib.nullcontext()
        data = {
            "pixels":  batch["pixels"].to(device, non_blocking=True),
            "proprio": batch["proprio"].to(device, non_blocking=True),
        }
        with context:
            out = dinowm.encode(
                data, target="embed", pixels_key="pixels", proprio_key="proprio"
            )
        z = out["embed"][:, -1].mean(dim=1)  # [B, D]
        return z.float()

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
        
    LATENT_DIM = dinowm.backbone.config.hidden_size + dinowm.proprio_encoder.emb_dim
    # can make this inferrable from first batch
    ACTION_DIM = len(train_data.dataset['action'][0])

    # action = (x, y)
    print('latent_dim, action_dim:', LATENT_DIM, ACTION_DIM)
    action_head = MLP(LATENT_DIM, ACTION_DIM).to(device)

    optimizer = torch.optim.AdamW(action_head.parameters(), lr=3e-4, weight_decay=1e-4)

    EPOCHS = 25
    for epoch in range(1, EPOCHS + 1):
        # train
        action_head.train()
        for i, batch in enumerate(train_loader):
            latent = encode(batch)
            action = batch['action'][:,0].to(device)
            pred = action_head(latent)
            loss = F.mse_loss(pred, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                sync()
                elapsed = time.perf_counter() - t0
                steps_per_sec   = i / max(1e-9, elapsed)
                samples_per_sec = n_seen / max(1e-9, elapsed)
                eta_steps = (len(train_loader) - i) / max(1e-9, steps_per_sec)
                print(
                    f"ep{epoch} step{i}/{len(train_loader)} "
                    f"loss={loss.item():.4f} "
                    f"sps={steps_per_sec:.1f} ips={samples_per_sec:.1f} "
                    f"eta={eta_steps/60:.1f}m"
                )
        
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
        print(f'epoch {epoch}: RMSE: {val_rmse:.6f}')

if __name__ == "__main__":
    run()