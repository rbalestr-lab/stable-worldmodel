#!/usr/bin/env python3
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm

import stable_worldmodel as swm
from stable_worldmodel.data import StepsDataset
import stable_pretraining as spt

import torch
from torch.utils.data import DataLoader
import time
import torch.multiprocessing as mp

# better than no_grad?
@torch.inference_mode()
def encode(dinowm, batch, device):
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        out = dinowm.encode(
            {"pixels": batch["pixels"].to(device), "proprio": batch["proprio"].to(device)},
            target="embed",
            pixels_key="pixels",
            proprio_key="proprio",
        )
    z_pix = out["pixels_embed"].mean(2).to(torch.float32).cpu()
    z_prp = out["proprio_embed"].to(torch.float32).cpu()
    return z_pix, z_prp

def step_transform(num_steps):
    transforms = []
    for t in range(num_steps):
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

def main():
    # params
    DATASET_NAME = "pusht_expert_dataset_val"
    CHECKPOINT_NAME = "dinowm_pusht_object.ckpt"
    OUTPUT_DIR = "latents_val"

    NUM_STEPS = 1 # i think this should be enforced, you can create stepsdataset post-encoding?
    BATCH_SIZE = 512
    NUM_WORKERS = 10
    SHARD_SIZE = 10000

    cache_dir = swm.data.get_cache_dir()
    out_dir = cache_dir / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device =torch.device("cpu")
    print("device:", device)

    dinowm = torch.load(cache_dir / CHECKPOINT_NAME, map_location=device, weights_only=False)
    dinowm = dinowm.to(device).eval()
    for p in dinowm.parameters():
        p.requires_grad_(False)

    data = StepsDataset(DATASET_NAME,
                        num_steps=NUM_STEPS,
                        transform=step_transform(NUM_STEPS),
                        cache_dir=cache_dir)
    # hacky fix
    data.data_dir = data.data_dir.parent

    loader = DataLoader(data,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS,
                        persistent_workers = (NUM_WORKERS > 0),
                        pin_memory = (device.type == "cuda"))

    probe = next(iter(loader))
    z_pix0, z_prop0 = encode(dinowm, probe, device) # [B,T,D], [B,T,Dp_emb]
    
    T, D = z_pix0.shape[1], z_pix0.shape[2]
    Dp_raw = probe["proprio"].shape[-1]
    Da = probe["action"].shape[-1]
    Dp_emb = z_prop0.shape[-1]

    features = Features({
        "episode_idx": Value("int32"),
        "step_idx": Value("int32"),
        "pixels_embed": Sequence(Sequence(Value("float32"), length=D), length=T), # [T,D_pixel_embed]
        "proprio_embed": Sequence(Sequence(Value("float32"), length=Dp_emb), length=T), # [T,D_proprio_embed]
        "proprio": Sequence(Sequence(Value("float32"), length=Dp_raw), length=T), # [T,D_proprio]
        "action": Sequence(Sequence(Value("float32"), length=Da), length=T), # [T,Da]
    })

    shards = sorted([p for p in out_dir.glob("shard_*") if p.is_dir()])
    start_shard = len(shards)
    skip_rows   = start_shard * SHARD_SIZE
    print(f"existing shards: {start_shard}  → skipping {skip_rows} rows")

    rows, written, shard_id, seen = [], 0, start_shard, 0
    t0 = time.perf_counter()

    for batch in tqdm(loader, total=len(loader), desc=f"dump {DATASET_NAME}"):
        bsz = batch["action"].shape[0]
        if seen < skip_rows:
            seen += bsz
            continue

        z_pix, z_prop = encode(dinowm, batch, device)

        epi_list = batch["episode_idx"][:, -1].cpu().tolist()
        step_list = batch["step_idx"][:, -1].cpu().tolist()
        prop = batch["proprio"].cpu().tolist()
        act = batch["action"].cpu().tolist()

        for i in range(bsz):
            rows.append({
                "episode_idx": int(epi_list[i]),
                "step_idx": int(step_list[i]),
                "pixels_embed": z_pix[i].tolist(),
                "proprio_embed": z_prop[i].tolist(),
                "proprio": prop[i],
                "action": act[i],
            })
            written += 1
            if written % SHARD_SIZE == 0:
                shard_path = out_dir / f"shard_{shard_id:05d}" 
                Dataset.from_list(rows, features=features).save_to_disk(str(shard_path))
                rows.clear(); shard_id += 1

    if rows:
        shard_path = out_dir / f"shard_{shard_id:05d}"
        Dataset.from_list(rows, features=features).save_to_disk(str(shard_path))

    print(f"Finished! took {time.perf_counter()-t0:.1f}s; wrote {shard_id - start_shard} shards")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
