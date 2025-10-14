from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

import stable_worldmodel as swm


def get_data(dataset_name):
    """Return data and action space dim for training predictor."""

    N_STEPS = 2

    def get_img_pipeling(key, target, img_size=224):
        return spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                source=key,
                target=target,
            ),
            spt.data.transforms.Resize(img_size, source=key, target=target),
            spt.data.transforms.CenterCrop(img_size, source=key, target=target),
        )

    def norm_col_transform(dataset, col="pixels"):
        data = dataset[col][:]
        mean = data.mean(0).unsqueeze(0)
        std = data.std(0).unsqueeze(0)
        return lambda x: (x - mean) / std

    dataset = swm.data.StepsDataset(
        dataset_name,
        num_steps=N_STEPS,
        frameskip=5,
        transform=None,
    )

    IMG_SIZE = (224 // 16) * 14

    norm_action_transform = norm_col_transform(dataset.dataset, "action")
    norm_proprio_transform = norm_col_transform(dataset.dataset, "proprio")

    transform = spt.data.transforms.Compose(
        *[get_img_pipeling(f"{col}.{i}", f"{col}.{i}", IMG_SIZE) for col in ["pixels"] for i in range(N_STEPS)],
        spt.data.transforms.WrapTorchTransform(
            norm_action_transform,
            source="action",
            target="action",
        ),
        spt.data.transforms.WrapTorchTransform(
            norm_proprio_transform,
            source="proprio",
            target="proprio",
        ),
    )

    # override dataset transform
    dataset.transform = transform

    train_set, val_set = spt.data.random_split(dataset, lengths=[0.9, 0.1])

    print(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}")

    train = DataLoader(
        train_set,
        batch_size=32,
        num_workers=10,  # Reduced from 10 to avoid OOM
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=1,
        shuffle=True,
    )
    val = DataLoader(
        val_set,
        batch_size=32,
        num_workers=10,
        persistent_workers=False,
        pin_memory=True,
    )  # Reduced workers
    data_module = spt.data.DataModule(train=train, val=val)

    # -- determine action space dimension
    action = dataset[0]["action"]
    action_dim = action[0].shape[-1]
    return data_module, action_dim


def forward(self, batch, stage):
    """Forward pass for predictor training"""

    proprio_key = "proprio" if "proprio" in batch else None

    # make action and proprio NaN (last action) at 0
    if proprio_key is not None:
        nan_mask = torch.isnan(batch[proprio_key])
        batch[proprio_key][nan_mask] = 0.0

    if "action" in batch:
        nan_mask = torch.isnan(batch["action"])
        batch["action"][nan_mask] = 0.0

    batch = self.model.encode(
        batch,
        target="embed",
        pixels_key="pixels",
        proprio_key=proprio_key,
        action_key="action",
    )

    # predictions
    embedding = batch["embed"][:, :-1, :, :]  # (B, history_size, P, d)
    pred_embedding = self.model.predict(embedding)

    # targets values
    target_embedding = batch["embed"][:, 1:, :, :]  # (B, T-history_size, P, d)

    # == pixels loss
    pixels_dim = batch["pixels_embed"].shape[-1]
    pixels_loss = F.mse_loss(pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim].detach())

    batch["pixels_loss"] = pixels_loss
    loss = pixels_loss

    # == proprio loss
    if proprio_key is not None:
        proprio_dim = batch["proprio_embed"].shape[-1]
        proprio_loss = F.mse_loss(
            pred_embedding[..., pixels_dim : pixels_dim + proprio_dim],
            target_embedding[..., pixels_dim : pixels_dim + proprio_dim].detach(),
        )
        batch["proprio_loss"] = proprio_loss
        loss = loss + proprio_loss

    batch["loss"] = loss
    prefix = "" if self.training else "val_"
    # == logging
    losses_dict = {f"{prefix}{k}": v.item() for k, v in batch.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, on_epoch=True, sync_dist=True)

    return batch


def get_world_model(cfg):
    """Return stable_spt module with world model"""

    # encoder = swm.wm.dinowm.DinoV2Encoder("dinov2_vits14", feature_key="x_norm_patchtokens")
    encoder = AutoModel.from_pretrained("facebook/dinov2-small")
    embedding_dim = encoder.config.hidden_size

    num_patches = (cfg.image_size // cfg.patch_size) ** 2  # 256 for 224×224
    embedding_dim += cfg.dinowm.proprio_embed_dim + cfg.dinowm.action_embed_dim

    logging.info(f"Encoder: {encoder}, emb_dim: {embedding_dim}, num_patches: {num_patches}")

    predictor = swm.wm.dinowm.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim,
        **cfg.predictor,
    )

    logging.info(f"Predictor: {predictor}")

    # -- create action encoder
    effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
    action_encoder = swm.wm.dinowm.Embedder(in_chans=effective_act_dim, emb_dim=cfg.dinowm.action_embed_dim)

    logging.info(f"Action dim: {effective_act_dim}, action emb dim: {cfg.dinowm.action_embed_dim}")

    # -- create proprioceptive encoder
    proprio_encoder = swm.wm.dinowm.Embedder(in_chans=cfg.dinowm.proprio_dim, emb_dim=cfg.dinowm.proprio_embed_dim)
    logging.info(f"Proprio dim: {cfg.dinowm.proprio_dim}, proprio emb dim: {cfg.dinowm.proprio_embed_dim}")

    world_model = swm.wm.DINOWM(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
        device="cuda",
    )

    # -- world model as a stable_spt module

    def add_opt(module_name, lr):
        return {
            "modules": str(module_name),
            "optimizer": {"type": "AdamW", "lr": lr},
        }

    world_model = spt.Module(
        model=world_model,
        forward=forward,
        optim={
            "predictor_opt": add_opt("model.predictor", cfg.predictor_lr),
            "proprio_opt": add_opt("model.proprio_encoder", cfg.proprio_encoder_lr),
            "action_opt": add_opt("model.action_encoder", cfg.action_encoder_lr),
        },
    )
    return world_model


def setup_pl_logger(cfg):
    if not cfg.wandb.enable:
        return None

    wandb_run_id = cfg.wandb.get("run_id", None)
    wandb_logger = WandbLogger(
        name="dino_wm",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        resume="allow" if wandb_run_id else None,
        id=wandb_run_id,
    )

    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    return wandb_logger


@hydra.main(version_base=None, config_path="./", config_name="config")
def run(cfg):
    """Run training of predictor"""

    wandb_logger = setup_pl_logger(cfg)

    data, action_dim = get_data(cfg.dataset_name)
    world_model = get_world_model(cfg)

    cache_dir = swm.data.get_cache_dir()
    checkpoint_callback = ModelCheckpoint(dirpath=cache_dir, filename=f"{cfg.output_model_name}_weights")

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
        logger=wandb_logger,
        precision="16-mixed",
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer, module=world_model, data=data
    )  # , ckpt_path=f"{cfg.output_model_name}_ckpt")
    manager()

    if hasattr(cfg, "dump_object") and cfg.dump_object:
        # -- save the world model object
        output_path = Path(cache_dir, f"{cfg.output_model_name}_object.ckpt")
        torch.save(world_model.to("cpu"), output_path)
        print(f"Saved world model object to {output_path}")


if __name__ == "__main__":
    run()
