import hydra
import stable_pretraining as spt
import torch
from einops import rearrange
from loguru import logger as logging
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

import stable_worldmodel as swm


DINO_PATCH_SIZE = 14  # DINO encoder uses 14x14 patches


def get_data(cfg):
    """Setup dataset with image transforms and normalization."""

    def get_img_pipeline(key, target, img_size=224):
        return spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                **spt.data.dataset_stats.ImageNet,
                source=key,
                target=target,
            ),
            spt.data.transforms.Resize(img_size, source=key, target=target),
            spt.data.transforms.CenterCrop(img_size, source=key, target=target),
        )

    def norm_col_transform(dataset, col="pixels"):
        """Normalize column to zero mean, unit variance."""
        data = dataset[col][:]
        mean = data.mean(0).unsqueeze(0)
        std = data.std(0).unsqueeze(0)
        return lambda x: (x - mean) / std

    dataset = swm.data.FrameDataset(
        cfg.dataset.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cfg.get("cache_dir", None),
    )

    all_norm_transforms = []
    for key in cfg.dinowm.get("encoding", {}):
        trans_fn = norm_col_transform(dataset.dataset, key)
        trans_fn = spt.data.transforms.WrapTorchTransform(trans_fn, source=key, target=key)
        all_norm_transforms.append(trans_fn)

    # Image size must be multiple of DINO patch size (14)
    img_size = (cfg.image_size // cfg.patch_size) * DINO_PATCH_SIZE

    # Apply transforms to all steps
    transform = spt.data.transforms.Compose(
        *[get_img_pipeline(f"{col}.{i}", f"{col}.{i}", img_size) for col in ["pixels"] for i in range(cfg.n_steps)],
        *all_norm_transforms,
    )

    dataset.transform = transform
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    visual_set, _ = spt.data.random_split(
        dataset, lengths=[cfg.dataset.visual_split, 1 - cfg.dataset.visual_split], generator=rnd_gen
    )
    logging.info(f"Visual: {len(visual_set)}")

    visual = DataLoader(
        visual_set,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )

    return visual


# ============================================================================
# Model Architecture
# ============================================================================


def get_world_model(cfg):
    """Load and setup world model.
    For visualization, we only need the model to implement the `encode` method."""

    model = swm.policy.AutoCostModel(cfg.model_name).to(cfg.get("device", "cpu"))
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    class DinoV2Encoder(torch.nn.Module):
        def __init__(self, name, feature_key):
            super().__init__()
            self.name = name
            self.base_model = torch.hub.load("facebookresearch/dinov2", name)
            self.feature_key = feature_key
            self.emb_dim = self.base_model.num_features
            if feature_key == "x_norm_patchtokens":
                self.latent_ndim = 2
            elif feature_key == "x_norm_clstoken":
                self.latent_ndim = 1
            else:
                raise ValueError(f"Invalid feature key: {feature_key}")

            self.patch_size = self.base_model.patch_size

        def forward(self, x):
            emb = self.base_model.forward_features(x)[self.feature_key]
            if self.latent_ndim == 1:
                emb = emb.unsqueeze(1)  # dummy patch dim
            return emb

    ckpt = torch.load(swm.data.utils.get_cache_dir() / "dinowm_pusht_weights.ckpt")
    model.backbone = DinoV2Encoder("dinov2_vits14", feature_key="x_norm_patchtokens").to(cfg.get("device", "cpu"))
    model.predictor.load_state_dict(ckpt["predictor"], strict=False)
    model.action_encoder.load_state_dict(ckpt["action_encoder"])
    model.proprio_encoder.load_state_dict(ckpt["proprio_encoder"])
    model = model.to(cfg.get("device", "cpu"))
    model = model.eval()
    return model


@hydra.main(version_base=None, config_path="./", config_name="config")
def run(cfg):
    """Run visualization script."""

    data = get_data(cfg)
    world_model = get_world_model(cfg)

    # go through the dataset and encode all frames
    # embeddings will be stored in a list
    embeddings = []
    logging.info("Encoding dataset...")
    for batch in tqdm(data, desc="Processing batches"):
        # TODO make sure transform are applied
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(cfg.get("device", "cpu"))
        batch = world_model.encode(batch, target="embed")
        embeddings.append(batch["embed"].cpu().detach())

    # now we compute t-SNE on the embeddings
    logging.info("Computing t-SNE...")
    embeddings = torch.cat(embeddings, dim=0).numpy()
    # flatten the embeddings
    # TODO make sure we dont have action embedding here
    embeddings = rearrange(embeddings, "b ... -> b (...)")
    tsne = TSNE(n_components=2, random_state=cfg.seed)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d


if __name__ == "__main__":
    run()
