import hydra
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from einops import rearrange
from loguru import logger as logging
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

import stable_worldmodel as swm


DINO_PATCH_SIZE = 14  # DINO encoder uses 14x14 patches

# ============================================================================
# Data Loading
# ============================================================================


def get_data(cfg, dataset_cfg):
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

    # Use dataset_cfg for specific dataset parameters
    dataset = swm.data.FrameDataset(
        dataset_cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cfg.get("cache_dir", None),
    )

    all_norm_transforms = []
    # Use global cfg for encoding keys to ensure consistency
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

    # Use dataset_cfg for split and loader settings
    visual_set, _ = spt.data.random_split(
        dataset, lengths=[dataset_cfg.visual_split, 1 - dataset_cfg.visual_split], generator=rnd_gen
    )
    logging.info(f"Visual ({dataset_cfg.dataset_name}): {len(visual_set)}")

    visual = DataLoader(
        visual_set,
        batch_size=dataset_cfg.batch_size,
        num_workers=dataset_cfg.num_workers,
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


def get_world_model(cfg, model_cfg):
    """Load and setup world model.
    For visualization, we only need the model to implement the `encode` method."""

    model = swm.policy.AutoCostModel(model_cfg.name).to(cfg.get("device", "cpu"))
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

    # Load specific checkpoint from experiment config
    ckpt_path = swm.data.utils.get_cache_dir() / model_cfg.ckpt_path
    ckpt = torch.load(ckpt_path)

    model.backbone = DinoV2Encoder("dinov2_vits14", feature_key="x_norm_patchtokens").to(cfg.get("device", "cpu"))
    model.predictor.load_state_dict(ckpt["predictor"], strict=False)
    model.action_encoder.load_state_dict(ckpt["action_encoder"])
    model.proprio_encoder.load_state_dict(ckpt["proprio_encoder"])
    model = model.to(cfg.get("device", "cpu"))
    model = model.eval()
    return model


# ============================================================================
# Embedding Collection and Visualization
# ============================================================================


def collect_embeddings(cfg, exp_cfg):
    """
    Loads a specific dataset and its corresponding world model,
    encodes the data, and returns the flattened embeddings.
    """
    data = get_data(cfg, exp_cfg.dataset)
    world_model = get_world_model(cfg, exp_cfg.model)

    logging.info(f"Encoding dataset: {exp_cfg.dataset.dataset_name} using model: {exp_cfg.model.name}...")
    dataset_embeddings = []

    # Process batches and collect embeddings
    for batch in tqdm(data, desc=f"Processing {exp_cfg.dataset.dataset_name}"):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(cfg.get("device", "cpu"))

        # Encode
        batch = world_model.encode(batch, target="embed")
        if cfg.get("backbone_only", False):  # use only vision backbone embeddings
            dataset_embeddings.append(batch["pixels_embed"].cpu().detach())
        else:  # use full model embeddings (proropio + action + vision)
            dataset_embeddings.append(batch["embed"].cpu().detach())

    # Consolidate and flatten embeddings for this dataset
    if len(dataset_embeddings) > 0:
        dataset_embeddings = torch.cat(dataset_embeddings, dim=0)
        # Flatten: b ... -> b (...)
        dataset_embeddings = rearrange(dataset_embeddings, "b ... -> b (...)")
        return dataset_embeddings

    return None


# ============================================================================
# Main Visualization Script
# ============================================================================


def plot_joint_tsne(embeddings_2d, labels, output_file="joint_tsne.pdf"):  # <-- Changed default extension to .pdf
    """
    Plots the 2D t-SNE embeddings, coloring points by their dataset label.

    Args:
        embeddings_2d (np.ndarray): Shape (N, 2) containing 2D coordinates.
        labels (list or np.ndarray): Shape (N,) containing dataset names for each point.
        output_file (str): Path to save the resulting plot.
    """
    plt.figure(figsize=(12, 10))

    # Get unique datasets to iterate over for the legend
    unique_datasets = np.unique(labels)

    # Use a colormap suitable for categorical data
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_datasets)))

    for dataset_name, color in zip(unique_datasets, colors):
        # Select indices belonging to this dataset
        indices = np.where(labels == dataset_name)

        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            label=dataset_name,
            c=[color],
            alpha=0.6,
            s=10,  # Marker size
        )

    plt.title("Joint t-SNE of Dataset Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Datasets", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    logging.info(f"Saving plot to {output_file}")
    # Matplotlib automatically handles the output format based on the file extension
    plt.savefig(output_file, dpi=300)
    plt.close()


@hydra.main(version_base=None, config_path="./configs", config_name="config_datasets")
def run(cfg):
    """Run visualization script for multiple datasets and compute joint t-SNE."""

    all_embeddings_list = []
    all_labels_list = []  # Added to track source datasets

    # Iterate over all defined datasets and collect embeddings
    for exp_cfg in cfg.datasets:
        embeddings = collect_embeddings(cfg, exp_cfg)

        if embeddings is not None:
            all_embeddings_list.append(embeddings)

            # Create a label entry for every point in this batch
            dataset_name = exp_cfg.dataset.dataset_name
            num_points = embeddings.shape[0]
            all_labels_list.extend([dataset_name] * num_points)

    # Concatenate all datasets for joint TSNE
    if not all_embeddings_list:
        logging.warning("No embeddings generated from any experiment.")
        return

    # Ensure dimensions match before concatenating
    ref_dim = all_embeddings_list[0].shape[1]
    for i, emb in enumerate(all_embeddings_list):
        if emb.shape[1] != ref_dim:
            raise ValueError(
                f"Dimension mismatch in dataset {i}: expected {ref_dim}, got {emb.shape[1]}. "
                "Ensure image_size, patch_size, and model architecture are consistent across all datasets."
            )

    logging.info("Computing Joint t-SNE...")
    # Concatenate all tensors then convert to numpy for TSNE
    full_embeddings = torch.cat(all_embeddings_list, dim=0).numpy()
    all_labels = np.array(all_labels_list)  # Convert labels to numpy array

    # now we compute t-SNE on the embeddings
    tsne = TSNE(n_components=2, random_state=cfg.seed)
    embeddings_2d = tsne.fit_transform(full_embeddings)

    logging.info(f"t-SNE completed. Output shape: {embeddings_2d.shape}")

    # Plot the results
    plot_joint_tsne(embeddings_2d, all_labels)

    return


if __name__ == "__main__":
    run()
