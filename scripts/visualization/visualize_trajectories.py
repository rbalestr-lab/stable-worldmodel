from collections import OrderedDict
from pathlib import Path

import datasets
import hydra
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from einops import rearrange
from loguru import logger as logging
from omegaconf import open_dict
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoModel, AutoModelForImageClassification

import stable_worldmodel as swm


DINO_PATCH_SIZE = 14  # DINO encoder uses 14x14 patches

# ============================================================================
# Data Loading
# ============================================================================


def get_episodes_length(dataset, episodes):
    episode_idx = dataset["episode_idx"][:]
    step_idx = dataset["step_idx"][:]
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_data(cfg, dataset_cfg, model_cfg):
    """Setup dataset with image transforms and normalization."""

    # Load dataset
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir(), dataset_cfg.dataset_name)
    dataset = datasets.load_from_disk(dataset_path).with_format("numpy")

    # Get unique episode indices
    ep_indices = np.unique(dataset["episode_idx"][:])

    # Compute episode lengths
    episode_len = get_episodes_length(dataset, ep_indices)

    # Randomly sample episodes
    g = np.random.default_rng(cfg.seed)
    assert dataset_cfg.n_trajectories <= len(ep_indices), (
        f"Requested number of trajectories ({dataset_cfg.n_trajectories}) "
        f"exceeds available episodes ({len(ep_indices)}) in dataset {dataset_cfg.dataset_name}."
    )
    num_trajectories = dataset_cfg.n_trajectories
    sampled_ep_indices = g.choice(ep_indices, size=num_trajectories, replace=False)

    # Build start and end steps for full trajectories

    ep_len_dict = {ep_id: episode_len[i] for i, ep_id in enumerate(ep_indices)}

    start_steps = np.zeros(num_trajectories, dtype=int)
    end_steps = []
    for ep_id in sampled_ep_indices:
        # we make sure that the end step allows for full frameskip intervals
        end_steps.append(ep_len_dict[ep_id] - (ep_len_dict[ep_id] % cfg.frameskip))

    end_steps = np.array(end_steps, dtype=int)

    # Load dataset and keep only needed trajectories

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
        num_steps=dataset_cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cfg.get("cache_dir", None),
    )

    all_norm_transforms = []
    # Use global cfg for encoding keys to ensure consistency
    for key in model_cfg.get("encoding", {}):
        trans_fn = norm_col_transform(dataset.dataset, key)
        trans_fn = spt.data.transforms.WrapTorchTransform(trans_fn, source=key, target=key)
        all_norm_transforms.append(trans_fn)

    # Image size must be multiple of DINO patch size (14)
    img_size = (cfg.image_size // cfg.patch_size) * DINO_PATCH_SIZE

    # Apply transforms to all steps
    traj_list = []
    for j in range(num_trajectories):
        transform = spt.data.transforms.Compose(
            *[
                get_img_pipeline(f"{col}.{i}", f"{col}.{i}", img_size)
                for col in ["pixels"]
                for i in range(end_steps[j] // cfg.frameskip)
            ],
            *all_norm_transforms,
        )

        dataset.transform = transform
        data = dataset.load_chunk(
            episode=sampled_ep_indices[j],
            start=start_steps[j],
            end=end_steps[j],
        )
        traj_list.append(data[0])

    with open_dict(model_cfg) as model_cfg:
        model_cfg.extra_dims = {}
        for key in model_cfg.get("encoding", {}):
            if key not in dataset.dataset.column_names:
                raise ValueError(f"Encoding key '{key}' not found in dataset columns.")
            inpt_dim = dataset.dataset[0][key].numel()
            model_cfg.extra_dims[key] = inpt_dim if key != "action" else inpt_dim * cfg.frameskip

    return traj_list


# ============================================================================
# Model Architecture
# ============================================================================


def get_encoder(cfg, model_cfg):
    """Factory function to create encoder based on backbone type."""

    # Define encoder configurations
    ENCODER_CONFIGS = {
        "resnet": {
            "prefix": "microsoft/resnet-",
            "model_class": AutoModelForImageClassification,
            "embedding_attr": lambda model: model.config.hidden_sizes[-1],
            "post_init": lambda model: setattr(model.classifier, "1", torch.nn.Identity()),
            "interpolate_pos_encoding": False,
        },
        "vit": {"prefix": "google/vit-"},
        "dino": {"prefix": "facebook/dino-"},
        "dinov2": {"prefix": "facebook/dinov2-"},
        "dinov3": {"prefix": "facebook/dinov3-"},  # TODO handle resnet base in dinov3
        "mae": {"prefix": "facebook/vit-mae-"},
        "ijepa": {"prefix": "facebook/ijepa"},
        "vjepa2": {"prefix": "facebook/vjepa2-vit"},
        "siglip2": {"prefix": "google/siglip2-"},
    }

    # Find matching encoder
    encoder_type = None
    for name, config in ENCODER_CONFIGS.items():
        if model_cfg.backbone.name.startswith(config["prefix"]):
            encoder_type = name
            break

    if encoder_type is None:
        raise ValueError(f"Unsupported backbone: {model_cfg.backbone.name}")

    config = ENCODER_CONFIGS[encoder_type]

    # Load model
    backbone = config.get("model_class", AutoModel).from_pretrained(model_cfg.backbone.name)

    # CLIP style model
    if hasattr(backbone, "vision_model"):
        backbone = backbone.vision_model

    # Post-initialization if needed (e.g., ResNet)
    if "post_init" in config:
        config["post_init"](backbone)

    # Get embedding dimension
    embedding_dim = config.get("embedding_attr", lambda model: model.config.hidden_size)(backbone)

    # Determine number of patches
    is_cnn = encoder_type == "resnet"
    num_patches = 1 if is_cnn else (cfg.image_size // cfg.patch_size) ** 2

    interp_pos_enc = config.get("interpolate_pos_encoding", True)

    return backbone, embedding_dim, num_patches, interp_pos_enc


def get_world_model(cfg, model_cfg):
    """Load and setup world model.
    For visualization, we only need the model to implement the `encode` method."""

    if model_cfg.model_name is not None:
        model = swm.policy.AutoCostModel(model_cfg.model_name).to(cfg.get("device", "cpu"))
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

        ckpt = torch.load(swm.data.utils.get_cache_dir() / model_cfg.ckpt_path)
        model.backbone = DinoV2Encoder("dinov2_vits14", feature_key="x_norm_patchtokens").to(cfg.get("device", "cpu"))
        model.predictor.load_state_dict(ckpt["predictor"], strict=False)
        model.action_encoder.load_state_dict(ckpt["action_encoder"])
        model.proprio_encoder.load_state_dict(ckpt["proprio_encoder"])
        model = model.to(cfg.get("device", "cpu"))
        model = model.eval()
    else:
        encoder, embedding_dim, num_patches, interp_pos_enc = get_encoder(cfg, model_cfg)
        embedding_dim += sum(emb_dim for emb_dim in model_cfg.get("encoding", {}).values())  # add all extra dims

        logging.info(f"Patches: {num_patches}, Embedding dim: {embedding_dim}")

        # Build causal predictor (transformer that predicts next latent states)

        print(">>>> DIM PREDICTOR:", embedding_dim)

        predictor = swm.wm.pyro.CausalPredictor(
            num_patches=num_patches,
            num_frames=model_cfg.history_size,
            dim=embedding_dim,
            **model_cfg.predictor,
        )

        # Build action and proprioception encoders
        extra_encoders = OrderedDict()
        for key, emb_dim in model_cfg.get("encoding", {}).items():
            inpt_dim = model_cfg.extra_dims[key]
            extra_encoders[key] = swm.wm.pyro.Embedder(in_chans=inpt_dim, emb_dim=emb_dim)
            print(f"Build encoder for {key} with input dim {inpt_dim} and emb dim {emb_dim}")

        extra_encoders = torch.nn.ModuleDict(extra_encoders)

        # Assemble world model
        model = swm.wm.PYRO(
            encoder=spt.backbone.EvalOnly(encoder),
            predictor=predictor,
            extra_encoders=extra_encoders,
            history_size=model_cfg.history_size,
            num_pred=model_cfg.num_preds,
            interpolate_pos_encoding=interp_pos_enc,
        )
        model.to(cfg.get("device", "cpu"))
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
    # load trajectories of the dataset
    trajs = get_data(cfg, exp_cfg.dataset, exp_cfg.world_model)
    world_model = get_world_model(cfg, exp_cfg.world_model)

    logging.info(f"Encoding dataset: {exp_cfg.dataset.dataset_name} using model: {exp_cfg.world_model.model_name}...")
    trajs_embeddings = []  # list to store the embeddings of the trajectories (T x D)
    predicted_embeddings = []  # list to store the predicted embeddings of the trajectories (T x D)
    trajs_pixels = []  # list to store the pixels of the trajectories (T x C x H x W)

    # Process batches and collect embeddings
    for traj in tqdm(trajs, desc=f"Processing {exp_cfg.dataset.dataset_name}"):
        init_state = {}
        for key in traj:
            if isinstance(traj[key], torch.Tensor):
                traj[key] = traj[key].unsqueeze(0).to(cfg.get("device", "cpu"))
                init_state[key] = traj[key][:, :1].unsqueeze(0)

        # store pixels for visualization
        trajs_pixels.append(traj["pixels"].squeeze(0).cpu().detach())

        # Encode trajectory
        traj = world_model.encode(traj, target="embed")
        if exp_cfg.world_model.get("backbone_only", False):  # use only vision backbone embeddings
            flat_embed = rearrange(traj["pixels_embed"][0], "t p d ->t (p d)")
            trajs_embeddings.append(flat_embed.cpu().detach())
        else:  # use full model embeddings (proropio + action + vision)
            flat_embed = rearrange(traj["embed"][0], "t p d ->t (p d)")
            trajs_embeddings.append(flat_embed.cpu().detach())

        # predict trajectory
        traj = world_model.rollout(init_state, traj["action"].unsqueeze(0))
        if exp_cfg.world_model.get("backbone_only", False):  # use only vision backbone embeddings
            flat_predicted = rearrange(traj["predicted_pixels_embed"][0, 0], "t p d ->t (p d)")
            predicted_embeddings.append(flat_predicted.cpu().detach())
        else:  # use full model embeddings (proropio + action + vision)
            flat_predicted = rearrange(traj["predicted_embedding"][0, 0], "t p d ->t (p d)")
            predicted_embeddings.append(flat_predicted.cpu().detach())

    return trajs_embeddings, predicted_embeddings, trajs_pixels


# ============================================================================
# Main Visualization Script
# ============================================================================


def plot_trajectories(embeddings_2d, labels, output_file="trajectories_tsne.pdf"):
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


# TODO have a function that makes a mp4 with pixels of the trajectory next to the tsne point moving in the latent space


@hydra.main(version_base=None, config_path="./configs", config_name="config_trajectories")
def run(cfg):
    """Run visualization script for multiple datasets and compute joint t-SNE."""

    all_embeddings_list = []
    all_predicted_embeddings_list = []
    all_pixels_list = []
    all_labels_list = []  # Added to track source datasets

    # Iterate over all defined datasets and collect embeddings
    for key in cfg.datasets:
        print(f"Processing dataset key: {key}, with {cfg.datasets[key].dataset.n_trajectories} trajectories")
    for exp_cfg in cfg.datasets.values():
        embeddings, predicted_embeddings, trajs_pixels = collect_embeddings(cfg, exp_cfg)

        if embeddings is not None:
            all_embeddings_list.append(embeddings)
            all_predicted_embeddings_list.append(predicted_embeddings)
            all_pixels_list.append(trajs_pixels)

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
    plot_trajectories(embeddings_2d, all_labels)

    return


if __name__ == "__main__":
    run()
