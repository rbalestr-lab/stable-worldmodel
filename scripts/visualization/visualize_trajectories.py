from collections import OrderedDict
from pathlib import Path

import datasets
import hydra
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from einops import rearrange
from loguru import logger as logging
from omegaconf import open_dict
from scipy.interpolate import interp1d
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
        )[0]
        data["id"] = torch.ones(data["pixels"].shape[0], 1) * sampled_ep_indices[j]
        traj_list.append(data)

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
# Visualization Functions
# ============================================================================


def reconstruct_trajectories(flat_embeddings_2d, trajectory_lengths):
    """
    Splits the flattened t-SNE results back into a list of trajectory arrays.
    """
    trajs_2d = []
    start_idx = 0
    for length in trajectory_lengths:
        end_idx = start_idx + length
        trajs_2d.append(flat_embeddings_2d[start_idx:end_idx])
        start_idx = end_idx
    return trajs_2d


def plot_static_trajectories(trajs_2d, labels, output_file="trajectories_tsne.pdf", interp_factor=10):
    """
    Plots all trajectories as sequences of points on the same 2D plane.

    Args:
        trajs_2d: List of (N, 2) numpy arrays.
        labels: List of labels corresponding to trajs_2d.
        output_file: Filename for the saved plot.
        interp_factor: How many times to increase point density.
                       1 = original points, 10 = 10x denser.
    """
    plt.figure(figsize=(12, 10))

    unique_labels = np.unique(labels)
    # Use colormaps directly
    cmap = plt.cm.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))

    added_labels = set()

    for traj, label in zip(trajs_2d, labels):
        # Skip trajectories that are too short to interpolate
        if len(traj) < 2:
            continue

        base_color = label_color_map[label]
        rgb = mcolors.to_rgb(base_color)

        # --- Interpolation Step ---
        # Original indices (0, 1, 2, ... N-1)
        x_indices = np.arange(len(traj))

        # New finer indices (0, 0.1, 0.2, ... N-1)
        # We generate 'interp_factor' times more points
        new_indices = np.linspace(0, len(traj) - 1, num=len(traj) * interp_factor)

        # Create interpolation functions for x and y coordinates
        # kind='linear' ensures we strictly follow the straight path between points
        f_x = interp1d(x_indices, traj[:, 0], kind="linear")
        f_y = interp1d(x_indices, traj[:, 1], kind="linear")

        # Generate the new dense trajectory
        traj_dense_x = f_x(new_indices)
        traj_dense_y = f_y(new_indices)

        num_points = len(traj_dense_x)

        # --- Color Gradient Step ---
        point_colors = np.zeros((num_points, 4))
        point_colors[:, :3] = rgb
        # Alpha increases from 0.05 to 1.0 (start lighter since points are denser)
        point_colors[:, 3] = np.linspace(0.05, 1.0, num_points)

        # Plot the dense points
        # s=5: reduced size because points are now much closer together
        plt.scatter(traj_dense_x, traj_dense_y, c=point_colors, s=5, edgecolors="none")

        # --- Markers ---
        # Plot markers at the REAL original start and end locations
        plt.scatter(
            traj[0, 0], traj[0, 1], color=base_color, marker="o", s=40, alpha=1.0, edgecolors="white", zorder=10
        )
        plt.scatter(traj[-1, 0], traj[-1, 1], color=base_color, marker="x", s=40, alpha=1.0, linewidth=2, zorder=10)

        # Legend handling
        if label not in added_labels:
            plt.scatter([], [], color=base_color, label=label, s=30)
            added_labels.add(label)

    plt.title("Trajectory Embeddings (Interpolated)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Static trajectory plot saved to {output_file}")


def create_video_visualization(trajs_2d, trajs_pixels, labels, output_file="trajectories_video.mp4", max_trajs=5):
    """
    Creates an MP4 with pixel video on left and moving t-SNE trail on right.
    - Colors match the static plot (per dataset).
    - Latent space shows a 'trail' where past points fade out.
    """
    # Select a subset of trajectories to visualize
    indices = np.linspace(0, len(trajs_2d) - 1, min(len(trajs_2d), max_trajs), dtype=int)

    subset_trajs_2d = [trajs_2d[i] for i in indices]
    subset_pixels = [trajs_pixels[i] for i in indices]
    subset_labels = [labels[i] for i in indices]

    n_rows = len(subset_trajs_2d)
    max_len = max([t.shape[0] for t in subset_trajs_2d])

    # --- Color Setup ---
    unique_labels = np.unique(labels)
    # Using the same colormap as the static plot for consistency
    cmap = plt.cm.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))

    # Setup Figure
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Pre-calculate bounds
    all_points = np.vstack(subset_trajs_2d)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    margin = (x_max - x_min) * 0.1

    img_plots = []
    scatter_plots = []

    for i in range(n_rows):
        ax_vid = axes[i, 0]
        ax_lat = axes[i, 1]

        label = subset_labels[i]
        base_color = label_color_map[label]

        # --- Left: Video Frame ---
        # Initialize with first frame
        frame0 = subset_pixels[i][0].permute(1, 2, 0).numpy()
        frame0 = (frame0 - frame0.min()) / (frame0.max() - frame0.min() + 1e-6)

        img_plot = ax_vid.imshow(frame0)
        ax_vid.set_title(f"{label}")
        ax_vid.axis("off")
        img_plots.append(img_plot)

        # --- Right: Latent Space ---
        # Plot the FULL static trajectory faintly in the background for context
        ax_lat.plot(
            subset_trajs_2d[i][:, 0], subset_trajs_2d[i][:, 1], color=base_color, alpha=0.15, linewidth=1, zorder=1
        )

        # Initialize the dynamic scatter plot (the "trail")
        # We start with empty data
        scat_plot = ax_lat.scatter([], [], color=base_color, s=20, zorder=2)

        ax_lat.set_xlim(x_min - margin, x_max + margin)
        ax_lat.set_ylim(y_min - margin, y_max + margin)
        ax_lat.set_title("Latent Space Trail")
        scatter_plots.append(scat_plot)

    def update(frame_idx):
        artists = []
        for i in range(n_rows):
            traj_len = len(subset_trajs_2d[i])
            idx = min(frame_idx, traj_len - 1)

            # --- Update Video ---
            frame = subset_pixels[i][idx].permute(1, 2, 0).numpy()
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
            img_plots[i].set_data(frame)
            artists.append(img_plots[i])

            # --- Update Latent Trail ---
            # Get the path history up to the current frame
            current_path = subset_trajs_2d[i][: idx + 1]

            # Create the gradient colors
            base_rgb = mcolors.to_rgb(label_color_map[subset_labels[i]])
            num_points = len(current_path)

            rgba_colors = np.zeros((num_points, 4))
            rgba_colors[:, :3] = base_rgb
            # Alpha increases from 0.05 (oldest) to 1.0 (current/newest)
            rgba_colors[:, 3] = np.linspace(0.05, 1.0, num_points)

            # Update position and color
            scatter_plots[i].set_offsets(current_path)
            scatter_plots[i].set_color(rgba_colors)

            # Make the current "head" point slightly larger if desired (optional)
            # scatter_plots[i].set_sizes(...)

            artists.append(scatter_plots[i])

        return artists

    logging.info(f"Generating animation with {n_rows} trajectories over {max_len} frames...")
    # interval=100ms -> 10fps
    ani = animation.FuncAnimation(fig, update, frames=max_len, interval=100, blit=True)
    ani.save(output_file, fps=10, extra_args=["-vcodec", "libx264"])
    plt.close()
    logging.info(f"Animation saved to {output_file}")


# ============================================================================
# Main Run Loop
# ============================================================================


@hydra.main(version_base=None, config_path="./configs", config_name="config_trajectories")
def run(cfg):
    """Run visualization script for multiple datasets and compute joint t-SNE."""

    all_embeddings_list = []
    # all_predicted_embeddings_list = []
    all_pixels_list = []
    all_labels_list = []  # Added to track source datasets
    trajectory_lengths = []  # Track lengths to split t-SNE results later

    # Iterate over all defined datasets and collect embeddings
    if cfg.datasets:
        for key in cfg.datasets:
            print(f"Processing dataset key: {key}")
            exp_cfg = cfg.datasets[key]

            # Note: Assuming collect_embeddings returns list of tensors
            embeddings, _, trajs_pixels = collect_embeddings(cfg, exp_cfg)

            if embeddings is not None:
                all_embeddings_list.extend(embeddings)
                all_pixels_list.extend(trajs_pixels)

                dataset_name = exp_cfg.dataset.dataset_name
                num_traj = len(embeddings)

                # Store labels for each trajectory
                all_labels_list.extend([dataset_name] * num_traj)

                # Store lengths
                for emb in embeddings:
                    trajectory_lengths.append(emb.shape[0])

    # Concatenate all datasets for joint TSNE
    if not all_embeddings_list:
        logging.warning("No embeddings generated from any experiment.")
        return

    # Check dims
    ref_dim = all_embeddings_list[0].shape[1]
    for i, emb in enumerate(all_embeddings_list):
        if emb.shape[1] != ref_dim:
            raise ValueError(f"Dimension mismatch. Expected {ref_dim}, got {emb.shape[1]}")

    logging.info(f"Computing Joint t-SNE on {len(all_embeddings_list)} trajectories...")

    # Flatten everything into (N_total_frames, Dim)
    full_embeddings = torch.cat(all_embeddings_list, dim=0).numpy()

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=cfg.seed)
    embeddings_2d_flat = tsne.fit_transform(full_embeddings)

    logging.info(f"t-SNE completed. Output shape: {embeddings_2d_flat.shape}")

    # Reconstruct list of (T, 2) arrays
    trajs_2d = reconstruct_trajectories(embeddings_2d_flat, trajectory_lengths)

    # Plot 1: Static 2D Latent Space
    plot_static_trajectories(trajs_2d, all_labels_list, output_file="trajectories_tsne.pdf")

    # Plot 2: Video Visualization
    # Note: Pass max_trajs to avoid creating a video with 100 rows
    create_video_visualization(
        trajs_2d, all_pixels_list, all_labels_list, output_file="trajectories_video.mp4", max_trajs=4
    )

    return


if __name__ == "__main__":
    run()
