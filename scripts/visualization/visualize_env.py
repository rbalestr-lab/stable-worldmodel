import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from loguru import logger as logging
from sklearn import preprocessing
from sklearn.manifold import TSNE
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht.env import PushT
from stable_worldmodel.envs.two_room.env import TwoRoomEnv
from stable_worldmodel.wrappers import MegaWrapper, VariationWrapper


DINO_PATCH_SIZE = 14  # DINO encoder uses 14x14 patches

# ============================================================================
# Setting up Environment, transform and processing
# ============================================================================


def img_transform():
    transform = transforms.Compose(
        [
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform


def get_env(cfg):
    """Setup dataset with image transforms and normalization."""

    env = gym.make_vec(
        cfg.env.env_name,
        num_envs=1,
        vectorization_mode="sync",
        wrappers=[
            lambda x: MegaWrapper(
                x,
                image_shape=(cfg.image_size, cfg.image_size),
                pixels_transform=None,
                goal_transform=None,
                history_size=cfg.env.history_size,
                frame_skip=cfg.env.frame_skip,
            )
        ]
        + ([]),
        max_episode_steps=50,
        render_mode="rgb_array",
    )

    env = VariationWrapper(env)
    env.unwrapped.autoreset_mode = gym.vector.AutoresetMode.DISABLED

    # create the transform
    transform = {
        "pixels": img_transform(),
        "goal": img_transform(),
    }

    # create the processing
    ACTION_MEAN = np.array([-0.0087, 0.0068])
    ACTION_STD = np.array([0.2019, 0.2002])
    PROPRIO_MEAN = np.array([236.6155, 264.5674, -2.93032027, 2.54307914])
    PROPRIO_STD = np.array([101.1202, 87.0112, 74.84556075, 74.14009094])

    action_process = preprocessing.StandardScaler()
    action_process.mean_ = ACTION_MEAN
    action_process.scale_ = ACTION_STD

    proprio_process = preprocessing.StandardScaler()
    proprio_process.mean_ = PROPRIO_MEAN
    proprio_process.scale_ = PROPRIO_STD

    process = {
        "action": action_process,
        "proprio": proprio_process,
        "goal_proprio": proprio_process,
    }

    return env, process, transform


def prepare_info(info_dict, process, transform):
    # pre-process and transform observations
    for k, v in info_dict.items():
        is_numpy = isinstance(v, (np.ndarray | np.generic))

        if k in process:
            if not is_numpy:
                raise ValueError(f"Expected numpy array for key '{k}' in process, got {type(v)}")

            # flatten extra dimensions if needed
            shape = v.shape
            if len(shape) > 2:
                v = v.reshape(-1, *shape[2:])

            # process and reshape back
            v = process[k].transform(v)
            v = v.reshape(shape)

        # collapse env and time dimensions for transform (e, t, ...) -> (e * t, ...)
        # then restore after transform
        if k in transform:
            shape = None
            if is_numpy or torch.is_tensor(v):
                if v.ndim > 2:
                    shape = v.shape
                    v = v.reshape(-1, *shape[2:])

            v = torch.stack([transform[k](x) for x in v])
            is_numpy = isinstance(v, (np.ndarray | np.generic))

            if shape is not None:
                v = v.reshape(*shape[:2], *v.shape[1:])

        if is_numpy and v.dtype.kind not in "USO":
            v = torch.from_numpy(v)

        info_dict[k] = v

    return info_dict


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


# ============================================================================
# Computing Embeddings
# ============================================================================


def get_state_from_grid(env, grid_element, dim: int | list = 0):
    # first retrieve the reference state depending on the env type
    if isinstance(dim, int):
        dim = [dim]
    if isinstance(env, PushT):
        reference_state = np.concatenate(
            [
                env.variation_space["agent"]["start_position"].value.tolist(),
                env.variation_space["block"]["start_position"].value.tolist(),
                [env.variation_space["block"]["angle"].value],
                env.variation_space["agent"]["velocity"].value.tolist(),
            ]
        )
        # get the positions of the block and the agent closer
        reference_state[2:4] = reference_state[0:2] + 0.3 * (reference_state[2:4] - reference_state[0:2])
    elif isinstance(env, TwoRoomEnv):
        reference_state = np.concatenate(
            [env.variation_space["agent"]["position"].value, env.variation_space["goal"]["position"].value]
        )
    # computing the state from a grid element
    grid_state = reference_state.copy()
    for i, d in enumerate(dim):
        grid_state[d] = grid_element[i]
    if isinstance(env, PushT):
        # relative position of agent and block remains the same
        # we set the position of the block accordingly
        grid_state[2:4] = grid_state[0:2] + (reference_state[2:4] - reference_state[0:2])
    elif isinstance(env, TwoRoomEnv):
        # TODO should check position is feasible
        grid_state
    return grid_state


def get_state_grid(env, grid_size: int = 10):
    logging.info(f"Generating state grid for env type: {type(env)}")

    if isinstance(env, PushT):
        dim = [0, 1]  # Agent X, Y
        # Extract low/high limits for the specified dims
        min_val = [env.variation_space["agent"]["start_position"].low[d] for d in dim]
        max_val = [env.variation_space["agent"]["start_position"].high[d] for d in dim]
        range_val = [max_v - min_v for min_v, max_v in zip(min_val, max_val)]
        # decrease range a bit to avoid unreachable states
        min_val = [min_v + 0.15 * r for min_v, r in zip(min_val, range_val)]
        max_val = [max_v - 0.15 * r for max_v, r in zip(max_val, range_val)]
    elif isinstance(env, TwoRoomEnv):
        dim = [0, 1]  # Agent X, Y
        # Extract low/high limits for the specified dims
        min_val = [env.variation_space["agent"]["position"].low[d] for d in dim]
        max_val = [env.variation_space["agent"]["position"].high[d] for d in dim]
    else:
        raise NotImplementedError(f"State grid generation not implemented for env type: {type(env)}")

    # Create linear spaces for each dimension
    linspaces = [np.linspace(mn, mx, grid_size) for mn, mx in zip(min_val, max_val)]

    # Create the meshgrid and reshape to (N, 2)
    # Using indexing='ij' ensures x varies with axis 0, y with axis 1
    mesh = np.meshgrid(*linspaces, indexing="ij")
    grid = np.stack(mesh, axis=-1).reshape(-1, len(dim))

    # Convert grid points to full state vectors
    state_grid = [get_state_from_grid(env, x, dim) for x in grid]

    return grid, state_grid


def collect_embeddings(world_model, env, process, transform, cfg):
    """Go through the environment and collect embeddings using the world model."""

    grid, state_grid = get_state_grid(env.unwrapped.envs[0].unwrapped, cfg.env.grid_size)
    embeddings = []
    pixels = []
    for state in tqdm(state_grid, desc="Collecting embeddings"):
        options = {"state": state}
        _, infos = env.reset(options=options)
        infos = prepare_info(infos, process, transform)
        for key in infos:
            if isinstance(infos[key], torch.Tensor):
                infos[key] = infos[key].to(cfg.get("device", "cpu"))
        infos = world_model.encode(infos, target="embed")
        embeddings.append(infos["embed"].cpu().detach())
        pixels.append(infos["pixels"][0].cpu().detach())

    return grid, embeddings, pixels


# ============================================================================
# Dimensionality Reduction
# ============================================================================


def compute_tsne(embeddings, cfg):
    """
    Computes t-SNE projection on the collected embeddings.
    """
    logging.info("Computing t-SNE...")
    # Flatten if embeddings are spatial (e.g. from patch tokens)
    # Shape: (N_samples, Embedding_Dim)
    embeddings = rearrange(embeddings, "b ... -> b (...)")

    n_samples = embeddings.shape[0]

    # Perplexity must be < n_samples. Default is 30, which breaks for small grids.
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1

    # Initialize and fit t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=cfg.get("seed", 42))
    embeddings_2d = tsne.fit_transform(embeddings)

    return embeddings_2d


# ============================================================================
# Visualization
# ============================================================================


def plot_distance_maps(grid, embeddings, pixels, save_path="distance_maps.pdf"):
    """
    Plots pairs of (Reference Image, Distance Heatmap).

    Args:
        grid: (H, W, 2) array of physical state coordinates.
        embeddings: (H, W, D) array of high-dim embeddings.
        pixels: (H, W, C, H_img, W_img) or (H, W, H_img, W_img, C) array of images.
    """
    height, width = grid.shape[0], grid.shape[1]
    X = grid[:, :, 0]
    Y = grid[:, :, 1]

    # Define Reference Indices (Top-Left, Top-Right, Center, Bottom-Center)
    ref_indices = [(0, 0), (0, width - 1), (height // 2, width // 2), (height - 1, width // 2)]

    # Create subplots: 4 rows (one per reference), 2 columns (Image, Heatmap)
    # You can also do 1 row with 8 cols, but 4x2 is usually clearer for pairs.
    fig, axes = plt.subplots(len(ref_indices), 2, figsize=(10, 4 * len(ref_indices)))

    for i, (r_idx, c_idx) in enumerate(ref_indices):
        ax_img = axes[i, 0]
        ax_map = axes[i, 1]

        # --- A. Plot Reference Image ---
        ref_img = pixels[r_idx, c_idx]

        # Handle format: Un-normalize and Channels First/Last
        if ref_img.min() < 0:
            ref_img = (ref_img * 0.5) + 0.5

        if ref_img.shape[0] in [1, 3]:
            ref_img = np.moveaxis(ref_img, 0, -1)

        ax_img.imshow(ref_img.clip(0, 1))
        ax_img.set_title("Reference Observation")
        ax_img.axis("off")

        # --- B. Plot Distance Heatmap ---
        ref_emb = embeddings[r_idx, c_idx]
        dists = np.linalg.norm(embeddings - ref_emb, axis=-1)

        contour = ax_map.contourf(X, Y, dists, levels=50, cmap="viridis")

        # Mark reference on the map too
        ref_x = X[r_idx, c_idx]
        ref_y = Y[r_idx, c_idx]
        ax_map.scatter(ref_x, ref_y, c="red", marker="X", s=100, edgecolors="white")

        ax_map.set_title(f"Distance from ({ref_x:.2f}, {ref_y:.2f})")
        ax_map.axis("off")

        # Add colorbar for this row
        cbar = fig.colorbar(contour, ax=ax_map, fraction=0.046, pad=0.04)
        cbar.set_label("L2 Distance")

    plt.suptitle("Latent Distances vs. Visual Input", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save as PDF
    plt.savefig(save_path, format="pdf")
    logging.info(f"Distance maps saved to {save_path}")
    plt.close(fig)


def plot_representations(grid, representations_2d, title_suffix="Latent Space", save_path="latent_vis.pdf"):
    """
    Plots the ground truth grid and the 2D representations side-by-side.
    Colors are generated based on the grid position to visualize topology.
    """
    # Create colors based on grid position (Normalized x, y -> R, G, 0)
    grid_norm = (grid - grid.min(axis=0)) / (grid.max(axis=0) - grid.min(axis=0) + 1e-6)

    colors = np.zeros((len(grid), 4))
    colors[:, 0] = grid_norm[:, 0]  # Red varies with dimension 0
    colors[:, 1] = grid_norm[:, 1]  # Green varies with dimension 1
    colors[:, 2] = 0.5  # Constant Blue
    colors[:, 3] = 1.0  # Alpha

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Ground Truth Grid (Physical State)
    axes[0].scatter(grid[:, 0], grid[:, 1], c=colors, s=50, edgecolor="k", alpha=0.8)
    axes[0].set_title("Physical State Grid")
    axes[0].set_xlabel("State Dim 0")
    axes[0].set_ylabel("State Dim 1")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Plot 2: 2D Representation
    axes[1].scatter(representations_2d[:, 0], representations_2d[:, 1], c=colors, s=50, edgecolor="k", alpha=0.8)
    axes[1].set_title(f"2D Projection ({title_suffix})")
    axes[1].set_xlabel("Projected Dim 1")
    axes[1].set_ylabel("Projected Dim 2")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    # Save as PDF
    plt.savefig(save_path, format="pdf")
    logging.info(f"Visualization saved to {save_path}")
    plt.close(fig)


# ===========================================================================
# Main function
# ===========================================================================


@hydra.main(version_base=None, config_path="./configs", config_name="config_envs")
def run(cfg):
    """Run visualization script."""
    cache_dir = swm.data.utils.get_cache_dir()
    cfg.cache_dir = cache_dir

    env, process, transform = get_env(cfg)
    world_model = get_world_model(cfg)

    # --- Define filenames using environment name ---
    env_name = type(env.unwrapped.envs[0].unwrapped).__name__
    tsne_save_path = f"{env_name}_latent_space_tsne_vis.pdf"
    distmap_save_path = f"{env_name}_latent_distance_maps.pdf"

    logging.info("Computing embeddings from environment...")
    grid, embeddings_list, pixels_list = collect_embeddings(world_model, env, process, transform, cfg)

    # Convert lists to numpy
    embeddings = torch.cat(embeddings_list, dim=0).numpy()
    embeddings = rearrange(embeddings, "b ... -> b (...)")

    # Process pixels: Stack -> Numpy
    # Assuming pixels are Tensors (B, C, H, W), we stack them into (N*N, C, H, W)
    pixels = torch.cat(pixels_list, dim=0).cpu().numpy()

    logging.info(f"Collected {len(embeddings)} samples. Shape: {embeddings.shape}")

    # --- Visualization 1: t-SNE ---
    representations_2d = compute_tsne(embeddings, cfg)
    plot_representations(
        grid,
        representations_2d,
        title_suffix="t-SNE Projection",
        save_path=tsne_save_path,  # Use PDF path
    )

    # --- Visualization 2: Distance Fields ---
    logging.info("Generating distance maps...")

    grid_size = cfg.env.grid_size

    # Reshape all data to (Grid_H, Grid_W, ...)
    grid_reshaped = grid.reshape(grid_size, grid_size, -1)
    embeddings_reshaped = embeddings.reshape(grid_size, grid_size, -1)
    pixels_reshaped = pixels.reshape(grid_size, grid_size, *pixels.shape[1:])

    plot_distance_maps(
        grid_reshaped,
        embeddings_reshaped,
        pixels_reshaped,
        save_path=distmap_save_path,  # Use PDF path
    )

    return


if __name__ == "__main__":
    run()
