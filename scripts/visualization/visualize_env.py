import gymnasium as gym
import hydra
import numpy as np
import torch
from einops import rearrange
from loguru import logger as logging
from sklearn import preprocessing
from sklearn.manifold import TSNE
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

import stable_worldmodel as swm
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

    return env, transform, process


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


def get_state_from_grid(env, grid_element):
    grid_state = np.zeros(env.state_dim, dtype=np.float32)
    # fill in the relevant dimensions from grid_element
    return grid_state


def get_state_grid(env, grid_size: int = 10, dim: int | list = 0):
    # for each dimension in dim, create a linspace from min to max with grid_size points
    if isinstance(dim, int):
        dim = [dim]
    linspaces = []
    for d in dim:
        min_val, max_val = env.state_bounds[d]
        linspaces.append(np.linspace(min_val, max_val, grid_size))
    grid = np.array(np.meshgrid(*linspaces)).T.reshape(-1, len(dim))
    state_grid = [get_state_from_grid(env, x) for x in grid]
    return state_grid


def collect_embeddings(world_model, env, process, transform, cfg):
    """Go through the environment and collect embeddings using the world model."""

    state_grid = get_state_grid(env.unwrapped.envs[0], cfg.env.grid_size)
    embeddings = []
    for state in tqdm(state_grid, desc="Collecting embeddings"):
        options = {"state": state}
        _, infos = env.reset(options=options)
        infos = prepare_info(infos, process, transform)
        for key in infos:
            if isinstance(infos[key], torch.Tensor):
                infos[key] = infos[key].to(cfg.get("device", "cpu"))
        infos = world_model.encode(infos, target="embed")
        embeddings.append(infos["embed"].cpu().detach())

    return embeddings


# ===========================================================================
# Main function
# ===========================================================================


@hydra.main(version_base=None, config_path="./", config_name="config")
def run(cfg):
    """Run visualization script."""

    cache_dir = swm.data.utils.get_cache_dir()
    cfg.cache_dir = cache_dir

    env, process, transform = get_env(cfg)
    world_model = get_world_model(cfg)

    # go through the dataset and encode all frames
    # embeddings will be stored in a list
    logging.info("Computing embeddings from environment...")
    embeddings = collect_embeddings(world_model, env, process, transform, cfg)
    # TODO should also return the corresponding states for coloring

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
