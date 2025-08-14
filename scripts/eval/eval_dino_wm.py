import hydra
import lightning as pl
import minari
import stable_ssl as ssl
import torch
import xenoworlds
from einops import rearrange, repeat
from loguru import logger as logging
from stable_ssl.data import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel

from xenoworlds.predictor import CausalPredictor

## TAKEN FROM THEIR REPO
# precomputed dataset stats
ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])
STATE_MEAN = torch.tensor([
    236.6155,
    264.5674,
    255.1307,
    266.3721,
    1.9584,
    -2.93032027,
    2.54307914,
])
STATE_STD = torch.tensor([
    101.1202,
    87.0112,
    52.7054,
    57.4971,
    1.7556,
    74.84556075,
    74.14009094,
])
PROPRIO_MEAN = torch.tensor([236.6155, 264.5674, -2.93032027, 2.54307914])
PROPRIO_STD = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])


class Config:
    """Configuration for the training script"""

    # encoder
    img_size: int = 224
    num_hist: int = 3
    num_pred: int = 1
    frameskip: int = 5
    num_patches: int = 1
    action_emb_dim: int = 10
    proprio_emb_dim: int = 10
    normalize_action: True


class Embedder(torch.nn.Module):
    def __init__(
        self,
        num_frames=1,
        tubelet_size=1,
        in_chans=8,
        emb_dim=10,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim

        self.patch_embed = torch.nn.Conv1d(
            in_chans, emb_dim, kernel_size=tubelet_size, stride=tubelet_size
        )

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)  # (B, T, B) -> (B, D, T)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        return x


class DINOWM(torch.nn.Module):
    def __init__(
        self, encoder, predictor, action_encoder, proprio_encoder, device="cpu"
    ):
        super().__init__()
        self.device = device
        self.backbone = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.proprio_encoder = proprio_encoder

    def forward(self, states, actions):  # obs, actions, proprio):
        """world model forward pass"""

        obs = torch.from_numpy(states["pixels"]).float()  # (B, T, C, H, W)
        proprio = torch.from_numpy(states["proprio"]).float()  # (B, T, P)

        obs = obs.unsqueeze(1)  # (B, T, C, H, W)
        proprio = proprio.unsqueeze(1)  # (B, T, P)
        actions = actions.unsqueeze(1)  # (B, T, A)

        # normalize proprio and actions
        actions = (actions - ACTION_MEAN) / ACTION_STD
        proprio = (proprio - PROPRIO_MEAN) / PROPRIO_STD

        obs = obs.to(self.device)  # (B, T, C, H, W)
        actions = actions.to(self.device)  # (B, T, A)
        proprio = proprio.to(self.device)  # (B, T, P)

        def encode(obs, actions, proprio):
            # -- encode actions
            actions = self.action_encoder(actions)  # (B,T,A) -> (B,T,A_emb)

            # -- encode proprioceptive
            proprio = self.proprio_encoder(proprio)  # (B,T,P) -> (B,T,P_emb)

            # -- encode observations to get states
            B, T, C, H, W = obs.shape
            obs = rearrange(obs, "b t ... -> (b t) ...")

            # get the state
            state = self.backbone(obs).last_hidden_state  # (B*T, n_patches, D)
            state = state[:, 1:, :]  # drop cls token

            # state = torch.nn.functional.normalize(state, dim=-1)

            state = z = rearrange(state, "(b t) p d -> b t p d", b=B)

            # -- merge state, action, proprio
            n_patches = state.shape[2]
            # share action/proprio embedding accros patches for each time step
            proprio_repeat = repeat(
                proprio.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches
            )
            actions_repeat = repeat(
                actions.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches
            )
            # z (B, T, P, dim+A_emb+P_emb)
            z = torch.cat([state, actions_repeat, proprio_repeat], dim=3)
            return z

        # -- predict the next state
        def predict(z):
            T = z.shape[1]
            z = rearrange(z, "b t p d -> b (t p) d")
            preds = self.predictor(z)
            preds = rearrange(preds, "b (t p) d -> b t p d", t=T)
            return preds

        # -- preprocess inputs
        if type(actions) is dict:
            actions = [a.flatten(2) for a in actions.values()]
            actions = torch.cat(actions, -1)
        else:
            actions.flatten(2)

        # -- compute prediction error
        z = encode(obs, actions, proprio)
        z_preds = predict(z)

        # TODO should check from their code
        # z_src = z[:, : Config.num_hist, :, :]
        # z_tgt = z[:, Config.num_pred :, :, :]

        # keep only the part corresponding to the visual features
        z_pred_visual = z_preds[..., : -Config.action_emb_dim - Config.proprio_emb_dim]

        return z_pred_visual

    def encode_goal(self, goal_pixel):
        """Preprocess goal pixel images for the world model"""
        goal_pixel = goal_pixel.to(self.device)  # (B, T, C, H, W)
        goal = self.backbone(goal_pixel).last_hidden_state
        # drop cls token
        goal = goal[:, 1:, :]  # (B, n_patches, D
        return goal.unsqueeze(1)  # (B, 1, n_patches, D)


def get_world_model(action_dim, proprio_dim, device="cpu"):
    """Return stable_ssl module with world model"""

    def load_ckpt(module, name):
        module.load_state_dict(torch.load(name, map_location="cpu"))
        return

    config = AutoConfig.from_pretrained("facebook/dinov2-small")
    encoder = AutoModel.from_config(config)
    emb_dim = config.hidden_size
    patch_size = 16  # config.patch_size they used 16!
    num_patches = (Config.img_size // patch_size) ** 2

    logging.info(f"Encoder: {encoder}, emb_dim: {emb_dim}, num_patches: {num_patches}")

    encoder.train(False)
    encoder.requires_grad_(False)

    encoder.eval()

    # -- create predictor
    predictor = CausalPredictor(
        num_patches=num_patches,
        num_frames=Config.num_hist,
        dim=emb_dim + Config.proprio_emb_dim + Config.action_emb_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        pool="mean",
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    )
    logging.info(f"Predictor: {predictor}")

    # -- create action encoder
    action_encoder = Embedder(in_chans=action_dim, emb_dim=Config.action_emb_dim)
    logging.info(f"Action dim: {action_dim}, action emb dim: {Config.action_emb_dim}")

    # -- create proprioceptive encoder
    proprio_encoder = Embedder(in_chans=proprio_dim, emb_dim=Config.proprio_emb_dim)
    logging.info(
        f"Proprio dim: {proprio_dim}, proprio emb dim: {Config.proprio_emb_dim}"
    )

    # NOTE: can add a decoder here if needed

    # load_ckpt(
    #     action_encoder, "dino_wm_ckpt/pusht/checkpoints/action_encoder.pth"
    # )

    load_ckpt(
        proprio_encoder,
        "dino_wm_ckpt/pusht/checkpoints/proprio_encoder.pth",
    )
    load_ckpt(predictor, "dino_wm_ckpt/pusht/checkpoints/predictor.pth")

    # -- world model as a stable_ssl module
    world_model = DINOWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        device=device,
    )

    return world_model.to(device)


# @hydra.main(version_base=None, config_path="./", config_name="slurm")
def run():
    """Run training of predictor"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- make transform operations
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    wrappers = [
        lambda x: xenoworlds.wrappers.RecordVideo(x, video_folder="./videos"),
        lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: xenoworlds.wrappers.TransformObservation(x, mean=mean, std=std),
    ]

    world = xenoworlds.World(
        "xenoworlds/PushT-v1", num_envs=1, wrappers=wrappers, max_episode_steps=100
    )

    action_dim = world.single_action_space.shape[-1]
    proprio_dim = world.single_observation_space["proprio"].shape[-1]

    print(f"Action space dim: {action_dim}")
    print(f"Proprioceptive space dim: {proprio_dim}")

    world_model = get_world_model(action_dim, proprio_dim, device=device)

    print(f"World model: {world_model}")

    # world_model = xenoworlds.DummyWorldModel(
    #     image_shape=(3, 224, 224), action_dim=world.single_action_space.shape[0]
    # )

    # -- create a planning policy with a gradient descent solver
    # solver = xenoworlds.solver.GDSolver(world_model, n_steps=100, action_space=world.action_space)
    # policy = xenoworlds.policy.PlanningPolicy(world, solver)
    # -- create a random policy
    # policy = xenoworlds.policy.RandomPolicy(world)
    planning_solver = xenoworlds.solver.GDSolver(
        world_model, n_steps=100, action_space=world.action_space
    )
    planning_policy = xenoworlds.policy.PlanningPolicy(world, planning_solver)

    # -- run evaluation
    evaluator = xenoworlds.evaluator.Evaluator(world, planning_policy)
    data = evaluator.run(episodes=1)

    # data will be a dict with all the collected metrics

    # # visualize a rollout video (e.g. for debugging purposes)
    # xenoworlds.utils.save_rollout_videos(data["frames_list"])


if __name__ == "__main__":
    run()
