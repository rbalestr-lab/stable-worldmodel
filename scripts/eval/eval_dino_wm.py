import torch
from loguru import logger as logging
from transformers import AutoConfig, AutoModel

import xenoworlds


class Config:
    """Configuration for PUSHT Eval"""

    # encoder
    img_size: int = 224
    num_hist: int = 3
    num_pred: int = 1
    frameskip: int = 5
    horizon: int = 5
    num_patches: int = 1
    action_emb_dim: int = 10
    proprio_emb_dim: int = 10
    normalize_action: True

    # -- precomputed pusht dataset stats
    # code taken from original repo
    action_mean = torch.tensor([-0.0087, 0.0068])
    action_std = torch.tensor([0.2019, 0.2002])
    proprio_mean = torch.tensor([236.6155, 264.5674, -2.93032027, 2.54307914])
    proprio_std = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])
    state_mean = torch.tensor([
        236.6155,
        264.5674,
        255.1307,
        266.3721,
        1.9584,
        -2.93032027,
        2.54307914,
    ])
    state_std = torch.tensor([
        101.1202,
        87.0112,
        52.7054,
        57.4971,
        1.7556,
        74.84556075,
        74.14009094,
    ])


def get_world_model(action_dim, proprio_dim, device="cpu"):
    """Return stable_ssl module with world model"""

    def load_ckpt(module, name):
        module.load_state_dict(torch.load(name, map_location="cpu"))
        return

    # config = AutoConfig.from_pretrained("facebook/dinov2-small")
    # encoder = AutoModel.from_config(config)
    encoder = xenoworlds.wm.DinoV2Encoder(
        "dinov2_vits14", feature_key="x_norm_patchtokens"
    )
    emb_dim = 384  # config.hidden_size
    patch_size = 16  # config.patch_size they used 16!
    num_patches = (Config.img_size // patch_size) ** 2

    logging.info(f"Encoder: {encoder}, emb_dim: {emb_dim}, num_patches: {num_patches}")

    encoder.train(False)
    encoder.requires_grad_(False)

    encoder.eval()

    # -- create predictor
    predictor = xenoworlds.wm.CausalPredictor(
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
    action_encoder = xenoworlds.wm.Embedder(
        in_chans=action_dim * Config.frameskip, emb_dim=Config.action_emb_dim
    )

    logging.info(f"Action dim: {action_dim}, action emb dim: {Config.action_emb_dim}")

    # -- create proprioceptive encoder
    proprio_encoder = xenoworlds.wm.Embedder(
        in_chans=proprio_dim, emb_dim=Config.proprio_emb_dim
    )
    logging.info(
        f"Proprio dim: {proprio_dim}, proprio emb dim: {Config.proprio_emb_dim}"
    )

    # NOTE: can add a decoder here if needed

    load_ckpt(action_encoder, "dino_wm_ckpt/pusht/checkpoints/action_encoder.pth")
    load_ckpt(proprio_encoder, "dino_wm_ckpt/pusht/checkpoints/proprio_encoder.pth")
    load_ckpt(predictor, "dino_wm_ckpt/pusht/checkpoints/predictor.pth")

    # -- world model as a stable_ssl module
    world_model = xenoworlds.wm.DINOWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        action_emb_dim=Config.action_emb_dim,
        proprio_emb_dim=Config.proprio_emb_dim,
        history_size=Config.num_hist,
        frameskip=Config.frameskip,
        device=device,
    ).to(device)

    return world_model


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

    goal_wrappers = [
        lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: xenoworlds.wrappers.TransformObservation(x),
    ]

    world = xenoworlds.World(
        "xenoworlds/PushT-v1",
        num_envs=1,
        wrappers=wrappers,
        max_episode_steps=100,
        goal_wrappers=goal_wrappers,
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

    # -- create a random policy
    # policy = xenoworlds.policy.RandomPolicy(world)
    planning_solver = xenoworlds.solver.GDSolver(
        world_model,
        n_steps=100,
        action_space=world.action_space,
        horizon=Config.horizon,
    )
    policy = xenoworlds.policy.PlanningPolicy(world, planning_solver)

    # -- run evaluation
    evaluator = xenoworlds.evaluator.Evaluator(world, policy, device=device)
    data = evaluator.run(episodes=2)

    # data will be a dict with all the collected metrics

    # # visualize a rollout video (e.g. for debugging purposes)
    # xenoworlds.utils.save_rollout_videos(data["frames_list"])


if __name__ == "__main__":
    run()
