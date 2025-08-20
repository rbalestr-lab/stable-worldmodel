from pathlib import Path
import torch
from loguru import logger as logging
from transformers import AutoConfig, AutoModel
import torchvision.transforms.v2 as transforms
import xenoworlds


class Config:
    """Configuration for PUSHT Eval"""

    # experiments
    root_dir: str = "./results/"
    exp_name: str = "dinowm_agent_color"

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

    encoder = xenoworlds.wm.DinoV2Encoder(
        "dinov2_vits14", feature_key="x_norm_patchtokens"
    )

    emb_dim = encoder.emb_dim  # 384 for vits14
    patch_size = 16  # 16 size for create 14 patches
    num_patches = (Config.img_size // patch_size) ** 2  # 256 for 224×224

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

    decoder = xenoworlds.wm.VQVAE(
        channel=384,
        n_embed=2048,
        n_res_block=4,
        n_res_channel=128,
        quantize=False,
        emb_dim=384,
    )
    logging.info(f"Decoder: {decoder}")

    load_ckpt(action_encoder, "dino_wm_ckpt/pusht/checkpoints/action_encoder.pth")
    load_ckpt(proprio_encoder, "dino_wm_ckpt/pusht/checkpoints/proprio_encoder.pth")
    load_ckpt(predictor, "dino_wm_ckpt/pusht/checkpoints/predictor.pth")
    load_ckpt(decoder, "dino_wm_ckpt/pusht/checkpoints/decoder.pth")

    # -- world model as a stable_ssl module
    world_model = xenoworlds.wm.DINOWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        decoder=decoder,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        action_emb_dim=Config.action_emb_dim,
        proprio_emb_dim=Config.proprio_emb_dim,
        history_size=Config.num_hist,
        image_size=Config.img_size,
        frameskip=Config.frameskip,
        device=device,
        action_mean=Config.action_mean,
        action_std=Config.action_std,
        proprio_mean=Config.proprio_mean,
        proprio_std=Config.proprio_std,
    ).to(device)

    return world_model


# @hydra.main(version_base=None, config_path="./", config_name="slurm")
def run():
    """Run training of predictor"""

    exp_path = Path(Config.root_dir) / Config.exp_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- make transform operations
    def default_transform(img_size=224):
        return transforms.Compose([
            transforms.ToImage(),
            transforms.Lambda(lambda img: img.clone()),
            transforms.Lambda(lambda img: img / 255.0),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    deform_wrappers = [
        lambda x: xenoworlds.ColorDeform(
            x,
            target=["agent"],
            every_k_steps=-1,
        ),
    ]

    wrappers = deform_wrappers + [
        lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: xenoworlds.wrappers.TransformObservation(
            x, transform=default_transform()
        ),
    ]

    goal_wrappers = deform_wrappers + [
        lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: xenoworlds.wrappers.TransformObservation(
            x, transform=default_transform()
        ),
    ]

    world = xenoworlds.World(
        "xenoworlds/PushT-v1",
        num_envs=10,
        wrappers=wrappers,
        max_episode_steps=Config.horizon * Config.frameskip,
        goal_wrappers=goal_wrappers,
        seed=torch.randint(0, 10000, (1,)).item(),
        output_dir=exp_path,
    )

    # -- reproduce dino-wm pusht results (world wrapper)
    expert_dataset = xenoworlds.tmp.PushTDataset(
        data_path="pusht_noise/val/", normalize_action=False
    )

    world = xenoworlds.PushTRolloutCompletion(
        world,
        horizon=Config.horizon * Config.frameskip,
        transforms=default_transform(),
        dataset=expert_dataset,
    )
    world = xenoworlds.SaveInitAndGoal(world)

    action_dim = world.single_action_space.shape[-1]
    proprio_dim = world.single_observation_space["proprio"].shape[-1]
    world_model = get_world_model(action_dim, proprio_dim, device=device)

    cem_solver = xenoworlds.solver.CEMSolver(
        world_model,
        horizon=Config.horizon,
        num_samples=300,
        var_scale=1.0,
        opt_steps=30,
        action_dim=action_dim * Config.frameskip,
        topk=30,
        device=device,
    )

    cem_solver = xenoworlds.solver.MPCWrapper(cem_solver, n_mpc_actions=1)

    policy = xenoworlds.policy.PlanningPolicy(world, cem_solver, output_dir=exp_path)

    # -- run evaluation
    evaluator = xenoworlds.evaluator.Evaluator(
        world, policy, output_dir=exp_path, device=device
    )
    data = evaluator.run(episodes=1)


if __name__ == "__main__":
    run()
