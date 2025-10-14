if __name__ == "__main__":
    import numpy as np
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms

    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/PushT-v1",
        num_envs=2,
        image_shape=(224, 224),
        max_episode_steps=50,
        render_mode="rgb_array",
    )

    print("Available variations: ", world.single_variation_space.names())

    # #######################
    # ##  Data Collection  ##
    # #######################

    # world.set_policy(swm.policy.RandomPolicy())
    # world.record_dataset(
    #     "example-pusht",
    #     episodes=5000,
    #     seed=2347,
    #     options=None,
    # )

    ################
    ##  Pretrain  ##
    ################

    # swm.pretraining(
    #     "scripts/train/dinowm.py",
    #     dataset_name="pusht_expert",  # "example-pusht",
    #     output_model_name="dummy_pusht",
    #     dump_object=True,
    # )

    ################
    ##  Evaluate  ##
    ################

    # NOTE for user: make sure to match action_block with the one used during training!

    ######### TO-DEL #####

    # import stable_pretraining as spt
    # import torch
    # from transformers import AutoModel

    # encoder = AutoModel.from_pretrained("facebook/dinov2-small")
    # emb_dim = encoder.config.hidden_size  # encoder.emb_dim  # 384 for vits14

    # HISTORY_SIZE = 3
    # PREDICTION_HORIZON = 1
    # proprio_dim = 4
    # proprio_emb_dim = 10
    # action_emb_dim = 10
    # image_size = 224  # 224 for dinov2_vits14
    # patch_size = 16  # 16 size for create 14 patches

    # num_patches = (image_size // patch_size) ** 2  # 256 for 224×224

    # # -- create predictor
    # predictor = swm.wm.dinowm.CausalPredictor(
    #     num_patches=num_patches,
    #     num_frames=HISTORY_SIZE,
    #     dim=emb_dim + proprio_emb_dim + action_emb_dim,
    #     depth=6,
    #     heads=16,
    #     mlp_dim=2048,
    #     pool="mean",
    #     dim_head=64,
    #     dropout=0.1,
    #     emb_dropout=0.0,
    # )

    # # -- create action encoder
    # action_encoder = swm.wm.dinowm.Embedder(in_chans=10, emb_dim=action_emb_dim)

    # # -- create proprioceptive encoder
    # proprio_encoder = swm.wm.dinowm.Embedder(in_chans=proprio_dim, emb_dim=proprio_emb_dim)

    # #### LOAD CKPT
    # checkpoint = torch.load("../xenoworlds/dino_wm_pusht_all.pth", map_location="cpu")

    # action_encoder.load_state_dict(checkpoint["action_encoder"])
    # proprio_encoder.load_state_dict(checkpoint["proprio_encoder"])
    # predictor.load_state_dict(checkpoint["predictor"])

    # model = swm.wm.DINOWM(
    #     encoder=spt.backbone.EvalOnly(encoder),
    #     predictor=predictor,
    #     action_encoder=action_encoder,
    #     proprio_encoder=proprio_encoder,
    #     history_size=HISTORY_SIZE,
    #     num_pred=PREDICTION_HORIZON,
    #     device="cuda",
    # ).to("cuda")
    ############### TO DEL

    def img_transform():
        transform = transforms.Compose(
            [
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transform

    action_process = preprocessing.StandardScaler()
    action_process.mean_ = np.array([-0.0087, 0.0068])
    action_process.scale_ = np.array([0.2019, 0.2002])

    proprio_process = preprocessing.StandardScaler()
    proprio_process.mean_ = np.array([236.6155, 264.5674, -2.93032027, 2.54307914])
    proprio_process.scale_ = np.array([101.1202, 87.0112, 74.84556075, 74.14009094])

    transform = {
        "pixels": img_transform(),
        "goal": img_transform(),
    }

    process = {
        "action": action_process,
        "proprio": proprio_process,
        "goal_proprio": proprio_process,
    }

    model = swm.policy.AutoCostModel("dummy_pusht").to("cuda")
    config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    solver = swm.solver.CEMSolver(model, num_samples=300, var_scale=1.0, n_steps=3, topk=30, device="cuda")
    # solver = swm.solver.GDSolver(model, n_steps=10, action_noise=0.0, device="cuda")
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    world.set_policy(policy)
    results = world.evaluate(episodes=3, seed=2347)
    world.record_video(
        "./",
        seed=2347,
    )

    print(results)
