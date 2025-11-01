if __name__ == "__main__":
    import datasets
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms

    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/PushT-v1",
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=25,
        render_mode="rgb_array",
    )

    print("Available variations: ", world.single_variation_space.names())

    # #######################
    # ##  Data Collection  ##
    # #######################

    world.set_policy(swm.policy.RandomPolicy())
    # world.record_dataset(
    #     "example-pusht",
    #     episodes=10,
    #     seed=2347,
    #     options=None,
    # )

    # world.record_video_from_dataset(
    #     "./",
    #     "example-pusht",
    #     episode_idx=[0, 1],
    # )

    ################
    ##  Pretrain  ##
    ################

    # swm.pretraining(
    #     "scripts/train/dinowm.py",
    #     dataset_name="example-pusht",
    #     output_model_name="dummy_pusht",
    #     dump_object=True,
    # )

    #########################
    ##  Transform/Process  ##
    #########################

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

    transform = {
        "pixels": img_transform(),
        "goal": img_transform(),
    }

    dataset_path = swm.data.get_cache_dir() / "full_pusht"  # "example-pusht"
    dataset = datasets.load_from_disk(dataset_path).with_format("numpy")

    action_process = preprocessing.StandardScaler()
    action_process.fit(dataset["action"][:])

    proprio_process = preprocessing.StandardScaler()
    proprio_process.fit(dataset["proprio"][:])

    process = {
        "action": action_process,
        "proprio": proprio_process,
        "goal_proprio": proprio_process,
    }

    ################
    ##  Evaluate  ##
    ################

    model = swm.policy.AutoCostModel("world_model_epoch_99").to("cuda")
    config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    solver = swm.solver.CEMSolver(model, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device="cuda")
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    world.set_policy(policy)

    metrics = world.evaluate_from_dataset(
        "pusht_eval",
        start_steps=[135],
        episodes_idx=[11345],
        num_steps=25,
        callables={"_set_state": "state", "_set_goal_state": "goal_state"},
    )

    print("Evaluation Metrics: ", metrics)
