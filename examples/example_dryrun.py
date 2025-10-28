if __name__ == "__main__":
    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/SimplePointMaze-v0",
        num_envs=5,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )

    print("Available variations: ", world.single_variation_space.names())

    # #######################
    # ##  Data Collection  ##
    # #######################
    policy = swm.policy.RandomPolicy(42)
    world.set_policy(policy)

    world.evaluate_from_dataset(
        "simple-pointmaze",
        episodes_idx=[0, 1, 2, 2, 2],
        start_steps=[0, 1, 2, 3, 4],
        num_steps=3,
    )

    # world.record_dataset("simple-pointmaze", episodes=10, seed=2347, options=None)

    # world.set_policy(policy)
    # world.record_video(
    #     "./",
    #     seed=2347,
    #     options={"variation": ("walls.number", "walls.shape", "walls.positions")},
    # )

    # world.record_video_from_dataset(
    #     "./",
    #     "simple-pointmaze",
    #     episode_idx=list(range(10)),
    # )
    # world.record_video(
    #     "./",
    #     seed=2347,
    #     options={"variation": ("walls.number", "walls.shape", "walls.positions")},
    # )

    # ################
    # ##  Pretrain  ##
    # ################

    # # pre-train world model
    # swm.pretraining(
    #     "scripts/train/dummy.py",
    #     dataset_name="simple-pointmaze",
    #     output_model_name="dummy_test",
    #     dump_object=True,
    # )

    # ################
    # ##  Evaluate  ##
    # ################

    # model = swm.policy.AutoCostModel("dummy_test")  # auto-policy name is confusing
    # config = swm.PlanConfig(horizon=10, receding_horizon=5, action_block=5)
    # solver = swm.solver.GDSolver(model, n_steps=10)
    # policy = swm.policy.WorldModelPolicy(solver=solver, config=config)
    # world.set_policy(policy)
    # results = world.evaluate(episodes=2, seed=2347)  # , options={...})

    # print(results)
