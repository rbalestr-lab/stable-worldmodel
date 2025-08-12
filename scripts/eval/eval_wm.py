import xenoworlds

if __name__ == "__main__":
    wrappers = [
        lambda x: xenoworlds.wrappers.RecordVideo(x, video_folder="./videos"),
        lambda x: xenoworlds.wrappers.AddRenderObservation(
            x, render_only=False, render_key="goal_pixels", obs_key="goal"
        ),
        lambda x: xenoworlds.wrappers.TransformObservation(
            x, source_key="goal_pixels", target_key="goal_pixels"
        ),
        lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: xenoworlds.wrappers.TransformObservation(x),
    ]
    world = xenoworlds.World(
        "xenoworlds/PushT-v1", num_envs=1, wrappers=wrappers, max_episode_steps=500
    )

    world_model = xenoworlds.DummyWorldModel(
        image_shape=(3, 224, 224), action_dim=world.single_action_space.shape[0]
    )

    # -- create a gradient descent solver
    action_space = world.action_space
    # solver = xenoworlds.solver.GDSolver(
    #     world_model, n_steps=100, action_space=action_space
    # )
    # planning_policy = xenoworlds.policy.PlanningPolicy(world, solver)
    random_policy = xenoworlds.policy.RandomPolicy(world)

    # -- run evaluation
    evaluator = xenoworlds.Evaluator(world, random_policy)
    data = evaluator.run(episodes=5)
    # data will be a dict with all the collected metrics
