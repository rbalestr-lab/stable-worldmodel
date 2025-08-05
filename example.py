# import os
# os.environ["MUJOCO_GL"] = "egl"
import xenoworlds

if __name__ == "__main__":
    # run with MUJOCO_GL=egl python example.py

    # gym.register_envs(gymnasium_robotics)
    # envs = gym.envs.registry.keys()
    # print(envs)
    # asdf
    wrappers = [
        # lambda x: RecordVideo(x, video_folder="./videos"),
        lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: xenoworlds.wrappers.TransformObservation(x),
    ]
    world = xenoworlds.World(
        "dinowm/pusht_noise-v0", num_envs=4, wrappers=wrappers, max_episode_steps=10
    )

    world_model = xenoworlds.DummyWorldModel(
        image_shape=(3, 224, 224), action_dim=world.single_action_space.shape[0]
    )

    # -- create a gradient descent solver
    action_space = world.action_space
    solver = xenoworlds.GDSolver(world_model, n_steps=100, action_space=action_space)

    # -- run evaluation
    evaluator = xenoworlds.Evaluator(world, solver)
    data = evaluator.run(episodes=5)
    # data will be a dict with all the collected metrics
