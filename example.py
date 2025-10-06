import os
os.environ['MUJOCO_GL'] = 'egl'
import ogbench

if __name__ == "__main__":
    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/OGBCube-v0",
        num_envs=1,
        image_shape=(64, 64),
        # render_mode="rgb_array",
        max_episode_steps=200,
        
        env_type='single',
        ob_type='pixels',
        width=64,
        height=64,
        visualize_info=False,
    )

    print("Available variations: ", world.single_variation_space.names())

    #######################
    ##  Data Collection  ##
    #######################

    world.set_policy(swm.policy.RandomPolicy())
    world.record_dataset(
        "ogbench-cube-single",
        episodes=10,
        seed=2347,
        options=dict(variation=("cube.color", "cube.size", "agent.color")),
    )
    world.record_video(
        "./",
        seed=2347,
        options=dict(variation=("cube.color", "cube.size", "agent.color")),
    )

    ################
    ##  Pretrain  ##
    ################

    # pre-train world model
    swm.pretraining(
        "scripts/train/dummy.py",
        "++dump_object=True dataset_name=ogbench-cube-single output_model_name=dummy_test",
    )

    ################
    ##  Evaluate  ##
    ################

    model = swm.policy.AutoCost("dummy_test")  # auto-cost is confusing
    config = swm.PlanConfig(horizon=10, receding_horizon=5, action_block=5)
    solver = swm.solver.GDSolver(model, n_steps=10)
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config)
    world.set_policy(policy)
    results = world.evaluate(episodes=2, seed=2347)  # , options={...})

    print(results)
