if __name__ == "__main__":
    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/PushT-v1",
        num_envs=10,
        image_shape=(224, 224),
        max_episode_steps=200,
        render_mode="rgb_array",
    )

    print("Available variations: ", world.single_variation_space.names())

    # #######################
    # ##  Data Collection  ##
    # #######################

    world.set_policy(swm.policy.RandomPolicy())
    world.record_dataset(
        "example-record",
        episodes=2,
        seed=2347,
        options=None,
    )

    dataset = swm.data.FrameDataset("example-record")
    world.record_video_from_dataset(
        "./",
        dataset,
        episode_idx=[0, 1],
    )
