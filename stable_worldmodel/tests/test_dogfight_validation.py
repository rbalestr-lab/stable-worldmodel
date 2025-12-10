"""Validation tests for PyFlyt Dogfight environment."""

import tempfile

import stable_worldmodel as swm


def test_dogfight_basic_functionality():
    """Test that PyFlyt Dogfight environment works with all basic features."""

    # Test 1: Environment creation
    world = swm.World(
        'swm/PyFlytDogfight-v0',
        num_envs=2,
        image_shape=(224, 224),
        max_episode_steps=10,
        render_mode='rgb_array',
        verbose=0,
    )

    # Test 2: Reset
    world.reset(seed=42)
    assert world.single_observation_space is not None
    assert world.single_action_space is not None
    assert world.single_variation_space is not None

    # Test 3: Policy and stepping
    policy = swm.policy.RandomPolicy(42)
    world.set_policy(policy)

    for i in range(5):
        world.step()

    # Test 4: Dataset recording
    temp_dir = tempfile.mkdtemp()
    world.record_dataset("test_dogfight_dataset", episodes=1, seed=123, cache_dir=temp_dir)

    # Test 5: Video recording
    world.reset(seed=456)
    import os
    video_dir = f"{temp_dir}/test_video"
    os.makedirs(video_dir, exist_ok=True)
    world.record_video(video_dir, max_steps=10, seed=456)

    # Test 6: Domain randomization
    world.reset(seed=789, options={'variation': ['all']})

    print("✓ All PyFlyt Dogfight environment features working correctly")


def test_dogfight_observation_structure():
    """Test that observations have the correct structure."""

    world = swm.World(
        'swm/PyFlytDogfight-v0',
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=10,
        render_mode='rgb_array',
        verbose=0,
    )

    world.reset(seed=42)

    # Check infos structure
    assert "action" in world.infos
    assert "pixels" in world.infos

    # Check image shape
    assert world.infos["pixels"].shape[:4] == (1, 1, 224, 224)

    policy = swm.policy.RandomPolicy(42)
    world.set_policy(policy)
    world.step()

    # Check after step
    assert world.infos["pixels"].shape[:4] == (1, 1, 224, 224)

    print("✓ PyFlyt Dogfight observations have correct structure")


def test_dogfight_variation_space():
    """Test that variation space is properly defined."""

    world = swm.World(
        'swm/PyFlytDogfight-v0',
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=10,
        render_mode='rgb_array',
        verbose=0,
    )

    var_space = world.single_variation_space

    # Check that key variation parameters exist
    assert "team1" in var_space.spaces
    assert "team2" in var_space.spaces
    assert "environment" in var_space.spaces
    assert "lighting" in var_space.spaces

    # Check team variations
    assert "color" in var_space.spaces["team1"].spaces
    assert "spawn_height_range" in var_space.spaces["team1"].spaces

    # Check environment variations
    assert "spawn_separation" in var_space.spaces["environment"].spaces

    # Check lighting variations
    assert "ambient_color" in var_space.spaces["lighting"].spaces

    print("✓ PyFlyt Dogfight variation space properly defined")
