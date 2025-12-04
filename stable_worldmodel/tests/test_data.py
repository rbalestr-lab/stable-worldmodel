import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import PIL.Image
import pytest
import torch
from datasets import Dataset

from stable_worldmodel.data import (
    StepsDataset,
    dataset_info,
    delete_dataset,
    delete_model,
    get_cache_dir,
    is_image,
    list_datasets,
    list_models,
    list_worlds,
    world_info,
)


###########################
## is_image tests        ##
###########################


def test_is_image_valid_rgb():
    """Test is_image with valid RGB image."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert is_image(img)


def test_is_image_valid_grayscale():
    """Test is_image with valid grayscale image."""
    img = np.zeros((64, 64, 1), dtype=np.uint8)
    assert is_image(img)


def test_is_image_valid_rgba():
    """Test is_image with valid RGBA image."""
    img = np.zeros((64, 64, 4), dtype=np.uint8)
    assert is_image(img)


def test_is_image_wrong_dtype():
    """Test is_image with wrong dtype."""
    img = np.zeros((64, 64, 3), dtype=np.float32)
    assert not is_image(img)


def test_is_image_wrong_dimensions():
    """Test is_image with wrong number of dimensions."""
    img = np.zeros((64, 64), dtype=np.uint8)
    assert not is_image(img)


def test_is_image_wrong_channels():
    """Test is_image with wrong number of channels."""
    img = np.zeros((64, 64, 5), dtype=np.uint8)
    assert not is_image(img)


def test_is_image_not_array():
    """Test is_image with non-array input."""
    assert not is_image("not an array")
    assert not is_image([1, 2, 3])
    assert not is_image(None)


def test_is_image_four_dimensions():
    """Test is_image with 4D array (batch of images)."""
    img = np.zeros((10, 64, 64, 3), dtype=np.uint8)
    assert not is_image(img)


###########################
## get_cache_dir tests   ##
###########################


def test_get_cache_dir_default():
    """Test get_cache_dir returns default directory."""
    with patch.dict(os.environ, {}, clear=True):
        cache_dir = get_cache_dir()
        assert cache_dir == Path(os.path.expanduser("~/.stable_worldmodel"))


def test_get_cache_dir_custom_env():
    """Test get_cache_dir with custom environment variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_path = str(Path(tmpdir) / "custom_cache")
        with patch.dict(os.environ, {"STABLEWM_HOME": custom_path}):
            cache_dir = get_cache_dir()
            assert cache_dir == Path(custom_path)


def test_get_cache_dir_creates_directory():
    """Test get_cache_dir creates directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test_cache"
        with patch.dict(os.environ, {"STABLEWM_HOME": str(test_path)}):
            cache_dir = get_cache_dir()
            assert cache_dir.exists()
            assert cache_dir.is_dir()


###########################
## list_datasets tests   ##
###########################


def test_list_datasets_empty():
    """Test list_datasets with empty cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            datasets = list_datasets()
            assert datasets == []


def test_list_datasets_with_datasets():
    """Test list_datasets with multiple datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test dataset directories
        (Path(tmpdir) / "dataset1").mkdir()
        (Path(tmpdir) / "dataset2").mkdir()
        (Path(tmpdir) / "dataset3").mkdir()
        # Create a file (should be ignored)
        (Path(tmpdir) / "not_a_dataset.txt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            datasets = list_datasets()
            assert set(datasets) == {"dataset1", "dataset2", "dataset3"}


def test_list_datasets_only_directories():
    """Test list_datasets only returns directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "dataset1").mkdir()
        (Path(tmpdir) / "file.txt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            datasets = list_datasets()
            assert datasets == ["dataset1"]


###########################
## list_models tests     ##
###########################


def test_list_models_empty():
    """Test list_models with empty cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            models = list_models()
            assert models == []


def test_list_models_with_weights():
    """Test list_models with weight checkpoint files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test checkpoint files
        (Path(tmpdir) / "model1_weights.ckpt").touch()
        (Path(tmpdir) / "model2_weights-epoch10.ckpt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            models = list_models()
            assert set(models) == {"model1", "model2"}


def test_list_models_with_objects():
    """Test list_models with object checkpoint files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "model1_object.ckpt").touch()
        (Path(tmpdir) / "model2_object.ckpt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            models = list_models()
            assert set(models) == {"model1", "model2"}


def test_list_models_mixed():
    """Test list_models with mixed checkpoint types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "model1_weights.ckpt").touch()
        (Path(tmpdir) / "model2_object.ckpt").touch()
        (Path(tmpdir) / "model3_weights-best.ckpt").touch()
        (Path(tmpdir) / "not_a_model.txt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            models = list_models()
            assert set(models) == {"model1", "model2", "model3"}


def test_list_models_sorted():
    """Test list_models returns sorted list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "zebra_weights.ckpt").touch()
        (Path(tmpdir) / "alpha_object.ckpt").touch()
        (Path(tmpdir) / "beta_weights.ckpt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            models = list_models()
            assert models == ["alpha", "beta", "zebra"]


def test_list_models_case_insensitive_pattern():
    """Test list_models pattern matching is case insensitive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "Model1_WEIGHTS.CKPT").touch()
        (Path(tmpdir) / "Model2_Object.ckpt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            models = list_models()
            assert set(models) == {"Model1", "Model2"}


###########################
## list_worlds tests     ##
###########################


def test_list_worlds_returns_sorted_list():
    """Test list_worlds returns sorted list of worlds."""
    worlds = list_worlds()
    assert isinstance(worlds, list)
    assert len(worlds) > 0
    assert worlds == sorted(worlds)


def test_list_worlds_contains_expected_worlds():
    """Test list_worlds contains worlds from swm.envs.WORLDS."""
    from stable_worldmodel import envs

    worlds = list_worlds()
    for world in worlds:
        assert world in envs.WORLDS


###########################
## delete_dataset tests  ##
###########################


def test_delete_dataset_success(monkeypatch):
    """Test delete_dataset successfully deletes a dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test dataset structure
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()
        records_path = dataset_path / "records"
        records_path.mkdir()

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.cleanup_cache_files = MagicMock()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with patch("stable_worldmodel.data.load_from_disk", return_value=mock_dataset):
                delete_dataset("test_dataset")

                # Verify dataset was cleaned up
                mock_dataset.cleanup_cache_files.assert_called_once()
                # Verify directory was deleted
                assert not dataset_path.exists()


def test_delete_dataset_not_exists():
    """Test delete_dataset prints error when dataset doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with patch("stable_worldmodel.data.print") as mock_print:
                delete_dataset("nonexistent_dataset")
                # Should print error message
                assert any("Error" in str(call) for call in mock_print.call_args_list)


def test_delete_dataset_error_handling(monkeypatch):
    """Test delete_dataset handles errors gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()
        records_path = dataset_path / "records"
        records_path.mkdir()

        mock_dataset = MagicMock()
        mock_dataset.cleanup_cache_files = MagicMock(side_effect=Exception("Test error"))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with patch("stable_worldmodel.data.load_from_disk", return_value=mock_dataset):
                with patch("stable_worldmodel.data.print") as mock_print:
                    delete_dataset("test_dataset")
                    # Should print error message
                    assert any("Error" in str(call) for call in mock_print.call_args_list)


###########################
## delete_model tests    ##
###########################


def test_delete_model_success():
    """Test delete_model successfully deletes model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test model files
        (Path(tmpdir) / "model1_weights.ckpt").touch()
        (Path(tmpdir) / "model1_object.ckpt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with patch("stable_worldmodel.data.print") as mock_print:
                delete_model("model1")

                # Verify files were deleted
                assert not (Path(tmpdir) / "model1_weights.ckpt").exists()
                assert not (Path(tmpdir) / "model1_object.ckpt").exists()
                # Verify success message was printed
                assert mock_print.call_count == 2


def test_delete_model_partial_match():
    """Test delete_model only deletes matching files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "model1_weights.ckpt").touch()
        (Path(tmpdir) / "model2_weights.ckpt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            delete_model("model1")

            # Only model1 should be deleted
            assert not (Path(tmpdir) / "model1_weights.ckpt").exists()
            assert (Path(tmpdir) / "model2_weights.ckpt").exists()


def test_delete_model_error_handling():
    """Test delete_model handles errors gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "model1_weights.ckpt"
        model_file.touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with patch("os.remove", side_effect=Exception("Permission denied")):
                with patch("stable_worldmodel.data.print") as mock_print:
                    delete_model("model1")
                    # Should print error message
                    assert any("Error" in str(call) for call in mock_print.call_args_list)


def test_delete_model_no_files():
    """Test delete_model when no matching files exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with patch("stable_worldmodel.data.print") as mock_print:
                delete_model("nonexistent_model")
                # No files deleted, so print shouldn't be called
                mock_print.assert_not_called()


###########################
## dataset_info tests    ##
###########################


def test_dataset_info_not_found():
    """Test dataset_info raises error when dataset not found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with pytest.raises(ValueError, match="not found"):
                dataset_info("nonexistent_dataset")


def test_dataset_info_success():
    """Test dataset_info returns correct information."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset" / "records"
        dataset_path.mkdir(parents=True)

        # Create mock dataset
        mock_data = {
            "episode_idx": [0, 0, 1, 1, 1],
            "step_idx": [0, 1, 0, 1, 2],
            "episode_len": [2, 2, 3, 3, 3],
            "pixels": [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)],
            "action": [np.array([0.0, 1.0]) for _ in range(5)],
            "goal": [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            info = dataset_info("test_dataset")

            assert info["name"] == "test_dataset"
            assert info["num_episodes"] == 2
            assert info["num_steps"] == 5
            assert "pixels" in info["columns"]
            assert "action" in info["columns"]
            assert info["obs_shape"] == (64, 64, 3)
            assert info["action_shape"] == (2,)


def test_dataset_info_with_variation():
    """Test dataset_info detects variation columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset" / "records"
        dataset_path.mkdir(parents=True)

        mock_data = {
            "episode_idx": [0],
            "step_idx": [0],
            "episode_len": [1],
            "pixels": [np.zeros((64, 64, 3), dtype=np.uint8)],
            "action": [np.array([0.0])],
            "goal": [np.zeros((64, 64, 3), dtype=np.uint8)],
            "variation.color": ["red"],
            "variation.size": ["large"],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            info = dataset_info("test_dataset")

            assert info["variation"]["has_variation"]
            assert set(info["variation"]["names"]) == {"color", "size"}


def test_dataset_info_missing_required_columns():
    """Test dataset_info with missing required columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset" / "records"
        dataset_path.mkdir(parents=True)

        # Dataset missing 'action' column
        mock_data = {
            "episode_idx": [0],
            "step_idx": [0],
            "episode_len": [1],
            "pixels": [np.zeros((64, 64, 3), dtype=np.uint8)],
            "goal": [np.zeros((64, 64, 3), dtype=np.uint8)],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with pytest.raises(AssertionError, match="action"):
                dataset_info("test_dataset")


###########################
## world_info tests      ##
###########################


def test_world_info_invalid_world():
    """Test world_info raises error for invalid world."""
    # Clear the cache before running this test
    world_info.cache_clear()

    with pytest.raises(ValueError, match="not found"):
        world_info("nonexistent_world")


@patch("stable_worldmodel.data.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_success(mock_world_class):
    """Test world_info returns correct information."""
    # Clear the cache before running this test
    world_info.cache_clear()

    # Mock the World instance
    mock_world = MagicMock()
    mock_world.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    mock_world.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    mock_world.single_variation_space = None
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    info = world_info("test_world")

    assert info["name"] == "test_world"
    assert "observation_space" in info
    assert "action_space" in info
    assert info["variation"]["has_variation"] is False


@patch("stable_worldmodel.data.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_with_variation(mock_world_class):
    """Test world_info with variation space."""
    # Clear the cache before running this test
    world_info.cache_clear()

    mock_world = MagicMock()
    mock_world.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    mock_world.single_action_space = gym.spaces.Discrete(4)

    # Mock variation space
    mock_variation_space = MagicMock()
    mock_variation_space.names = MagicMock(return_value=["color", "size"])
    mock_world.single_variation_space = mock_variation_space
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    info = world_info("test_world")

    assert info["variation"]["has_variation"]
    assert info["variation"]["names"] == ["color", "size"]


@patch("stable_worldmodel.data.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_closes_world(mock_world_class):
    """Test world_info properly closes the world."""
    # Clear the cache before running this test
    world_info.cache_clear()

    mock_world = MagicMock()
    mock_world.single_observation_space = gym.spaces.Discrete(5)
    mock_world.single_action_space = gym.spaces.Discrete(4)
    mock_world.single_variation_space = None
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    world_info("test_world")
    mock_world.close.assert_called_once()


@patch("stable_worldmodel.data.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_closes_world_on_error(mock_world_class):
    """Test world_info closes world even if error occurs."""
    # Clear the cache before running this test
    world_info.cache_clear()

    mock_world = MagicMock()
    mock_world.single_observation_space = None
    mock_world.single_action_space = None
    mock_world.single_variation_space = None
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    # Should not raise, and should close world
    world_info("test_world")
    mock_world.close.assert_called_once()


@patch("stable_worldmodel.data.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_caching(mock_world_class):
    """Test world_info uses caching for repeated calls."""
    # Clear the cache before running this test
    world_info.cache_clear()

    mock_world = MagicMock()
    mock_world.single_observation_space = gym.spaces.Discrete(5)
    mock_world.single_action_space = gym.spaces.Discrete(4)
    mock_world.single_variation_space = None
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    # First call
    info1 = world_info("test_world")
    # Second call should use cache
    info2 = world_info("test_world")

    # Should only create world once due to caching
    assert mock_world_class.call_count == 1
    assert info1 == info2


@patch("stable_worldmodel.data.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_with_dict_space(mock_world_class):
    """Test world_info with Dict observation space."""
    # Clear the cache before running this test
    world_info.cache_clear()

    mock_world = MagicMock()
    # Create a Dict space with multiple sub-spaces
    mock_world.single_observation_space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "vector": gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        }
    )
    mock_world.single_action_space = gym.spaces.Discrete(4)
    mock_world.single_variation_space = None
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    info = world_info("test_world")

    assert info["name"] == "test_world"
    assert "observation_space" in info
    # Check that Dict space was properly handled
    assert isinstance(info["observation_space"], dict)
    assert "image" in info["observation_space"]
    assert "vector" in info["observation_space"]


@patch("stable_worldmodel.data.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_with_tuple_space(mock_world_class):
    """Test world_info with Tuple action space."""
    # Clear the cache before running this test
    world_info.cache_clear()

    mock_world = MagicMock()
    mock_world.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    # Create a Tuple space with multiple sub-spaces
    mock_world.single_action_space = gym.spaces.Tuple(
        (
            gym.spaces.Discrete(4),
            gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        )
    )
    mock_world.single_variation_space = None
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    info = world_info("test_world")

    assert info["name"] == "test_world"
    assert "action_space" in info
    # Check that Tuple space was properly handled as a list
    assert isinstance(info["action_space"], list)
    assert len(info["action_space"]) == 2


###########################
## StepsDataset tests    ##
###########################


def test_steps_dataset_initialization():
    """Test StepsDataset initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()

        # Create mock dataset
        mock_data = {
            "episode_idx": [0, 0, 0, 1, 1, 1],
            "step_idx": [0, 1, 2, 0, 1, 2],
            "action": [np.array([0.0, 1.0]) for _ in range(6)],
            "pixels": [f"img_{i}.png" for i in range(6)],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            steps_dataset = StepsDataset("test_dataset", num_steps=2, frameskip=1)

            assert steps_dataset.num_steps == 2
            assert steps_dataset.frameskip == 1
            assert len(steps_dataset.episodes) == 2


def test_steps_dataset_missing_required_columns():
    """Test StepsDataset raises error for missing required columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()

        # Missing 'episode_idx' column
        mock_data = {
            "step_idx": [0, 1, 2],
            "action": [np.array([0.0]) for _ in range(3)],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with pytest.raises(AssertionError, match="episode_idx"):
                StepsDataset("test_dataset", num_steps=2)


def test_steps_dataset_episode_too_short():
    """Test StepsDataset raises error when episode is too short."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()

        # Episode with only 1 step, but num_steps=2
        mock_data = {
            "episode_idx": [0],
            "step_idx": [0],
            "action": [np.array([0.0])],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with pytest.raises(ValueError, match="too short"):
                StepsDataset("test_dataset", num_steps=2, frameskip=1)


def test_steps_dataset_length():
    """Test StepsDataset __len__ returns correct length."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()

        # Episode with 5 steps
        mock_data = {
            "episode_idx": [0, 0, 0, 0, 0],
            "step_idx": [0, 1, 2, 3, 4],
            "action": [np.array([0.0]) for _ in range(5)],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            # num_steps=2, so we can have 4 possible slices (0-1, 1-2, 2-3, 3-4)
            steps_dataset = StepsDataset("test_dataset", num_steps=2, frameskip=1)
            assert len(steps_dataset) == 4


def test_steps_dataset_getitem():
    """Test StepsDataset __getitem__ returns correct data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()

        # Create actual image files
        for i in range(5):
            img = PIL.Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
            img.save(dataset_path / f"img_{i}.png")

        # Episode with 5 steps
        mock_data = {
            "episode_idx": [0, 0, 0, 0, 0],
            "step_idx": [0, 1, 2, 3, 4],
            "action": [np.array([float(i), float(i)]) for i in range(5)],
            "pixels": [f"img_{i}.png" for i in range(5)],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        # Define a transform that converts PIL images to tensors
        def image_to_tensor(batch):
            for key in ["pixels"]:
                if key in batch and isinstance(batch[key][0], PIL.Image.Image):
                    batch[key] = [
                        torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in batch[key]
                    ]
            return batch

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            steps_dataset = StepsDataset("test_dataset", num_steps=2, frameskip=1, transform=image_to_tensor)

            # Get first slice (steps 0-1)
            sample = steps_dataset[0]

            assert "action" in sample
            assert "pixels" in sample
            assert isinstance(sample["action"], torch.Tensor)
            assert sample["action"].shape == (2, 2)  # num_steps=2, action_dim=2


def test_steps_dataset_frameskip():
    """Test StepsDataset with frameskip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()

        # Create image files
        for i in range(10):
            img = PIL.Image.new("RGB", (32, 32), color=(i * 20, i * 20, i * 20))
            img.save(dataset_path / f"img_{i}.png")

        # Episode with 10 steps
        mock_data = {
            "episode_idx": [0] * 10,
            "step_idx": list(range(10)),
            "action": [np.array([float(i)]) for i in range(10)],
            "pixels": [f"img_{i}.png" for i in range(10)],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        # Define a transform that converts PIL images to tensors
        def image_to_tensor(batch):
            for key in ["pixels"]:
                if key in batch and isinstance(batch[key][0], PIL.Image.Image):
                    batch[key] = [
                        torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in batch[key]
                    ]
            return batch

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            # num_steps=2, frameskip=2 means we need 4 steps total (0, 2 for observations)
            steps_dataset = StepsDataset("test_dataset", num_steps=2, frameskip=2, transform=image_to_tensor)

            sample = steps_dataset[0]

            # Should have 2 frames (with frameskip=2)
            assert sample["pixels"].shape[0] == 2


def test_steps_dataset_multiple_episodes():
    """Test StepsDataset with multiple episodes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()

        # Create image files
        for i in range(8):
            img = PIL.Image.new("RGB", (32, 32))
            img.save(dataset_path / f"img_{i}.png")

        # Two episodes with 4 steps each
        mock_data = {
            "episode_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "step_idx": [0, 1, 2, 3, 0, 1, 2, 3],
            "action": [np.array([0.0]) for _ in range(8)],
            "pixels": [f"img_{i}.png" for i in range(8)],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            steps_dataset = StepsDataset("test_dataset", num_steps=2, frameskip=1)

            # Each episode has 4 steps, so 3 valid slices per episode = 6 total
            assert len(steps_dataset) == 6


def test_steps_dataset_infer_img_path_columns():
    """Test StepsDataset infer_img_path_columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()

        # Create image files
        img = PIL.Image.new("RGB", (32, 32))
        img.save(dataset_path / "img_0.png")
        img.save(dataset_path / "goal_0.jpg")

        mock_data = {
            "episode_idx": [0, 0, 0],
            "step_idx": [0, 1, 2],
            "action": [np.array([0.0]) for _ in range(3)],
            "pixels": ["img_0.png", "img_0.png", "img_0.png"],
            "goal": ["goal_0.jpg", "goal_0.jpg", "goal_0.jpg"],
            "other_data": [1, 2, 3],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            steps_dataset = StepsDataset("test_dataset", num_steps=2, frameskip=1)

            assert "pixels" in steps_dataset.img_cols
            assert "goal" in steps_dataset.img_cols
            assert "other_data" not in steps_dataset.img_cols


def test_steps_dataset_with_transform():
    """Test StepsDataset with transform function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()

        # Create image files
        for i in range(3):
            img = PIL.Image.new("RGB", (32, 32))
            img.save(dataset_path / f"img_{i}.png")

        mock_data = {
            "episode_idx": [0, 0, 0],
            "step_idx": [0, 1, 2],
            "action": [np.array([0.0]) for _ in range(3)],
            "pixels": [f"img_{i}.png" for i in range(3)],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        # Define a simple transform
        def mock_transform(batch):
            batch["transformed"] = True
            # Also convert images to tensors to avoid stacking error
            for key in ["pixels"]:
                if key in batch and isinstance(batch[key][0], PIL.Image.Image):
                    batch[key] = [
                        torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in batch[key]
                    ]
            return batch

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            steps_dataset = StepsDataset("test_dataset", num_steps=2, frameskip=1, transform=mock_transform)

            sample = steps_dataset[0]
            assert "transformed" in sample
            assert sample["transformed"] is True


def test_steps_dataset_custom_cache_dir():
    """Test StepsDataset with custom cache_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_cache = Path(tmpdir) / "custom_cache"
        dataset_path = custom_cache / "test_dataset"
        dataset_path.mkdir(parents=True)

        mock_data = {
            "episode_idx": [0, 0, 0],
            "step_idx": [0, 1, 2],
            "action": [np.array([0.0]) for _ in range(3)],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        steps_dataset = StepsDataset("test_dataset", num_steps=2, frameskip=1, cache_dir=str(custom_cache))

        assert steps_dataset.data_dir == dataset_path


def test_steps_dataset_action_reshape():
    """Test StepsDataset correctly reshapes actions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()

        # Actions with shape (2,) that span multiple steps
        # With num_steps=2 and frameskip=1, we need actions for both observation steps
        # The dataset stores actions at full resolution
        mock_data = {
            "episode_idx": [0, 0, 0],
            "step_idx": [0, 1, 2],
            "action": [np.array([float(i), float(i) * 2]) for i in range(3)],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            steps_dataset = StepsDataset("test_dataset", num_steps=2, frameskip=1)

            sample = steps_dataset[0]

            # Action should be reshaped to (num_steps, action_dim)
            assert sample["action"].shape == (2, 2)


###########################
## FinancialDataset tests##
###########################


def test_alpaca_client_initialization_without_credentials():
    """Test that data module handles missing Alpaca credentials gracefully."""
    import subprocess
    import sys

    # Run a subprocess that imports the module without credentials
    # This ensures lines 63-64 (the else clause) get executed
    test_code = """
import os
# Remove Alpaca credentials
os.environ.pop('ALPACA_API_KEY', None)
os.environ.pop('ALPACA_SECRET_KEY', None)

# Now import the module - this will execute lines 63-64
from stable_worldmodel.data import FinancialDataset
# If we get here, module loaded successfully without credentials
print('SUCCESS')
"""

    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True,
        env={**os.environ, "ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": ""},
    )

    # Check that module loaded successfully
    assert "SUCCESS" in result.stdout or result.returncode == 0


def test_financial_dataset_encode_bfloat16():
    """Test encode_bfloat16 with valid price series."""
    try:
        from stable_worldmodel.data import FinancialDataset
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    import pandas as pd

    prices = pd.Series([100.0, 101.5, 99.8, 102.3])
    encoded = FinancialDataset.encode_bfloat16(prices)

    assert isinstance(encoded, np.ndarray)
    assert encoded.dtype == np.uint16
    assert len(encoded) == len(prices)


def test_financial_dataset_encode_bfloat16_no_ml_dtypes():
    """Test encode_bfloat16 raises ImportError when ml_dtypes not available."""
    from stable_worldmodel.data import FinancialDataset

    with patch("stable_worldmodel.data.ml_dtypes", None):
        import pandas as pd

        prices = pd.Series([100.0, 101.5, 99.8])
        with pytest.raises(ImportError, match="ml_dtypes required for bfloat16 encoding"):
            FinancialDataset.encode_bfloat16(prices)


def test_financial_dataset_decode_bfloat16():
    """Test decode_bfloat16 with valid encoded data."""
    try:
        from stable_worldmodel.data import FinancialDataset
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    import pandas as pd

    prices = pd.Series([100.0, 101.5, 99.8, 102.3])
    encoded = FinancialDataset.encode_bfloat16(prices)
    anchor = float(prices.iloc[0])

    decoded = FinancialDataset.decode_bfloat16(encoded, anchor)

    # Check decoded values are close to originals (bfloat16 has lower precision)
    assert len(decoded) == len(prices)
    np.testing.assert_allclose(decoded, prices.values, rtol=1e-2)


def test_financial_dataset_decode_bfloat16_no_ml_dtypes():
    """Test decode_bfloat16 raises ImportError when ml_dtypes not available."""
    from stable_worldmodel.data import FinancialDataset

    with patch("stable_worldmodel.data.ml_dtypes", None):
        encoded = np.array([1, 2, 3], dtype=np.uint16)
        anchor = 100.0
        with pytest.raises(ImportError, match="ml_dtypes required for bfloat16 decoding"):
            FinancialDataset.decode_bfloat16(encoded, anchor)


def test_financial_dataset_encode_financial_data():
    """Test encode_financial_data with OHLC DataFrame."""
    try:
        from stable_worldmodel.data import FinancialDataset
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    import pandas as pd

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )

    df_encoded, metadata = FinancialDataset.encode_financial_data(df)

    # Check encoded columns exist
    assert "open_encoded" in df_encoded.columns
    assert "high_encoded" in df_encoded.columns
    assert "low_encoded" in df_encoded.columns
    assert "close_encoded" in df_encoded.columns

    # Check original OHLC columns removed
    assert "open" not in df_encoded.columns
    assert "high" not in df_encoded.columns
    assert "low" not in df_encoded.columns
    assert "close" not in df_encoded.columns

    # Check metadata has anchors
    assert "open_anchor" in metadata
    assert "high_anchor" in metadata
    assert "low_anchor" in metadata
    assert "close_anchor" in metadata

    # Check timestamp and volume preserved
    assert "timestamp" in df_encoded.columns
    assert "volume" in df_encoded.columns


def test_financial_dataset_decode_financial_data():
    """Test decode_financial_data with encoded DataFrame."""
    try:
        from stable_worldmodel.data import FinancialDataset
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    import pandas as pd

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )

    df_encoded, metadata = FinancialDataset.encode_financial_data(df)
    df_decoded = FinancialDataset.decode_financial_data(df_encoded, metadata)

    # Check OHLC columns restored
    assert "open" in df_decoded.columns
    assert "high" in df_decoded.columns
    assert "low" in df_decoded.columns
    assert "close" in df_decoded.columns

    # Check encoded columns removed
    assert "open_encoded" not in df_decoded.columns
    assert "high_encoded" not in df_decoded.columns
    assert "low_encoded" not in df_decoded.columns
    assert "close_encoded" not in df_decoded.columns

    # Check values are close to originals
    np.testing.assert_allclose(df_decoded["open"].values, df["open"].values, rtol=1e-2)
    np.testing.assert_allclose(df_decoded["high"].values, df["high"].values, rtol=1e-2)
    np.testing.assert_allclose(df_decoded["low"].values, df["low"].values, rtol=1e-2)
    np.testing.assert_allclose(df_decoded["close"].values, df["close"].values, rtol=1e-2)


def test_financial_dataset_load_with_existing_data():
    """Test load method with existing cached data."""
    try:
        from stable_worldmodel.data import FinancialDataset
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        storage_dir = base_dir / "numpyformat_bfloat16"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Create mock data file
        symbol = "AAPL"
        date = "2024-01-17"
        timestamps = pd.date_range("2024-01-17 09:30", periods=10, freq="min")
        prices_open = np.array([100.0 + i * 0.5 for i in range(10)])
        prices_high = prices_open + 0.5
        prices_low = prices_open - 0.3
        prices_close = prices_open + 0.2
        volumes = np.array([1000 + i * 100 for i in range(10)])

        # Encode data
        df_raw = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": prices_open,
                "high": prices_high,
                "low": prices_low,
                "close": prices_close,
                "volume": volumes,
            }
        )

        df_encoded, metadata = FinancialDataset.encode_financial_data(df_raw)

        # Save to file
        data_dict = {
            "timestamps": df_encoded["timestamp"].values,
            "open_encoded": df_encoded["open_encoded"].values,
            "high_encoded": df_encoded["high_encoded"].values,
            "low_encoded": df_encoded["low_encoded"].values,
            "close_encoded": df_encoded["close_encoded"].values,
            "volume": df_encoded["volume"].values,
        }
        for key, value in metadata.items():
            data_dict[key] = np.array([value])

        file_path = storage_dir / f"{symbol}_{date}.npz"
        np.savez_compressed(file_path, **data_dict)

        # Test load
        result = FinancialDataset.load(stocks=[symbol], dates=[date], base_dir=str(base_dir))

        assert not result.empty
        assert len(result) == 10
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns


def test_financial_dataset_load_with_date_range():
    """Test load method with date range including weekends."""
    try:
        from stable_worldmodel.data import FinancialDataset
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        storage_dir = base_dir / "numpyformat_bfloat16"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Create mock data for weekdays only (2024-01-15 is Monday, 01-19 is Friday)
        symbol = "TSLA"
        weekdays = [
            "2024-01-15",
            "2024-01-16",
            "2024-01-17",
            "2024-01-18",
            "2024-01-19",
        ]

        for date in weekdays:
            timestamps = pd.date_range(f"{date} 09:30", periods=5, freq="min")
            prices_open = np.array([200.0 + i * 0.5 for i in range(5)])
            df_raw = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": prices_open,
                    "high": prices_open + 1.0,
                    "low": prices_open - 0.5,
                    "close": prices_open + 0.3,
                    "volume": np.array([2000 + i * 100 for i in range(5)]),
                }
            )

            df_encoded, metadata = FinancialDataset.encode_financial_data(df_raw)
            data_dict = {
                "timestamps": df_encoded["timestamp"].values,
                "open_encoded": df_encoded["open_encoded"].values,
                "high_encoded": df_encoded["high_encoded"].values,
                "low_encoded": df_encoded["low_encoded"].values,
                "close_encoded": df_encoded["close_encoded"].values,
                "volume": df_encoded["volume"].values,
            }
            for key, value in metadata.items():
                data_dict[key] = np.array([value])

            file_path = storage_dir / f"{symbol}_{date}.npz"
            np.savez_compressed(file_path, **data_dict)

        # Test load with date range that includes weekend
        result = FinancialDataset.load(
            stocks=[symbol],
            dates=[("2024-01-15", "2024-01-21")],
            base_dir=str(base_dir),
        )

        assert not result.empty
        # Should only load weekdays (5 days * 5 data points = 25)
        assert len(result) == 25


def test_financial_dataset_load_empty_result():
    """Test load method returns empty DataFrame when no data available."""
    import pandas as pd

    from stable_worldmodel.data import FinancialDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Test with no cached data and no alpaca client
        with patch("stable_worldmodel.data._alpaca_client", None):
            result = FinancialDataset.load(stocks=["AAPL"], dates=["2024-01-17"], base_dir=str(base_dir))

            assert isinstance(result, pd.DataFrame)
            assert result.empty


def test_financial_dataset_load_with_alpaca_download():
    """Test load method with Alpaca API download for missing data."""
    try:
        from stable_worldmodel.data import FinancialDataset
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        storage_dir = base_dir / "numpyformat_bfloat16"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Mock Alpaca client
        mock_client = MagicMock()
        mock_response = MagicMock()

        # Create mock DataFrame with proper structure
        timestamps = pd.date_range("2024-01-17 09:30", periods=5, freq="min", tz="UTC")
        mock_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [150.0, 150.5, 151.0, 151.5, 152.0],
                "high": [150.5, 151.0, 151.5, 152.0, 152.5],
                "low": [149.5, 150.0, 150.5, 151.0, 151.5],
                "close": [150.2, 150.7, 151.2, 151.7, 152.2],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        mock_response.df.reset_index.return_value = mock_df
        mock_client.get_stock_bars.return_value = mock_response

        with patch("stable_worldmodel.data._alpaca_client", mock_client):
            with patch("stable_worldmodel.data.StockBarsRequest"):
                result = FinancialDataset.load(stocks=["NVDA"], dates=["2024-01-17"], base_dir=str(base_dir))

                # Check that Alpaca API was called
                assert mock_client.get_stock_bars.called
                assert not result.empty

                # Check that data was saved to file
                file_path = storage_dir / "NVDA_2024-01-17.npz"
                assert file_path.exists()


def test_financial_dataset_load_with_alpaca_empty_response():
    """Test load handles empty response from Alpaca API."""
    try:
        from stable_worldmodel.data import FinancialDataset
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Mock Alpaca client returning empty DataFrame
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.df.reset_index.return_value = pd.DataFrame()
        mock_client.get_stock_bars.return_value = mock_response

        with patch("stable_worldmodel.data._alpaca_client", mock_client):
            with patch("stable_worldmodel.data.StockBarsRequest"):
                result = FinancialDataset.load(stocks=["TEST"], dates=["2024-01-17"], base_dir=str(base_dir))

                # Should return empty DataFrame
                assert result.empty


def test_financial_dataset_load_with_alpaca_exception():
    """Test load handles exceptions from Alpaca API gracefully."""
    try:
        from stable_worldmodel.data import FinancialDataset
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Mock Alpaca client that raises exception
        mock_client = MagicMock()
        mock_client.get_stock_bars.side_effect = Exception("API error")

        with patch("stable_worldmodel.data._alpaca_client", mock_client):
            with patch("stable_worldmodel.data.StockBarsRequest"):
                result = FinancialDataset.load(stocks=["TEST"], dates=["2024-01-17"], base_dir=str(base_dir))

                # Should return empty DataFrame
                assert result.empty


def test_financial_dataset_load_multiple_stocks_and_dates():
    """Test load with multiple stocks and dates."""
    try:
        from stable_worldmodel.data import FinancialDataset
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        storage_dir = base_dir / "numpyformat_bfloat16"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Create data for multiple stocks and dates
        symbols = ["AAPL", "TSLA"]
        dates = ["2024-01-17", "2024-01-18"]

        for symbol in symbols:
            for date in dates:
                timestamps = pd.date_range(f"{date} 09:30", periods=3, freq="min")
                prices = np.array([100.0 + i for i in range(3)])
                df_raw = pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "open": prices,
                        "high": prices + 0.5,
                        "low": prices - 0.3,
                        "close": prices + 0.2,
                        "volume": np.array([1000 + i * 100 for i in range(3)]),
                    }
                )

                df_encoded, metadata = FinancialDataset.encode_financial_data(df_raw)
                data_dict = {
                    "timestamps": df_encoded["timestamp"].values,
                    "open_encoded": df_encoded["open_encoded"].values,
                    "high_encoded": df_encoded["high_encoded"].values,
                    "low_encoded": df_encoded["low_encoded"].values,
                    "close_encoded": df_encoded["close_encoded"].values,
                    "volume": df_encoded["volume"].values,
                }
                for key, value in metadata.items():
                    data_dict[key] = np.array([value])

                file_path = storage_dir / f"{symbol}_{date}.npz"
                np.savez_compressed(file_path, **data_dict)

        result = FinancialDataset.load(stocks=symbols, dates=dates, base_dir=str(base_dir))

        assert not result.empty
        # 2 stocks * 2 dates * 3 data points = 12
        assert len(result) == 12


def test_financial_dataset_load_default_base_dir():
    """Test load with default base_dir (None)."""
    try:
        from stable_worldmodel.data import FinancialDataset
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock DATA_DIR to point to our temp directory
        with patch.object(FinancialDataset, "DATA_DIR", tmpdir):
            storage_dir = Path(tmpdir) / "numpyformat_bfloat16"
            storage_dir.mkdir(parents=True, exist_ok=True)

            # Create mock data
            symbol = "MSFT"
            date = "2024-01-17"
            timestamps = pd.date_range(f"{date} 09:30", periods=3, freq="min")
            prices = np.array([300.0, 301.0, 302.0])
            df_raw = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": prices,
                    "high": prices + 1.0,
                    "low": prices - 0.5,
                    "close": prices + 0.5,
                    "volume": np.array([5000, 5100, 5200]),
                }
            )

            df_encoded, metadata = FinancialDataset.encode_financial_data(df_raw)
            data_dict = {
                "timestamps": df_encoded["timestamp"].values,
                "open_encoded": df_encoded["open_encoded"].values,
                "high_encoded": df_encoded["high_encoded"].values,
                "low_encoded": df_encoded["low_encoded"].values,
                "close_encoded": df_encoded["close_encoded"].values,
                "volume": df_encoded["volume"].values,
            }
            for key, value in metadata.items():
                data_dict[key] = np.array([value])

            file_path = storage_dir / f"{symbol}_{date}.npz"
            np.savez_compressed(file_path, **data_dict)

            # Test with base_dir=None (should use DATA_DIR)
            result = FinancialDataset.load(stocks=[symbol], dates=[date], base_dir=None)

            assert not result.empty
            assert len(result) == 3
