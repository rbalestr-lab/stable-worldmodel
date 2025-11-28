"""
Data collection script for the OGBench environments adjusted to SWM format.
See original script at: https://github.com/seohongpark/ogbench/blob/master/data_gen_scripts/generate_manipspace.py
"""

import os
from collections import defaultdict
from pathlib import Path

import datasets
import gymnasium
import imageio.v3 as iio
import numpy as np
import ogbench.manipspace  # noqa
from absl import app, flags
from datasets import Dataset, Features, Value
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle
from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle
from tqdm import trange

import stable_worldmodel as swm
from stable_worldmodel.data import is_image


os.environ["MUJOCO_GL"] = "egl"

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "swm/OGBCube-v0", "Environment name.")
flags.DEFINE_string("dataset_type", "play", "Dataset type.")
flags.DEFINE_string("dataset_name", "ogb-cube-v0", "Save path.")
flags.DEFINE_string("save_dir", None, "Save path.")
flags.DEFINE_float("noise", 0.1, "Action noise level.")
flags.DEFINE_float("noise_smoothing", 0.5, "Action noise smoothing level for PlanOracle.")
flags.DEFINE_float("min_norm", 0.4, "Minimum action norm for MarkovOracle.")
flags.DEFINE_float("p_random_action", 0, "Probability of selecting a random action.")
flags.DEFINE_integer("num_episodes", 1000, "Number of episodes.")
flags.DEFINE_integer("max_episode_steps", 1001, "Maximum episode length.")
flags.DEFINE_list("variation_list", [], "List of variations to employ during data collection.")

#### Command examples to produce datasets. ####
# OGBCube: python ogbench_data_collection.py --env_name=swm/OGBCube-v0 --dataset_name=ogb-cube-v0 --save_dir=dataset/ --num_episodes=1000 --max_episode_steps=1001 --dataset_type=play --variation_list=cube.color,cube.size
# OGBScene: python ogbench_data_collection.py --env_name=swm/OGBScene-v0 --dataset_name=ogb-scene-v0 --save_dir=dataset/ --num_episodes=1000 --max_episode_steps=1001 --dataset_type=noisy --p_random_action=0.1 --variation_list=all
###############################################


def main(_):
    assert FLAGS.env_name in ["swm/OGBCube-v0", "swm/OGBScene-v0"]
    assert FLAGS.dataset_type in ["play", "noisy"]
    # 'play': Use a non-Markovian oracle (PlanOracle) that follows a pre-computed plan.
    # 'noisy': Use a Markovian, closed-loop oracle (MarkovOracle) with Gaussian action noise.

    # Initialize environment.
    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        mode="data_collection",
        ob_type="pixels",
        max_episode_steps=FLAGS.max_episode_steps,
    )

    # Initialize oracles.
    oracle_type = "plan" if FLAGS.dataset_type == "play" else "markov"
    has_button_states = hasattr(env.unwrapped, "_cur_button_states")
    if "Cube" in FLAGS.env_name:
        if oracle_type == "markov":
            agents = {
                "cube": CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm),
            }
        else:
            agents = {
                "cube": CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
            }
    elif "Scene" in FLAGS.env_name:
        if oracle_type == "markov":
            agents = {
                "cube": CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm, max_step=100),
                "button": ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                "drawer": DrawerMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                "window": WindowMarkovOracle(env=env, min_norm=FLAGS.min_norm),
            }
        else:
            agents = {
                "cube": CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                "button": ButtonPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                "drawer": DrawerPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                "window": WindowPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
            }
    elif "puzzle" in FLAGS.env_name:
        if oracle_type == "markov":
            agents = {
                "button": ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm, gripper_always_closed=True),
            }
        else:
            agents = {
                "button": ButtonPlanOracle(
                    env=env,
                    noise=FLAGS.noise,
                    noise_smoothing=FLAGS.noise_smoothing,
                    gripper_always_closed=True,
                ),
            }

    # Collect data.
    records = defaultdict(list)
    total_steps = 0
    num_episodes = FLAGS.num_episodes
    for ep_idx in trange(num_episodes):
        # Have an additional while loop to handle rare cases with undesirable states (for the Scene environment).
        while True:
            ob, info = env.reset(options={"variation": FLAGS.variation_list})

            # Set the cube stacking probability for this episode.
            if "single" in FLAGS.env_name:
                p_stack = 0.0
            elif "double" in FLAGS.env_name:
                p_stack = np.random.uniform(0.0, 0.25)
            elif "triple" in FLAGS.env_name:
                p_stack = np.random.uniform(0.05, 0.35)
            elif "quadruple" in FLAGS.env_name:
                p_stack = np.random.uniform(0.1, 0.5)
            elif "octuple" in FLAGS.env_name:
                p_stack = np.random.uniform(0.0, 0.35)
            else:
                p_stack = 0.5

            if oracle_type == "markov":
                # Set the action noise level for this episode.
                xi = np.random.uniform(0, FLAGS.noise)

            agent = agents[info["privileged/target_task"]]
            agent.reset(ob, info)

            done = False
            step = 0
            ep_qpos = []

            while not done:
                if np.random.rand() < FLAGS.p_random_action:
                    # Sample a random action.
                    action = env.action_space.sample()
                else:
                    # Get an action from the oracle.
                    action = agent.select_action(ob, info)
                    action = np.array(action)
                    if oracle_type == "markov":
                        # Add Gaussian noise to the action.
                        action = action + np.random.normal(0, [xi, xi, xi, xi * 3, xi * 10], action.shape)
                action = np.clip(action, -1, 1)
                next_ob, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if agent.done:
                    # Set a new task when the current task is done.
                    agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                    agent = agents[agent_info["privileged/target_task"]]
                    agent.reset(agent_ob, agent_info)

                records["episode_idx"].append(ep_idx)
                records["step_idx"].append(step)
                records["pixels"].append(ob)
                records["actions"].append(action)
                records["terminals"].append(done)
                records["qpos"].append(info["prev_qpos"])
                records["qvel"].append(info["prev_qvel"])
                if has_button_states:
                    records["button_states"].append(info["prev_button_states"])
                ep_qpos.append(info["prev_qpos"])

                ob = next_ob
                step += 1

            if "scene" in FLAGS.env_name:
                # Perform health check. We want to ensure that the cube is always visible unless it's in the drawer.
                # Otherwise, the test-time goal images may become ambiguous.
                is_healthy = True
                ep_qpos = np.array(ep_qpos)
                block_xyzs = ep_qpos[:, 14:17]
                if (block_xyzs[:, 1] >= 0.29).any():
                    is_healthy = False  # Block goes too far right.
                if ((block_xyzs[:, 1] <= -0.3) & ((block_xyzs[:, 2] < 0.06) | (block_xyzs[:, 2] > 0.08))).any():
                    is_healthy = False  # Block goes too far left, without being in the drawer.

                if is_healthy:
                    break
                else:
                    # Remove the last episode and retry.
                    print("Unhealthy episode, retrying...", flush=True)
                    for k in records.keys():
                        records[k] = records[k][:-step]
            else:
                break

        records["episode_len"].extend([step] * step)
        total_steps += step

    print("Total steps:", total_steps)

    ########################
    # Save dataset to disk #
    ########################

    cache_dir = FLAGS.save_dir or swm.data.get_cache_dir()
    dataset_path = Path(cache_dir, FLAGS.dataset_name)
    dataset_path.mkdir(parents=True, exist_ok=True)

    assert "pixels" in records, "pixels key is required in records"
    assert "episode_idx" in records, "episode_idx key is required in records"
    assert "step_idx" in records, "step_idx key is required in records"
    assert "episode_len" in records, "episode_len key is required in records"

    # Create the dataset directory structure
    dataset_path.mkdir(parents=True, exist_ok=True)

    # save all jpeg images
    image_cols = {col for col in records if is_image(records[col][0])}

    # pre-create all directories
    for ep_idx in set(records["episode_idx"]):
        img_folder = dataset_path / "img" / f"{ep_idx}"
        img_folder.mkdir(parents=True, exist_ok=True)

    # dump all data
    for i in range(len(records["episode_idx"])):
        ep_idx = records["episode_idx"][i]
        step_idx = records["step_idx"][i]
        for img_col in image_cols:
            img = records[img_col][i]
            img_folder = dataset_path / "img" / f"{ep_idx}"
            img_path = img_folder / f"{step_idx}_{img_col.replace('.', '_')}.jpeg"
            iio.imwrite(img_path, img)

            # replace image in records with relative path
            records[img_col][i] = str(img_path.relative_to(dataset_path))

    def determine_features(records):
        features = {
            "episode_idx": Value("int32"),
            "step_idx": Value("int32"),
            "episode_len": Value("int32"),
        }

        for col_name in records:
            if col_name in features:
                continue

            first_elem = records[col_name][0]

            if type(first_elem) is str:
                features[col_name] = Value("string")

            elif isinstance(first_elem, np.ndarray):
                if first_elem.ndim == 1:
                    state_feature = datasets.Sequence(
                        feature=Value(dtype=first_elem.dtype.name),
                        length=len(first_elem),
                    )
                elif 2 <= first_elem.ndim <= 6:
                    feature_cls = getattr(datasets, f"Array{first_elem.ndim}D")
                    state_feature = feature_cls(shape=first_elem.shape, dtype=first_elem.dtype.name)
                else:
                    state_feature = Value(first_elem.dtype.name)
                features[col_name] = state_feature

            elif isinstance(first_elem, (np.generic)):
                features[col_name] = Value(first_elem.dtype.name)
            else:
                features[col_name] = Value(type(first_elem).__name__)

        return Features(features)

    records_feat = determine_features(records)
    records_ds = Dataset.from_dict(records, features=records_feat)

    # save dataset
    records_path = dataset_path  # / "records"
    num_chunks = num_episodes // 50
    records_path.mkdir(parents=True, exist_ok=True)
    records_ds.save_to_disk(records_path, num_shards=num_chunks or 1)

    print(f"Dataset saved to {dataset_path} with {num_episodes} episodes!")


if __name__ == "__main__":
    app.run(main)
