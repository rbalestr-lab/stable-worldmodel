# Running PyFlyt Dogfight Environment with Docker

This guide shows you how to run the PyFlyt Dogfight environment using Docker, which bypasses all ARM64 macOS pybullet compilation issues.

## Quick Start

### Option 1: Using Docker Compose (Easiest)

```bash
# Build and start the container
docker-compose up -d pyflyt

# Run the test script
docker-compose exec pyflyt python test_dogfight_docker.py

# Get a bash shell inside the container
docker-compose exec pyflyt bash

# Stop the container
docker-compose down
```

### Option 2: Using Docker Directly

```bash
# Build the Docker image
docker build -f Dockerfile.pyflyt -t pyflyt-dogfight .

# Run the container interactively
docker run -it -v $(pwd):/workspace pyflyt-dogfight bash

# Inside the container, run the test
python test_dogfight_docker.py
```

## What's Included

The Docker container includes:
- ✅ Python 3.11
- ✅ pybullet (latest version, compiles on Linux)
- ✅ PyFlyt
- ✅ stable-worldmodel (installed in editable mode)
- ✅ All dependencies

## Running Tests

Inside the Docker container:

```bash
# Test the PyFlyt Dogfight environment
python test_dogfight_docker.py

# Run check_world.py tests
pytest stable_worldmodel/tests/check_world.py::test_each_env[swm/PyFlytDogfight-v0] -v

# Use the environment in Python
python -c "
import gymnasium as gym
env = gym.make('swm/PyFlytDogfight-v0', render_mode='human')
obs, info = env.reset(seed=42)
print('Environment working!', obs.keys())
"
```

## Using with stable-worldmodel World

```python
import stable_worldmodel as swm

# Create world
world = swm.World(
    "swm/PyFlytDogfight-v0",
    num_envs=1,
    image_shape=(224, 224),
    max_episode_steps=100,
)

# Set random policy
policy = swm.policy.RandomPolicy(seed=42)
world.set_policy(policy)

# Record dataset
world.record_dataset("dogfight_data", episodes=10, seed=42)

# Record video
world.record_video("dogfight_video.mp4", seed=42)
```

## Development Workflow

1. Edit code on your Mac (files are mounted as volumes)
2. Run/test inside Docker container
3. Changes are immediately reflected (no rebuild needed)

## Troubleshooting

### Container won't start
```bash
# Clean up and rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Display/rendering issues
The container uses headless rendering (DISPLAY=:99). For GUI rendering, you may need to set up X11 forwarding.

### Out of memory
```bash
# Increase Docker memory limit in Docker Desktop settings
# Preferences > Resources > Memory > 8GB+
```

## Why Docker?

Docker solves the ARM64 macOS pybullet issue because:
- Containers run Linux (not macOS)
- pybullet compiles perfectly on Linux
- No architecture compatibility issues
- Consistent environment across all platforms

## Performance

Docker on Apple Silicon uses native ARM64 virtualization, so performance is excellent for CPU-bound tasks like physics simulation.
