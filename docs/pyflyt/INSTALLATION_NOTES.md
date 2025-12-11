# PyFlyt Dogfight Environment - Installation Notes

## Current Status

### ✅ Completed
1. **Environment Wrapper**: Fully implemented at `stable_worldmodel/envs/pyflyt_dogfight.py`
2. **Registration**: Added to `stable_worldmodel/envs/__init__.py` as `swm/PyFlytDogfight-v0`
3. **Documentation**: Comprehensive guide in `PYFLYT_DOGFIGHT_README.md`
4. **PyFlyt**: Installed successfully
5. **FFmpeg**: Installed via Homebrew (version 8.0.1)
6. **torchcodec**: Reinstalled successfully

### ⚠️ Remaining Issue: pybullet

**Problem**: pybullet has no pre-built ARM64 wheel for macOS and must be compiled from source. Compilation is failing with clang errors.

**Error**: `error: command '/usr/bin/clang' failed with exit code 1`

## Solutions to Try

### Option 1: Use Mamba (Currently Installing)
Mamba has better dependency resolution and may have pre-built pybullet binaries:

```bash
# Once mamba is installed:
mamba install -c conda-forge pybullet -y
```

### Option 2: Use Docker (Recommended for Development)
The easiest solution is to use Docker with a pre-configured environment:

```dockerfile
# Dockerfile
FROM continuumio/miniconda3:latest

RUN conda install -c conda-forge pybullet pyflyt -y
RUN pip install stable-worldmodel

CMD ["/bin/bash"]
```

```bash
docker build -t swm-pyflyt .
docker run -it swm-pyflyt
```

### Option 3: Fix Compilation (Advanced)
If you must compile pybullet:

1. **Ensure Xcode Command Line Tools are up to date**:
   ```bash
   xcode-select --install
   sudo xcode-select --reset
   ```

2. **Set compiler flags**:
   ```bash
   export CC=clang
   export CXX=clang++
   export CFLAGS="-arch arm64"
   export CXXFLAGS="-arch arm64"
   pip install pybullet --no-cache-dir
   ```

3. **Try building with specific SDK**:
   ```bash
   export SDKROOT=$(xcrun --show-sdk-path)
   pip install pybullet --no-cache-dir
   ```

### Option 4: Use Alternative Physics Engine
Since pybullet is problematic on ARM64 macOS, consider alternatives:

1. **Test on Linux/x86_64**: Where pybullet has better support
2. **Use MuJoCo**: Some environments in stable-worldmodel already use MuJoCo
3. **Wait for pybullet ARM64 support**: Track https://github.com/bulletphysics/bullet3/issues

### Option 5: Use Rosetta 2 (x86_64 Emulation)
Install Python via Rosetta and use x86_64 packages:

```bash
# Install Homebrew for x86_64
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Use x86_64 Python
arch -x86_64 conda create -n swm-x86 python=3.11 -y
arch -x86_64 conda activate swm-x86
arch -x86_64 pip install pybullet PyFlyt
```

## Testing Without pybullet

You can still test that the wrapper is correctly structured:

```bash
python test_dogfight_simple.py
```

This will confirm the wrapper loads and has the correct structure, even though it can't run without pybullet.

## Once pybullet is Installed

Run the full integration test:

```bash
# Test basic functionality
python -c "
import gymnasium as gym
env = gym.make('swm/PyFlytDogfight-v0')
obs, info = env.reset(seed=42)
print('✓ Environment works!')
print('  Observation keys:', list(obs.keys()))
print('  Action space:', env.action_space)
env.close()
"

# Run check_world.py
pytest stable_worldmodel/tests/check_world.py::test_each_env[swm/PyFlytDogfight-v0] -v
```

## Current Background Processes

Mamba is currently being installed which should provide better package resolution.

## Recommendations

1. **For Quick Testing**: Use Docker (Option 2)
2. **For Development**: Either:
   - Run on Linux where pybullet compiles easily
   - Use x86_64 emulation via Rosetta (Option 5)
3. **For Production**: Wait for pybullet to officially support ARM64 macOS

## Additional Resources

- PyFlyt Documentation: https://taijunjet.com/PyFlyt/
- pybullet Issues: https://github.com/bulletphysics/bullet3/issues
- Conda-forge pybullet: https://anaconda.org/conda-forge/pybullet
