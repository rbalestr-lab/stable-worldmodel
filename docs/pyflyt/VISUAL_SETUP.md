# How to See PyFlyt Dogfight Visualization

Unfortunately, PyFlyt Dogfight has a **visual rendering limitation**:
- ✅ Works perfectly for state-based observations (proprio/state)
- ❌ Does NOT support headless pixel rendering (no 'rgb_array' mode)
- ❌ Only supports 'human' mode which requires a GUI window

## The Issue

PyFlyt only supports `render_mode='human'` which opens a PyBullet GUI window. This requires:
1. A physical display (not headless)
2. pybullet to be installed (doesn't compile on ARM64 macOS)
3. X11/display server access

## Why Videos are Black

The videos you saw are black because:
1. PyFlyt doesn't support `render_mode='rgb_array'` (headless pixel rendering)
2. In Docker, there's no display, so we use `render_mode=None`
3. With `render_mode=None`, PyFlyt doesn't render anything
4. The environment returns all-black frames (zeros)

## What Actually Works

The PyFlyt Dogfight environment is **fully functional** for:
- ✅ **State observations**: `proprio` and `state` keys (34-dimensional vectors)
- ✅ **Actions**: RPYT commands (roll, pitch, yaw, throttle)
- ✅ **Physics simulation**: Aircraft flight, combat, collisions
- ✅ **Rewards**: Damage dealt, kills, team cooperation
- ✅ **Domain randomization**: Team colors, spawn positions, lighting
- ✅ **Episode termination**: Based on health/time
- ✅ **Dataset recording**: State-based trajectories
- ❌ **Pixel observations**: Not available in headless mode

## Comparison with Other Environments

Most stable-worldmodel environments (PushT, TwoRoom, etc.) support `render_mode='rgb_array'` which allows:
- Headless pixel rendering
- Video recording with actual visuals
- Dataset collection with images

PyFlyt Dogfight is different because it's based on PyBullet's older rendering system which requires a display.

## What This Means for Your Use Case

If you're using stable-worldmodel for:
- **Training world models from pixels**: PyFlyt Dogfight won't work (no pixels in headless mode)
- **Training RL agents from state**: PyFlyt Dogfight works perfectly! ✅
- **Collecting state-based datasets**: Works great ✅
- **Testing domain randomization**: Works ✅

## Alternative: State-Based Usage

Here's how to use PyFlyt Dogfight for state-based RL:

```python
import stable_worldmodel as swm

# Create world
world = swm.World(
    'swm/PyFlytDogfight-v0',
    num_envs=4,
    image_shape=(224, 224),  # Required but won't have pixels
    max_episode_steps=200,
)

policy = swm.policy.RandomPolicy(42)
world.set_policy(policy)
world.reset(seed=42)

# State observations are available
for _ in range(100):
    world.step()

    # Access state observations (not pixels!)
    proprio = world.infos['proprio']  # Shape: (4, 1, 34)
    state = world.infos['state']      # Shape: (4, 1, 34)

    # These work fine:
    # - proprio: Aircraft's own state (position, velocity, etc.)
    # - state: Same as proprio in this environment
    # - rewards: Combat rewards
    # - terminated/truncated: Episode end flags
```

## Potential Solutions (Advanced)

To get visual rendering, you would need to:

1. **Modify PyFlyt library** to add 'rgb_array' support (requires PyFlyt development)
2. **Use EGL rendering** in Docker (complex setup, may not work)
3. **Run on a machine with display** and use 'human' mode (opens GUI window)

## Recommendation

Since PyFlyt Dogfight doesn't support headless visual rendering, I recommend:

**Option A**: Use it for **state-based RL** (what it does well)
- Collect state trajectories
- Train agents on state observations
- Use domain randomization for robust policies

**Option B**: Choose a **different aerial environment** if you need pixels:
- Look for environments with native 'rgb_array' support
- Or environments built on modern rendering (not PyBullet GUI)

**Option C**: Accept the limitation and document it
- The environment works perfectly for state-based training
- Visual rendering is a known limitation of PyFlyt
- All other features work correctly

## Summary

The PyFlyt Dogfight environment integration is **complete and functional** for state-based use cases. The only limitation is visual/pixel rendering in headless mode, which is a constraint of the PyFlyt library itself, not our implementation.

If your stable-worldmodel use case requires pixel observations, PyFlyt Dogfight isn't the right choice. But for state-based RL, it's ready to use! ✅
