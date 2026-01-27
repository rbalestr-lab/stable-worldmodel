---
title: OGBench
summary: TBD
external_links:
    arxiv: https://arxiv.org/pdf/2410.20092
    github: https://github.com/seohongpark/ogbench
---

<!-- ![ogb](assets/env/ogb/normal.gif) -->

## [Cube]

### Description

A robotic manipulation task featuring a UR5e arm with a Robotiq gripper performing cube manipulation. The agent must move colored cubes to target positions, with tasks ranging from simple pick-and-place to complex multi-cube stacking and cyclic rearrangements.

**Success criteria**: All cubes within 4cm of their respective target positions.

### Specs

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, shape=(7,))` - Arm joint velocities + gripper |
| Observation Space | `pixels` (224x224) or `state` (proprioception + object states) |
| Reward | Count of successful cubes minus total cubes |
| Success Metric | All cubes within 4cm of targets |

| Type | Cubes | Task Examples |
|------|-------|---------------|
| `single` | 1 | Horizontal, vertical, diagonal movements |
| `double` | 2 | Pick-place, swap, stack |
| `triple` | 3 | Pick-place, unstack, cycle, stack |
| `quadruple` | 4 | Multi-cube pick-place, cycle, stack |
| `octuple` | 8 | Complex rearrangements and stacking |

### Datasets

| Type | Source | Download Link |
|------|--------|---------------|
| Expert | DINO-WM |            |

----------------------------------------

## [Scene]

### Description

A robotic manipulation task in a rich tabletop scene with multiple interactive objects: a cube, two lockable buttons, a drawer, and a sliding window. The buttons act as locks - pressing them unlocks/locks the corresponding drawer or window. Tasks require coordinating multiple object interactions.

**Success criteria**: All objects (cube position, button states, drawer position, window position) match their target configurations.

### Specs

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, shape=(7,))` - Arm joint velocities + gripper |
| Observation Space | `pixels` (224x224) or `state` (proprioception + object states) |
| Reward | Count of successful conditions minus total conditions |
| Success Metric | All objects in target state |

| Object | Description |
|--------|-------------|
| Cube | Movable block that can be placed on table or in drawer |
| Button 1 | Toggles drawer lock (red = locked, white = unlocked) |
| Button 2 | Toggles window lock (red = locked, white = unlocked) |
| Drawer | Sliding drawer, openable when unlocked |
| Window | Sliding window, openable when unlocked |

### Datasets

| Type | Source | Download Link |
|------|--------|---------------|
| Expert | DINO-WM |            |

----------------------------------------

