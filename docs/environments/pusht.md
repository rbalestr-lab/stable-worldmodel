---
title: Pusht
summary: TBD
external_links:
    arxiv: tbd
---

![pusht](assets/env/pusht/normal.gif)

## Description

A 2D contact-rich manipulation task where an agent controls a circular agent to push a T-shaped block into a target pose. The environment uses [Pymunk]() physics simulation with realistic friction and collision dynamics.

The agent must push the block to match both the target **position** and **orientation**, making this a challenging task that requires planning multi-step pushing sequences rather than simple point-to-point control.

**Success criteria**: The episode is successful when the block is within 20 pixels of the goal position AND the orientation error is less than π/9 radians.

## Environment Specs

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, shape=(2,))` - 2D velocity control |
| Observation Space | `Dict(proprio=(4,), state=(7,))` |
| Reward | Negative distance to goal |
| Episode Length | 200 steps (default) |
| Success Metric | Position < 20px, Angle < π/9 |

## Dataset

| Type | Source | Download Link |
|------|--------|---------------|
| Expert | DINO-WM |            |


## Variation Space


