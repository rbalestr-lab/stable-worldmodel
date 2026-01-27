---
title: Two-Room
summary: TBD
external_links:
    arxiv: tbd
---

<!-- ![tworoom](assets/env/tworoom/normal.gif) -->

## Description

A 2D navigation task where a circular agent must reach a goal position in the other room by navigating through doorways. The environment uses [Pymunk](http://www.pymunk.org/) physics simulation with a wall dividing the space into two rooms connected by one or more doors.

The agent starts in one room and must navigate to the goal in the other room while managing its limited energy. The task requires planning a path through the door openings rather than simple point-to-point navigation.

**Success criteria**: The episode is successful when the agent overlaps with the goal by at least 50% (either the agent covers 50% of the goal, or the goal covers 50% of the agent).

## Environment Specs

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, shape=(2,))` - 2D velocity control |
| Observation Space | `Dict(proprio=(4,), state=(6,))` |
| Reward | +1 on success, -0.01 per step |
| Episode Length | Up to 100 steps (default energy) |
| Success Metric | ≥50% overlap between agent and goal |

## Dataset

| Type | Source | Download Link |
|------|--------|---------------|
| Expert | DINO-WM |            |


## Variation Space

