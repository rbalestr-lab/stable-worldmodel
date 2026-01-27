---
title: Stable World-Model
summary: World Model Research Made Simple
sidebar_title: Home
---

!!! danger ""
    **The library is still in active development!**

Stable World-Model is an open-source library to conduct world model research.  You can install `stable-worldmodel` directly from PyPI:

=== "uv"

        :::bash
        uv add stable-worldmodel

=== "pip"

        :::bash
        pip install stable-worldmodel

=== "uv (all dependencies)"

        :::bash
        uv add stable-worldmodel --all-extras

=== "pip (all dependencies)"

        :::bash
        pip install stable-worldmodel[env, train]


!!! note ""
    (!) The base installation does not include environment (`env`) or training (`train`) dependencies. Install them separately or use the "all dependencies" option above if you need to run simulations or train models.

A **world model** is a learned simulator that predicts how an environment evolves in response to actions, enabling agents to plan by imagining future outcomes. Stable World-Model provides a unified research ecosystem that simplifies the entire pipeline: from data collection to model training and evaluation.

**Why another library?** World models have recently gained a lot of attention from the community. However, each new article re-implements over and over the same baselines, evaluation protocols, and data processing logic. We took that as an opportunity to provide a clean, documented, and tested library that researchers can trust for evaluation or training. More than just re-implementation, stable-worldmodel provides a complete ecosystem for world model research, from data collection to evaluation. We also extended the range of test-beds by providing researchers with a lean and simple API to fully customize the environments in which agents operate: from colors, to shapes, to physics properties. Everything is customizable, allowing for easy continual learning, out-of-distribution, or zero-shot robustness evaluation.

## Development Setup
---

Setup a ready-to-go development environment to contribute to the library:

```bash
git clone https://github.com/galilai-group/stable-worldmodel
cd stable-worldmodel/
uv venv --python=3.10
source .venv/bin/activate
uv sync --all-extras --group dev
```

!!! warning ""
    All datasets and model will be saved in the `$STABLEWM_HOME` environment variable.
    By default the corresponding location is `~/.stable-wm/`. We encourage every user to adapt that directory according to their need and storage.


## Next Steps
---

After you have installed stable-worldmodel, try the [Quick Start Guide](quick_start.md). You can also explore other part of the documentation:

| | |
|---|---|
| **[Tutorials](tutorial/collect_data.md)** | Step-by-step guides for data collection, training, and adding new environments. |
| **[Environments](envs/pusht.md)** | Explore the included environments: PushT, TwoRoom, OGBench, and more. |
| **[API Reference](api/world.md)** | Detailed documentation for World, Policy, Solver, Dataset, and other modules. |

## Citation

If you wish to cite our [pre-print](#):

```bibtex
@article{swm_maes2026,
  title={stable-world model},
  author={Lucas Maes, Quentin LeLidec, Dan Haramati, Nassim Massaudi, Yann LeCun, Randall Balestriero},
  booktitle={stable-worldmodel: World Model Research Made Simple},
  year={2026},
}
```