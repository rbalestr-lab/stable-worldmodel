---
title: stable-worldmodel
summary: World Model Research Made Simple
sidebar_title: Getting Started
---

!!! danger ""
    **The library is still in active development!**

Stable World-Model is an open-source library for world model research. It provides a unified interface for training, evaluating, and deploying world models across various environments.

## Installation

=== "uv"

    ```bash
    uv add stable-worldmodel
    ```

=== "pip"

    ```bash
    pip install stable-worldmodel
    ```

!!! info ""
    By default the library comes without environment or training dependencies. To add them, use:

    `uv add stable-worldmodel --all-extras`


#### Development Setup

Setup a ready-to-go devolpment environment to contribute to the library:

```bash
git clone https://github.com/galilai-group/stable-worldmodel
cd stable-worldmodel/
uv venv --python=3.10
source .venv/bin/activate
uv sync --all-extras --group dev
```


#### Cache Directory

All datasets and model will be saved in the `$STABLEWM_HOME` environment variable.

By default the corresponding location is `~/.stable-wm/`. We encourage every user to adapt that directory according to their need and storage.


## Citation

```bibtex
@article{swm_maes2026,
  title={stable-world model},
  author={Lucas Maes},
  booktitle={...},
  year={2026},
}
```