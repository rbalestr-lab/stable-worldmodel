"""Utility functions for stable_worldmodel."""

import os
import shlex
import subprocess
import sys
from collections.abc import Iterable
from typing import Any

from loguru import logger as logging


def pretraining(
    script_path: str,
    dataset_name: str,
    output_model_name: str,
    dump_object: bool = True,
    args: str = "",
) -> int:
    """Run a pretraining script as a subprocess."""
    if not os.path.isfile(script_path):
        raise ValueError(f"Script {script_path} does not exist.")

    logging.info(f"🏃🏃🏃 Running pretraining script: {script_path} with args: {args} 🏃🏃🏃")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    args = f"{args} ++dump_object={dump_object} dataset_name={dataset_name} output_model_name={output_model_name}"
    cmd = [sys.executable, script_path] + shlex.split(args)
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

    logging.info("🏁🏁🏁 Pretraining script finished 🏁🏁🏁")
    return


def flatten_dict(d, parent_key="", sep="."):
    """Flatten a nested dictionary into a single-level dictionary."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def get_in(mapping: dict, path: Iterable[str]) -> Any:
    """Retrieve a value from a nested dictionary using a sequence of keys."""
    cur = mapping
    for key in list(path):
        cur = cur[key]
    return cur
