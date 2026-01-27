[![PyPI](https://img.shields.io/pypi/v/stable-worldmodel.svg)](https://pypi.python.org/pypi/stable-worldmodel/#history)

# Stable World-Model

World model research made simple. From data collection to training and evaluation.

```bash
pip install stable-worldmodel
```

> **Note:** The library is still in active development.

## Quick Example

```python
import stable_worldmodel as swm

world = swm.World('swm/PushT-v1', num_envs=8)
world.set_policy(your_policy)
world.record_dataset(dataset_name='pusht_demo', episodes=100)

# ... train your world model ...

results = world.evaluate(episodes=50)
print(f"Success Rate: {results['success_rate']:.1f}%")
```

## Documentation

See the full documentation at [stable-worldmodel.github.io](https://stable-worldmodel.github.io).

## Contributing

Setup your codebase:

```bash
uv venv --python=3.10
source .venv/bin/activate
uv sync --all-extras --group dev
```

## Citation

```bibtex
@article{swm_maes2026,
  title={stable-world model},
  author={Lucas Maes, Quentin LeLidec, Dan Haramati, Nassim Massaudi, Yann LeCun, Randall Balestriero},
  booktitle={stable-worldmodel: World Model Research Made Simple},
  year={2026},
}
```

## Questions

If you have a question, please [file an issue](https://github.com/lucas-maes/swm/issues).
