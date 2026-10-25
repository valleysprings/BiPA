# Optimizer sources

Upstream sources of the optimizers, at the commits the runs used. `src/adapters/__init__.py` puts them on `sys.path`; the adapters in `src/adapters/` run their loops on our batches, executor and model client. Experiment settings live in the adapters, never in these checkouts.

| Optimizer | Source | Commit | Adapter |
|---|---|---|---|
| SkillOpt | [skillopt](skillopt/) | `79124b37e9a6371e13b753f8bcd7adb1e493ade1` | [adapters/skillopt.py](../src/adapters/skillopt.py) |
| SkillAdam | [skilladam](skilladam/) | `72ef9ba48bbadc059c8a2d099f1b7d531fc8288f` | [adapters/skilladam.py](../src/adapters/skilladam.py) |
| GEPA | [gepa](gepa/) | `d771eb21b5dd3228bc3f567293d2ccfc423fc900` | [adapters/gepa.py](../src/adapters/gepa.py) |
| Evo-Harness | [evoharness](evoharness/) | `3c7c7b8c62a3f07da1d4407aa03f0ca079954bcf` | none; not used by this code |

Their dependencies, from the repository root:

```sh
pip install -e optimizers/skillopt -e 'optimizers/skilladam[backend]'
pip install --no-deps -e optimizers/gepa
```
