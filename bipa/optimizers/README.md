# Official optimizer sources

Canonical upstream sources owned by BiPA. BACH comparison candidates have their own [optimizer directory](../../bach/optimizers/README.md); their implementations are not yet integrated. Preserve upstream algorithms and record adaptations in local adapters.

| Optimizer | Source | Pinned commit | Current adapter |
|---|---|---|---|
| SkillOpt | [skillopt](skillopt/) | `79124b37e9a6371e13b753f8bcd7adb1e493ade1` | [adapters/skillopt.py](../adapters/skillopt.py) |
| SkillAdam | [skilladam](skilladam/) | `72ef9ba48bbadc059c8a2d099f1b7d531fc8288f` | [adapters/skilladam.py](../adapters/skilladam.py) |
| GEPA | [gepa](gepa/) | `d771eb21b5dd3228bc3f567293d2ccfc423fc900` | [adapters/gepa.py](../adapters/gepa.py) |
| Evo-Harness | [evoharness](evoharness/) | `3c7c7b8c62a3f07da1d4407aa03f0ca079954bcf` | Historical reference; outside the current comparison |

The current adapters call upstream optimizer loops. They also adapt task panels, feedback, transport and caching; upstream source preservation does not establish behavioral equivalence. See the [runner guide](../README.md) and each adapter's `DEVIATIONS`. Do not modify upstream source checkouts to implement experiment policy. Evo-Harness remains a historical reference, not an active BiPA method.

For a fresh environment, install from the `harness-mitigation` root:

```sh
python -m pip install -e baselines/bipa/optimizers/skillopt -e 'baselines/bipa/optimizers/skilladam[backend]' python-dotenv socksio
python -m pip install --no-deps -e baselines/bipa/optimizers/gepa
```

The entire directory was moved from `assets/optimizers/`, including its Git checkouts. Historical provenance is not rewritten after source moves; `baselines/bipa/layout.py` resolves old paths. Existing editable installations must be refreshed after a move.
