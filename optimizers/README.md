# Optimizer dependencies

Install the external optimizer dependencies at the pinned commits used by the runs.

| Optimizer | Source | Pinned commit |
|---|---|---|
| SkillOpt | [microsoft/SkillOpt](https://github.com/microsoft/SkillOpt) | `79124b37e9a6371e13b753f8bcd7adb1e493ade1` |
| SkillAdam | [ruc-datalab/SkillAdam](https://github.com/ruc-datalab/SkillAdam) | `72ef9ba48bbadc059c8a2d099f1b7d531fc8288f` |
| GEPA | [gepa-ai/gepa](https://github.com/gepa-ai/gepa) | `d771eb21b5dd3228bc3f567293d2ccfc423fc900` |

From the repository root:

```sh
pip install 'skillopt @ git+https://github.com/microsoft/SkillOpt.git@79124b37e9a6371e13b753f8bcd7adb1e493ade1' \
    'skilladam[backend] @ git+https://github.com/ruc-datalab/SkillAdam.git@72ef9ba48bbadc059c8a2d099f1b7d531fc8288f'
pip install --no-deps 'gepa @ git+https://github.com/gepa-ai/gepa.git@d771eb21b5dd3228bc3f567293d2ccfc423fc900'
```
