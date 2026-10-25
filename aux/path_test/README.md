# path-test

Test runs of the skills along the tau3 optimization paths on the updated DeepSeek release, the seed rerun on that release, and what they say about the release shift. `run_path_test` and `run_xbench_seed` call the model (dry run / `--plan` by default; `--execute` runs them); `collect_path_test`, `pairs`, `estimate_gap` and `analyze` (with `xlib`) only read.
Run from the repo root, e.g. `python -m aux.path_test.collect_path_test`, `python -m aux.path_test.pairs`, `python -m aux.path_test.estimate_gap [--selftest]`. The raw passes live in `$BIPA_ASSETS/artifacts/path-test/{out,cache,logs}`; `launch.sh` holds the launch commands used.
