# path_test

Test runs of the skills along the tau3 optimization paths on the updated DeepSeek release, the seed rerun on that release, and the size of the release shift.
`run_path_test` calls the model (a dry run by default; `--execute` runs it); `launch.sh` holds the launch commands used. The raw passes live in `$BIPA_ASSETS/artifacts/path-test/{out,cache,logs}`.
`python -m aux.path_test.collect_path_test` writes `path_test.json` and `path_test_detail.json` (the data of the path-test figure); `python -m aux.path_test.estimate_gap` fits the shift of the seed's mean success between the two releases.
`python -m aux.path_test.pathtest_fig` (after `collect_path_test`) draws `mech_pathtest.{pdf,svg}`: the reported dev gain and the test difference to the seed rerun along the tau3 paths of SkillOpt and GEPA.
The evaluations name their request records by the absolute paths of the machine that ran them; the readers look the records up under the pass's own run directory instead.
