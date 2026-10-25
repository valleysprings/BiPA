"""Episode selection for the shortcut audit (shortcut_audit.py). Run:  python -m aux.contamination_and_loading.shortcut_load
Candidates = convergent gains (convergent_gains.json, group 'converged') whose learned-skill mean turns are below the seed's (rR < 1): 33.
Control    = 33 non-convergent gains drawn at random (numpy seed 0) from the 53 in the same file.
Audited episodes = every verified-successful run (of 3) of the learned skill on the task; reference = one successful seed run on the task
(the one with the most turns' median, i.e. the median-turn successful seed run), when the seed solved it at least once.
Reads OUT/transfer_and_trajectories/convergent_gains.json; writes OUT/contamination_and_loading/shortcut/episodes.json:
one row per audited episode with the paths needed to build the judge view."""
import json, re
from pathlib import Path

import numpy as np

from src import layout
from src.analysis import dynamics as D

HERE = layout.OUT / "contamination_and_loading"
_orig = D.ep_metrics
D.ep_metrics = lambda raw: {**_orig(raw), "raw": raw}

OLD_SEED = re.compile(r"^.*/harness-bench/assets/artifacts/execution/runs/seed/(?P<m>[^/]+)/opencode/(?P<b>[^/]+)/")
NEW_SEED = str(layout.EXECUTION) + "/{m}/native-seed/{b}/"


def remap(path):
    """Where a recorded evidence path is now (the DeepSeek seed run's refs name its pre-move location under harness-bench)."""
    m = OLD_SEED.match(path)
    return NEW_SEED.format(**m.groupdict()) + path[m.end():] if m else str(layout.relocate(path))


def trial_dir(raw):
    """The native trial directory of the scored attempt. A slot retried after a failed attempt lists several trial
    directories; the scored one is the last whose verifier reward equals the episode's recorded reward."""
    dirs = []
    for ref in raw.get("evidence_refs", []):
        m = re.match(r"(.*/native/[^/]+/[^/]+)/agent/", ref.split("#")[0])
        if m and remap(m.group(1)) not in dirs:
            dirs.append(remap(m.group(1)))
    want = (raw.get("backend_outcome") or {}).get("verifier_rewards", {}).get("reward")
    for d in sorted(dirs, reverse=True):
        p = Path(d) / "verifier/result.json"
        if p.exists() and json.loads(p.read_text()).get("reward") == want:
            return d
    d = _trial_dir(raw)
    return remap(d) if d else d


def _trial_dir(raw):
    for ref in raw.get("evidence_refs", []):
        m = re.match(r"(.*/native/[^/]+/[^/]+)/agent/", ref.split("#")[0])
        if m:
            return m.group(1)
    for r in raw.get("requests", []):
        for ref in r.get("evidence_refs", []):
            m = re.match(r"(.*/full)/attempts/", ref.split("#")[0])
            if m:
                cands = sorted(Path(remap(m.group(1))).glob("native/*/*/agent"))
                if cands:
                    return str(cands[-1].parent)
    return None

def main():
    rows = json.load(open(layout.OUT / "transfer_and_trajectories/convergent_gains.json"))
    cand = [r for r in rows if r["group"] == "converged" and r["rR"] < 1]
    nonconv = [r for r in rows if r["group"] == "not converged"]
    rng = np.random.default_rng(0)
    ctrl = [nonconv[i] for i in sorted(rng.choice(len(nonconv), size=len(cand), replace=False))]
    cache, out = {}, []
    def cell(model, arm, bench):
        k = (model, arm, bench)
        if k not in cache:
            cache[k] = D.load_cell(model, arm, bench)
        return cache[k]
    for set_name, items in (("candidate", cand), ("control", ctrl)):
        for r in items:
            own, seed = cell(r["model"], r["arm"], r["bench"]), cell(r["model"], "native-seed", r["bench"])
            seed_ok = [(k, seed[k][r["task"]]) for k in sorted(seed) if r["task"] in seed[k] and seed[k][r["task"]]["success"]]
            ref = None
            if seed_ok:
                seed_ok.sort(key=lambda kv: kv[1]["R"])
                k, ep = seed_ok[len(seed_ok) // 2]
                ref = {"rep": k, "trial": trial_dir(ep["raw"]), "R": ep["R"], "N": ep["N"]}
            for k in sorted(own):
                ep = own[k].get(r["task"])
                if not ep or not ep["success"]:
                    continue
                out.append({"set": set_name, **{x: r[x] for x in ("model", "bench", "arm", "task", "pattern", "rR")},
                            "rep": k, "trial": trial_dir(ep["raw"]), "R": ep["R"], "N": ep["N"],
                            "seed_ref": ref, "verifier": ep["raw"].get("backend_outcome", {}).get("verifier_rewards")})
    (HERE / "shortcut").mkdir(parents=True, exist_ok=True)
    (HERE / "shortcut/episodes.json").write_text(json.dumps(out, indent=1))
    print(len(cand), len(ctrl), len(out), sum(1 for o in out if o["trial"] is None), sum(1 for o in out if o["seed_ref"] is None))

if __name__ == "__main__":
    main()
