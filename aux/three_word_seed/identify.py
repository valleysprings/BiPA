"""Map each from-scratch run's returned skill (final/SKILL.md) to its version id via the delivery identity files (SKILL.md sha256)."""
import glob, hashlib, json
from src import layout

EXE = str(layout.EXECUTION / "gpt-6-luna")
OUT = layout.OUT / "three_word_seed"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for m in ("skillopt", "gepa"):
        for b in ("automationbench", "tau3-bench"):
            run = f"{EXE}/{m}/{b}/fromscratch"
            fin = hashlib.sha256(open(run + "/final/SKILL.md", "rb").read()).hexdigest()
            ids = {}
            for p in glob.glob(run + "/results/skills/deliveries/evaluations/*/*/identity.json"):
                d = json.load(open(p))
                ids[d["version_id"][:12]] = d["files"]["SKILL.md"]
            match = [v for v, h in ids.items() if h == fin]
            s = json.load(open(run + "/summary.json"))
            out[f"{m}/{b}"] = dict(final_sha=fin[:12], returned_vid=match, n_versions_with_identity=len(ids),
                                   summary_test=s["evaluated_panels"]["test"], extra={k: s["extra"].get(k) for k in ("best_selection_hard", "final_selection_hard", "best_step", "test_hard", "final_test_hard", "best_idx")},
                                   final_words=s["final_words"], wc=len(open(run + "/final/SKILL.md").read().split()))
            print(m, b, json.dumps(out[f"{m}/{b}"]))
    json.dump(out, open(OUT / "identify.json", "w"), indent=1)


if __name__ == "__main__":
    main()
