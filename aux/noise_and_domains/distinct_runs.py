"""Cluster the seed's test evaluations into distinct executions by a content signature (per-request input/output tokens), since copies of
one run into optimizer directories re-hash request ids.  Run: python -m aux.noise_and_domains.distinct_runs"""
import json
from aux.noise_and_domains import reruns_check as R
from src.analysis import selection as S

def content(e):
    return tuple((r.get("input_tokens"), r.get("output_tokens")) for r in e["requests"])


def main():
    R.OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for bench, _ in S.BENCHES:
        runs = R.PAPER_TEST[bench] + R.EXTRA_TEST[bench]
        sig = {}
        for run in runs:
            d = {}
            for f in R.eval_files(run):
                for e in json.load(open(f))["episodes"]:
                    if S.verified(e) and e["task_id"] not in d:
                        d[e["task_id"]] = (int(bool(e["success"])), content(e))
            if d:
                sig[run] = d
        clusters = []
        for run, d in sig.items():
            for c in clusters:
                ref = sig[c[0]]
                com = [t for t in d if t in ref]
                same = sum(d[t][1] == ref[t][1] for t in com)
                if com and same >= 0.5 * len(com):
                    c.append(run); break
            else:
                clusters.append([run])
        print("==", bench)
        for c in clusters:
            d = sig[c[0]]
            print(f"  n={len(d)} succ={sum(v for v, _ in d.values())}  members={c}")
        out[bench] = [{"members": c, "n": len(sig[c[0]]), "solved": sum(v for v, _ in sig[c[0]].values())} for c in clusters]
    json.dump(out, open(R.OUT / "distinct_runs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
