"""Trigger-rate cells: rate, first trigger turn, success by triggered, input tokens vs the seed's on-demand cell, fingerprints.

Run:  python -m aux.contamination_and_loading.trigger_rate
"""
import collections, json, math, re

from src.layout import relocate
from src.utils.dataloader import DS

T = DS / "trigger-check"
FP = re.compile(r'"system_fingerprint":\s*"([0-9a-f]+)"')


def rows(cell):
    p = json.loads((cell / "trigger-summary.json").read_text())["evaluation_path"]
    out = {}
    for ep in json.loads(relocate(p).read_text())["episodes"]:
        reqs = [r for r in {r["request_id"]: r for r in ep["requests"]}.values() if r["input_tokens"] is not None]
        sd = ep["skill_delivery"]
        fp = None
        if reqs and relocate(reqs[0]["record_path"]).is_file():
            m = FP.search(relocate(reqs[0]["record_path"]).read_text()); fp = m.group(1)[:8] if m else None
        out[ep["task_id"]] = {"S": ep["success"], "trig": sd.get("triggered"), "first": sd.get("first_trigger_request"),
                              "X": sum(int(r["input_tokens"]) for r in reqs), "R": len(reqs), "fp": fp}
    return out


def main():
    for bench in ("tau3-bench", "automationbench"):
        seed = rows(T / bench / "seed")
        for cell in ("seed", {"tau3-bench": "skillopt-tau3", "automationbench": "skillopt-autob"}[bench], "gepa-tau3", "gepa-autob"):
            r = rows(T / bench / cell)
            tr = [v for v in r.values() if v["trig"]]; nt = [v for v in r.values() if not v["trig"]]
            q = lambda vs: 100 * sum(bool(v["S"]) for v in vs) / len(vs) if vs else float("nan")
            common = [t for t in r if t in seed and r[t]["X"] and seed[t]["X"]]
            rho = math.exp(sum(math.log(r[t]["X"] / seed[t]["X"]) for t in common) / len(common))
            firsts = sorted(v["first"] for v in tr)
            print(f"{bench[:5]} {cell:15s} trig {len(tr)}/{len(r)} first turn median {firsts[len(firsts)//2] if firsts else '-'} "
                  f"Q {q(list(r.values())):.1f} (trig {q(tr):.1f} / not {q(nt):.1f}) rho_X vs seed {rho:.3f} R {sum(v['R'] for v in r.values())/len(r):.1f} "
                  f"fp {dict(collections.Counter(v['fp'] for v in r.values()))}")


if __name__ == "__main__":
    main()
