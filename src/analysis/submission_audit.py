"""Read-only audit of a submission manifest (analysis/submission_protocol.py): re-hash everything it names and list what is pending or inconsistent.

Checks per complete cell: summary / protocol / history / final hashes unchanged and the final skill, history, summary belong to the pinned run (same dir, config_hash, bench, model); the manifest's run is the pinned tag;
every avg@3 row has repetitions 0,1,2 with three distinct request-id signatures (recomputed from the evaluation files), the same panel per repetition as the Seed row, and no signature shared with another row;
a returned seed reuses the Seed row's executions explicitly; the BACH cell is not the old v8 run. Pending cells are listed. Exit 0 only if all cells are complete and consistent.
"""
import argparse, sys
from src.analysis import common
from src.analysis.common import *
from src.analysis.submission_protocol import execution, fsha, hlist, HISTORY

def audit(man):
    bad, pend = [], []
    def chk(ok, msg):
        if not ok: bad.append(msg)
    def same(a, b): return a == b
    for bench, row in man["cells"].items():
        seed = row["native-seed"]
        for m, c in row.items():
            tag = f"{bench}/{m}"
            if c["status"] != "complete": pend.append(f"{tag}: {c.get('reason')}"); continue
            run = Path(c["run"]); s = json.loads((run/"summary.json").read_text())
            chk(fsha(run/"summary.json") == c["summary_sha256"], f"{tag}: summary.json hash changed")
            chk(fsha(run/"native_protocol.json") == c["protocol_sha256"] or fsha(run/"split.json") == c["protocol_sha256"], f"{tag}: protocol/split hash changed")
            for h, v in c["history"].items(): chk(fsha(run/h) == v, f"{tag}: history {h} hash changed")
            chk(run.name == c["run_name"] and (m == "ours" and run.name == man["bach_tag"] or m in ("gepa", "skillopt", "skilladam") and run.name == man["baseline_tag"] or m in ("native-noseed", "native-seed")), f"{tag}: run {run.name} is not the pinned tag")
            chk(s.get("benchmark") == bench and s.get("model") == man["model"], f"{tag}: summary is for another bench/model")
            chk(not c["config_hash"] or s.get("config_hash", c["config_hash"]) == c["config_hash"], f"{tag}: summary config_hash differs from the protocol's")
            f = c["final"]; text = None if m == "native-noseed" else SEED_TEXT if m == "native-seed" else s["final_skill"]
            chk(vid(text) == f["version_id"], f"{tag}: summary final skill is not the pinned version")
            chk(m in ("native-noseed", "native-seed") or fsha(run/"final/SKILL.md") == f["file_sha256"], f"{tag}: final/SKILL.md hash changed")
            chk(f["returned_seed"] == (m not in ("native-seed",) and text == SEED_TEXT), f"{tag}: returned_seed flag wrong")
            if c.get("executions_from"):
                chk(f["returned_seed"] and c["executions_from"] == "native-seed" and "executions" not in c, f"{tag}: returned-seed row must reuse the seed row's executions explicitly"); continue
            ex = c["executions"]; chk(sorted(ex) == ["0", "1", "2"], f"{tag}: repetitions {sorted(ex)} != 0,1,2")
            sigs = []
            for rep, e in ex.items():
                now = dict(execution(e["file"]), source_run=e["source_run"]); sigs.append(now["signature"])
                chk(now == e, f"{tag} rep {rep}: evaluation.json changed since the freeze")
                chk(now["reps"] == [int(rep)], f"{tag} rep {rep}: file holds repetitions {now['reps']}")
                se = seed["executions"].get(rep) if seed["status"] == "complete" else None
                chk(se is None or now["panel_hash"] == se["panel_hash"], f"{tag} rep {rep}: test panel differs from the Seed row's")
            chk(None not in sigs and len(set(sigs)) == 3, f"{tag}: three distinct executions needed, signatures {sigs}")
        sigs = {}
        for m, c in row.items():
            for rep, e in (c.get("executions") or {}).items(): sigs.setdefault(e["signature"], []).append(f"{m}/{rep}")
        for sg, who in sigs.items(): chk(sg is None or len(who) == 1, f"{bench}: rows share one execution {who}")
        for k in ("train", "sel", "test"):
            hs = {m: c["panels"].get(k, {}).get("sha256") for m, c in row.items() if c["status"] == "complete" and m != "ours"}
            chk(len(set(hs.values())) <= 1, f"{bench}: {k} panel differs across rows {hs}")
    for bench, c in man["bach_v8_reference"].items(): chk(c.get("not_for_table") and c.get("run_name", REF) != man["bach_tag"], f"{bench}: v8 reference not marked or confused with v9")
    return bad, pend
REF = "bach-v8"

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--manifest", required=True, help="protocol.json from submission_protocol.py")
    man = json.loads(Path(ap.parse_args().manifest).read_text()); bad, pend = audit(man)
    for p in pend: print("PENDING", p)
    for b in bad: print("INCONSISTENT", b)
    n = sum(c["status"] == "complete" for r in man["cells"].values() for c in r.values())
    print(f"{n} cells complete, {len(pend)} pending, {len(bad)} inconsistent"); sys.exit(1 if bad or pend else 0)

if __name__ == "__main__": main()
