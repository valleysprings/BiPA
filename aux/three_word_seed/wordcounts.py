"""(b) Word counts (whitespace tokens, the convention of summary.final_words and of skilltext.py: len(text.split())) of the returned skills and the seeds.
'file' = whole SKILL.md including the YAML front matter; 'body' = text after the closing '---' of the front matter; 'delivered' = guidance.md the harness hands the agent (wrapper added).
Run: python -m aux.three_word_seed.wordcounts"""
import glob, json
from pathlib import Path
from src import layout
from src.utils import dataloader as C
EXE = layout.EXECUTION / "gpt-6-luna"
OUT = layout.OUT / "three_word_seed"
def body(t): return t.split("---", 2)[2] if t.startswith("---") else t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for opt, ret in (("skillopt", {"automationbench": "bb509d254dad", "tau3-bench": "dfd18dc47eb0"}), ("gepa", {"automationbench": "52d485532050", "tau3-bench": "d1ee2d3a774f"})):
        for bench in ("automationbench", "tau3-bench"):
            run = EXE / opt / bench / "fromscratch"
            t = (run / "final/SKILL.md").read_text()
            s = json.load(open(run / "summary.json"))
            g = glob.glob(str(run / f"results/skills/deliveries/evaluations/test/{ret[bench]}-*/guidance.md"))
            out[f"{opt}|{bench}"] = dict(file_words=len(t.split()), body_words=len(body(t).split()), summary_final_words=s["final_words"], lines=len(t.splitlines()), chars=len(t),
                                         delivered_guidance_words=len(Path(g[0]).read_text().split()) if g else None)
    seed3 = (EXE / "skillopt/automationbench/fromscratch/skillopt/seed.md").read_text()
    out["3-word seed"] = dict(file_words=len(seed3.split()), body_words=len(body(seed3).split()), summary_seed_words=json.load(open(EXE / "skillopt/automationbench/fromscratch/summary.json"))["seed_words"], text=seed3)
    out["hand-written seed (dataloader.SEED_TEXT, vid %s)" % C.vid(C.SEED_TEXT)[:12]] = dict(file_words=len(C.SEED_TEXT.split()), body_words=len(body(C.SEED_TEXT).split()))
    json.dump(out, open(OUT / "wordcounts.json", "w"), indent=1)
    for k, v in out.items(): print(k, {a: b for a, b in v.items() if a != "text"})


if __name__ == "__main__":
    main()
