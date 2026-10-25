"""(f) Split each returned skill (final/SKILL.md, front matter and headings removed) into sentences, number them, write sentences.json.
Run: python -m aux.three_word_seed.sentences"""
import json, re
from src import layout

EXE = layout.EXECUTION / "gpt-6-luna"
OUT = layout.OUT / "three_word_seed"
def clean(line):
    t = line.strip()
    t = re.sub(r'^(#{1,6}\s*|[-*+]\s+|\d+[.)]\s+)', '', t)
    return t.strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SPLIT = re.compile(r'(?<=[.!?;])\s+(?=[A-Z0-9`"\'(*\-])')
    out = {}
    for opt in ("skillopt", "gepa"):
        for bench in ("automationbench", "tau3-bench"):
            txt = (EXE / opt / bench / "fromscratch/final/SKILL.md").read_text()
            body = txt.split("---", 2)[2] if txt.startswith("---") else txt
            sents, headings = [], []
            for line in body.splitlines():
                if not line.strip():
                    continue
                if line.lstrip().startswith("#"):
                    headings.append(line.strip()); continue
                t = clean(line)
                for s in SPLIT.split(t):
                    s = s.strip()
                    if len(s.split()) >= 3:
                        sents.append(s)
            out[f"{opt}|{bench}"] = dict(headings=headings, sentences=sents)
    json.dump(out, open(OUT / "sentences.json", "w"), indent=1, ensure_ascii=False)
    for k, v in out.items():
        print("=====", k, len(v["sentences"]), "sentences;", v["headings"])
        for i, s in enumerate(v["sentences"]):
            print(f"{i:2d}. {s}")


if __name__ == "__main__":
    main()
