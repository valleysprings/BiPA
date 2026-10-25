"""Build the sentence tables for the seed-path comparison (Luna / AutoBench, five runs).
Reads OUT/skill_paths/paths/*.json (read-only); writes into OUT/skill_paths/compare/ sentences_all.json (with run/step keys),
sentences_blind.json (id + text, shuffled, no run information) and blind_key.json (id -> run, step, candidate)."""
import json, random, re
from pathlib import Path
from src.layout import OUT


if __name__ == "__main__":
    D = OUT / "skill_paths/compare"
    P = OUT / 'skill_paths/paths'
    RUNS = {
     'H-gepa':     'gpt-6-luna__gepa__automationbench__native-20261012.json',
     'H-skillopt': 'gpt-6-luna__skillopt__automationbench__native-20261012.json',
     'H-skilladam':'gpt-6-luna__skilladam__automationbench__native-20261012.json',
     'N-gepa':     'gpt-6-luna__gepa__automationbench__fromscratch.json',
     'N-skillopt': 'gpt-6-luna__skillopt__automationbench__fromscratch.json',
    }
    SPLIT = re.compile(r'(?<=[.!?;])\s+(?=[A-Z0-9`"\'(*\-])')
    def clean(line):
        t = line.strip()
        t = re.sub(r'^(#{1,6}\s*|[-*+]\s+|\d+[.)]\s+)', '', t)
        t = re.sub(r'^(name|description):\s*', '', t)
        return t.strip()
    rows = []
    for tag, fn in RUNS.items():
        d = json.load(open(P/fn))
        for s in d['steps']:
            if s['step'] == 0: continue
            for li, line in enumerate(s['added_lines']):
                if line.strip() in ('---',): continue
                t = clean(line)
                if not t: continue
                for si, sent in enumerate(SPLIT.split(t)):
                    sent = sent.strip()
                    if len(sent.split()) < 3: continue
                    rows.append(dict(run=tag, step=s['step'], candidate_id=s['candidate_id'], kept=s['kept'],
                                     line=li, sent=si, text=sent))
    D.mkdir(parents=True, exist_ok=True)
    random.Random(20261023).shuffle(rows)
    for i, r in enumerate(rows): r['id'] = f'S{i:04d}'
    json.dump(rows, open(D / 'sentences_all.json', 'w'), ensure_ascii=False, indent=1)
    json.dump([{'id': r['id'], 'text': r['text']} for r in rows], open(D / 'sentences_blind.json', 'w'), ensure_ascii=False, indent=1)
    json.dump({r['id']: {k: r[k] for k in ('run','step','candidate_id','kept')} for r in rows}, open(D / 'blind_key.json', 'w'), indent=1)
    from collections import Counter
    c = Counter(r['run'] for r in rows)
    print(len(rows), dict(c))
    print("steps with sentences per run:", {t: len({r['step'] for r in rows if r['run']==t}) for t in RUNS})
