"""(c, d) How independent are the 3-word seed's two tau3 evaluations (one in the SkillOpt run, one in the GEPA run)?  Episodes are cached by (skill text, task, split), so many are the same episode.
Run: python -m aux.three_word_seed.tau3_seed_overlap"""
import json, glob
from src import layout
from src.analysis import selection as S
EXE = str(layout.EXECUTION / 'gpt-6-luna')
OUT = layout.OUT / "three_word_seed"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out={}
    for split in ('test','dev'):
        sig={}; succ={}
        for opt in ('skillopt','gepa'):
            f=glob.glob(f'{EXE}/{opt}/tau3-bench/fromscratch/results/evaluations/{split}/cf7b932190ed-*/evaluation.json')[0]
            d=json.load(open(f))
            sig[opt]={e['task_id']:S.signature(e) for e in d['episodes'] if S.verified(e)}
            succ[opt]={e['task_id']:int(bool(e['success'])) for e in d['episodes'] if S.verified(e)}
        common=set(sig['skillopt'])&set(sig['gepa'])
        same=[t for t in common if sig['skillopt'][t]==sig['gepa'][t]]
        diff=[t for t in common if sig['skillopt'][t]!=sig['gepa'][t]]
        out[split]=dict(common=len(common),identical=len(same),different=len(diff),outcome_flips_among_different=sum(succ['skillopt'][t]!=succ['gepa'][t] for t in diff),
                        solved_skillopt_run=sum(succ['skillopt'].values()),solved_gepa_run=sum(succ['gepa'].values()))
        print(split,out[split])
    json.dump(out,open(OUT / 'tau3_seed_overlap.json','w'))


if __name__ == "__main__":
    main()
