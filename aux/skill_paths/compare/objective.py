"""Judge-free metrics for the seed-path comparison: service-name breadth and size of added sentences, per run.
Uses sentences_all.json (run, step, kept) and app_vocab.json (built from AutoBench allowed_services).
Ambiguous words (wave, buffer, monday, zoom as verb) are excluded from the alias table; recorded here."""
import json, re
from collections import defaultdict, Counter
from src.layout import OUT


if __name__ == "__main__":
    D = OUT / "skill_paths/compare"
    rows = json.load(open(D / 'sentences_all.json'))
    vocab = json.load(open(D / 'app_vocab.json'))['services']
    SKIP = {'wave', 'buffer', 'monday', 'canva'}          # common words or too rare to matter
    EXTRA = {'google_sheets': ['sheets'], 'google_drive': ['drive'], 'helpscout': ['help scout'], 'zoho_desk': ['zoho'],
             'facebook_pages': ['facebook'], 'basecamp3': ['basecamp'], 'monday': ['monday.com']}
    alias = {}
    for s in vocab:
        if s in SKIP: continue
        names = {s, s.replace('_', ' ')} | set(EXTRA.get(s, []))
        alias[s] = re.compile(r'\b(' + '|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r')\b', re.I)
    def apps_in(t): return sorted(s for s, rx in alias.items() if rx.search(t))
    by = defaultdict(list)
    for r in rows:
        r['apps'] = apps_in(r['text']); r['words'] = len(r['text'].split()); by[r['run']].append(r)
    def summ(rs):
        if not rs: return None
        n = len(rs); ap = Counter(a for r in rs for a in r['apps'])
        return dict(n=n, words_mean=round(sum(r['words'] for r in rs) / n, 1),
                    share_app=round(sum(1 for r in rs if r['apps']) / n, 3),
                    apps_per_sentence=round(sum(len(r['apps']) for r in rs) / n, 3),
                    distinct_apps=len(ap), top=ap.most_common(4))
    out = {}
    for run, rs in by.items():
        out[run] = dict(all=summ(rs), kept=summ([r for r in rs if r['kept']]), rejected=summ([r for r in rs if not r['kept']]))
        steps = sorted({r['step'] for r in rs})
        out[run]['first_kept_step'] = min((r['step'] for r in rs if r['kept']), default=None)
        out[run]['n_steps_with_sentences'] = len(steps)
        fk = out[run]['first_kept_step']
        out[run]['first_kept'] = summ([r for r in rs if r['kept'] and r['step'] == fk])
    json.dump(out, open(D / 'objective.json', 'w'), indent=1)
    for run, o in out.items():
        print(run, 'first kept step', o['first_kept_step'], 'steps', o['n_steps_with_sentences'])
        for k in ('all', 'kept', 'rejected', 'first_kept'): print('  ', k, o[k])
