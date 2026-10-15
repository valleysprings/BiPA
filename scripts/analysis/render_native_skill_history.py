#!/usr/bin/env python3
"""Export each cell's current run into results/<model>/<method>/<bench>/ (summary.json, rounds.json, per-round skills, final). Read-only on runs, idempotent, no API calls."""
import argparse, hashlib, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bipa.layout import SKILLS, BASELINE_ROOT, relocate, result_dir, current_run

ARMS = ('none', 'seed', 'skillopt', 'gepa', 'skilladam', 'ours')
MODELS = ('deepseek-flash', 'gpt-6-luna')
BENCHES = ('tau3-bench', 'automationbench', 'terminal-bench-2-1')


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Reader:
    """Persisted optimizer objects; skill refs carry their exact text."""
    def bytes(self, path):
        try:
            return relocate(path).read_bytes()
        except FileNotFoundError:
            return None

    def json(self, path, default=None):
        data = self.bytes(path)
        try:
            return json.loads(data) if data is not None else default
        except (ValueError, UnicodeError):
            return default

    def text(self, text, source):
        if not isinstance(text, str):
            return {'status': 'unwitnessed', 'source': str(source)}
        return {'status': 'persisted', 'sha256': digest(text.encode()), 'text': text, 'source': str(source)}

    def skill(self, path):
        data = self.bytes(path)
        return self.text(data.decode() if data is not None else None, relocate(path))


def skillopt(reader, run):
    folder = run / 'skillopt'
    runtime = reader.json(folder / 'runtime_state.json', {})
    history = reader.json(folder / 'history.json', [])
    by_step = {row['step']: row for row in history if isinstance(row, dict) and 'step' in row}
    completed = runtime.get('last_completed_step')
    initial = reader.skill(folder / 'skills/skill_v0000.md')
    if initial['status'] != 'persisted':
        initial = reader.skill(folder / 'seed.md')
    events = []
    for directory in sorted((folder / 'steps').glob('step_*')):
        if not re.fullmatch(r'step_\d+', directory.name):
            continue
        step = int(directory.name.split('_')[1])
        record = reader.json(directory / 'step_record.json', {})
        row = by_step.get(step, {})
        candidate = reader.skill(directory / 'candidate_skill.md')
        parent = reader.skill(folder / 'skills' / f'skill_v{step - 1:04d}.md')
        selected = reader.skill(folder / 'skills' / f'skill_v{step:04d}.md')
        issues = []
        declared = row.get('candidate_hash') or record.get('candidate_hash')
        if declared and candidate.get('sha256') and not candidate['sha256'].startswith(declared):
            issues.append('candidate hash does not match persisted decision')
        committed = (isinstance(completed, int) and step <= completed and bool(row) and bool(record)
                     and row.get('action') == record.get('action') and selected['status'] == 'persisted' and not issues)
        scores = {k: row[k] for k in ('rollout_hard', 'rollout_n', 'selection_hard', 'candidate_gate_score', 'current_score', 'best_score') if k in row}
        events.append({'round': step, 'status': 'committed' if committed else 'pending',
                       'decision': row.get('action') if committed else None,
                       'uncommitted_action': row.get('action') if not committed else None,
                       'scores': scores, 'parent': parent, 'candidate': candidate,
                       'selected': selected if committed else {'status': 'unwitnessed'}, 'issues': issues})
    return initial, events


def gepa(reader, run):
    folder = run / 'gepa'
    pool = reader.json(folder / 'candidates.json', [])
    logs = reader.json(folder / 'run_log.json', [])
    refs = [reader.text(row.get('current_candidate'), f'{folder}/candidates.json#/{i}') for i, row in enumerate(pool)]
    registry = {r['sha256']: r for r in refs if r.get('sha256')}
    for path in sorted((run / 'round-evidence').glob('*.json')):
        event = reader.json(path, {})
        skill = event.get('skill') or {}
        if event.get('arm') == 'gepa' and isinstance(event.get('step'), int):
            ref = reader.text(skill.get('text'), str(path))
            if ref.get('sha256') and ref['sha256'] == skill.get('sha256'):
                registry[ref['sha256']] = ref
    raw = reader.bytes(folder / 'run_log.txt')
    lines = (raw.decode() if raw is not None else '').splitlines(keepends=True)
    selected, proposed, blocks = {}, {}, {}
    for index, line in enumerate(lines):
        m = re.match(r'Iteration (\d+): Selected program (\d+) score:', line)
        if m:
            selected[int(m[1])] = int(m[2])
        m = re.match(r'Iteration (\d+): Proposed new text for current_candidate: ', line)
        if m:
            step = int(m[1])
            stop = next((i for i in range(index + 1, len(lines)) if lines[i].startswith(f'Iteration {step}: ')), len(lines))
            text = line.split('Proposed new text for current_candidate: ', 1)[1] + ''.join(lines[index + 1:stop])
            text = text[:-1] if text.endswith('\n') else text
            proposed[step] = registry.get(digest(text.encode()), {'status': 'unwitnessed'})
        m = re.match(r'Iteration (\d+): ', line)
        if m:
            blocks.setdefault(int(m[1]), []).append(line)
    records = {row['i'] + 1: row for row in logs if isinstance(row, dict) and isinstance(row.get('i'), int)}
    events = []
    for step in sorted((set(records) | set(selected)) - {0}):
        record = records.get(step)
        pidx = record.get('selected_program_candidate') if record else selected.get(step)
        parent = refs[pidx] if isinstance(pidx, int) and 0 <= pidx < len(refs) else {'status': 'unwitnessed'}
        candidate = proposed.get(step, {'status': 'unwitnessed'})
        nidx = record.get('new_program_idx') if record else None
        block = ''.join(blocks.get(step, []))
        if isinstance(nidx, int) and 0 <= nidx < len(refs):
            status, decision, chosen, candidate = 'committed', 'accepted_into_pool', refs[nidx], refs[nidx]
        elif 'not better than old score' in block:
            status, decision, chosen = 'committed', 'rejected_by_subsample', parent
        elif 'All subsample scores perfect' in block:
            status, decision, chosen = 'committed', 'skipped_perfect_subsample', parent
        elif 'did not propose' in block:
            status, decision, chosen = 'committed', 'no_proposal', parent
        elif 'Traceback (most recent call last)' in block:
            status, decision, chosen = 'committed', 'reflection_failed', parent
        else:
            status, decision, chosen = 'pending', None, {'status': 'unwitnessed'}
        scores = {}
        if record:
            old, new = record.get('subsample_scores'), record.get('new_subsample_scores')
            scores = {'minibatch_parent': sum(old) if old else None, 'minibatch_candidate': sum(new) if new else None,
                      'minibatch_n': len(old) if old else None}
        events.append({'round': step, 'status': status, 'decision': decision, 'uncommitted_action': None,
                       'scores': scores, 'parent': parent, 'candidate': candidate, 'selected': chosen, 'issues': []})
    return (refs[0] if refs else {'status': 'unwitnessed'}), events


def skilladam(reader, run):
    """Committed iterations of upstream's checkpoint; parent/candidate/selected rebuilt from the persisted patches by
    upstream's applier (audit_skill_history.skilladam_rounds); the gate scores are on the training batch itself."""
    from audit_skill_history import skilladam_rounds
    folder = run / 'skilladam'
    if not (folder / 'checkpoint.json').is_file():
        return reader.skill(folder / 'seed.md'), []
    events, source = [], f'{folder}/checkpoint.json'
    for x in skilladam_rounds(folder):
        row, base, cand = x['record'], x['record']['baseline'], x['record']['candidate']
        events.append({'round': row['iteration'] + 1, 'status': 'committed',
                       'decision': 'accepted' if row['accepted'] else 'patch_failed' if row['patch_mode'] == 'failed' else 'rejected',
                       'uncommitted_action': None, 'reason': row.get('reason'),
                       'scores': {'batch_parent': base.get('primary'), 'batch_candidate': cand.get('primary'),
                                  'batch_n': len(row['validation_case_ids']), 'edit_budget': row.get('edit_budget'),
                                  'v_ema': row.get('v_ema_after')},
                       'parent': reader.text(x['before'], source), 'candidate': reader.text(x['proposed'], x['patch_path']),
                       'selected': reader.text(x['after'], source), 'patch': x['patch'], 'issues': []})
    return reader.skill(folder / 'seed.md'), events


def export(reader, arm, model, benchmark, run):
    files = {}
    def put(rel, text):
        files[rel] = text.encode() if isinstance(text, str) else text
    final = reader.skill(run / 'final/SKILL.md')
    if arm not in ('skillopt', 'gepa', 'skilladam'):
        fixed = final if final['status'] == 'persisted' else (reader.text('', 'none') if arm == 'none' else reader.skill(SKILLS / 'seed/SKILL.md'))
        initial, events = fixed, []
    else:
        initial, events = {'skillopt': skillopt, 'gepa': gepa, 'skilladam': skilladam}[arm](reader, run)
    rounds = [{'round': 0, 'status': 'committed', 'decision': None, 'selected_sha256': initial.get('sha256'), 'scores': {}}]
    if initial['status'] == 'persisted':
        put('skills/round-00/SKILL.md', initial['text'])
    for e in events:
        n = e['round']
        row = {'round': n, 'status': e['status'], 'decision': e['decision'], 'uncommitted_action': e['uncommitted_action'],
               'parent_sha256': e['parent'].get('sha256'), 'candidate_sha256': e['candidate'].get('sha256'),
               'selected_sha256': e['selected'].get('sha256'), 'scores': e['scores'], 'issues': e['issues']}
        if e.get('reason'):
            row['reason'] = e['reason']
        rounds.append(row)
        for name in ('parent', 'candidate', 'selected'):
            if e[name]['status'] == 'persisted':
                put(f'skills/round-{n:02d}/{name}.md', e[name]['text'])
        if e.get('patch'):
            put(f'skills/round-{n:02d}/patch.diff', e['patch'])
    if final['status'] == 'persisted':
        put('final/SKILL.md', final['text'])
    summary = reader.json(run / 'summary.json', {})
    put('summary.json', json.dumps({**summary, 'source_run': str(run)}, ensure_ascii=False, indent=1) + '\n')
    put('rounds.json', json.dumps({'arm': arm, 'model': model, 'benchmark': benchmark, 'run_id': run.name, 'run_path': str(run),
        'final_sha256': final.get('sha256'), 'rounds': rounds,
        'semantics': 'Decisions are the optimizer\'s own persisted state; status pending means the step is not committed. selected is the per-step lineage, not the GEPA pool best. Not a certificate.'},
        ensure_ascii=False, indent=1) + '\n')
    return files


def sync(dest, files):
    """Write only what changed; drop owned files that are no longer produced."""
    owned = [p for p in (dest / 'rounds.json', dest / 'summary.json') if p.is_file()] + [p for d in ('skills', 'final') if (dest / d).is_dir() for p in (dest / d).rglob('*') if p.is_file()]
    for p in owned:
        if p.relative_to(dest).as_posix() not in files:
            p.unlink()
    for rel, data in files.items():
        p = dest / rel
        if not p.is_file() or p.read_bytes() != data:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    for d in sorted((dest / 'skills').rglob('*'), reverse=True) if (dest / 'skills').is_dir() else []:
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    reader, count = Reader(), 0
    for model in MODELS:
        for arm in ARMS:
            for bench in BENCHES:
                run = current_run(arm, model, bench)
                if run is not None:
                    sync(result_dir(arm, model, bench), export(reader, arm, model, bench, run))
                    count += 1
    print(count, 'cells exported')


if __name__ == '__main__':
    main()
