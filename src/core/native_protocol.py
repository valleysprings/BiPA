"""Freeze benchmark-migration protocol; never adopt legacy search checkpoints."""
import hashlib
from collections import Counter
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.core.layout import MITIGATION_ROOT, OPTIMIZERS, relocate


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def source_files(tree):
    for directory, children, names in os.walk(tree):
        children[:] = [name for name in children if name not in ('.venv', 'venv', 'assets', '__pycache__', '.git', 'node_modules')]
        for name in sorted(names):
            if name.endswith('.py'):
                yield Path(directory) / name


def prepare(args, out, sample, schedule, seed):
    base = Path(__file__).resolve().parent  # bipa/src/core
    files = [base / name for name in ('../run.py', 'native_protocol.py', '../utils/executor.py', 'panels.py',
                                     '../utils/llm.py', 'native_trace.py', '../../configs/tasks.yaml')]
    if args.arm in ('skillopt', 'gepa', 'skilladam'):
        files.append(base / '../adapters' / (args.arm + '.py'))
    if args.arm == 'skilladam':  # the prompts its proposer and tracker read: upstream's core prompts and our 5 files
        files.extend(sorted((OPTIMIZERS / 'skilladam/skilladam/core/prompts').glob('*.md')))
        files.extend(sorted((base / '../prompts/skilladam').rglob('*.md')))
    files.extend(base / '../dynamics' / name for name in ('audit_native_run.py', 'audit_skill_history.py'))
    upstream = OPTIMIZERS / args.arm
    if upstream.exists():
        files.extend(sorted(upstream.rglob('*.py')))
        files.extend(sorted(upstream.rglob('*.yaml')))
    import harness_bench
    from src.core.panels import CONFIG
    backend_roots = {"import": Path(harness_bench.__file__).resolve().parent.parent,
                     "worker": Path(CONFIG["backend"]["bench_root"]).resolve()}
    worker_python = os.environ.get("HARNESS_PYTHON") or str(Path(CONFIG["backend"]["bench_root"]) / ".venv/runtime/bin/python")
    backend_code = {}
    for label, tree in backend_roots.items():
        entries = {str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(source_files(tree))}
        backend_code[label] = {"root": str(tree), "tree_sha256": digest(entries), "files": entries}
    spec = {'schema': 'native-benchmark-migration-v1', 'protocol': args.protocol,
            'feedback_mode': args.feedback_mode, 'arm': args.arm, 'model': args.model,
            'benchmark': args.benchmark, 'harness': args.harness, 'scale': args.scale,
            'panels': sample, 'batches': schedule, 'train_exclude': list(getattr(args, 'train_exclude', [])), 'seed_sha256': hashlib.sha256(seed.encode()).hexdigest(),
            'cache_root': str(args.cache_root.resolve()), 'backend_code': backend_code,
            'unknown_outcome_retry_policy': {'schema': 'native-unknown-infrastructure-retry-v1',
                'splits': ['train', 'dev', 'test'], 'max_extra_attempts': 2,
                'eligibility': 'unknown verdict only; known success and known failure never resampled',
                'fixed': ['task_id', 'repetition', 'skill_version', 'split'],
                'archive': 'official retry-k/evaluation.json and replaced_episodes lineage',
                'exhausted': 'abort; manual review; resume does not replenish budget'},
            'llm_answer_replay': False,
            'resume_semantics': 'upstream persisted checkpoints and partial proposal artifacts are reused; newly requested LLM calls are fresh, never occurrence replay',
            'worker_python': worker_python, 'worker_python_resolved': str(Path(worker_python).resolve()),
            'transport_adaptation': 'recorded streaming OpenAI SDK; inherited per-arm temperature and max_tokens; infrastructure retry ledger separate from proposals',
            'code': {str(p.relative_to(MITIGATION_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'interpretation': 'unchanged upstream optimization algorithm on migrated benchmark/model; not published dataset reproduction'}
    spec['config_hash'] = digest(spec)
    marker = out / 'native_protocol.json'
    if args.resume and marker.exists():
        runtime_code = spec['code']
        spec = json.loads(marker.read_text())  # a resumed run keeps its recorded protocol
        for key in ('arm', 'model', 'benchmark', 'harness', 'scale', 'protocol', 'feedback_mode'):
            if spec.get(key) != getattr(args, key):
                raise ValueError('Resume differs from frozen protocol: ' + key)
        if relocate(spec['cache_root']).resolve() != args.cache_root.resolve():
            raise ValueError('Resume differs from frozen cache root')
        with (out / 'native-resumes.jsonl').open('a') as stream:
            stream.write(json.dumps({'resumed_utc': datetime.now(timezone.utc).isoformat(),
                                     'runtime_code': runtime_code,
                                     'semantics': 'upstream-native-resume; new LLM requests fresh'}, sort_keys=True) + '\n')
    else:
        out.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n')
    return spec


def require_complete(rows, requested):
    """Every requested task has a row; unknown outcomes stay recorded as unknown (Q is known-only)."""
    expected = Counter((row, 0) if isinstance(row, str) else (row['task_id'], row.get('repetition', 0))
                       for row in requested)
    actual = Counter((row['id'], row.get('repetition', 0)) for row in rows)
    if actual != expected or any(count != 1 for count in expected.values()):
        raise RuntimeError('native final panel incomplete or mismatched; summary not committed')
