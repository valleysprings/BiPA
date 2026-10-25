"""Build the list of services AutoBench tasks allow (breadth vocabulary) from allowed_services in each task's agent_config.json.
Reads only that key of the 600 AutoBench tasks; writes app_vocab.json before any judge output exists."""
import json
from pathlib import Path
from collections import Counter
from src.layout import HARNESS_BENCH, OUT


if __name__ == "__main__":
    D = OUT / "skill_paths/compare"
    T = HARNESS_BENCH / 'assets/benchmarks/automationbench/tasks'
    c = Counter(); n = 0; per_domain = {}
    for d in sorted(T.iterdir()):
        f = d / 'environment' / 'agent_config.json'
        if not f.is_file(): continue
        n += 1
        svc = json.load(open(f)).get('allowed_services') or []
        for s in svc: c[s] += 1
        per_domain.setdefault(d.name.split('-')[0], Counter()).update(svc)
    print(n, 'tasks;', len(c), 'services')
    for k, v in c.most_common(): print(v, k)
    D.mkdir(parents=True, exist_ok=True)
    json.dump(dict(n_tasks=n, services=dict(c.most_common()), domains=list(per_domain)), open(D / 'app_vocab.json', 'w'), indent=1)
