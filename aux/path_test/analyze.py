"""Unified 1017 $ structure per bench. usage: python -m aux.path_test.analyze out.json name=rootdir [name=rootdir ...]
Episodes = <trial>/agent/api.jsonl with <trial>/result.json holding verifier_result.rewards, not under /runtime or *.attempts.
Token weights for message content are chars (tiktoken absent); tool-result share of miss tokens = chars of tool results first sent / chars of all
messages first sent, applied to the measured (usage) miss tokens."""
import sys, os, json, re, collections, statistics as st
from multiprocessing import Pool
from aux.path_test.xlib import *
SH = 32
def bash_kind(cmd):
    if 'api_search' in cmd: return 'bash:api_search'
    if 'api_fetch' in cmd:
        m = re.search(r'--url\s+\\?["\']?https?://([^/\s"\']+)', cmd); return 'bash:api_fetch'
    c = re.sub(r'^\s*(cd [^;&]+(;|&&)\s*)+', '', cmd.strip()); c = re.sub(r'^(\w+=\S+\s+)+', '', c)
    w = c.split()[:1]; return 'bash:' + (w[0] if w else '?')
def kind_of(name, args):
    if name == 'bash':
        try: return bash_kind(json.loads(args).get('command', ''))
        except Exception: return 'bash:?'
    return name or '?'
def shingles(s):
    return {hash(s[i:i + SH]) for i in range(0, max(0, len(s) - SH + 1))}
def work(job):
    bench, d = job
    try:
        res = json.load(open(d + '/result.json')); rw = (res.get('verifier_result') or {}).get('rewards')
        if not rw: return None
        e = load_episode(d + '/agent/api.jsonl')
    except Exception as ex: return None
    if not e or not e['reqs'] or not e['complete']: return None
    rq = e['reqs']; hit = sum(r['hit'] for r in rq); miss = sum(r['miss'] for r in rq); out = sum(r['out'] for r in rq)
    reas = [r['reas'] for r in rq]; reas_ok = all(x is not None for x in reas); reas_t = sum(x or 0 for x in reas)
    # message accounting
    seen_all = set(); seen_tool = set(); tool_chars = 0; all_chars = 0
    per = collections.defaultdict(lambda: [0, 0, 0])  # kind -> [bytes, redundant_any, redundant_tool]
    last = rq[-1]; tpc = (last['hit'] + last['miss']) / max(1, sum(len(m['text']) + len(m['reasoning']) + sum(len(x) + len(y) for x, y in m['tcs']) for m in e['msgs']))
    n = len(rq); save_rw = 0.0; first5 = sum(P['hit'] * r['hit'] + P['miss'] * r['miss'] + P['out'] * r['out'] for r in rq[:5])
    rcost = [P['hit'] * r['hit'] + P['miss'] * r['miss'] + P['out'] * r['out'] for r in rq]
    for m in e['msgs']:
        s = m['text'] + m['reasoning'] + ''.join(a + b for a, b in m['tcs']); all_chars += len(s)
        if m['role'] == 'tool':
            t = m['text']; name, args = e['callmap'].get(m['tcid'], ('?', '')); k = kind_of(name, args)
            sh = shingles(t); n = len(sh)
            tool_chars += len(s)
            if n:
                per[k][0] += len(t); per[k][1] += len(t) * len(sh & seen_all) / n; per[k][2] += len(t) * len(sh & seen_tool) / n
            else: per[k][0] += len(t)
            if n: save_rw += len(t) * len(sh & seen_all) / n * tpc * (P['miss'] + P['hit'] * max(0, len(rq) - m['first'] - 1))
            seen_tool |= sh; seen_all |= sh
        else:
            seen_all |= shingles(s)
    return dict(bench=bench, task=res.get('task_name'), trial=res.get('trial_name'), reward=rw.get('reward'), n_req=len(rq), hit=hit, miss=miss, out=out,
                reas=reas_t if reas_ok else None, usd=usd(rq), tool_chars=tool_chars, all_chars=all_chars, per={k: list(v) for k, v in per.items()}, fps=e['fps'], save_rw=save_rw, first5=first5, rcost=rcost)
def find(root):
    jobs = []
    for r, ds, fs in os.walk(root):
        ds[:] = [x for x in ds if x != 'runtime' and not x.endswith('.attempts')]
        if r.endswith('/agent') and 'api.jsonl' in fs: jobs.append(os.path.dirname(r))
    return jobs
def summarize(eps):
    n = len(eps); H = sum(e['hit'] for e in eps) * P['hit']; M = sum(e['miss'] for e in eps) * P['miss']; O = sum(e['out'] for e in eps) * P['out']; T = H + M + O
    ro = [e for e in eps if e['reas'] is not None]
    rs = sum(e['reas'] for e in ro) / max(1, sum(e['out'] for e in ro))
    tc = sum(e['tool_chars'] for e in eps) / sum(e['all_chars'] for e in eps)
    # tool-result share of miss tokens, per-episode char share weighted by that episode's miss tokens
    tcs = sum(e['miss'] * e['tool_chars'] / e['all_chars'] for e in eps) / sum(e['miss'] for e in eps)
    per = collections.defaultdict(lambda: [0, 0, 0])
    for e in eps:
        for k, v in e['per'].items():
            for i in range(3): per[k][i] += v[i]
    tb = sum(v[0] for v in per.values()) or 1
    kinds = sorted(per.items(), key=lambda kv: -kv[1][0])
    reqs = sorted(e['n_req'] for e in eps)
    TT = sum(e['usd'] for e in eps)
    t90 = reqs[int(.9 * (n - 1))]; t75 = reqs[int(.75 * (n - 1))]
    tail90 = sum(sum(e['rcost'][t90:]) for e in eps) / TT; tail75 = sum(sum(e['rcost'][t75:]) for e in eps) / TT
    fail = [e for e in eps if e['reward'] != 1]
    ss = sorted((e['usd'] for e in eps), reverse=True); top10 = sum(ss[:max(1, n // 10)]) / TT
    return dict(n=n, tasks=len({e['task'] for e in eps}), usd_ep=T / n, hit_pct=100 * H / T, miss_pct=100 * M / T, out_pct=100 * O / T,
                out_reason_pct_of_out=100 * rs, out_reason_pct_of_total=100 * rs * O / T, out_visible_pct_of_total=100 * (1 - rs) * O / T,
                toolres_share_of_miss_tokens=100 * tcs, toolres_miss_usd_pct_of_total=100 * tcs * M / T,
                toolres_redundancy_any_pct=100 * sum(v[1] for v in per.values()) / tb, toolres_redundancy_toolonly_pct=100 * sum(v[2] for v in per.values()) / tb,
                toolres_MB_per_ep=tb / n / 1e6,
                by_tool=[dict(kind=k, byte_share_pct=100 * v[0] / tb, red_any_pct=100 * v[1] / max(1, v[0]), red_tool_pct=100 * v[2] / max(1, v[0])) for k, v in kinds[:8]],
                turns_mean=st.mean(reqs), turns_med=st.median(reqs), turns_p10=reqs[int(.1 * (n - 1))], turns_p90=reqs[int(.9 * (n - 1))], turns_max=reqs[-1],
                success_pct=100 * sum(1 for e in eps if e['reward'] == 1) / n, mean_reward=st.mean(e['reward'] or 0 for e in eps),
                replay_rewrite_ub_pct=100 * sum(e['save_rw'] for e in eps) / TT, replay_tail_p90_ub_pct=100 * tail90, replay_tail_p75_ub_pct=100 * tail75,
                fail_usd_share_pct=100 * sum(e['usd'] for e in fail) / TT, fail_mean_usd_over_succ_mean=(st.mean(e['usd'] for e in fail) / st.mean(e['usd'] for e in eps if e['reward'] == 1)) if fail and len(fail) < n else None,
                top10pct_eps_usd_share=100 * top10, first5req_usd_share_pct=100 * sum(e['first5'] for e in eps) / TT,
                fps=sorted({f for e in eps for f in e['fps']}), usd_total=T)
if __name__ == '__main__':
    out = sys.argv[1]; R = {}
    for a in sys.argv[2:]:
        name, root = a.split('=', 1); jobs = [(name, d) for d in find(root)]
        with Pool(16) as p: eps = [x for x in p.imap_unordered(work, jobs, chunksize=1) if x]
        print(name, len(jobs), 'candidate', len(eps), 'usable', flush=True)
        R[name] = dict(summary=summarize(eps), n_candidates=len(jobs), eps=[{k: v for k, v in e.items() if k != 'per'} for e in eps])
    json.dump(R, open(out, 'w'), indent=1)
    for k, v in R.items(): print(k, json.dumps(v['summary'], indent=1))
