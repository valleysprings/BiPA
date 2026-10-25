"""Unified $ structure parser for api.jsonl episodes (1017 prices: hit .003 / miss .15 / out .6 per 1M)."""
import json, base64, hashlib, collections
P = dict(hit=0.003e-6, miss=0.15e-6, out=0.6e-6)
def text_of(c):
    if c is None: return ''
    if isinstance(c, str): return c
    if isinstance(c, list): return '\n'.join(p.get('text', '') if isinstance(p, dict) else str(p) for p in c)
    return json.dumps(c, ensure_ascii=False)
def parse_usage(chunks):
    u = None; fp = ''
    for line in ''.join(chunks).split('\n'):
        if not line.startswith('data: ') or line.startswith('data: [DONE]'): continue
        try: d = json.loads(line[6:])
        except Exception: continue
        fp = fp or d.get('system_fingerprint') or ''
        if d.get('usage'): u = d['usage']
    return u, fp
def load_episode(apipath):
    """-> dict(reqs=[{hit,miss,out,reas}], msgs (ordered unique), fps, complete) or None"""
    reqs = {}; chunks = collections.defaultdict(list); ends = set()
    for l in open(apipath):
        r = json.loads(l); t = r['type']; i = r['request_index']
        if t == 'request': reqs[i] = r
        elif t == 'response_chunk': chunks[i].append(base64.b64decode(r['data']).decode('utf-8', 'replace'))
        elif t == 'response_end': ends.add(i)
    if not reqs: return None
    rq = []; fps = set(); callmap = {}; seen = {}; order = []; ri = -1
    for ix in sorted(reqs):
        b = json.loads(reqs[ix]['body']); u, fp = parse_usage(chunks[ix]); fps.add(fp[:6])
        if not u: continue
        ri += 1
        rq.append(dict(hit=u.get('prompt_cache_hit_tokens') or 0, miss=u.get('prompt_cache_miss_tokens') or 0, out=u.get('completion_tokens') or 0,
                       reas=((u.get('completion_tokens_details') or {}).get('reasoning_tokens')), nmsg=len(b['messages'])))
        for m in b['messages']:
            cc = text_of(m.get('content')); tcs = m.get('tool_calls') or []
            key = hashlib.blake2b(json.dumps([m.get('role'), m.get('tool_call_id'), cc, [(t['id']) for t in tcs]]).encode(), digest_size=10).hexdigest()
            if key in seen: continue
            for tc in tcs: callmap[tc['id']] = (tc['function'].get('name'), tc['function'].get('arguments') or '')
            seen[key] = 1; order.append(dict(first=ri, role=m['role'], text=cc, tcid=m.get('tool_call_id'), reasoning=m.get('reasoning_content') or '', tcs=[(t['function'].get('name'), t['function'].get('arguments') or '') for t in tcs]))
    return dict(reqs=rq, msgs=order, callmap=callmap, fps=sorted(fps), complete=set(reqs) <= ends)
def usd(rq):
    return sum(P['hit'] * r['hit'] + P['miss'] * r['miss'] + P['out'] * r['out'] for r in rq)
