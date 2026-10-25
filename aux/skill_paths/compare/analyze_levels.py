"""Merge the two blind judges (judge_J1.jsonl, judge_J2.jsonl in OUT/skill_paths/compare) with the run key; decision rules fixed before any judge output.
Order: (1) read both judges, (2) weighted kappa and the two-level kappa, (3) only then open blind_key.json,
(4) D = share(S2+S3 | hand-written) - share(S2+S3 | near-empty) per optimizer, for each judge separately,
(5) first-kept-candidate composition (H3), (6) cluster bootstrap over candidates (record only)."""
import json, random
from collections import Counter, defaultdict

from src.layout import OUT


if __name__ == "__main__":
    D = OUT / "skill_paths/compare"

    ORDER = ['S0', 'S1', 'S2', 'S3']


    def load(fn):
        d = {}
        for line in open(D / fn, encoding='utf-8'):
            r = json.loads(line); d[r['id']] = r
        return d


    J = {'J1': load('judge_J1.jsonl'), 'J2': load('judge_J2.jsonl')}
    ids = sorted(set(J['J1']) & set(J['J2']))
    print('judged by both:', len(ids), 'of', len(json.load(open(D / 'sentences_blind.json'))))


    def wkappa(a, b, levels):
        """Linear-weighted Cohen kappa on an ordered scale; sentences labelled X by either judge are dropped."""
        pairs = [(x, y) for x, y in zip(a, b) if x in levels and y in levels]
        k = len(levels); idx = {l: i for i, l in enumerate(levels)}
        n = len(pairs)
        obs = [[0] * k for _ in range(k)]
        for x, y in pairs: obs[idx[x]][idx[y]] += 1
        ra = [sum(obs[i]) for i in range(k)]; cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
        num = den = 0.0
        for i in range(k):
            for j in range(k):
                w = abs(i - j) / (k - 1)
                num += w * obs[i][j] / n
                den += w * ra[i] * cb[j] / (n * n)
        return (1 - num / den if den else float('nan')), n


    a = [J['J1'][i]['level'] for i in ids]; b = [J['J2'][i]['level'] for i in ids]
    k4, n4 = wkappa(a, b, ORDER)
    two = lambda l: l if l == 'X' else ('broad' if l in ('S0', 'S1') else 'specific')
    k2, n2 = wkappa([two(x) for x in a], [two(x) for x in b], ['broad', 'specific'])
    print('weighted kappa, four levels: %.3f (n=%d); two levels broad(S0+S1) vs specific(S2+S3): %.3f (n=%d)' % (k4, n4, k2, n2))
    print('exact agreement four levels: %.3f' % (sum(x == y for x, y in zip(a, b)) / len(ids)))
    for j in J: print(j, dict(Counter(J[j][i]['level'] for i in ids)))
    ok4 = k4 >= 0.6; ok2 = k2 >= 0.6
    mode = 'four' if ok4 else ('two' if ok2 else 'none')
    print('fixed rule: kappa>=0.6 four-level ->', ok4, '; else two-level ->', ok2, '; analysis mode =', mode)

    # ---- only now open the key
    key = json.load(open(D / 'blind_key.json'))
    rows = []
    for i in ids:
        k = key[i]
        rows.append(dict(id=i, run=k['run'], step=k['step'], cand=k['candidate_id'], kept=k['kept'],
                         L1=J['J1'][i]['level'], L2=J['J2'][i]['level'],
                         et1=J['J1'][i].get('echo_task'), et2=J['J2'][i].get('echo_task'),
                         es1=J['J1'][i].get('echo_seed'), es2=J['J2'][i].get('echo_seed')))
    spec = lambda L: L in ('S2', 'S3')
    valid = lambda L: L in ORDER


    def share(rs, who):
        rs = [r for r in rs if valid(r[who])]
        return (sum(spec(r[who]) for r in rs) / len(rs) if rs else float('nan')), len(rs)


    def boot(rsH, rsN, who, B=2000, seed=20261023):
        """Cluster bootstrap over candidates (sentences of one candidate are correlated); record only."""
        rnd = random.Random(seed)
        def clusters(rs):
            c = defaultdict(list)
            for r in rs: c[(r['run'], r['cand'])].append(r)
            return list(c.values())
        cH, cN = clusters(rsH), clusters(rsN)
        ds = []
        for _ in range(B):
            sH = [r for c in (rnd.choice(cH) for _ in cH) for r in c]
            sN = [r for c in (rnd.choice(cN) for _ in cN) for r in c]
            ds.append(share(sH, who)[0] - share(sN, who)[0])
        ds.sort()
        return ds[int(.025 * B)], ds[int(.975 * B)]


    print('\n=== D = share(S2+S3 | hand-written) - share(S2+S3 | near-empty), kept candidates only ===')
    res = {}
    for opt in ('gepa', 'skillopt'):
        for who, nm in (('L1', 'J1'), ('L2', 'J2')):
            H = [r for r in rows if r['run'] == 'H-' + opt and r['kept']]
            N = [r for r in rows if r['run'] == 'N-' + opt and r['kept']]
            sH, nH = share(H, who); sN, nN = share(N, who)
            lo, hi = boot(H, N, who)
            res[(opt, nm)] = sH - sN
            print('%-9s %s  H: %.3f (n=%d)  N: %.3f (n=%d)  D=%+.3f  cluster-bootstrap [%+.3f, %+.3f]' % (opt, nm, sH, nH, sN, nN, sH - sN, lo, hi))

    print('\n=== level mix among kept additions (J1 / J2) ===')
    for run in ('H-gepa', 'N-gepa', 'H-skillopt', 'N-skillopt', 'H-skilladam'):
        rs = [r for r in rows if r['run'] == run and r['kept']]
        for who, nm in (('L1', 'J1'), ('L2', 'J2')):
            c = Counter(r[who] for r in rs); n = sum(c.values())
            print('%-12s %s n=%-4d %s' % (run, nm, n, {l: round(c[l] / n, 3) for l in ORDER + ['X']}))

    print('\n=== rejected candidates (same runs) S2+S3 share ===')
    for run in ('H-gepa', 'N-gepa', 'H-skillopt', 'N-skillopt'):
        rs = [r for r in rows if r['run'] == run and not r['kept']]
        print(run, ' '.join('%s %.3f (n=%d)' % ((nm,) + share(rs, who)) for who, nm in (('L1', 'J1'), ('L2', 'J2'))))

    print('\n=== H3: first kept candidate, share S0+S1 ===')
    for run in ('H-gepa', 'N-gepa', 'H-skillopt', 'N-skillopt'):
        ks = sorted({r['step'] for r in rows if r['run'] == run and r['kept']})
        if not ks: continue
        rs = [r for r in rows if r['run'] == run and r['kept'] and r['step'] == ks[0]]
        out = []
        for who, nm in (('L1', 'J1'), ('L2', 'J2')):
            v = [r for r in rs if valid(r[who])]
            out.append('%s %.3f (n=%d)' % (nm, sum(not spec(r[who]) for r in v) / len(v), len(v)))
        print(run, 'first kept step', ks[0], ' '.join(out))

    print('\n=== echo flags (share of kept additions, J1 / J2) ===')
    for run in ('H-gepa', 'N-gepa', 'H-skillopt', 'N-skillopt', 'H-skilladam'):
        rs = [r for r in rows if r['run'] == run and r['kept']]
        n = len(rs)
        print('%-12s echo_task %.3f / %.3f   echo_seed %.3f / %.3f' % (run, sum(bool(r['et1']) for r in rs) / n, sum(bool(r['et2']) for r in rs) / n,
                                                                      sum(bool(r['es1']) for r in rs) / n, sum(bool(r['es2']) for r in rs) / n))

    print('\n=== verdict under the fixed rules ===')
    for opt in ('gepa', 'skillopt'):
        d1, d2 = res[(opt, 'J1')], res[(opt, 'J2')]
        same = (d1 > 0) == (d2 > 0)
        big = min(abs(d1), abs(d2)) >= 0.10
        print(opt, 'D(J1)=%+.3f D(J2)=%+.3f  same sign: %s  both |D|>=0.10: %s' % (d1, d2, same, big))
    print('H1/H2 (near-empty broader, hand-written more specific) predict D > 0.')
