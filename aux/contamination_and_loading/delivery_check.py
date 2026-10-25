"""Delivery-paradigm 2x2 on tau3 test (DeepSeek): always-resident vs on-demand skill.

Per-episode metrics follow luna_avg3.luna_row: X input tokens over distinct completed requests,
N executed tool calls, R distinct completed requests (turns). Ratios: per-task geometric mean
against the reference cell on tasks with known X, N > 0 in both.
Run:  python -m aux.contamination_and_loading.delivery_check
"""
import collections, json, math

from src.layout import relocate
from src.utils.dataloader import DS as H

DC = H / "delivery-check/tau3-bench"
CELLS = {
    "seed/A": DC / "seed-guidance/results/evaluations/test/4fd5ca441c55-57817cd02585",
    "seed/B": H / "skillzip/tau3-bench/native-20261018/results/evaluations/test/89a426c6e82d-57817cd02585",
    "skillopt/A": DC / "skillopt-guidance/results/evaluations/test/a0bdead535e7-57817cd02585",
    "skillopt/B": DC / "skillopt-native/results/evaluations/test/3b2b1fe3d99c-57817cd02585",
}


def known(ep):
    return type(ep.get("success")) is bool and ep.get("outcome_valid", True) is True


def episodes(d):
    """Base evaluation, unknown slots replaced by retry-* supplements."""
    eps = {}
    for p in sorted(d.glob("**/evaluation.json"), key=lambda p: len(p.parts)):
        for ep in json.loads(p.read_text())["episodes"]:
            k = ep["task_id"]
            if k not in eps or (not known(eps[k]) and known(ep)):
                eps[k] = ep
    return eps


def row(ep):
    reqs = [r for r in {r["request_id"]: r for r in ep.get("requests", [])}.values() if r["input_tokens"] is not None]
    calls = [c for r in reqs for c in r.get("tool_calls", []) if c.get("executed") is not False]
    skill_idx = [i for i, r in enumerate(reqs) for c in r.get("tool_calls", []) if c.get("tool") == "skill"]
    fps = set()
    for r in reqs:
        try:
            rec = json.loads(relocate(r["record_path"]).read_text())
            s = json.dumps(rec)
            i = s.find('"system_fingerprint": "')
            if i >= 0:
                fps.add(s[i + 23:i + 55])
        except Exception:
            pass
    return {"success": bool(ep["success"]) if known(ep) else None,
            "X": sum(int(r["input_tokens"]) for r in reqs) if reqs else None,
            "N": len(calls) if reqs else None, "R": len(reqs) if reqs else None,
            "skill_calls": len(skill_idx), "first_skill": skill_idx[0] if skill_idx else None,
            "fps": fps, "admitted": (ep.get("admission") or {}).get("admitted_at", "")[:10]}


def gm_ratio(a, b, key):
    common = [t for t in a if t in b and a[t][key] and b[t][key]]
    return math.exp(sum(math.log(a[t][key] / b[t][key]) for t in common) / len(common)), len(common)


def main():
    rows = {c: {t: row(ep) for t, ep in episodes(d).items()} for c, d in CELLS.items()}
    for c, rs in rows.items():
        k = [r for r in rs.values() if r["success"] is not None]
        fps = collections.Counter(f for r in rs.values() for f in r["fps"])
        trig = [r for r in rs.values() if r["skill_calls"] > 0]
        first = collections.Counter(r["first_skill"] for r in trig)
        print(f"{c}: n={len(rs)} known={len(k)} Q={100*sum(r['success'] for r in k)/len(k):.1f} "
              f"meanX={sum(r['X'] or 0 for r in rs.values())/len(rs):.0f} meanR={sum(r['R'] or 0 for r in rs.values())/len(rs):.1f} "
              f"meanN={sum(r['N'] or 0 for r in rs.values())/len(rs):.1f} triggered={len(trig)}/{len(rs)} "
              f"skill_calls={sum(r['skill_calls'] for r in rs.values())} first_idx={dict(first)} "
              f"dates={sorted(collections.Counter(r['admitted'] for r in rs.values()).items())} fps={dict(fps)}")
    pairs = [("seed/B", "seed/A"), ("skillopt/A", "seed/A"), ("skillopt/B", "seed/B"), ("skillopt/B", "skillopt/A"), ("skillopt/B", "seed/A")]
    for a, b in pairs:
        out = [f"{key}={gm_ratio(rows[a], rows[b], key)[0]:.3f}" for key in ("X", "R", "N")]
        print(f"{a} vs {b}: {' '.join(out)} (n={gm_ratio(rows[a], rows[b], 'X')[1]})")
    # paired success flips
    for a, b in pairs:
        both = [t for t in rows[a] if t in rows[b] and rows[a][t]["success"] is not None and rows[b][t]["success"] is not None]
        up = sum(rows[a][t]["success"] and not rows[b][t]["success"] for t in both)
        dn = sum(rows[b][t]["success"] and not rows[a][t]["success"] for t in both)
        print(f"flips {a} vs {b}: +{up} -{dn} of {len(both)}")


if __name__ == "__main__":
    main()
