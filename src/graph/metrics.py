"""Per-episode quantities read off the graph episodes, and the within-task slope fit."""
import collections

import numpy as np

from src.graph import episode as W


def metrics(ep):
    """R, N, X, prefix F, content per turn h, m = N / R, s_F, e_R and success; None without two turns, a call and history."""
    if ep.R < 2 or ep.N < 1:
        return None
    idx = [i for i, c in enumerate(ep.comps) if c in W.FIXED]
    F = float(ep.Xrg[:, idx].sum(1).mean()) if idx else float(ep.req_in[0])
    X = ep.X
    history = X - ep.R * F
    if history <= 0 or F <= 0:
        return None
    sF = ep.R * F / X
    return dict(R=ep.R, N=ep.N, X=X, F=F, h=history / (ep.R * (ep.R - 1) / 2), m=ep.N / ep.R, sF=sF,
                eR=1 + (1 - sF) * ep.R / (ep.R - 1), succ=ep.success)


def fe_reg(items):
    """Within-task slope of log y on log R over [(task, R, y)], with SE, and the demeaned points."""
    x = np.log([r[1] for r in items])
    y = np.log([r[2] for r in items])
    groups = collections.defaultdict(list)
    for i, it in enumerate(items):
        groups[it[0]].append(i)
    xs, ys = x.copy(), y.copy()
    for idx in groups.values():
        xs[idx] -= x[idx].mean()
        ys[idx] -= y[idx].mean()
    slope = float((xs * ys).sum() / (xs * xs).sum())
    res = ys - slope * xs
    se = float(np.sqrt((res ** 2).sum() / (len(xs) - len(groups) - 1) / (xs * xs).sum()))
    r2 = float(1 - (res ** 2).sum() / (ys ** 2).sum())
    return dict(slope=slope, se=se, n=len(xs), n_tasks=len(groups), r2=r2), (xs, ys)
