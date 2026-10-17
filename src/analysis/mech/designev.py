"""Paper 1 (mechanism paper): the empirical support of design principles P2, P3 and P4 for a cost-aware skill optimizer.

Role in the argument: the paper measures that optimizers inflate input tokens, mostly through extra turns and actions. This module asks what a better optimizer would
have to do about it, using only the logged baseline proposals (no new runs).
 A (P2, price the behavior a rule induces): over the 59 canonical dev edges (edges.load_dev_edges), how well does the change in log skill length, the change in
   log turns (dlogR) or in log actions (dlogN) explain the candidate's change in log input tokens dlogX?  R^2 and Spearman of simple regressions.  At test level
   (nine learned skills) the same for log rho of input tokens on log of the skill's word ratio vs the seed and on log rho_turns.  If skill length ranked candidates by
   their cost, a length penalty would price the right thing; if turns and actions explain far more, it prices the smaller quantity.
 B (P4, make proposals cost-aware): share of all 59 proposals (accepted and rejected) with dlogX > 0 by optimizer; dlogX of kept vs rejected; and a counterfactual gate
   applied to the logged proposals: accept iff dQ >= 0 and dlogX <= 0.  It reports how many accepted edges it keeps, how many rejected it would accept and the mean
   dlogX of the kept set.  If almost nothing passes, the gate has nothing to choose from: the extra tokens arrive with the proposals.
 C (P3, give the editor operators that cut cost): how many edges remove any text or shorten the skill, chars added vs removed by optimizer, and mean dlogX of
   removing/shortening edges vs pure additions.
Counterfactual, not an experiment: the logged proposals are those of the success-only optimizers; a cost-aware editor would propose different candidates.

Commands (from bipa):           .venv/bin/python -m src.analysis.mech.designev compute  -> assets/cache/analysis/mech/designev.json
                               python3 -m src.analysis.mech.designev draw -> fig/mech/mech_designev.{pdf,svg}, data/mech/mech_designev.md
"""
import argparse
import json

import numpy as np
from scipy import stats

from src.analysis.mech import common as MC
from src.analysis.mech import edges as E

OPTS = E.OPTS
NAMES = E.NAMES
SELF_CMD = ".venv/bin/python -m src.analysis.mech.designev compute\npython3 -m src.analysis.mech.designev draw"


def fit(x, y):
    """R^2 of the simple regression y ~ x and the Spearman correlation, with the sample size."""
    reg = stats.linregress(x, y)
    rho = stats.spearmanr(x, y)
    return dict(n=len(x), r2=float(reg.rvalue ** 2), slope=float(reg.slope), r=float(reg.rvalue),
                spearman=float(rho.statistic), spearman_p=float(rho.pvalue))


def part_a(edges):
    """What predicts dlogX: edge level (59 dev edges) and test level (nine learned skills)."""
    dlogx = np.array([e["dlogX"] for e in edges])
    dloglen = np.array([np.log(e["len_c"] / e["len_p"]) for e in edges])
    dlogr = np.array([e["dlogR"] for e in edges])
    dlogn = np.array([e["dlogN"] for e in edges])
    edge_level = {
        "skill length": fit(dloglen, dlogx),
        "turns": fit(dlogr, dlogx),
        "actions": fit(dlogn, dlogx),
    }
    # test level: log rho of input tokens vs log word ratio of the skill and log rho_turns
    attribution = json.loads((MC.CACHE / "attribution.json").read_text())["cells"]
    compaction = json.loads((MC.CACHE / "compaction.json").read_text())["cells"]
    sizes = json.loads((MC.CACHE / "skilltext.json").read_text())["sizes"]
    turns = {(c["method"], c["bench"]): c["rho_turns"] for c in compaction}
    cells = []
    for c in attribution:
        key = c["method"].lower() + "|" + c["bench"]
        cells.append(dict(method=c["method"], bench=c["bench"], log_rho=c["A"]["total"],
                          log_len=float(np.log(sizes[key]["words_ratio"])), log_rho_turns=float(np.log(turns[(c["method"], c["bench"])]))))
    log_rho = np.array([c["log_rho"] for c in cells])
    test_level = {
        "skill length": fit(np.array([c["log_len"] for c in cells]), log_rho),
        "turns": fit(np.array([c["log_rho_turns"] for c in cells]), log_rho),
    }
    return dict(edge_level=edge_level, test_level=test_level, test_cells=cells)


def part_b(edges):
    """Proposals above zero, kept vs rejected, and the cost-aware gate (accept iff dQ >= 0 and dlogX <= 0)."""
    by_opt = {}
    for opt in OPTS:
        group = [e for e in edges if e["opt"] == opt]
        by_opt[opt] = dict(n=len(group), n_above=int(sum(e["dlogX"] > 0 for e in group)),
                           share_above=float(np.mean([e["dlogX"] > 0 for e in group])) if group else None)
    accepted = [e for e in edges if e["accepted"]]
    rejected = [e for e in edges if not e["accepted"]]
    dq_ok = [e for e in edges if e["dQ"] >= 0]
    passing = [e for e in edges if e["dQ"] >= 0 and e["dlogX"] <= 0]
    kept_acc = [e for e in passing if e["accepted"]]
    kept_rej = [e for e in passing if not e["accepted"]]
    gate_set = kept_acc + kept_rej
    return dict(
        n=len(edges), n_above=int(sum(e["dlogX"] > 0 for e in edges)), share_above=float(np.mean([e["dlogX"] > 0 for e in edges])),
        by_opt=by_opt,
        accepted=dict(n=len(accepted), mean=float(np.mean([e["dlogX"] for e in accepted])), median=float(np.median([e["dlogX"] for e in accepted])),
                      share_above=float(np.mean([e["dlogX"] > 0 for e in accepted])), values=[e["dlogX"] for e in accepted]),
        rejected=dict(n=len(rejected), mean=float(np.mean([e["dlogX"] for e in rejected])), median=float(np.median([e["dlogX"] for e in rejected])),
                      share_above=float(np.mean([e["dlogX"] > 0 for e in rejected])), values=[e["dlogX"] for e in rejected]),
        mannwhitney_p=float(stats.mannwhitneyu([e["dlogX"] for e in accepted], [e["dlogX"] for e in rejected]).pvalue),
        n_dq_ok=len(dq_ok),
        gate=dict(n_pass=len(passing), accepted_kept=len(kept_acc), accepted_total=len(accepted),
                  rejected_taken=len(kept_rej), rejected_total=len(rejected),
                  mean_dlogX_kept=float(np.mean([e["dlogX"] for e in gate_set])) if gate_set else None,
                  mean_dlogX_accepted_kept=float(np.mean([e["dlogX"] for e in kept_acc])) if kept_acc else None,
                  passing=[dict(opt=e["opt"], bench=e["bench"], child=e["child"], accepted=e["accepted"], dlogX=e["dlogX"], dQ=e["dQ"]) for e in passing]),
        dlogx_min=float(min(e["dlogX"] for e in edges)),
    )


def part_c(edges):
    """Edges that remove text or shorten the skill, chars added vs removed, and their dlogX against pure additions."""
    removing = [e for e in edges if e["rem_chars"] > 0]
    shortening = [e for e in edges if e["len_c"] < e["len_p"]]
    additions = [e for e in edges if e["rem_chars"] == 0]
    by_opt = {}
    for opt in OPTS:
        group = [e for e in edges if e["opt"] == opt]
        by_opt[opt] = dict(n=len(group), added=int(sum(e["add_chars"] for e in group)), removed=int(sum(e["rem_chars"] for e in group)),
                           n_removing=int(sum(e["rem_chars"] > 0 for e in group)), n_shortening=int(sum(e["len_c"] < e["len_p"] for e in group)))

    def mean_of(group):
        return float(np.mean([e["dlogX"] for e in group])) if group else None

    return dict(n=len(edges), n_removing=len(removing), n_shortening=len(shortening), n_additions=len(additions),
                added=int(sum(e["add_chars"] for e in edges)), removed=int(sum(e["rem_chars"] for e in edges)),
                by_opt=by_opt, mean_dlogX_removing=mean_of(removing), mean_dlogX_shortening=mean_of(shortening),
                mean_dlogX_additions=mean_of(additions),
                shortening=[dict(opt=e["opt"], bench=e["bench"], accepted=e["accepted"], dlogX=e["dlogX"], dQ=e["dQ"]) for e in shortening])


def compute():
    """Parts A, B, C on the 59 canonical dev edges and the nine test-level skills."""
    edges = E.load_dev_edges()
    out = dict(n_edges=len(edges), A=part_a(edges), B=part_b(edges), C=part_c(edges))
    MC.write_json("designev", out)
    print("A edge", {k: round(v["r2"], 3) for k, v in out["A"]["edge_level"].items()}, "test", {k: round(v["r2"], 3) for k, v in out["A"]["test_level"].items()})
    print("B", out["B"]["share_above"], out["B"]["gate"]["n_pass"], out["B"]["gate"]["accepted_kept"], out["B"]["gate"]["rejected_taken"])
    print("C", out["C"]["n_removing"], out["C"]["n_shortening"], out["C"]["added"], out["C"]["removed"])
    return out


def panel_predict(ax, data):
    """R^2 of dlogX on skill-length change, turns and actions: dev edges (dark bars) and, for the length and turn ratios, the nine test-level skills (light bars)."""
    edge = data["A"]["edge_level"]
    test = data["A"]["test_level"]
    # two populations, drawn as two labelled groups: the dev-scored edits (top) and the nine learned skills (bottom)
    labels = ["skill length", "turns", "tool calls", "skill length", "turns"]
    values = [edge["skill length"]["r2"], edge["turns"]["r2"], edge["actions"]["r2"], test["skill length"]["r2"], test["turns"]["r2"]]
    colors = ["#999999", "#2166ac", "#2166ac", "#cccccc", "#92c5de"]
    ypos = np.array([5.6, 4.6, 3.6, 1.4, 0.4])
    ax.barh(ypos, values, color=colors, height=0.68)
    ax.text(-0.02, 6.45, "59 dev-scored edits", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.5, fontweight="bold")
    ax.text(-0.02, 2.25, "9 learned skills", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.5, fontweight="bold")
    ax.set_ylim(-0.3, 6.8)
    for y, v in zip(ypos, values):
        ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=6.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=6.5, linespacing=0.9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"token change explained ($R^2$)", fontsize=7, labelpad=1)
    ax.tick_params(axis="x", labelsize=6.5, length=2)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_gate(ax, data):
    """dlogX (x100 log points) of rejected and accepted proposals as a strip with a zero line; the share above zero and the cost-aware gate result are written on the panel."""
    b = data["B"]
    rng = np.random.default_rng(0)
    groups = [("rejected", b["rejected"], "#b2182b"), ("accepted", b["accepted"], "#2166ac")]
    for i, (name, g, color) in enumerate(groups):
        values = 100 * np.array(g["values"])
        jitter = rng.uniform(-0.17, 0.17, len(values))
        ax.scatter(i + jitter, values, s=7, color=color, alpha=0.75, linewidths=0)
        ax.hlines(100 * g["mean"], i - 0.3, i + 0.3, color="k", lw=1.0)
        ax.text(i, 112, f"{100 * g['share_above']:.0f}% add", ha="center", va="top", fontsize=6)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"rejected\n({b['rejected']['n']} edits)", f"kept\n({b['accepted']['n']} edits)"], fontsize=6.5, linespacing=0.9)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(-95, 115)
    ax.set_ylabel(r"change in log input tokens ($\times$100)", fontsize=7, labelpad=1)
    ax.tick_params(axis="y", labelsize=6.5, length=2)
    ax.tick_params(axis="x", length=0)
    g = b["gate"]
    ax.text(0.5, 0.02, f"a cost gate passes\n{g['accepted_kept']} of {g['accepted_total']} and {g['rejected_taken']} of {g['rejected_total']}",
            transform=ax.transAxes, fontsize=5.8, va="bottom", ha="center", linespacing=1.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw(data):
    """Standalone 3.25 in two-panel figure."""
    import matplotlib
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.25, 1.7), gridspec_kw=dict(width_ratios=[1, 1.15]))
    fig.subplots_adjust(left=0.2, right=0.99, bottom=0.2, top=0.97, wspace=0.8)
    panel_predict(a1, data)
    panel_gate(a2, data)
    MC.save_figure(fig, "designev")


def fmt(x, nd=2):
    """Number with nd decimals, 'n/a' for None."""
    return "n/a" if x is None else f"{x:.{nd}f}"


def record(data):
    """paper/assets/data/mech/mech_designev.md."""
    a, b, c = data["A"], data["B"], data["C"]
    rows_a = []
    for level, fits in (("dev edges (59), dlogX on", a["edge_level"]), ("nine skills, log rho on", a["test_level"])):
        for name, f in fits.items():
            rows_a.append([level, name, f["n"], fmt(f["r2"]), fmt(f["spearman"]), fmt(f["spearman_p"], 3), fmt(f["slope"])])
    table_a = MC.md_table(["level", "predictor", "n", "R^2", "Spearman", "p", "slope"], rows_a)
    rows_b = [[NAMES[o], v["n"], v["n_above"], fmt(v["share_above"])] for o, v in b["by_opt"].items()]
    rows_b.append(["all", b["n"], b["n_above"], fmt(b["share_above"])])
    table_b = MC.md_table(["optimizer", "proposals", "dlogX > 0", "share"], rows_b)
    rows_k = [[n, g["n"], fmt(100 * g["mean"], 1), fmt(100 * g["median"], 1), fmt(g["share_above"])] for n, g in (("accepted", b["accepted"]), ("rejected", b["rejected"]))]
    table_k = MC.md_table(["status", "n", "mean dlogX x100", "median x100", "share > 0"], rows_k)
    g = b["gate"]
    gate_lines = (f"Gate accept iff dQ >= 0 and dlogX <= 0: {g['n_pass']} of {b['n']} proposals pass (min dlogX over all proposals = {fmt(100 * b['dlogx_min'], 1)} x100). "
                  f"It keeps {g['accepted_kept']} of {g['accepted_total']} accepted edges and would accept {g['rejected_taken']} of {g['rejected_total']} rejected ones; "
                  f"mean dlogX of the kept set = {fmt(None if g['mean_dlogX_kept'] is None else 100 * g['mean_dlogX_kept'], 1)} x100. "
                  f"{b['n_dq_ok']} proposals have dQ >= 0. Mann-Whitney p (accepted vs rejected dlogX) = {fmt(b['mannwhitney_p'], 3)}.\n\n"
                  + MC.md_table(["optimizer", "child", "accepted", "dlogX x100", "dQ"], [[NAMES[p["opt"]] + " " + p["bench"], p["child"], p["accepted"], fmt(100 * p["dlogX"], 1), fmt(p["dQ"], 1)] for p in g["passing"]]))
    rows_c = [[NAMES[o], v["n"], v["n_removing"], v["n_shortening"], v["added"], v["removed"]] for o, v in c["by_opt"].items()]
    rows_c.append(["all", c["n"], c["n_removing"], c["n_shortening"], c["added"], c["removed"]])
    table_c = MC.md_table(["optimizer", "edges", "remove text", "shorten skill", "chars added", "chars removed"], rows_c)
    c_lines = (f"Mean dlogX x100: edges that remove text ({c['n_removing']}) = {fmt(None if c['mean_dlogX_removing'] is None else 100 * c['mean_dlogX_removing'], 1)}; "
               f"edges that shorten the skill ({c['n_shortening']}) = {fmt(None if c['mean_dlogX_shortening'] is None else 100 * c['mean_dlogX_shortening'], 1)}; "
               f"pure additions ({c['n_additions']}) = {fmt(100 * c['mean_dlogX_additions'], 1)}.")
    table = ("### A. what predicts dlogX\n\n" + table_a + "\n\n### B. proposals and the cost-aware gate\n\n" + table_b + "\n\n" + table_k + "\n\n" + gate_lines
             + "\n\n### C. operators: does the editor ever remove text?\n\n" + table_c + "\n\n" + c_lines)
    what = ("Empirical support for three design principles. P2: skill length is a poor price of a candidate's token tax; turns and actions explain it. P4: nearly all logged proposals add tokens, so a success-only gate "
            "has nothing cost-free to choose from, and a cost-aware gate applied to the same proposals has almost nothing to keep. P3: the editors almost only add text. A counterfactual on logged proposals, not a new run.")
    runs = "The 59 canonical dev edges of native-20261008-https (edges_table.json, SkillAdam x terminal-bench-2-1 excluded); test level: the nine learned skills of mech_attribution3 / mech_compaction / mech_skilltext."
    agg = ("A: scipy linregress R^2 and spearmanr of dlogX on log(len_c/len_p), dlogR, dlogN over edges; test level: log rho of input tokens (attribution total, Table 1 log rho_trj) on log(words of skill / words of seed) and on log rho_turns "
           "(compaction rho_turns). B: counts over all 59 edges, gate accept iff dQ >= 0 and dlogX <= 0. C: rem_chars > 0, len_c < len_p, sums of add_chars and rem_chars. No intervals; the sample is 59 edges and 9 skills.")
    MC.write_record("designev", what, runs, agg, table, SELF_CMD)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["compute", "draw"])
    step = ap.parse_args().step
    if step == "compute":
        compute()
        return
    data = json.loads((MC.CACHE / "designev.json").read_text())
    draw(data)
    record(data)


if __name__ == "__main__":
    main()
