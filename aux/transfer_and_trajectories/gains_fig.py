"""The gains figure (mech_gains2) and its numbers, written to the record mech_gains.md.

(a) tasks solved in at least one of three runs by one side only, seed over no skill and learned skills over
    seed (summed over GEPA, SkillOpt, SkillAdam), per (model, benchmark) cell and pooled; exact two-sided sign
    test on the pooled counts.   Source: pass3_ci.json (4 cells) + gains_tb2.json (DeepSeek TB2).
(b) share of tasks whose three runs agree more (diverge less) than under the reference arm, among tasks whose
    success went up / stayed the same / went down; stratified Mantel-Haenszel odds ratio (up vs same, strata
    = the reference arm's successes out of three).
    Source: explore_strat.json per_cell_pairs (4 cells) + gains_tb2.json pairs (DeepSeek TB2).
Luna TB2 is omitted (its three optimizers returned the seed text, so learned-over-seed would compare the seed with itself).
Record mech_gains.md: sections 1-6 from write_record(); section 7 (five-cell addendum: seed vs learned odds-ratio test, 2->3
shares, per-optimizer discordance) from addendum().
Inputs: pass3_ci.json, gains_tb2.json (aux.headline_numbers.pooled), explore_strat.json (this subset).
Run:  python -m aux.transfer_and_trajectories.gains_fig
"""
import functools
import json
import math

from aux.transfer_and_trajectories import plot_style as R
from aux.transfer_and_trajectories import explore_strat as ES  # mh(): Mantel-Haenszel OR with the Robins-Breslow-Greenland interval
from aux.headline_numbers import dynamics as D
from src.utils.dataloader import CACHE
from src.utils.fig import save_figure
from src.utils.logger import paperout as P

HERE = R.HERE

LEARNED_KEYS = ("gepa_vs_seed", "skillopt_vs_seed", "skilladam_vs_seed")
TB2 = "deepseek-flash|terminal-bench-2-1"
CELLS = [(TB2, "DeepSeek TB2")] + [(k, lab.replace("\n", " ")) for k, lab in R.CELLS]
TICKS = ["TB2", r"$\tau^3$", "AutoB", r"$\tau^3$", "AutoB", "total"]
FS_NUM, FS_TICK, FS_NOTE, FS_AX, FS_TITLE = 6.8, 7.2, 6.8, 7.5, 7.8
WIDTH_IN = 6.3  # printed at \textwidth (455.24 pt); every text at least 6.5 pt there


def sign_p(a, b):
    """Exact two-sided binomial (sign) test of a vs b discordant tasks, p = 1/2."""
    n, k = a + b, min(a, b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@functools.cache
def tb2():
    """gains_tb2.json plus the agreement pairs of each TB2 contrast (explore_strat's task rows on the same runs)."""
    t = json.loads((CACHE / "gains_tb2.json").read_text())
    arms = ("native-noseed", "native-seed", "gepa", "skillopt", "skilladam")
    for model, used in (("deepseek-flash", arms), ("gpt-6-luna", arms[:2])):
        rows = {arm: ES.task_rows(D.load_cell(model, arm, "terminal-bench-2-1")) for arm in used}
        keys = [("native-seed", "native-noseed", "seed_vs_noskill")] + [(a, "native-seed", f"{a}_vs_seed") for a in used[2:]]
        t[model]["pairs"] = json.loads(json.dumps({key: ES.pairs(rows[own], rows[ref]) for own, ref, key in keys}))
    return t


def coverage():
    cells = json.loads((CACHE / "pass3_ci.json").read_text())["cells"]
    cells[TB2] = tb2()["deepseek-flash"]
    rows = []
    for key, label in CELLS:
        c = cells[key]["contrasts"]
        s = (c["seed_vs_noskill"]["only_own"], c["seed_vs_noskill"]["only_ref"])
        l = (sum(c[k]["only_own"] for k in LEARNED_KEYS), sum(c[k]["only_ref"] for k in LEARNED_KEYS))
        rows.append((label, s, l))
    ps = tuple(sum(r[1][i] for r in rows) for i in range(2))
    pl = tuple(sum(r[2][i] for r in rows) for i in range(2))
    rows.append(("pooled", ps, pl))
    return rows, sign_p(*ps), sign_p(*pl)


def by_start_mh(pairs):
    """explore_strat.analyse's by_start MH: success up vs same, strata = the reference arm's successes out of 3."""
    strata = []
    for lvl in range(4):
        g = [c for kr, ko, c in pairs if ko > kr and kr == lvl]
        u = [c for kr, ko, c in pairs if ko == kr and kr == lvl]
        strata.append((sum(g), len(g) - sum(g), sum(u), len(u) - sum(u)))
    return ES.mh(strata)


def counts(pairs):
    out = {}
    for g, keep in (("gain", lambda kr, ko: ko > kr), ("tie", lambda kr, ko: ko == kr), ("loss", lambda kr, ko: ko < kr)):
        sel = [c for kr, ko, c in pairs if keep(kr, ko)]
        out[g] = (sum(sel), len(sel), 100 * sum(sel) / len(sel) if sel else float("nan"))
    return out


def agreement():
    """Pooled (k_ref, k_own, agrees more) pairs. Without TB2 they reproduce explore_strat.json exactly."""
    strat = json.loads((HERE / "explore_strat.json").read_text())
    cells = dict(strat["per_cell_pairs"])
    cells[TB2] = {"seed": tb2()["deepseek-flash"]["pairs"]["seed_vs_noskill"],
                  "learned": [p for k in LEARNED_KEYS for p in tb2()["deepseek-flash"]["pairs"][k]]}
    out = {}
    for name, skey in (("seed", "seed_over_noskill"), ("learned", "learned_over_seed")):
        four = [tuple(p) for key, c in cells.items() if key != TB2 for p in c[name]]
        five = four + [tuple(p) for p in cells[TB2][name]]
        mh4 = by_start_mh(four)
        assert abs(mh4["OR"] - strat[skey]["by_start"]["mh"]["OR"]) < 1e-9, (name, mh4)  # same pairs as explore_strat.json
        out[name] = counts(five)
        out[name]["mh"] = by_start_mh(five)
        out[name]["no_tb2"] = counts(four)
        out[name]["no_tb2"]["mh"] = mh4
        out[name]["tb2"] = counts([tuple(p) for p in cells[TB2][name]])
    return out


def draw(cov, ps, st):
    """(a) tasks solved in some run by one side only (up: the newer arm, down: the reference), the last group summed over
    the five cells; (b) share of tasks whose runs agree more, by how success moved, with the MH odds ratio."""
    plt = R.pyplot()
    figure, (axa, axb) = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.35), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.2))
    figure.subplots_adjust(left=0.06, right=0.995, bottom=0.255, top=0.905)
    xs = [0, 1, 2, 3, 4, 5.5]
    last, w, gap_last = len(xs) - 1, 0.38, 0.42  # wider gap in the total group so the p label clears the next bar
    for i, (x, (label, s, l)) in enumerate(zip(xs, cov)):
        gap = gap_last if i == last else 0.04
        for j, ((own, ref), color) in enumerate(((s, R.SEED), (l, R.LEARNED))):
            xx = x + (j - 0.5) * (w + gap)
            axa.bar(xx, own, width=w, color=color, edgecolor="white", linewidth=0.4)
            axa.bar(xx, -ref, width=w, color=color, alpha=0.38, edgecolor="white", linewidth=0.4)
            axa.text(xx, own + 1.5, f"{own}", ha="center", va="bottom", fontsize=FS_NUM, color="#222222")
            axa.text(xx, -ref - 1.5, f"{ref}", ha="center", va="top", fontsize=FS_NUM, color="#222222")
    xx = xs[last] - 0.5 * (w + gap_last)  # only the seed-over-no-skill p is printed
    axa.text(xx, cov[last][1][0] + 11, f"$p$={ps:.3f}" if ps < 0.01 else f"$p$={ps:.2f}", ha="center", va="bottom", fontsize=FS_NUM,
             color="#222222")
    axa.axhline(0, color="#333333", lw=0.6)
    axa.axvline((xs[4] + xs[5]) / 2, color="#BBBBBB", lw=0.6, ls=":")
    axa.set_xticks(xs)
    axa.set_xticklabels(TICKS, fontsize=FS_TICK)
    axa.tick_params(axis="x", length=0, pad=2)
    for x0, x1, group in ((xs[0], xs[2], "DeepSeek"), (xs[3], xs[4], "Luna"), (xs[5], xs[5], "all five pairs")):
        if x1 > x0:
            axa.plot([x0 - 0.35, x1 + 0.35], [-0.13, -0.13], transform=axa.get_xaxis_transform(), color="#999999", lw=0.5, clip_on=False)
        axa.text((x0 + x1) / 2, -0.16, group, transform=axa.get_xaxis_transform(), ha="center", va="top", fontsize=FS_TICK)
    axa.set_ylim(-88, 82)
    axa.set_yticks([-60, -30, 0, 30, 60])
    axa.set_yticklabels(["60", "30", "0", "30", "60"])
    axa.set_ylabel("tasks", fontsize=FS_AX, labelpad=1)
    axa.text(-0.58, 81, "solved only by the seed / learned skill", ha="left", va="top", fontsize=FS_NOTE, color="#444444")
    axa.text(-0.58, -87, "solved only by no skill / the seed", ha="left", va="bottom", fontsize=FS_NOTE, color="#444444")
    axa.set_xlim(-0.62, 6.2)
    axa.set_title("(a) tasks solved in some run by one side only", fontsize=FS_TITLE, loc="left")
    R.style(axa)
    axa.tick_params(axis="both", labelsize=FS_TICK)
    groups = [("gain", "success\nup"), ("tie", "success\nsame"), ("loss", "success\ndown")]
    arms = (("seed", R.SEED), ("learned", R.LEARNED))
    for j, (name, color) in enumerate(arms):
        for g, (k, _) in enumerate(groups):
            conv, n, v = st[name][k]
            x = g + (j - 0.5) * 0.36
            axb.bar(x, v, width=0.32, color=color, edgecolor="white", linewidth=0.4)
            axb.text(x, 3, f"{conv}/{n}", ha="center", va="bottom", fontsize=FS_NUM, color="white", rotation=90)
    axb.axhline(50, color="#333333", lw=0.6, ls=":")
    for i, (name, color) in enumerate(arms):
        y = 87 - 9 * i
        axb.plot(1.0, y, marker="s", ms=3.6, color=color, lw=0, clip_on=False)
        axb.text(1.1, y, f"{st[name]['mh']['OR']:.2f}", ha="left", va="center", fontsize=FS_NOTE, color="#222222")
    axb.text(0.9, 97, "odds ratio, up vs same", ha="left", va="center", fontsize=FS_NOTE, color="#444444")
    axb.set_xticks(range(3))
    axb.set_xticklabels([g[1] for g in groups], fontsize=FS_TICK)
    axb.tick_params(axis="x", length=0, pad=2)
    axb.set_xlim(-0.5, 2.5)
    axb.set_ylim(0, 100)
    axb.set_ylabel("share of tasks (%)", fontsize=FS_AX, labelpad=1)
    axb.set_title("(b) tasks whose episodes agree more", fontsize=FS_TITLE, loc="left")
    R.style(axb)
    axb.tick_params(axis="both", labelsize=FS_TICK)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=R.SEED, label="seed over no skill"),
               plt.Rectangle((0, 0), 1, 1, facecolor=R.LEARNED, label="learned skills over seed")]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, -0.02), ncol=2, frameon=False, fontsize=FS_TICK,
                  handlelength=0.9, handletextpad=0.3, columnspacing=1.2)
    figure.canvas.draw()
    scale = 455.24 / (WIDTH_IN * 72)
    small = [(t.get_text(), t.get_fontsize()) for t in figure.findobj(plt.Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() * scale < 6.5]
    assert not small, small
    save_figure(figure, "gains2")


def raw_paths():
    out = set()
    for bench in ("tau3-bench", "automationbench", "terminal-bench-2-1"):
        for arm in R.ARMS:
            a = arm[0]
            out.update(str(p) for p in D.deepseek_reps(a, bench).values())
            if bench == "terminal-bench-2-1" and a not in ("native-noseed", "native-seed"):
                continue  # Luna TB2: only the seed-over-no-skill counts in the notes use these runs
            run = D.LUNA / a / bench / D.LUNA_TAG[a]
            out.add(str(run))
            reps = D.LUNA / "test-reps" / f"{bench}__{a}__{D.LUNA_TAG[a]}"
            if reps.is_dir():
                out.add(str(reps))
    return sorted(out)


def fmt_or(mh):
    return f"{mh['OR']:.2f} [{mh['ci'][0]:.2f}, {mh['ci'][1]:.2f}]"


def addendum(st):
    """The 'Five-cell addendum' section of mech_gains.md: numbers over the five cells."""
    e = json.loads((HERE / "explore_strat.json").read_text())
    t = tb2()["deepseek-flash"]
    lp, sp = [], []
    for v in e["per_cell_pairs"].values():
        lp += [tuple(p) for p in v["learned"]]
        sp += [tuple(p) for p in v["seed"]]
    lp += [tuple(p) for k in LEARNED_KEYS for p in t["pairs"][k]]
    sp += [tuple(p) for p in t["pairs"]["seed_vs_noskill"]]
    rl, rs = ES.analyse(lp, "learned"), ES.analyse(sp, "seed")
    ml, ms = rl["by_start"]["mh"], rs["by_start"]["mh"]
    for m, name in ((ml, "learned"), (ms, "seed")):  # same pairs as panel (b)
        assert abs(m["OR"] - st[name]["mh"]["OR"]) < 1e-9, (name, m, st[name]["mh"])
    se = lambda m: (math.log(m["ci"][1]) - math.log(m["ci"][0])) / (2 * 1.96)
    z = (math.log(ml["OR"]) - math.log(ms["OR"])) / math.sqrt(se(ml) ** 2 + se(ms) ** 2)
    p = math.erfc(abs(z) / math.sqrt(2))
    tl, ts = rl["transitions"]["2->3"], rs["transitions"]["2->3"]
    four = json.loads((CACHE / "pass3_ci.json").read_text())["cells"]
    names = (("gepa_vs_seed", "GEPA"), ("skillopt_vs_seed", "SkillOpt"), ("skilladam_vs_seed", "SkillAdam"))
    d4 = {k: tuple(sum(c["contrasts"][k][f] for c in four.values()) for f in ("only_own", "only_ref")) for k, _ in names}
    dt = {k: (t["contrasts"][k]["only_own"], t["contrasts"][k]["only_ref"]) for k, _ in names}
    d5 = {k: (d4[k][0] + dt[k][0], d4[k][1] + dt[k][1]) for k, _ in names}
    disc = lambda d: ", ".join(f"{lab} {d[k][0]} / {d[k][1]}" for k, lab in names)
    sh = lambda x: f"{x['conv']}/{x['n']} = {x['share']:.1f}%"
    ci = lambda m: f"{m['OR']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"
    L = ["Numbers over the five cells of mech_gains (DeepSeek TB2, tau3, AutoBench; Luna tau3, AutoBench). "
         "Computed by gains_fig.addendum() from explore_strat.json, gains_tb2.json and pass3_ci.json.", "",
         f"- Seed-over-no-skill vs learned-over-seed odds ratios (MH, up vs same, strata = reference successes out of 3): learned {ci(ml)}, seed {ci(ms)}; "
         f"ratio {ml['OR'] / ms['OR']:.2f}, z = {z:.2f} on the log scale (SE from the RBG intervals), two-sided p = {p:.3f}.",
         f"- Tasks going from 2 to 3 successes whose runs agree more: learned {sh(tl)}, seed {sh(ts)} (paper: {tl['share']:.0f} and {ts['share']:.0f}%).",
         f"- Per-optimizer pass@3 discordance (gained / lost against the seed): {disc(d5)}; four cells (pass3_ci.json): {disc(d4)}; "
         f"DeepSeek TB2 (gains_tb2.json): {disc(dt)}."]
    return "\n".join(L)


def write_record(cov, ps, pl, st, cov4):
    t = tb2()
    luna = t["gpt-6-luna"]
    lp = luna["pairs"]["seed_vs_noskill"]
    lc = counts([tuple(x) for x in lp])
    P.SPECS["mech_gains"] = dict(
        pub="mech",
        shows="what the gains are: (a) tasks solved in some run by one side only, (b) share of tasks whose runs agree more than under the reference, among tasks whose success went up / stayed the same / went down, with the stratified Mantel-Haenszel odds ratio",
        files="fig/mech/mech_gains2.pdf, fig/mech/mech_gains2.svg; the attractors-only view is fig/mech/mech_explore.pdf",
        notes=["The only interval drawn is the MH odds ratio's, on which the claim 'learned-skill gains agree more than unchanged tasks' rests (it excludes 1).",
               "TB2, DeepSeek: all five arms have 3 runs with 18/18 verified outcomes (gains_tb2.py), so pass@3 discordance in (a) is computed exactly as on tau3 / AutoBench. In (b) the agreement measure needs 3 runs with at least one request on both sides; that holds on 17 (no skill, GEPA) or 18 (seed, SkillOpt, SkillAdam) of the 18 tasks, so the definition applies unchanged and TB2 is included. It adds few tasks whose success moves: 10 of the 18 tasks have the same successes out of 3 under all five arms (7 solved in every run, 3 in none).",
               "TB2, Luna: omitted from both panels. Its GEPA, SkillOpt and SkillAdam runs returned the seed text (dynamics.luna_seed_skill), so learned over seed would compare the seed with itself; showing only the seed contrast would leave the cell half empty. For the record, Luna TB2 seed over no skill (17 verified tasks): (a) %d / %d, (b) up %d/%d, same %d/%d, down %d/%d." % (
                   luna["contrasts"]["seed_vs_noskill"]["only_own"], luna["contrasts"]["seed_vs_noskill"]["only_ref"],
                   lc["gain"][0], lc["gain"][1], lc["tie"][0], lc["tie"][1], lc["loss"][0], lc["loss"][1]),
               "Learned-over-seed counts are summed over GEPA, SkillOpt and SkillAdam (three contrasts per cell); seed-over-no-skill is one contrast per cell.",
               "Runs and task sets: Table 1's avg@3 test runs via aux.headline_numbers.dynamics.load_cell; (a) tasks with a verified outcome in all 3 runs on both sides (pooled.pass3, pooled.tb2); (b) tasks with 3 verified runs with R > 0 on both sides (explore_strat.py, pooled.tb2)."])
    L = ["Panel (a). Tasks solved in at least one of three runs by the newer arm only / by the reference only (pass@3 discordant tasks).", "",
         P.mdtable(["cell", "seed over no skill: newer only / reference only", "learned over seed (3 optimizers summed): newer only / reference only"],
                   [[lab.replace(r"$\tau^3$", "tau3"), f"{s[0]} / {s[1]}", f"{l[0]} / {l[1]}"] for lab, s, l in cov]), "",
         f"Pooled exact two-sided sign test (5 cells, as drawn): seed over no skill {cov[-1][1][0]} vs {cov[-1][1][1]}, p = {ps:.4f}; "
         f"learned over seed {cov[-1][2][0]} vs {cov[-1][2][1]}, p = {pl:.3f}.",
         f"Without DeepSeek TB2 (4 cells): seed {cov4[0][0]} vs {cov4[0][1]}, p = {sign_p(*cov4[0]):.4f}; learned {cov4[1][0]} vs {cov4[1][1]}, p = {sign_p(*cov4[1]):.3f}.", "",
         "Panel (b). A task's runs agree more = its three runs diverge less (1 - mean pairwise edit similarity of executed tool-name sequences) than under the reference arm; success up / same / down = successes out of 3 higher / equal / lower than under the reference; pooled over the cells and the three optimizers.", ""]
    rows = []
    for name, lab in (("seed", "seed over no skill"), ("learned", "learned skills over seed")):
        for part, tag in ((st[name], "5 cells (drawn)"), (st[name]["tb2"], "DeepSeek TB2 alone"), (st[name]["no_tb2"], "4 cells (no DeepSeek TB2)")):
            r = [lab, tag]
            for g in ("gain", "tie", "loss"):
                conv, n, v = part[g]
                r.append(f"{conv}/{n} = {v:.1f}%" if n else "0/0")
            r.append(fmt_or(part["mh"]) if "mh" in part and part["mh"] else "-")
            rows.append(r)
    L += [P.mdtable(["contrast", "cells", "success up", "same", "down", "MH OR, up vs same (strata: reference successes out of 3; RBG 95% CI)"], rows), ""]
    L += ["Attractors-only view (mech_explore.pdf): numbers in mech_explore.md section A."]
    P.record("mech_gains", "figure", result="\n".join(L),
             commands=["python -m aux.headline_numbers.pooled gains",
                       "python -m aux.transfer_and_trajectories.explore_strat",
                       "python -m aux.transfer_and_trajectories.gains_fig"],
             raw=raw_paths(),
             source=["aux/transfer_and_trajectories/" + f for f in ("gains_fig.py", "explore_strat.py", "plot_style.py")] + ["aux/headline_numbers/pooled.py"],
             sections=[("Five-cell addendum", addendum(st))])


def main():
    cov, ps, pl = coverage()
    four = cov[1:-1]  # the four cells without DeepSeek TB2
    cov4 = [tuple(sum(r[1 + j][i] for r in four) for i in range(2)) for j in range(2)]
    st = agreement()
    draw(cov, ps, st)
    write_record(cov, ps, pl, st, cov4)
    for lab, s, l in cov:
        print(f"{lab:16s} seed {s[0]:3d}/{s[1]:3d}   learned {l[0]:3d}/{l[1]:3d}")
    print(f"sign test p: seed {ps:.4f}, learned {pl:.3f}; without TB2 {cov4} p {sign_p(*cov4[0]):.4f} {sign_p(*cov4[1]):.3f}")
    for name in ("seed", "learned"):
        print(name, {g: tuple(round(x, 1) for x in st[name][g]) for g in ("gain", "tie", "loss")}, fmt_or(st[name]["mh"]),
              "| no TB2", fmt_or(st[name]["no_tb2"]["mh"]), "| TB2", {g: st[name]["tb2"][g][:2] for g in ("gain", "tie", "loss")})


if __name__ == "__main__":
    main()
