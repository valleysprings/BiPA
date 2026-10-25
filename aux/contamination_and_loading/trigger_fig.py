"""The trigger figure (mech_trigger).

(a) Delivery 2x2 on tau3 test (DeepSeek-V4.1-Flash 1017 version, one run per cell): input-token ratio rho of SkillOpt over the
    seed when the skill is always resident (full text in the first user message) and when it is loaded on demand (OpenCode
    native --skill with an activation directive), and the on-demand overhead (on demand over always resident) per skill.
    Same cells, episodes and ratio as mech_delivery.md.
(b) Trigger cells (same model version, one run per cell, on demand without an activation instruction): share of episodes that
    load SKILL.md through the skill tool (bars), median first-load turn (label above the bar; turn index from 0 = the first
    request), and rho of the cell's input tokens over the same benchmark's seed cell (row at the top).
    Same cells and statistics as mech_trigger.md.
main() asserts that the values match those records.
Style and colors are those of aux.transfer_and_trajectories.plot_style; files go through src.utils.fig.save_figure (pdf + editable svg).
Run:  python -m aux.contamination_and_loading.trigger_fig
"""
import json
import math
import statistics
from pathlib import Path

from aux.transfer_and_trajectories import plot_style as R
from aux.headline_numbers.dynamics import known
from src.utils import fig
from src.utils.dataloader import DS as EXD
from src.utils.logger import paperout as P

DC = EXD / "delivery-check/tau3-bench"
DELIVERY = {"seed/A": DC / "seed-guidance/results/evaluations/test/4fd5ca441c55-57817cd02585",
            "seed/B": EXD / "skillzip/tau3-bench/native-20261018/results/evaluations/test/89a426c6e82d-57817cd02585",
            "skillopt/A": DC / "skillopt-guidance/results/evaluations/test/a0bdead535e7-57817cd02585",
            "skillopt/B": DC / "skillopt-native/results/evaluations/test/3b2b1fe3d99c-57817cd02585"}
TRIG = EXD / "trigger-check"
TRIG_CELLS = [("tau3-bench", "seed", "seed", R.SEED, False), ("tau3-bench", "skillopt-tau3", "SkillOpt", R.OC["SkillOpt"], False),
              ("tau3-bench", "gepa-tau3", "GEPA", R.OC["GEPA"], False), ("tau3-bench", "gepa-autob", "GEPA\nof AutoB", R.OC["GEPA"], True),
              ("automationbench", "seed", "seed", R.SEED, False), ("automationbench", "skillopt-autob", "SkillOpt", R.OC["SkillOpt"], False),
              ("automationbench", "gepa-autob", "GEPA", R.OC["GEPA"], False), ("automationbench", "gepa-tau3", "GEPA\n" r"of $\tau^3$", R.OC["GEPA"], True)]
# values printed in mech_delivery.md / mech_trigger.md (checked, not drawn from)
EXPECT_DELIVERY = {("skillopt/A", "seed/A"): 1.226, ("skillopt/B", "seed/B"): 1.230, ("seed/B", "seed/A"): 1.063, ("skillopt/B", "skillopt/A"): 1.066}
EXPECT_TRIG = {("tau3-bench", "seed"): (0, 75, None, 1.000), ("tau3-bench", "skillopt-tau3"): (0, 75, None, 1.076),
               ("tau3-bench", "gepa-tau3"): (60, 75, 1, 1.190), ("tau3-bench", "gepa-autob"): (2, 75, 34, 0.940),
               ("automationbench", "seed"): (21, 120, 33, 1.000), ("automationbench", "skillopt-autob"): (37, 120, 42, 1.217),
               ("automationbench", "gepa-autob"): (67, 120, 27, 1.203), ("automationbench", "gepa-tau3"): (79, 120, 1, 1.145)}


# ---------- (a) delivery ----------

def delivery_rows(d):
    eps = {}
    for p in sorted(d.glob("**/evaluation.json"), key=lambda p: len(p.parts)):  # base evaluation, unknown slots replaced by retries
        for ep in json.loads(p.read_text())["episodes"]:
            k = ep["task_id"]
            if k not in eps or (not known(eps[k]) and known(ep)):
                eps[k] = ep
    out = {}
    for t, ep in eps.items():
        reqs = [r for r in {r["request_id"]: r for r in ep.get("requests", [])}.values() if r["input_tokens"] is not None]
        out[t] = {"X": sum(int(r["input_tokens"]) for r in reqs) if reqs else None,
                  "success": bool(ep["success"]) if known(ep) else None}
    return out


def gm_ratio(a, b):
    common = [t for t in a if t in b and a[t]["X"] and b[t]["X"]]
    return math.exp(sum(math.log(a[t]["X"] / b[t]["X"]) for t in common) / len(common)), len(common)


def delivery():
    missing = [f"{c}: {d}" for c, d in DELIVERY.items() if not d.is_dir()]
    if missing:
        raise SystemExit("delivery cells missing under $BIPA_ASSETS/artifacts:\n  " + "\n  ".join(missing))
    rows = {c: delivery_rows(d) for c, d in DELIVERY.items()}
    out = {}
    for pair, want in EXPECT_DELIVERY.items():
        rho, n = gm_ratio(rows[pair[0]], rows[pair[1]])
        assert round(rho, 3) == want and n == 75, (pair, rho, n)
        out[pair] = (rho, n)
    q = {c: 100 * sum(r["success"] for r in rs.values() if r["success"] is not None) / sum(r["success"] is not None for r in rs.values())
         for c, rs in rows.items()}
    return out, q


# ---------- (b) trigger ----------

def trig_rows(cell):
    recorded = json.loads((cell / "trigger-summary.json").read_text())["evaluation_path"]
    p = cell / "results/evaluations/test" / Path(recorded).parent.name / "evaluation.json"
    assert p.is_file(), (cell, recorded)
    out = {}
    for ep in json.loads(p.read_text())["episodes"]:
        reqs = [r for r in {r["request_id"]: r for r in ep["requests"]}.values() if r["input_tokens"] is not None]
        sd = ep["skill_delivery"]
        out[ep["task_id"]] = {"S": ep["success"], "trig": sd.get("triggered"), "first": sd.get("first_trigger_request"),
                              "X": sum(int(r["input_tokens"]) for r in reqs)}
    return out, p


def trigger():
    out, paths = {}, []
    for bench in ("tau3-bench", "automationbench"):
        seed, _ = trig_rows(TRIG / bench / "seed")
        for b, cell, *_ in TRIG_CELLS:
            if b != bench:
                continue
            r, p = trig_rows(TRIG / bench / cell)
            paths.append(p)
            firsts = sorted(v["first"] for v in r.values() if v["trig"])
            common = [t for t in r if t in seed and r[t]["X"] and seed[t]["X"]]
            rho = math.exp(sum(math.log(r[t]["X"] / seed[t]["X"]) for t in common) / len(common))
            med = firsts[len(firsts) // 2] if firsts else None  # upper middle for an even count
            got = (len(firsts), len(r), med, round(rho, 3))
            assert got == EXPECT_TRIG[(bench, cell)], (bench, cell, got)
            q = 100 * sum(bool(v["S"]) for v in r.values()) / len(r)
            out[(bench, cell)] = {"loaded": len(firsts), "n": len(r), "median_first": med, "rho": rho, "Q": q,
                                  "first_q": (firsts[len(firsts) // 4], firsts[(3 * len(firsts)) // 4]) if firsts else None,
                                  "mean_first": statistics.mean(firsts) if firsts else None}
    return out, paths


# ---------- figure ----------

def draw(dl, tr):
    plt = R.pyplot()
    figure, (axa, axb) = plt.subplots(1, 2, figsize=(6.3, 2.0), gridspec_kw=dict(width_ratios=[0.8, 1.6], wspace=0.17))
    figure.subplots_adjust(left=0.065, right=0.995, bottom=0.3, top=0.86)
    # (a) bars from rho = 1
    bars = [(0, dl[("skillopt/A", "seed/A")][0], "always\nresident", R.OC["SkillOpt"]),
            (1.15, dl[("skillopt/B", "seed/B")][0], "on\ndemand", R.OC["SkillOpt"]),
            (2.65, dl[("seed/B", "seed/A")][0], "seed", R.SEED),
            (3.8, dl[("skillopt/B", "skillopt/A")][0], "SkillOpt", R.OC["SkillOpt"])]
    for x, v, _, color in bars:
        light = x > 2  # the on-demand overhead bars, drawn lighter
        axa.bar(x, v - 1, bottom=1, width=0.66, color=color, alpha=0.45 if light else 1, edgecolor="white", linewidth=0.4)
        axa.text(x, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=5.8, color="#222222")
    axa.axhline(1, color="#333333", lw=0.6)
    axa.set_xticks([b[0] for b in bars])
    axa.set_xticklabels([b[2] for b in bars], fontsize=6.2)
    axa.tick_params(axis="x", length=0, pad=2)
    for x0, x1, lab in ((0, 1.15, "SkillOpt / seed"), (2.65, 3.8, "on demand /\nalways resident")):
        axa.plot([x0 - 0.4, x1 + 0.4], [-0.25, -0.25], transform=axa.get_xaxis_transform(), color="#999999", lw=0.5, clip_on=False)
        axa.text((x0 + x1) / 2, -0.28, lab, transform=axa.get_xaxis_transform(), ha="center", va="top", fontsize=6.2)
    axa.set_ylim(0.98, 1.29)
    axa.set_yticks([1.0, 1.1, 1.2])
    axa.set_xlim(-0.55, 4.35)
    axa.set_ylabel(r"input-token ratio $\rho$", fontsize=7, labelpad=1)
    axa.set_title(r"(a) delivery route, $\tau^3$", fontsize=7, loc="left", pad=13)
    R.style(axa)
    # (b) share of episodes that load the skill
    xs = [0, 1, 2, 3, 4.6, 5.6, 6.6, 7.6]
    for x, (bench, cell, label, color, ood) in zip(xs, TRIG_CELLS):
        c = tr[(bench, cell)]
        v = 100 * c["loaded"] / c["n"]
        axb.bar(x, v, width=0.62, color="white" if ood else color, edgecolor=color, linewidth=0.9 if ood else 0.4)
        top = f"turn {c['median_first']}" if c["median_first"] is not None else "never"
        axb.text(x, v + 2, f"{c['loaded']}/{c['n']}\n{top}", ha="center", va="bottom", fontsize=5.6, color="#222222", linespacing=1.05)
        axb.text(x, 1.05, f"{c['rho']:.2f}", transform=axb.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.8, color="#222222")
    axb.text(-0.62, 1.05, r"$\rho$ vs seed", transform=axb.get_xaxis_transform(), ha="right", va="bottom", fontsize=5.8, color="#444444")
    axb.set_xticks(xs)
    axb.set_xticklabels([c[2] for c in TRIG_CELLS], fontsize=6.2)
    axb.tick_params(axis="x", length=0, pad=2)
    for x0, x1, lab in ((xs[0], xs[3], r"$\tau^3$ (75 tasks)"), (xs[4], xs[7], "AutoB (120 tasks)")):
        axb.plot([x0 - 0.35, x1 + 0.35], [-0.25, -0.25], transform=axb.get_xaxis_transform(), color="#999999", lw=0.5, clip_on=False)
        axb.text((x0 + x1) / 2, -0.28, lab, transform=axb.get_xaxis_transform(), ha="center", va="top", fontsize=6.2)
    axb.axvline((xs[3] + xs[4]) / 2, color="#BBBBBB", lw=0.6, ls=":")
    axb.set_xlim(-0.5, 8.1)
    axb.set_ylim(0, 100)
    axb.set_ylabel("episodes loading (%)", fontsize=7, labelpad=1)
    axb.set_title("(b) on demand without an activation instruction", fontsize=7, loc="left", pad=13)
    R.style(axb)
    fig.save_figure(figure, "trigger")


def write_record(dl, q, tr, paths):
    P.SPECS["mech_trigger_fig"] = dict(
        pub="mech",
        shows="(a) SkillOpt's input-token ratio over the seed under always-resident and on-demand delivery, and the on-demand overhead; (b) per trigger cell the share of episodes that load the skill, the median first-load turn and the input-token ratio over the seed cell",
        files="fig/mech/mech_trigger.pdf, fig/mech/mech_trigger.svg",
        notes=["All cells: DeepSeek-V4.1-Flash 1017 version (fingerprint e72d3c7f), one test run per cell; compare only within this figure (Table 1 runs are the earlier snapshot). Full per-cell tables: mech_delivery.md (panel a) and mech_trigger.md (panel b).",
               "rho = per-task geometric mean of input-token ratios on tasks with X > 0 in both cells (n = 75 for every pair in (a)). In (b) the reference is the same benchmark's seed cell of the trigger runs, so the two seed bars have rho = 1 by construction.",
               "Turn = request index from 0 (0 = the first request; in (a) the activation directive puts every load at turn 0). Median = the upper middle value for an even count; the quartile column for gepa-autob on tau3 lists its two load turns.",
               "(a) uses an explicit activation directive, so every on-demand episode loads the skill at turn 0 (75/75 in both on-demand cells); (b) has none, so loading is measured.",
               "Hollow bars in (b): off-distribution skills (GEPA's AutoBench skill on tau3, GEPA's tau3 skill on AutoBench). Light bars in (a): the on-demand overhead ratios.",
               "Noise and an artifact: opencode lists the bundle path, so the bundle directory name (seed, skillopt-autob, gepa-tau3, ...) was visible to the agent. Seed and skillopt-autob share name and description; their AutoBench loading (21 vs 37 of 120) bounds noise plus this artifact. On tau3, SkillOpt's skill is never loaded and still reads 1.076 times the seed cell's tokens, although its body never reaches the agent: one-run spread plus whatever the resident name, description and directory name change.",
               "Success is not drawn (one run per cell; see mech_trigger.md for Q, Q loaded / not loaded)."])
    rows_a = [["SkillOpt / seed, always resident", f"{dl[('skillopt/A', 'seed/A')][0]:.3f}", f"{q['skillopt/A']:.1f} vs {q['seed/A']:.1f}"],
              ["SkillOpt / seed, on demand", f"{dl[('skillopt/B', 'seed/B')][0]:.3f}", f"{q['skillopt/B']:.1f} vs {q['seed/B']:.1f}"],
              ["seed: on demand / always resident", f"{dl[('seed/B', 'seed/A')][0]:.3f}", f"{q['seed/B']:.1f} vs {q['seed/A']:.1f}"],
              ["SkillOpt: on demand / always resident", f"{dl[('skillopt/B', 'skillopt/A')][0]:.3f}", f"{q['skillopt/B']:.1f} vs {q['skillopt/A']:.1f}"]]
    rows_b = []
    for bench, cell, label, _, ood in TRIG_CELLS:
        c = tr[(bench, cell)]
        rows_b.append(["tau3" if bench == "tau3-bench" else "AutoB", cell + (" (off-distribution)" if ood else ""),
                       f"{c['loaded']}/{c['n']} = {100 * c['loaded'] / c['n']:.1f}%",
                       "-" if c["median_first"] is None else str(c["median_first"]),
                       "-" if c["first_q"] is None else f"{c['first_q'][0]}-{c['first_q'][1]}",
                       f"{c['rho']:.3f}", f"{c['Q']:.1f}"])
    L = ["Panel (a): delivery 2x2 on tau3 (75 tasks).", "",
         P.mdtable(["bar", "rho (input tokens)", "success Q, numerator vs denominator cell (not drawn)"], rows_a), "",
         "Panel (b): trigger cells, on demand without an activation instruction.", "",
         P.mdtable(["bench", "cell", "episodes loading the skill (bar)", "median first-load turn (label)", "quartiles of first-load turn (not drawn)",
                    "rho vs the seed cell (top row)", "Q (not drawn)"], rows_b)]
    P.record("mech_trigger_fig", "figure", result="\n".join(L),
             commands=["python -m aux.contamination_and_loading.trigger_fig"],
             raw=[str(p) for p in DELIVERY.values()] + [str(TRIG / b / c) for b, c, *_ in TRIG_CELLS] + paths,
             source=["aux/contamination_and_loading/trigger_fig.py", "aux/transfer_and_trajectories/plot_style.py"])


def main():
    dl, q = delivery()
    tr, paths = trigger()
    draw(dl, tr)
    write_record(dl, q, tr, paths)
    for k, v in dl.items():
        print(k, f"{v[0]:.3f} n={v[1]}")
    for k, v in tr.items():
        print(k, v)


if __name__ == "__main__":
    main()
