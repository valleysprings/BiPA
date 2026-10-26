"""Skill loading: a skill loaded on demand reads as many extra input tokens as one always resident, and without an instruction
to load it, its description decides whether the agent loads it at all (record mech_trigger, figure mech_trigger).

Runs: DeepSeek-V4.1-Flash on its updated release (served from 17 October), tau3-bench and AutoBench test, one run per cell,
with OpenCode's own skill tool (the SkillZip runtime). Delivery (aux_runs delivery): the seed and SkillOpt's tau3 skill,
each always resident (the full text in the first user message, as in every other run) and on demand (name and description
resident, the body loaded through the skill tool, with an activation instruction after the task); the on-demand seed is the
seed evaluation of a SkillZip run. Trigger (aux_runs trigger): on demand without the activation instruction, for the seed,
SkillOpt's and GEPA's skill of the benchmark and GEPA's skill of the other benchmark. A run's evaluation is identified by the
skill it delivered (its native SKILL.md), and its retries fill the slots the base evaluation left without a verified
outcome. Episodes count what they recorded (common.luna_row). Compare within these runs only: the main table ran on the
earlier release.

rho: the geometric mean over the tasks both cells have with input tokens of the per-task ratio, here of one run each. The
reference of a trigger cell is the seed cell of the same benchmark. A load is the agent's first call of the skill tool for
SKILL.md; its turn is the request index from 0, the median the upper middle value for an even count.

The runs were made with two scripts now in the git history (they call the model and need the SkillZip runtime, which is not
part of this repository): aux.contamination_and_loading.run_native BUNDLE OUT CACHE_ROOT evaluated a bundle through the
native route with the SkillZip activation instruction (always resident is harness_bench's default guidance route), and
run_trigger BUNDLE BENCH OUT [SLOTS] used the same route with an empty guidance file and the delivery gate lifted, so that an
episode that never loads the skill keeps its verdict and records whether and when it loaded SKILL.md
(skill_delivery.triggered, first_trigger_request)."""
import math

import numpy as np

from src.analysis import settings as S
from src.analysis.common import luna_row
from src.utils import fig
from src.utils.logger import md_table, record

TAU, AUTOB = "tau3-bench", "automationbench"
ROUTES = ("always resident", "on demand")


def delivered(run):
    """[(skill, route, {task: episode})] per test evaluation of a run: the skill is "seed", (method, source benchmark) of a
    DeepSeek skill, or None for another text; the route how it reached the agent."""
    skills = {S.SEED_TEXT: "seed"} | {S.skill(S.DEEPSEEK, b, m): (m, b) for _, b, m in S.learned([S.DEEPSEEK])}
    out = []
    for folder in sorted((run / "results/evaluations/test").iterdir()):
        eps = S.episodes({0: sorted(folder.glob("**/evaluation.json"), key=lambda f: len(f.parts))})[0]
        text = (run / "results/skills/deliveries/evaluations/test" / folder.name / "native/SKILL.md").read_text()
        modes = {e["skill_delivery"]["mode"] for e in eps.values()}
        out.append((skills.get(text), ROUTES[modes != {"guidance"}], eps))
    return out


def ratio(a, b):
    """(rho, tasks): geometric mean of input-token ratios over the tasks both have with input tokens."""
    x = {t: luna_row(e)["X"] for t, e in a.items()}
    y = {t: luna_row(e)["X"] for t, e in b.items()}
    common = [t for t in x if x[t] and y.get(t)]
    return math.exp(np.mean([math.log(x[t] / y[t]) for t in common])), len(common)


def success(eps):
    return 100 * np.mean([e["success"] for e in eps.values() if S.known(e)])


def compute():
    cells = {(skill if skill == "seed" else skill[0], route): eps for run in S.extra("delivery", S.DEEPSEEK, TAU)
             for skill, route, eps in delivered(run) if skill}
    pairs = [(("skillopt", r), ("seed", r)) for r in ROUTES] + [((s, ROUTES[1]), (s, ROUTES[0])) for s in ("seed", "skillopt")]
    delivery = [(a, b, *ratio(cells[a], cells[b]), success(cells[a]), success(cells[b])) for a, b in pairs]
    trigger = []
    for bench in (TAU, AUTOB):
        runs = [(run, *delivered(run)[0]) for run in S.extra("trigger", S.DEEPSEEK, bench)]
        seed = next(eps for _, skill, _, eps in runs if skill == "seed")
        for run, skill, _, eps in runs:
            firsts = sorted(e["skill_delivery"]["first_trigger_request"] for e in eps.values()
                            if e["skill_delivery"].get("triggered"))
            k = len(firsts)
            trigger.append(dict(bench=bench, run=run, skill=skill, loaded=k, n=len(eps), rho=ratio(eps, seed)[0], Q=success(eps),
                                median=firsts[k // 2] if k else None,
                                quartiles=(firsts[k // 4], firsts[3 * k // 4]) if k else None))
    return delivery, trigger


def name(c):
    """The tick label of a trigger cell: seed, the optimizer, and the source of a skill from the other benchmark."""
    if c["skill"] == "seed":
        return "seed"
    method, source = c["skill"]
    return S.ARM_NAME[method] + ("" if source == c["bench"] else f"\nof {fig.BENCH_TICK[source]}")


def off(c):
    return c["skill"] != "seed" and c["skill"][1] != c["bench"]


def draw(delivery, trigger):
    """(a) SkillOpt's rho over the seed by route, and the on-demand overhead; (b) share of episodes that load the skill, with
    loaded/total and the median first-load turn, and rho against the seed cell on top."""
    plt = fig.pyplot(7, tick=6.5)
    figure, (axa, axb) = plt.subplots(1, 2, figsize=(fig.TEXT_W, 2.0), gridspec_kw=dict(width_ratios=[0.8, 1.6], wspace=0.17))
    figure.subplots_adjust(left=0.065, right=0.995, bottom=0.3, top=0.86)
    xs = (0, 1.15, 2.65, 3.8)
    colors = (fig.ARM["skillopt"], fig.ARM["skillopt"], fig.ARM["native-seed"], fig.ARM["skillopt"])
    for i, (x, (_, _, rho, *_), color) in enumerate(zip(xs, delivery, colors)):  # the on-demand overhead bars drawn lighter
        axa.bar(x, rho - 1, bottom=1, width=0.66, color=color, alpha=0.45 if i >= 2 else 1, edgecolor="white", linewidth=0.4)
        axa.text(x, rho + 0.006, f"{rho:.3f}", ha="center", va="bottom", fontsize=5.8)
    axa.axhline(1, color=fig.DARK, lw=0.6)
    axa.set_xticks(xs)
    axa.set_xticklabels(("always\nresident", "on\ndemand", "seed", "SkillOpt"))
    for x0, x1, text in ((xs[0], xs[1], "SkillOpt / seed"), (xs[2], xs[3], "on demand /\nalways resident")):
        axa.plot([x0 - 0.4, x1 + 0.4], [-0.25, -0.25], transform=axa.get_xaxis_transform(), color=fig.MID, lw=0.5, clip_on=False)
        axa.text((x0 + x1) / 2, -0.28, text, transform=axa.get_xaxis_transform(), ha="center", va="top", fontsize=6.2)
    axa.set_ylim(0.98, 1.29)
    axa.set_yticks([1.0, 1.1, 1.2])
    axa.set_xlim(-0.55, 4.35)
    axa.set_ylabel(r"input-token ratio $\rho$", labelpad=1)
    axa.set_title(f"(a) delivery route, {fig.BENCH_TICK[TAU]}", loc="left", pad=13)
    split = sum(c["bench"] == TAU for c in trigger)
    xs = [i + (0.6 if i >= split else 0) for i in range(len(trigger))]
    for x, c in zip(xs, trigger):
        color = fig.ARM["native-seed" if c["skill"] == "seed" else c["skill"][0]]
        share = 100 * c["loaded"] / c["n"]
        axb.bar(x, share, width=0.62, color="white" if off(c) else color, edgecolor=color, linewidth=0.9 if off(c) else 0.4)
        top = f"turn {c['median']}" if c["median"] is not None else "never"
        axb.text(x, share + 2, f"{c['loaded']}/{c['n']}\n{top}", ha="center", va="bottom", fontsize=5.6, linespacing=1.05)
        axb.text(x, 1.05, f"{c['rho']:.2f}", transform=axb.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.8)
    axb.text(-0.62, 1.05, r"$\rho$ vs seed", transform=axb.get_xaxis_transform(), ha="right", va="bottom", fontsize=5.8,
             color=fig.DARK)
    axb.set_xticks(xs)
    axb.set_xticklabels([name(c) for c in trigger])
    for lo, hi, bench in ((0, split - 1, TAU), (split, len(xs) - 1, AUTOB)):
        axis = axb.get_xaxis_transform()
        axb.plot([xs[lo] - 0.35, xs[hi] + 0.35], [-0.25, -0.25], transform=axis, color=fig.MID, lw=0.5, clip_on=False)
        axb.text((xs[lo] + xs[hi]) / 2, -0.28, f"{fig.BENCH_TICK[bench]} ({trigger[lo]['n']} tasks)", transform=axis, ha="center",
                 va="top", fontsize=6.2)
    axb.axvline((xs[split - 1] + xs[split]) / 2, color=fig.LIGHT, lw=0.6, ls=":")
    axb.set_xlim(-0.5, xs[-1] + 0.5)
    axb.set_ylim(0, 100)
    axb.set_ylabel("episodes loading (%)", labelpad=1)
    axb.set_title("(b) on demand without an activation instruction", loc="left", pad=13)
    for ax in (axa, axb):
        fig.despine(ax)
        ax.tick_params(length=2, pad=1)
    fig.save_figure(figure, "trigger")


def main():
    delivery, trigger = compute()
    draw(delivery, trigger)
    label = {"skillopt": "SkillOpt", "seed": "seed"}
    rows_a = [[f"SkillOpt / seed, {a[1]}" if a[0] != b[0] else f"{label[a[0]]}: {a[1]} / {b[1]}", f"{rho:.3f}",
               f"{qa:.1f} vs {qb:.1f}", n] for a, b, rho, n, qa, qb in delivery]
    rows_b = [[S.BENCH_SHORT[c["bench"]], c["run"].name + (" (off-distribution)" if off(c) else ""),
               f"{c['loaded']}/{c['n']} = {100 * c['loaded'] / c['n']:.1f}%", "-" if c["median"] is None else c["median"],
               "-" if c["quartiles"] is None else "{}-{}".format(*c["quartiles"]), f"{c['rho']:.3f}", f"{c['Q']:.1f}"]
              for c in trigger]
    numbers = "\n".join([
        f"Delivery on {fig.BENCH_LABEL[TAU]}: input-token ratio and test success of the numerator and denominator cells.", "",
        md_table(["bar", "rho (input tokens)", "success Q, numerator vs denominator cell (not drawn)", "tasks"], rows_a), "",
        "On demand without an activation instruction.", "",
        md_table(["bench", "cell", "episodes loading the skill (bar)", "median first-load turn (label)",
                  "quartiles of first-load turn (not drawn)", "rho vs the seed cell (top row)", "Q (not drawn)"], rows_b)])
    record(__spec__.name, "trigger", numbers, S.runs_text([], S.extra("delivery", S.DEEPSEEK, TAU) + [c["run"] for c in trigger]))


if __name__ == "__main__":
    main()
