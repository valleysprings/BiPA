"""Shared style of the gains and trigger scripts: colors, the cells and arms of the gains record, and the matplotlib settings."""
from src import layout

HERE = layout.OUT / "transfer_and_trajectories"
OC = {"GEPA": "#009E73", "SkillOpt": "#0072B2", "SkillAdam": "#D55E00"}
NOSKILL, SEED, LEARNED = "#BBBBBB", "#555555", "#882255"
CELLS = [("deepseek-flash|tau3-bench", "DeepSeek\n" r"$\tau^3$"), ("deepseek-flash|automationbench", "DeepSeek\nAutoB"),
         ("gpt-6-luna|tau3-bench", "Luna\n" r"$\tau^3$"), ("gpt-6-luna|automationbench", "Luna\nAutoB")]
ARMS = [("native-noseed", "No skill", NOSKILL), ("native-seed", "Seed", SEED),
        ("gepa", "GEPA", OC["GEPA"]), ("skillopt", "SkillOpt", OC["SkillOpt"]), ("skilladam", "SkillAdam", OC["SkillAdam"])]


def pyplot():
    """matplotlib.pyplot with the Agg backend and the shared font settings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    return plt


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=6.5, length=2, pad=1)
