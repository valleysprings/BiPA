"""Figure style shared by the dev/test figures."""
FONT = "Liberation Serif"
INK = "#222222"
MUTED = "#5A5A5A"
GRID = "#E2E5EA"
BENCH_NAME = {"tau3-bench": r"$\tau^3$-bench", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2",
              "tau3": r"$\tau^3$-bench", "AutoB": "AutoBench", "TB2": "TB2"}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": FONT, "font.size": 7.5, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "mathtext.fontset": "stix", "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    return plt
