import re

from src.utils import fig as style


def test_settings_and_save(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(style, "FIG", tmp_path / "fig")
    monkeypatch.setattr(style, "CACHE", tmp_path / "cache")
    plt = style.pyplot(7.0)
    assert plt.rcParams["font.serif"][0] == style.FONT and plt.rcParams["font.size"] == 7.0
    figure, ax = plt.subplots(figsize=(style.TEXT_W, 1.5))
    ax.plot([0, 1], [0, 1], color=style.ARM["skillopt"])
    ax.text(0.5, 0.5, "small", fontsize=5.5)
    style.despine(ax)
    style.save_figure(figure, "test")
    assert {p.name for p in (tmp_path / "fig").iterdir()} == {"mech_test.pdf", "mech_test.svg"}
    assert (tmp_path / "cache/mech_test.png").is_file()
    assert "<text" in (tmp_path / "fig/mech_test.svg").read_text()  # text kept as text
    assert "mech_test: layout: small small 5.5" in capsys.readouterr().out


def test_printed_width_scales_the_size_check():
    plt = style.pyplot()
    figure, ax = plt.subplots(figsize=(7.1, 2))
    ax.set_axis_off()
    ax.text(0.5, 0.5, "label", fontsize=7.5)
    assert not [p for p in style.check(figure) if p[0] == "small"]
    assert [p for p in style.check(figure, width=0.92 * style.TEXT_W) if p[0] == "small"]  # 7.5 pt prints at 6.1 pt


def test_pyplot_sets_every_text_size_and_starts_from_the_defaults():
    plt = style.pyplot(7.5, title=8, tick=7)
    plt.rcParams["axes.linewidth"] = 3.0  # a figure that changed a setting by hand
    plt = style.pyplot(7.5, title=8, tick=7, legend=7.5)
    assert plt.rcParams["axes.linewidth"] == 0.5
    assert (plt.rcParams["axes.titlesize"], plt.rcParams["xtick.labelsize"], plt.rcParams["axes.labelsize"]) == (8, 7, 7.5)
    assert plt.rcParams["legend.fontsize"] == 7.5


def svg_text_sizes(svg):
    """{text: size in pt} of the text elements of a matplotlib svg (svg.fonttype none: 1 px = 1 pt)."""
    found = re.findall(r"<text[^>]*style=\"[^\"]*?([\d.]+)px[^\"]*\"[^>]*>(.*?)</text>", svg.read_text(), re.S)
    return {re.sub(r"<[^>]+>", "", body).strip(): float(size) for size, body in found}


def test_svg_text_sizes_are_read_back(tmp_path, monkeypatch):
    monkeypatch.setattr(style, "FIG", tmp_path / "fig")
    monkeypatch.setattr(style, "CACHE", tmp_path / "cache")
    plt = style.pyplot(7.5, title=8)
    figure, ax = plt.subplots(figsize=(2, 1))
    ax.set_title(style.BENCH_LABEL["tau3-bench"])
    style.save_figure(figure, "sizes")
    sizes = svg_text_sizes(tmp_path / "fig/mech_sizes.svg")
    assert sizes["τ³-bench"] == 8.0


def test_one_record_format(tmp_path, monkeypatch):
    from src.utils import logger
    monkeypatch.setattr(logger, "DATA", tmp_path)
    logger.record("src.utils.fig", "x", "| a | b |\n|---|---|\n| 1 | 2 |", "- run")
    text = (tmp_path / "mech_x.md").read_text()
    assert text.startswith("# mech_x\n\n## What it shows\n\nThe paper's one figure style")
    assert "## Aggregation\n\nEvery figure draws with these." in text and "scripts/main.sh table1 --execute" in text
