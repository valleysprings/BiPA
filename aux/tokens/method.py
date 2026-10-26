"""The method figure (Figure 2): the TikZ source method.tex, built with pdflatex into fig/mech/mech_method.pdf. Its
blocks, edge widths and bar values are illustrative; no recorded run is read."""
import shutil, subprocess, tempfile
from pathlib import Path

from src.utils.fig import FIG

SOURCE = Path(__file__).with_name("method.tex")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-jobname=mech_method", f"-output-directory={tmp}", str(SOURCE)],
                       check=True, stdout=subprocess.DEVNULL)
        shutil.copy(Path(tmp) / "mech_method.pdf", FIG / "mech_method.pdf")


if __name__ == "__main__":
    main()
