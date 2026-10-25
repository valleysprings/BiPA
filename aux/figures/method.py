"""The method figure: builds the TikZ source method.tex with pdflatex into fig/mech/mech_method.pdf."""
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.utils.fig import FIG

SOURCE = Path(__file__).with_name("method.tex")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-jobname=mech_method", f"-output-directory={tmp}",
                        str(SOURCE)], check=True, stdout=subprocess.DEVNULL)
        shutil.copy(Path(tmp) / "mech_method.pdf", FIG / "mech_method.pdf")
    print(FIG / "mech_method.pdf")


if __name__ == "__main__":
    main()
