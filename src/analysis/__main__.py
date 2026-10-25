"""Table 1 and the method's analyses in dependency order: python -m src.analysis [--list] [MODULE ...] (default: all)."""
import subprocess, sys

STEPS = [("table1",), ("stats",), ("slopes", "compute"), ("slopes", "draw-input"), ("attribution", "compute"),
         ("attribution", "attr")]

if __name__ == "__main__":
    names = [arg for arg in sys.argv[1:] if arg != "--list"]
    for module, *args in STEPS:
        if not names or module in names:
            print("python -m src.analysis." + " ".join([module, *args]), flush=True)
            if "--list" not in sys.argv:
                subprocess.run([sys.executable, "-m", f"src.analysis.{module}", *args], check=True)
