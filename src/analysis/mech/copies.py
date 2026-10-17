"""mech_nullsim, mech_attribution, mech_price: byte copies of the already baseline-only BACH-paper figures (appB_nullsim, appC_attribution, appC_price; pdf + svg)
under mech_ names, so paper 1 never references the BACH paper's files, plus a short audit record that points at the source record and repeats its result table.
  python3 -m src.analysis.mech.copies      (stdlib only; copies pdf/svg and writes data/mech/mech_<name>.md)"""
import shutil
from src.analysis.mech import common as M

COPIES = {
    "nullsim": ("appB_nullsim", "Zero-effect dev-selection replay: distribution of the dev-set gain when every candidate equals the seed (SkillOpt's strict-improvement rule), per benchmark, under the fitted task-difficulty spread and Beta concentration 4; vertical lines mark the observed SkillOpt and GEPA gains."),
    "attribution": ("appC_attribution", "Per baseline skill (GEPA, SkillOpt, SkillAdam) the change in log input tokens against the seed split into the four terms (more actions, turn-splitting, content per turn, skill text) and the measured total, annotated heatmap grouped by optimizer."),
    "price": ("appC_price", "Per baseline skill the input-token ratio against the cache-priced cost ratio (uncached input + 0.1 x cached input + 4 x output tokens), colored by benchmark."),
}


def result_section(text):
    """The '## 4. Result' section of a paperout record."""
    start = text.index("## 4. Result")
    end = text.index("## 5.", start)
    return text[start + len("## 4. Result"):end].strip()


def main():
    for name, (source, what) in COPIES.items():
        for ext in ("pdf", "svg"):
            shutil.copyfile(M.FIG / f"{source}.{ext}", M.FIG / f"mech_{name}.{ext}")
        record_path = M.find_existing(M.OLD_DATA_DIRS, f"{source}.md")
        table = result_section(record_path.read_text())
        M.write_record(
            name, what,
            f"Same as the source record: DeepSeek-V4.1-Flash in OpenCode; baseline runs only (GEPA, SkillOpt, SkillAdam; native-20261008-https, seed native-seed/<bench>/native-20261003-trace-final); no BACH rows in this figure.",
            f"Byte copy (cp) of fig/mech/{source}.pdf and .svg; nothing recomputed. The full audit record (runs, dates, raw files) is {record_path}.",
            table,
            f"cp paper/assets/fig/mech/{source}.pdf paper/assets/fig/mech/mech_{name}.pdf   (same for .svg)\npython3 -m src.analysis.mech.copies\n(source figure: see {source}.md, section 2)",
            f"Source record: paper/assets/data/mech/{source}.md.")
        print("copied", source, "->", f"mech_{name}")


if __name__ == "__main__":
    main()
