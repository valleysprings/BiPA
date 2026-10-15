"""BACH lineage bipartite graphs for the six `ours` cells (execution/<model>/ours/<bench>/<tag>, tags in bach-harness/config/ours_runs.json).
Stage 1 (slim venv python): bipartite_lineage.py compute -> attribution/<model>/ours/<bench>/bipartite/{components-edits,edits-blocks}.json
Stage 2 (system python3, matplotlib): bipartite_lineage.py plot -> the same names with .pdf and .png; then writes the README section of each cell.

components-edits: edit nodes (every candidate the loop screened: id, move, round, decision) x component groups; edge weight = the edit's rise
  Delta_g(e) = zo-result nodes[].edit.delta summed over the fine components of a group (they add up to the edit's change of mean log workload, lmdi.edge).
edits-blocks: edit nodes x blocks of the returned skill (and of the best over-budget archive point); an edge links a block to the edit that introduced
  or last rewrote it (prices.provenance on the node chain); p_j = theta * tok_j + sum_g abar_jg (pooled attributed costs, z = 0, theta from the seed's
  dev episodes of the same run), q_j = pooled success credit shrunk to 0 (ledger.Book.q); seed blocks are one node."""
import glob, json, math, re, sys, textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ATTR = ROOT / "assets/artifacts/attribution"
GROUPS = ["Reads & retrieval", "Bash & other tools", "History", "Skill text", "Fixed context"]
READ = ("tool:read.", "tool:KB_search", "tool:lookup", "tool:discover", "tool:grep", "tool:glob", "tool:bash.read")
MOVE = {"c": "search", "a": "rewrite", "b": "recombine", "d": "prune", "d_slack": "slack prune"}
DECISION = {"accept": "accepted", "reject_dominated": "dominated", "reject_screen": "failed screen"}
MARK = "<!-- bipartite-lineage -->"


def group(g):
    if g.startswith(READ): return GROUPS[0]
    if g.startswith("tool:"): return GROUPS[1]
    return {"history": GROUPS[2], "skill": GROUPS[3]}.get(g, GROUPS[4])


def cells():
    ours = json.loads((ROOT / "bach-harness/config/ours_runs.json").read_text())
    return [(m, b, ours[m][b]) for m in ("deepseek-flash", "gpt-6-luna") for b in ("tau3-bench", "automationbench", "terminal-bench-2-1")]


def outdir(m, b): return ATTR / m / "ours" / b / "bipartite"


def compute():
    sys.path[:0] = [str(ROOT / "bach-harness/src")]
    from bach import decompose as D, workload as M
    for m, b, tag in cells():
        run = current_run("zo", m, b)  # tag = ours_runs.json
        d = json.loads((run / "zo-result.json").read_text())
        nodes = {n["id"]: n for n in d["nodes"]}
        kappa = d["config"]["kappa"]
        dec = {r["child"]: r["decision"] for r in d["rounds"] if r.get("child")}
        chain = lambda i: ([] if nodes[i]["parent"] is None else chain(nodes[i]["parent"])) + [i]
        ret = d["returned"]
        pool = [i for i in dict.fromkeys(d["archive"] + d["front"]) if i != ret and nodes[i]["sel"] and nodes[i]["sel"]["L"] > kappa]
        over = max(pool, key=lambda i: (nodes[i]["sel"]["Q"], nodes[i]["sel"]["S"], -nodes[i]["sel"]["L"])) if pool else None
        on = {"returned": chain(ret), "over_budget": chain(over) if over else []}
        # components x edits
        edits = []
        for n in d["nodes"]:
            if n["edit"] is None: continue
            e = n["edit"]; gd = {g: 0.0 for g in GROUPS}
            for g, v in e["delta"].items(): gd[group(g)] += v
            sc = n["screen"]
            edits.append({"id": n["id"], "parent": n["parent"], "kind": n["kind"], "move": MOVE[n["kind"]], "round": n["round"], "decision": dec[n["id"]], "screen_pass": sc["pass"],
                          "screen_n": sc["n"], "dQ": e["dQ"], "dS": e["dS"], "total": sum(e["delta"].values()), "groups": gd, "fine": e["delta"], "se_fine": e["se"],
                          "lineage_of": [k for k, c in on.items() if n["id"] in c], "sel": {k: n["sel"][k] for k in ("Q", "S", "L", "n")} if n["sel"] else None})
        out = outdir(m, b); out.mkdir(parents=True, exist_ok=True)
        (out / "components-edits.json").write_text(json.dumps({"model": m, "bench": b, "run": str(run.relative_to(ROOT)), "groups": GROUPS, "returned": ret, "over_budget": over, "edits": edits}, indent=1))
        # edits x blocks
        sd = nodes["n0"]
        seed_eps = M.load(max(glob.glob(f"{run}/results/evaluations/dev/{sd['version']}*/evaluation.json")))
        th, th_se = D.theta([e for e in seed_eps if e.success is not None])
        T = d["blocks"]

        def rec(k):
            r = T.get(k)
            if r is None: return None
            q = D.pool(r["s"])
            a = {g: D.pool(v)[0] for g, v in r["a"].items() if g != "skill"}
            m_, v_, _ = q
            return {"tok": r["tok"], "q": r["q"], "sd": r["sd"], "behavior": sum(a.values()), "text_price": th * r["tok"], "p": th * r["tok"] + sum(a.values()),
                    "by_group": {gg: sum(v for g, v in a.items() if group(g) == gg) for gg in GROUPS}, "edits": r["edit"], "repl": r["repl"]}

        shown = {}  # key -> block node
        for tagname, nid in (("returned", ret), ("over_budget", over)):
            if nid is None: continue
            ch = chain(nid); texts = [nodes[i]["text"] for i in ch]
            for bl, k in zip(D.blocks(texts[-1]), D.provenance(texts)):
                key = D.key(bl["text"])
                node = shown.setdefault(key, {"key": key, "text": bl["text"], "tok": bl["tokens"], "in": [], "edit": ch[k] if k else "seed"})
                node["in"].append(tagname)
        for key, node in shown.items(): node["table"] = rec(key)
        fallback = not any(n["edit"] != "seed" for n in shown.values())
        if fallback:  # no non-seed block in the returned / over-budget skill: show the blocks the screened edits produced
            for key, r in T.items(): shown.setdefault(key, {"key": key, "text": r["text"], "tok": r["tok"], "in": [], "edit": r["edit"][-1], "table": rec(key)})
        blocks = list(shown.values())
        eids = [e["id"] for e in edits if e["id"] in {x["edit"] for x in blocks} or any(e["id"] in c for c in on.values())]
        (out / "edits-blocks.json").write_text(json.dumps({
            "model": m, "bench": b, "run": str(run.relative_to(ROOT)), "returned": ret, "over_budget": over, "kappa": kappa, "theta": th, "theta_se": th_se, "fallback_all_table_blocks": fallback,
            "returned_sel": nodes[ret]["sel"] and {k: nodes[ret]["sel"][k] for k in ("Q", "S", "L")}, "over_budget_sel": over and {k: nodes[over]["sel"][k] for k in ("Q", "S", "L")},
            "edit_nodes": [{"id": "seed", "kind": "seed"}] + [{"id": i, "kind": nodes[i]["kind"], "round": nodes[i]["round"], "decision": dec[i], "lineage_of": [k for k, c in on.items() if i in c]} for i in eids],
            "blocks": blocks, "seed_blocks": {"n": sum(1 for x in blocks if x["edit"] == "seed"), "tokens": sum(x["tok"] for x in blocks if x["edit"] == "seed")},
            "table_blocks_total": len(T)}, indent=1))
        print(m, b, tag, "returned", ret, "over", over, "edits", len(edits), "blocks shown", len(blocks), "fallback", fallback, flush=True)


# ------------------------------------------------------------------ plotting
BLACK, OPT, GEPA, ADAM, BACH = "black", "#E69F00", "#56B4E9", "#009E73", "#D55E00"
GCOL = dict(zip(GROUPS, ["#0072B2", "#CC79A7", "#F0E442", "#FFFFFF", "#BBBBBB"]))
POS, NEG = "#C0392B", "#2A9D6F"
ML = {"deepseek-flash": "Flash", "gpt-6-luna": "Luna"}
BL = {"tau3-bench": "tau3-bench", "automationbench": "AutomationBench", "terminal-bench-2-1": "Terminal-Bench 2.1"}


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.5, "pdf.fonttype": 42, "font.family": "serif"})
    return plt


def edit_box(ax, x, y, w, h, e, fs=5.5):
    from matplotlib.patches import FancyBboxPatch
    d = e.get("decision")
    face = {"accept": "#F6C9A8", "reject_dominated": "#DDDDDD", "reject_screen": "#FFFFFF"}.get(d, "#FFFFFF")
    ls, lw, ec = "-", 0.5, "#555555"
    lin = e.get("lineage_of") or []
    if "returned" in lin: lw, ec = 1.6, BLACK
    elif "over_budget" in lin: lw, ec, ls = 1.4, GEPA, "--"
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.015", fc=face, ec=ec, lw=lw, ls=ls, zorder=3))
    return d


def plot_components(plt, m, b, gmax):
    d = json.loads((outdir(m, b) / "components-edits.json").read_text())
    E = d["edits"]; n = len(E)
    H = max(2.3, 0.50 * n + 0.75)
    fig, ax = plt.subplots(figsize=(3.4, H)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    top, bot = 0.88, 0.08
    ey = [top - (top - bot) * i / max(n - 1, 1) for i in range(n)] if n > 1 else [0.5]
    cy = [top - (top - bot) * i / 4 for i in range(5)]
    xe, xc, we, wc = 0.17, 0.80, 0.30, 0.30
    hh = min(0.075, 0.42 / max(n, 1))
    for e, y in zip(E, ey):
        for gi, g in enumerate(GROUPS):
            v = e["groups"][g]
            ax.plot([xe + we / 2, xc - wc / 2], [y, cy[gi]], color=POS if v > 0 else NEG, ls="-" if v > 0 else (0, (3, 1)), lw=max(abs(v) / gmax * 5.0, 0.12), alpha=0.8, solid_capstyle="butt", zorder=1)
            if abs(v) >= 0.03:
                t = 0.78
                ax.text(xe + we / 2 + t * (xc - wc / 2 - xe - we / 2), y + t * (cy[gi] - y), f"{v:+.2f}", fontsize=4.8, ha="center", va="center", zorder=4,
                        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9))
        edit_box(ax, xe, y, we, hh * 2, e)
        ax.text(xe, y + hh * 0.45, f"{e['id']} {e['move']} (r{e['round']})", ha="center", va="center", fontsize=5.4, zorder=5)
        ax.text(xe, y - hh * 0.5, f"{DECISION[e['decision']]}", ha="center", va="center", fontsize=4.8, zorder=5, color="#333333")
        ax.text(xe - we / 2 - 0.012, y, f"$\\Sigma\\Delta$ {e['total']:+.2f}\ndQ {e['dQ']:+.2f}", ha="right", va="center", fontsize=4.8, color="#333333")
    from matplotlib.patches import Rectangle
    for g, y in zip(GROUPS, cy):
        ax.add_patch(Rectangle((xc - wc / 2, y - 0.04), wc, 0.08, fc=GCOL[g], ec="black", lw=0.4, hatch="////" if g == GROUPS[3] else None, zorder=3))
        ax.text(xc + wc / 2 + 0.01, y, g, ha="left", va="center", fontsize=5.4)
    ax.text(0, 1.0, f"{ML[m]}, {BL[b]}: edits $\\times$ components", ha="left", va="bottom", fontsize=6.5, fontweight="bold", transform=ax.transAxes)
    ax.text(0, 0.0, f"edge = rise $\\Delta_g(e)$ of mean log workload (solid red up, dashed green down); width $\\propto|\\Delta|$ (5 pt = {gmax:.2f}).\n"
            "box: orange accepted, grey dominated, white failed screen;\nblack border = returned lineage, blue dashed = over-budget archive point.", ha="left", va="top", fontsize=4.6, transform=ax.transAxes)
    out = outdir(m, b) / "components-edits"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight"); plt.close(fig)


def snippet(t, n=34):
    t = re.sub(r"\s+", " ", re.sub(r"[#*`|]", "", t)).strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def plot_blocks(plt, m, b):
    d = json.loads((outdir(m, b) / "edits-blocks.json").read_text())
    B, E = d["blocks"], d["edit_nodes"]
    rank = {e["id"]: i for i, e in enumerate(E)}
    B = sorted(B, key=lambda x: (rank[x["edit"]] if x["edit"] in rank else 99, 0))
    nb = max(len(B), 1)
    # seed group is one row on the block side
    seed = [x for x in B if x["edit"] == "seed"]; rest = [x for x in B if x["edit"] != "seed"]
    rows = ([{"seed": seed}] if seed else []) + [{"b": x} for x in rest]
    nr = len(rows)
    H = max(2.3, 0.33 * nr + 0.9, 0.34 * len(E) + 0.9)
    fig, ax = plt.subplots(figsize=(3.4, H)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    top, bot = 0.90, 0.07
    ry = [top - (top - bot) * i / max(nr - 1, 1) for i in range(nr)] if nr > 1 else [0.5]
    ey = {}
    for i, e in enumerate(E): ey[e["id"]] = top - (top - bot) * i / max(len(E) - 1, 1) if len(E) > 1 else 0.5
    # a block row's edit node is placed at the mean y of its blocks, so edges stay short; edits without surviving blocks keep an even slot
    xe, xb, we = 0.10, 0.45, 0.17
    hh = min(0.05, 0.40 / max(len(E), 1))
    own = {}
    for r, y in zip(rows, ry):
        eid = "seed" if "seed" in r else r["b"]["edit"]
        own.setdefault(eid, []).append(y)
    placed = {}
    for e in E:
        placed[e["id"]] = sum(own[e["id"]]) / len(own[e["id"]]) if e["id"] in own else None
    # pack edits top to bottom in lineage order with a minimum separation; an edit without a surviving block takes the next free slot
    order = [e["id"] for e in E]
    ys = dict(placed)
    prev = None
    for eid in order:
        if ys[eid] is None: ys[eid] = (prev - 2 * hh - 0.01) if prev is not None else top
        if prev is not None and ys[eid] > prev - 2 * hh - 0.01: ys[eid] = prev - 2 * hh - 0.01
        prev = ys[eid]
    lo = min(ys.values())
    ymin = min(lo - hh, bot - 0.05)
    ax.set_ylim(min(ymin, 0) - 0.04, 1)
    from matplotlib.patches import Rectangle
    for e in E:
        if e["id"] == "seed":
            from matplotlib.patches import FancyBboxPatch
            ax.add_patch(FancyBboxPatch((xe - we / 2, ys["seed"] - hh), we, 2 * hh, boxstyle="round,pad=0,rounding_size=0.015", fc="white", ec=BLACK, lw=1.0, zorder=3))
            ax.text(xe, ys["seed"], "seed\nskill", ha="center", va="center", fontsize=5.4, zorder=5)
        else:
            edit_box(ax, xe, ys[e["id"]], we, 2 * hh, e)
            ax.text(xe, ys[e["id"]] + hh * 0.35, f"{e['id']} {MOVE[e['kind']]}", ha="center", va="center", fontsize=5.2, zorder=5)
            ax.text(xe, ys[e["id"]] - hh * 0.45, f"r{e['round']} {DECISION[e['decision']]}", ha="center", va="center", fontsize=4.4, zorder=5, color="#333333")
    for r, y in zip(rows, ry):
        if "seed" in r:
            s = r["seed"]; tk = sum(x["tok"] for x in s)
            ax.plot([xe + we / 2, xb - 0.015], [ys["seed"], y], color="#777777", lw=0.8, zorder=1)
            ax.add_patch(Rectangle((xb - 0.015, y - 0.022), 0.03, 0.044, fc="#BBBBBB", ec="black", lw=0.4, zorder=3))
            ax.text(xb + 0.03, y, f"{len(s)} seed blocks ({tk:.0f} tok), kept; no price or credit", ha="left", va="center", fontsize=5.0)
            continue
        x = r["b"]; t = x["table"]
        mem = x["in"]
        ec = BLACK if "returned" in mem else GEPA if "over_budget" in mem else "#999999"
        ax.plot([xe + we / 2, xb - 0.015], [ys[x["edit"]], y], color=POS if t and t["p"] > 0 else "#777777", lw=0.5 + 2.0 * (x["tok"] / max(1, max(z["tok"] for z in rest))), alpha=0.55, zorder=1, solid_capstyle="butt")
        ax.add_patch(Rectangle((xb - 0.015, y - 0.022), 0.03, 0.044, fc=BACH if "returned" in mem else ("#FFFFFF" if not mem else "#9ED2F2"), ec=ec, lw=1.0 if mem else 0.4, zorder=3))
        lab = textwrap.shorten(re.sub(r"\s+", " ", re.sub(r"[#*`|]|[^\x00-\u206f]", "", x["text"])).strip(), 40, placeholder="…")
        nums = f"{x['tok']:.0f} tok" + (f"  p {t['p']:.3f}  q {t['q']:+.3f}" if t else "  (heading, no record)")
        ax.text(xb + 0.03, y + 0.011, lab, ha="left", va="bottom", fontsize=5.0)
        ax.text(xb + 0.03, y - 0.004, nums, ha="left", va="top", fontsize=4.6, color="#333333")
    ax.text(0, 1.0, f"{ML[m]}, {BL[b]}: edits $\\times$ blocks", ha="left", va="bottom", fontsize=6.5, fontweight="bold", transform=ax.transAxes)
    note = ("blocks of the returned skill (orange; none beyond the seed's if the seed was returned) and of the best over-budget archive point (blue); "
            "grey = block born by a screened edit but in neither skill" if d["fallback_all_table_blocks"] or any(not x["in"] for x in B) else
            "blocks of the returned skill (orange) and of the best over-budget archive point (blue)")
    ax.text(0, ax.get_ylim()[0], f"edge: edit that introduced / last rewrote the block; width $\\propto$ tokens.\np = price (work units), q = success credit.\n{textwrap.fill(note, 78)}\n"
            f"box: orange accepted, grey dominated, white failed screen;\nblack = returned lineage, blue dashed = over-budget point.", ha="left", va="top", fontsize=4.4)
    out = outdir(m, b) / "edits-blocks"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight"); plt.close(fig)


def readme(m, b):
    p = ATTR / m / "ours" / b / "README.md"
    ce = json.loads((outdir(m, b) / "components-edits.json").read_text()); eb = json.loads((outdir(m, b) / "edits-blocks.json").read_text())
    req = sorted(x.name for x in outdir(m, b).glob("episode-*") if x.suffix == ".png")
    sect = f"""{MARK}
## Bipartite graphs ({ML[m]}, {BL[b]}; run `{ce['run']}`)

The BACH run is read through three bipartite structures: requests x components (what one episode spends), components x edits (which edit raised which part of the workload), and edits x blocks (which edit wrote which text block).

- Requests x components: the per-episode graph (model requests against harness components; invocation edges = tool calls, exposure edges = content re-sent to later requests; c = calls + input tokens/1000 split exactly over components){" is in `bipartite/" + req[0] + "` (seed and BACH skill side by side; `components.png` pools the change by group)." if req else " is drawn by the separate request-graph step into `bipartite/requests-components*` (not part of this file set)."} It shows where one run's workload lands, for the seed and for a final skill.
- `bipartite/components-edits.{{pdf,png,json}}`: components x edits. Every candidate the loop screened ({len(ce['edits'])} edits, returned `{ce['returned']}`{', over-budget archive point `' + ce['over_budget'] + '`' if ce['over_budget'] else ''}) is a node (id, move, round, screen decision) linked to the five component groups by its rise Delta_g(e) of mean log workload (red solid = up, green dashed = down, width proportional to |Delta|); the weights come from `zo-result.json` `nodes[].edit.delta`.
- `bipartite/edits-blocks.{{pdf,png,json}}`: edits x blocks. Each block of the returned skill (and of the best over-budget archive point) is linked to the edit that introduced or last rewrote it (provenance), annotated with its price p_j = theta*tok + attributed behaviour cost and its success credit q_j from the blocks table; seed blocks are one node{'. The returned skill is the seed, so the graph shows the blocks of the screened edits instead' if eb['fallback_all_table_blocks'] else ''}.
{MARK}
"""
    old = p.read_text() if p.exists() else f"# ours, {ML[m]}, {BL[b]}\n\n"
    if MARK in old: old = re.sub(rf"{MARK}.*?{MARK}\n?", "", old, flags=re.S)
    p.write_text(old.rstrip("\n") + "\n\n" + sect)


def plot():
    plt = style()
    gmax = max(abs(v) for m, b, _ in cells() for e in json.loads((outdir(m, b) / "components-edits.json").read_text())["edits"] for v in e["groups"].values())
    for m, b, _ in cells():
        plot_components(plt, m, b, gmax); plot_blocks(plt, m, b); readme(m, b)
        print("plotted", m, b, flush=True)


if __name__ == "__main__":
    {"compute": compute, "plot": plot}[sys.argv[1]]()
