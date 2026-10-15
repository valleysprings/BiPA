"""SkillOpt as published (microsoft/SkillOpt @79124b37): upstream's ReflACTTrainer, unchanged, on our sample, with
an EnvAdapter over our executor, a loader serving the shared panels and batches, and our LLM as upstream's client."""

import json, random, os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import skillopt
from skillopt.config import flatten_config, load_config
from skillopt.datasets.base import BaseDataLoader, BatchSpec
from skillopt.engine.trainer import ReflACTTrainer
from skillopt.envs.base import EnvAdapter
from skillopt.model import openai_compatible_backend as transport

from src.utils import executor
LIMIT = 80000  # chars of conversation text per trajectory handed to the analyst
DEVIATIONS = [
    "rollouts run on our executor (harness_bench: the cell's model and harness), not upstream's envs/*/rollout.py",
    "sample: train batches = shared B_1..B_T (stratified folds, upstream b=40; 160 train episodes: one pass over 160, TB2 reshuffled passes over 62), not shuffle-and-chunk;"
    " selection split = shared D_sel (30 dev tasks; TB2 all 9); test split = shared test panel",
    "predictions/<id> as officeqa writes them, from the final request: target_{system,user}_prompt.txt, "
    "task_description = first user message minus the skill, conversation.json = later turns + verifier [verification]",
    f"a trajectory's texts are water-filled to {LIMIT} chars in all, longer ones cut head+tail (seed ones reach 1M)",
    "hard = soft = verifier success (no partial credit); unknown outcome = failure; task_type = benchmark name",
    "optimizer LLM = our recorded LLM as upstream's openai_compatible client: temperature 1.0 (all arms), max_tokens "
    "16384 (upstream's request), no reasoning_effort (as that backend); our retries nest inside upstream's",
    "config = upstream default.yaml but out_root, env, skill_init, models, optimizer_backend, upstream b=40 and minibatch_size=8, train_size = Σ|B_t| (one upstream epoch = B_1..B_T; TB2 tasks recur across its 62)",
    "episodes are cached by (skill text, task, split) on every split; upstream caches only selection scores by hash",
]


STRICT_POLICY = {"schema": "skillopt-outcome-policy-v1", "policy": "abort-before-trainer-on-unknown"}

class UnknownOutcome(RuntimeError):
    pass


class NativeIncomplete(BaseException):
    pass


def outcome_policy(root: Path, strict: bool):
    """Strict starts in an empty optimizer directory; resumes only its own policy."""
    marker = root / "outcome_policy.json"
    if marker.exists():
        if not strict or json.loads(marker.read_text()) != STRICT_POLICY:
            raise RuntimeError("SkillOpt outcome policy mismatch; use a separate run directory")
    elif strict:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("strict SkillOpt cannot adopt an existing optimizer directory")
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(STRICT_POLICY, indent=2) + "\n")


def require_known(rows, root: Path, split: str, out_dir, native=False):
    unknown = [{"id": row["id"], "reason": "executor returned no verified outcome"}
               for row in rows if row["success"] is None]
    if unknown:
        record = {"policy": STRICT_POLICY, "split": split, "out_dir": str(out_dir),
                  "requested": len(rows), "unknown": unknown,
                  "semantics": "whole panel aborted; no rows filtered or scored; evaluations may consume episodes"}
        (root / "unknown_outcome.json").write_text(json.dumps(record, indent=2) + "\n")
        error = NativeIncomplete if native else UnknownOutcome
        raise error(f"{split}: {len(unknown)} unknown outcomes; retry complete panel before continuing")


def items(ids: list, split: str) -> list[dict]:  # upstream split -> executor split
    return [{"id": i, "split": {"train": "train", "valid_seen": "dev", "valid_unseen": "test"}[split]} for i in ids]


def clip(turns: list[dict]) -> list[dict]:
    """Water-fill one cap over a trajectory's texts so they total <= LIMIT; a longer text keeps head and tail."""
    sizes = sorted(len(t[k]) for t in turns for k in ("content", "cmd", "obs") if k in t)
    cap = next(((LIMIT - sum(sizes[:i])) // (len(sizes) - i) for i in range(len(sizes))
                if sizes[i] * (len(sizes) - i) > LIMIT - sum(sizes[:i])), LIMIT)
    cut = lambda s: (s if len(s) <= cap else
                     f"{s[:cap // 2]}\n…[{len(s) - cap} chars omitted]…\n{s[len(s) - cap // 2:]}")
    return [{k: cut(v) if k in ("content", "cmd", "obs") else v for k, v in t.items()} for t in turns]


def conversation(row: dict, full=False) -> tuple[str, str, list[dict]]:
    """(system prompt, first user message, the rest as upstream conversation entries, clipped)."""
    messages = row["messages"]
    head = {m["role"]: m["content"] for m in messages[:2]}
    outputs = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
    turns = []
    for m in messages[2:]:
        if m["role"] == "assistant":
            turns += [{"type": "message", "content": m["content"]}] if m["content"] else []
            turns += [{"type": "tool_call", "cmd": f"{c['function']['name']}({c['function']['arguments']})",
                       "obs": outputs.get(c["id"], "")} for c in m.get("tool_calls", [])]
        elif m["role"] != "tool":
            turns.append({"role": m["role"], "content": m["content"]})
    turns += [{"role": "system", "content": row["verifier"]}] if row["verifier"] else []  # upstream: [verification]
    return head.get("system", ""), head.get("user", ""), turns if full else clip(turns)


class Harness(EnvAdapter, BaseDataLoader):  # upstream's env adapter, and its dataloader
    def __init__(self, cell, panels: dict, batches: list, cfg: dict):
        self.cell, self.panels, self.batches, self.episodes = cell, panels, batches, Counter()
        self.strict, self.root = cfg.get("strict_outcomes", False), Path(cfg["out_root"])
        self.full = os.environ.get("HARNESS_FEEDBACK_MODE") == "full"
        vars(self).update({k: cfg[k] for k in ("analyst_workers", "failure_only", "minibatch_size", "edit_budget")})

    def get_dataloader(self):
        return self

    def plan_train_epoch(self, *, epoch, steps_per_epoch, seed, **_):
        planned = self.batches[(epoch - 1) * steps_per_epoch: epoch * steps_per_epoch]
        return [BatchSpec("train", "train", seed + epoch * 1000 + i + 1, len(batch), items(batch, "train"))
                for i, batch in enumerate(planned)]

    def build_train_batch(self, batch_size, seed, **_):  # SplitDataLoader.build_train_batch
        drawn = items(self.panels["train"], "train")
        random.Random(seed).shuffle(drawn)
        return BatchSpec("train", "train", seed, len(drawn[:batch_size]), drawn[:batch_size])

    def build_eval_batch(self, env_num, split, seed, **_):
        drawn = items(self.panels[{"valid_seen": "sel", "valid_unseen": "test"}[split]], split)
        return BatchSpec("eval", split, seed, len(drawn), drawn)

    def build_env_from_batch(self, batch, **_):
        return batch.payload

    def build_train_env(self, batch_size, seed, **_):
        return self.build_train_batch(batch_size, seed).payload

    def build_eval_env(self, env_num, split, seed, **_):
        return self.build_eval_batch(env_num, split, seed).payload

    def get_task_types(self):
        return [self.cell.benchmark]

    def rollout(self, env, skill, out_dir, **_):
        split, results = env[0]["split"], []
        self.episodes[split] += len(env)
        rows = executor.evaluate(self.cell, skill, [item["id"] for item in env], split)
        if self.strict:
            if len(rows) != len(env) or sorted(row["id"] for row in rows) != sorted(item["id"] for item in env):
                raise NativeIncomplete("Executor returned an incomplete/mismatched panel")
            if split != "test":  # a test outcome that stays unknown (e.g. a verifier that always times out) is left out of Q, as in every cell
                require_known(rows, self.root, split, out_dir, native=os.environ.get("HARNESS_PROTOCOL") == "native-v1")
        if os.environ.get("HARNESS_PROTOCOL") == "native-v1":
            import re
            from src.native_trace import record_round_evaluation
            match = re.search(r"step_(\d+)", str(out_dir))
            record_round_evaluation(self.cell.root, arm="skillopt", step=int(match[1]) if match else None,
                                    role=str(Path(out_dir).relative_to(self.root)), skill_text=skill,
                                    evaluation_path=rows[0]["evaluation_path"])
        for row in rows:
            system, user, turns = conversation(row, full=self.full)
            folder = Path(out_dir, "predictions", row["id"])
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "conversation.json").write_text(json.dumps(turns, ensure_ascii=False, indent=1), "utf-8")
            (folder / "target_system_prompt.txt").write_text(system, "utf-8")
            (folder / "target_user_prompt.txt").write_text(user, "utf-8")
            reason = {1: "", 0: "verifier: task failed", None: "unknown: no outcome"}[row["success"]]
            results.append({"id": row["id"], "hard": float(not reason), "soft": float(not reason),
                            "fail_reason": reason, "task_description": executor._SKILL.sub("", user).strip(),
                            "task_type": self.cell.benchmark, "n_turns": len(turns)})
        return results


def client(llm) -> SimpleNamespace:
    def create(*, messages, max_tokens, temperature, **_):  # what upstream's openai_compatible backend calls
        # a failed call raises into upstream (reflect.py drops that analyst minibatch; merges fall back)
        text, _, usage = llm.complete(messages, temperature=temperature, max_tokens=max_tokens)
        message = SimpleNamespace(role="assistant", content=text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(index=0, message=message)], usage=SimpleNamespace(**usage))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def run(cell, panels: dict, batches: list, llm, out: Path) -> dict:
    root = out / "skillopt"
    strict = os.environ.get("SKILLOPT_STRICT_OUTCOMES", "0") == "1"
    outcome_policy(root, strict)
    root.mkdir(parents=True, exist_ok=True)
    (root / "seed.md").write_text(executor.SEED, encoding="utf-8")
    cfg = flatten_config(load_config(str(Path(skillopt.__file__).parents[1] / "configs/_base_/default.yaml")))
    cfg.update(out_root=str(root), strict_outcomes=strict, env="harness", skill_init=str(root / "seed.md"), optimizer_model=llm.model,
               optimizer_backend="openai_compatible", target_model=cell.model, batch_size=len(batches[0]),
               train_size=sum(map(len, batches)), num_epochs=1)  # one upstream epoch = B_1..B_T
    transport.configure_openai_compatible(optimizer_model=llm.model, max_tokens=16384, temperature=1.0)
    transport._build_client = lambda config: client(llm)
    DEVIATIONS[:] = [item.replace("unknown outcome aborts the whole panel before scoring (strict policy)", "unknown outcome = failure") for item in DEVIATIONS]
    if strict:
        DEVIATIONS[:] = [item.replace("unknown outcome = failure", "unknown outcome aborts the whole panel before scoring (strict policy)") for item in DEVIATIONS]
    if os.environ.get("HARNESS_FEEDBACK_MODE") == "full":
        DEVIATIONS[:] = [item for item in DEVIATIONS if not item.startswith("a trajectory") ]
        DEVIATIONS.append("feedback: complete conversation entries in upstream officeqa layout; no adapter clipping")
    harness = Harness(cell, panels, batches, cfg)
    summary = ReflACTTrainer(cfg, harness).train()
    keys = ("baseline_selection_hard", "best_selection_hard", "final_selection_hard", "best_step", "best_origin",
            "total_accepts", "total_rejects", "total_skips", "baseline_test_hard", "test_hard", "final_test_hard")
    return {"final_skill": (root / "best_skill.md").read_text(encoding="utf-8"), "episodes": dict(harness.episodes),
            "llm": {"calls": llm.usage["calls"], "tokens": llm.usage["prompt_tokens"] + llm.usage["completion_tokens"]},
            "extra": {**{key: summary[key] for key in keys}, "strict_outcomes": strict,
                      "outcome_policy": STRICT_POLICY if strict else "legacy_unknown_as_failure"}}
