"""SkillOpt (microsoft/SkillOpt): upstream's ReflACTTrainer on our panels, executor and optimizer LLM."""

import json, math, random, re
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
from src.utils.skill_frame import SKILL_FRAME

SPLITS = {"train": "train", "valid_seen": "dev", "valid_unseen": "test"}  # upstream split -> executor split
KEYS = ("baseline_selection_hard", "best_selection_hard", "final_selection_hard", "best_step", "best_origin",
        "total_accepts", "total_rejects", "total_skips", "baseline_test_hard", "test_hard", "final_test_hard")


class Incomplete(BaseException):
    """A missing or unknown train/dev outcome; not an Exception, so upstream's handlers cannot swallow it."""


def items(ids: list, split: str) -> list[dict]:
    return [{"id": i, "split": SPLITS[split]} for i in ids]


def conversation(row: dict) -> tuple[str, str, list[dict]]:
    """(system prompt, first user message, the later turns and the verifier as upstream's officeqa entries)."""
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
    turns += [{"role": "system", "content": row["verifier"]}] if row["verifier"] else []
    return head.get("system", ""), head.get("user", ""), turns


class Harness(EnvAdapter, BaseDataLoader):
    """Upstream's env adapter and dataloader: the shared batches, D_sel and test panel, rolled out on our executor."""

    def __init__(self, cell, panels: dict, batches: list, cfg: dict):
        self.cell, self.panels, self.batches, self.root = cell, panels, batches, Path(cfg["out_root"])
        self.episodes_file = self.root / "arm_episodes.json"  # rollout folder -> [episode key, tasks requested]
        self.rollouts = json.loads(self.episodes_file.read_text()) if self.episodes_file.is_file() else {}
        vars(self).update({k: cfg[k] for k in ("analyst_workers", "failure_only", "minibatch_size", "edit_budget")})

    @property
    def episodes(self) -> Counter:
        """Episodes per split and stage; a rollout repeated on a resume replaces its earlier count."""
        counts = Counter()
        for key, n in self.rollouts.values():
            counts[key] += n
        return counts

    def get_dataloader(self):
        return self

    def plan_train_epoch(self, *, epoch, steps_per_epoch, seed, **_):
        planned = self.batches[(epoch - 1) * steps_per_epoch: epoch * steps_per_epoch]
        return [BatchSpec("train", "train", seed + epoch * 1000 + i + 1, len(batch), items(batch, "train"))
                for i, batch in enumerate(planned)]

    def build_train_batch(self, batch_size, seed, **_):  # the slow update's draw: upstream's seeded shuffle
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
        split, ids, role = env[0]["split"], [item["id"] for item in env], Path(out_dir).relative_to(self.root)
        stage = role.parts[0]  # steps, or slow_update / meta_skill from epoch 2
        self.rollouts[str(role)] = [split if split != "train" or stage == "steps" else f"train_{stage}", len(env)]
        self.episodes_file.write_text(json.dumps(self.rollouts))
        rows = executor.evaluate(self.cell, skill, ids, split)
        unknown = split != "test" and any(row["success"] is None for row in rows)  # a test one scores as a failure
        if unknown or sorted(row["id"] for row in rows) != sorted(ids):
            raise Incomplete(f"{role}: missing or unknown outcomes")
        step = re.search(r"step_(\d+)", str(out_dir))
        executor.record_round(self.cell.root, arm="skillopt", step=step and int(step[1]), role=str(role),
                              skill_text=skill, evaluation_path=rows[0]["evaluation_path"])
        return [self.prediction(row, out_dir) for row in rows]

    def prediction(self, row: dict, out_dir) -> dict:
        """Write predictions/<id> as upstream's officeqa env does and return upstream's result row."""
        system, user, turns = conversation(row)
        folder = Path(out_dir, "predictions", row["id"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "conversation.json").write_text(json.dumps(turns, ensure_ascii=False, indent=1), "utf-8")
        (folder / "target_system_prompt.txt").write_text(system, "utf-8")
        (folder / "target_user_prompt.txt").write_text(user, "utf-8")
        reason = {1: "", 0: "verifier: task failed", None: "unknown: no outcome"}[row["success"]]
        return {"id": row["id"], "hard": float(not reason), "soft": float(not reason), "fail_reason": reason,
                "task_description": SKILL_FRAME.sub("", user).strip(), "task_type": self.cell.benchmark,
                "n_turns": len(turns)}


def epochs(batches: list) -> dict:
    """Upstream's epochs over the batches: train_size distinct tasks, ceil(train_size / batch_size) steps each."""
    train_size, size = len({task for batch in batches for task in batch}), len(batches[0])
    num_epochs = math.ceil(sum(map(len, batches)) / train_size)
    if len(batches) != num_epochs * math.ceil(train_size / size):
        raise ValueError(f"{len(batches)} batches do not fill {num_epochs} epochs of {train_size} tasks at {size}")
    return {"batch_size": size, "train_size": train_size, "num_epochs": num_epochs}


def client(llm) -> SimpleNamespace:
    """Our LLM as the OpenAI client upstream's openai_compatible backend calls; a failed call raises into upstream."""
    def create(*, messages, max_tokens, temperature, **_):
        text, _, usage = llm.complete(messages, temperature=temperature, max_tokens=max_tokens)
        message = SimpleNamespace(role="assistant", content=text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(index=0, message=message)], usage=SimpleNamespace(**usage))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def run(cell, panels: dict, batches: list, llm, out: Path) -> dict:
    root = out / "skillopt"
    root.mkdir(parents=True, exist_ok=True)
    (root / "seed.md").write_text(executor.SEED, encoding="utf-8")
    cfg = flatten_config(load_config(str(Path(skillopt.__file__).parents[1] / "configs/_base_/default.yaml")))
    cfg.update(out_root=str(root), env="harness", skill_init=str(root / "seed.md"), optimizer_model=llm.model,
               optimizer_backend="openai_compatible", target_model=cell.model, **epochs(batches))
    transport.configure_openai_compatible(optimizer_model=llm.model, max_tokens=16384, temperature=1.0)
    transport._build_client = lambda config: client(llm)
    harness = Harness(cell, panels, batches, cfg)
    summary = ReflACTTrainer(cfg, harness).train()  # resumes from upstream's runtime_state.json in root
    return {"final_skill": (root / "best_skill.md").read_text(encoding="utf-8"), "episodes": dict(harness.episodes),
            "llm": {"calls": llm.usage["calls"], "tokens": llm.usage["prompt_tokens"] + llm.usage["completion_tokens"]},
            "extra": {key: summary[key] for key in KEYS}}
