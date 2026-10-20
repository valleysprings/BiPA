---
name: verification-before-completion
description: Use when about to claim work is complete, fixed, or passing, before committing or creating PRs, and when running long, destructive, or uncertain operations - preserve input evidence, produce the deliverable early, verify against the task's own oracle, and finish within budget; evidence before assertions always
---

# Verification Before Completion

## Overview

**Core principle:** Evidence before claims, always.

**Second principle:** Verify against the task's own oracle, and finish within budget.

**Third principle:** Never destroy the evidence you were given, and never let the clock run out with no deliverable.

**Violating the letter of this rule is violating the spirit of this rule.**

## The Iron Law

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

If you haven't run the verification command in this message, you cannot claim it passes.

## The Gate Function

```
BEFORE claiming any status or expressing satisfaction:

1. IDENTIFY: What command proves this claim? Prefer the task's own test/verifier.
2. RUN: Execute the FULL command (fresh, complete)
3. READ: Full output, check exit code, count failures
4. VERIFY: Does output confirm the claim?
   - If NO: State actual status with evidence
   - If YES: State claim WITH evidence
5. ONLY THEN: Make the claim

Skip any step = lying, not verifying
```

## Preserve Inputs - Back Up Before You Mutate

You cannot recover what you destroyed. Many tools mutate or delete their inputs as a side effect, and recovery tasks hinge on artifacts you only get one chance to read.

- BEFORE running anything that can write, checkpoint, flush, truncate, or delete, copy the inputs aside (`cp -a INPUT DEST`) and hash them. Treat the originals as read-only; work on copies.
- Recovery / forensics tasks: your FIRST action is the backup, not a convenient inspection with a stateful tool.
  - Opening a SQLite DB auto-recovers and can **delete** `*.db-wal` / `*.db-shm`. Copy the DB and its sidecar files together *before* invoking `sqlite3`.
  - `>` truncates; editors, daemons, indexers, and VCS checkouts rewrite files.
- If a file is irreplaceable, never let a tool touch the original before you have a copy.
- Once destroyed, do NOT burn the whole budget combing container layers, block devices, `/proc`, or host mounts for it — the only real fix was prevention. Acknowledge it and move on quickly.

## Produce the Deliverable Early, Then Refine

The grader reads the artifact, not your intent. A missing file always scores zero; a present-but-imperfect file can still score.

- Create the required output file (even a placeholder or best guess) as soon as you know its path.
- Update it in place as you learn more — never hold the answer "in your head" until the end.
- Before any long, risky, or uncertain exploration, write your current best answer to the required path.
- Reserve a final slice of the budget to (re)write and then verify the deliverable.
- Partial-but-present beats perfect-but-absent.

## Verify Against the Real Oracle

Your own checks can share the same misunderstanding as your implementation. A wrong mental model produces a wrong self-written test, so "my checker passed" is not evidence.

- If the environment provides a test/verifier (e.g. `test_outputs.py`, a Makefile target, a reference binary), run **exactly that** — not a paraphrase or re-implementation.
- If no verifier is provided, derive checks from the literal task text and known-good external references; prefer end-to-end behavior over internal consistency.
- When no oracle exists, write down your interpretation of the requirement and sanity-check the riskiest assumptions before investing heavily.
- Compare against ground truth where possible: diff produced vs expected, byte-compare artifacts, run the real consumer of your output.
- State which oracle you used. "Internally consistent" is weaker than "matches the specified verifier."

## Time-Box Long and Uncertain Work

A correct-but-unfinished solution scores zero. Watch the clock.

- Estimate cost before long operations (compiles, dependency trees, brute-force searches, large builds). If a step can take many minutes, launch it in the background and poll, or choose a faster path.
- Set a budget per approach. If two or three attempts at a hypothesis have not converged, **stop and pivot**: save the best result to the deliverable, then try a fundamentally different strategy — not another variation.
- Do not let tool-installation or exploration yak-shaving consume the budget while the deliverable is missing. If a tool will not arrive quickly, fall back to a simpler path that still produces output.
- Prefer the shortest path that satisfies the requirement; avoid optional gold-plating while a required piece is unsolved.
- If time is nearly exhausted, secure and verify the best partial result rather than opening a new rabbit hole.

## Common Failures

| Claim | Requires | Not Sufficient |
|-------|----------|----------------|
| Tests pass | Test command output: 0 failures | Previous run, "should pass" |
| Linter clean | Linter output: 0 errors | Partial check, extrapolation |
| Build succeeds | Build command: exit 0 | Linter passing, logs look good |
| Bug fixed | Test original symptom: passes | Code changed, assumed fixed |
| Regression test works | Red-green cycle verified | Test passes once |
| Agent completed | VCS diff shows changes | Agent reports "success" |
| Requirements met | Line-by-line checklist | Tests passing |
| Task verified | Ran the provided verifier | Self-written checks only |
| Inputs intact | Backup copy made before any mutation | "I'll restore it later" |
| Deliverable exists | File written at required path | Answer computed but unsaved |
| Done in time | Budget respected, output produced | "Almost working" at timeout |

## Red Flags - STOP

- Using "should", "probably", "seems to", "I'm confident"
- Expressing satisfaction before verification ("Great!", "Perfect!", "Done!", etc.)
- About to commit/push/PR without verification
- Trusting agent success reports
- Relying on partial verification
- Thinking "just this once"
- Tired and wanting work over
- **ANY wording implying success without having run verification**
- **Passing your own checker instead of the task's verifier**
- **Running a stateful tool on an original input before backing it up**
- **Searching for destroyed data instead of preventing its loss**
- **No output file on disk while the clock runs down**
- **Installing yet another tool while the deliverable is still missing**
- **Repeating a failing attempt/brute-force with no progress**
- **Assuming "internally consistent" means "correct"**

## Rationalization Prevention

| Excuse | Reality |
|--------|---------|
| "Should work now" | RUN the verification |
| "I'm confident" | Confidence ≠ evidence |
| "Just this once" | No exceptions |
| "Linter passed" | Linter ≠ compiler |
| "Agent said success" | Verify independently |
| "I'm tired" | Exhaustion ≠ excuse |
| "Partial check is enough" | Partial proves nothing |
| "Different words so rule doesn't apply" | Spirit over letter |
| "My custom test passed" | Run the task's own test |
| "Opening it won't change anything" | Stateful tools delete sidecar files; back up first |
| "I'll write the answer at the end" | Write it now, refine later |
| "It's almost done" | Finish or secure partial output before timeout |
| "One more attempt will work" | Pivot after repeated failure |

## Key Patterns

**Tests:**
```
✅ [Run task's test command] [See: 34/34 pass] "All tests pass"
❌ "Should pass now" / "Looks correct" / only my own smoke test
```

**Destructive / recovery commands:**
```
✅ cp -a input/ backup/ && sha256sum backup/*  → then inspect the copy
❌ sqlite3 original.db ...   (deletes *.db-wal before you read it)
```

**Deliverables:**
```
✅ write required file early → refine in place → re-verify at the end
❌ compute everything in memory and never save the artifact before timeout
```

**Regression tests (TDD Red-Green):**
```
✅ Write → Run (pass) → Revert fix → Run (MUST FAIL) → Restore → Run (pass)
❌ "I've written a regression test" (without red-green verification)
```

**Build:**
```
✅ [Run build] [See: exit 0] "Build passes"
❌ "Linter passed" (linter doesn't check compilation)
```

**Long operations:**
```
✅ Estimate → background/poll → check progress → pivot if stalled
❌ "It's still compiling" until the clock runs out
```

**Requirements:**
```
✅ Re-read plan → Create checklist → Verify each → Report gaps or completion
❌ "Tests pass, phase complete"
```

**Agent delegation:**
```
✅ Agent reports success → Check VCS diff → Verify changes → Report actual state
❌ Trust agent report
```

## When To Apply

**ALWAYS before:**
- ANY variation of success/completion claims
- ANY expression of satisfaction
- ANY positive statement about work state
- Committing, PR creation, task completion
- Moving to next task
- Delegating to agents
- Beginning any long-running, repeated, or destructive/mutating operation
- Running any tool that could alter or delete the inputs you must analyze

**Rule applies to:**
- Exact phrases
- Paraphrases and synonyms
- Implications of success
- ANY communication suggesting completion/correctness