---
name: verification-before-completion
description: Use when about to claim work is complete, fixed, or passing, before committing or creating PRs - requires running verification commands and confirming output before making any success claims; evidence before assertions always
---

# Verification Before Completion

## Overview

**Core principle:** Evidence before claims, always.

**Violating the letter of this rule is violating the spirit of this rule.**

## The Iron Law

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

If you haven't run the verification command in this message, you cannot claim it passes.

## The Gate Function

```
BEFORE claiming any status or expressing satisfaction:

1. IDENTIFY: What command proves this claim?
2. RUN: Execute the FULL command (fresh, complete)
3. READ: Full output, check exit code, count failures
4. VERIFY: Does output confirm the claim?
   - If NO: State actual status with evidence
   - If YES: State claim WITH evidence
5. ONLY THEN: Make the claim

Skip any step = lying, not verifying
```

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

## Red Flags - STOP

- Using "should", "probably", "seems to"
- Expressing satisfaction before verification ("Great!", "Perfect!", "Done!", etc.)
- About to commit/push/PR without verification
- Trusting agent success reports
- Relying on partial verification
- Thinking "just this once"
- Tired and wanting work over
- **ANY wording implying success without having run verification**

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

## Key Patterns

**Tests:**
```
✅ [Run test command] [See: 34/34 pass] "All tests pass"
❌ "Should pass now" / "Looks correct"
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

**Rule applies to:**
- Exact phrases
- Paraphrases and synonyms
- Implications of success
- ANY communication suggesting completion/correctness

## Confirm Independently Against a Provided Oracle

When the task supplies a reference — an original program to port, a provided binary, a known-answer oracle, a spec example, a checksum, a prefix — do not settle for spot checks or your own reasoning. A fresh run of the method that produced the result can repeat the same mistake, so confirm with a method that does **not** depend on how you produced it.

- Run the task's own reference implementation, checker, or provided tool where one exists, and report its literal output. Build an automated harness that runs BOTH the reference and your artifact on the same inputs and compares outputs (and exit codes) exactly.
- Generate MANY inputs, not one: the provided example, boundary cases (0, 1, empty, max, negative, truncation, overflow), randomized/fuzzed inputs, and fuzzed initial state (existing data files, prior refs/objects). Compare the exact artifact the grader will see — bytes, sorted order, field widths, date format, exit status.
- Use a second, independent solver/engine and require agreement (and when they disagree, report the disagreement instead of picking one). Prefer the provided tool over a from-scratch reimplementation; it is the thing that will judge you.
- If the task gives a partial answer, hint, prefix, checksum, or reference value, enumerate the plausible interpretations, compute each, and select the one that matches the oracle — do not silently pick one reading.
- For crypto/algorithmic tasks, confirm the recovered key/parameters by an independent route (the provided decrypt binary reproduces the known pairs; re-encrypting recovers the ciphertext) before declaring the answer.
- Run a statistically meaningful number of random trials (not one) for probabilistic methods.

```
✅ Run both paths over N cases incl. edges → 0 mismatches vs the reference
❌ "It matches on the provided example, so it's correct"
```

## Verify the State and Format You Actually Deliver

A claim about an output file must be backed by reading that exact file from its FINAL required path — not from memory, a variable, a temp copy, or a drafts directory.

- After writing the deliverable, open/`cat` the file at its FINAL required path and inspect the real bytes. Assert every format property the task specifies: exact filename and location, content character set, single-line vs. multi-line, no empty/extra lines, delimiter and field order, required header/footer, key set, row/record count, trailing newline, byte length and any size limit.
- Whenever you clean up, reset, restore, or remove test scaffolding, RE-RUN the full verification on the final state. Emptying a repo, restoring input files, or deleting credentials can silently break what you just proved.
- Leave the system in the state the grader expects: restore original inputs/data files, remove temporary keys/credentials you added, and confirm with a fresh read (hash/contents) that they match the originals.
- Exercise the real end-to-end workflow through the real interface with the user's exact invocations (clone → commit → push → curl; `python3 file.py N`; `gcc file.c -o cmain && ./cmain N`). Starting a process is not proof — connect to the port / call the interface and read the response.
- Re-measure stated limits (e.g. an action completes within a given time budget) on the delivered state; do not assume them.

## The Deliverable Must Exist

A task is not complete until its required output exists at the required path and you have read it back. A correct analysis, a plan, or "the code looks right" is not the deliverable.

- Before finishing, list/read the exact file(s), the exact command/interface the task names, and confirm format, count, and order. Confirm required inputs were not left modified.
- If a required input is missing, search broadly before concluding: other directories, `/logs` artifacts, mounted volumes, package registries/mirrors, and the harness itself. Report exactly what you searched.
- Only then state the blocker — and still leave any partial deliverable the task permits. Never fabricate content that depends on a missing input; report the real status with evidence.
- For long builds/training, run in the background and poll, and make sure the required output file and artifacts are actually produced before time runs out.
