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

## Verify the Outcome Against the Governing Source (Adversarial Pre-Completion Audit)

A tool call returning success — or a row / message / page / field now existing — proves only that an *action ran*, never that the *requirement was met*. Verifying that what you did landed only confirms your own (possibly truncated) plan.

Before claiming completion, act as a strict grader:

1. Re-read the task and EVERY artifact it references (policy/SOP sheet, config, schema, superseding email, channel history) from scratch, ignoring what you already did.
2. Convert the task and every governing source into a checklist of required outcomes — each deliverable, each exclusion/exemption, each "do NOT", each conditional branch — and independently compute the expected result for every record: which qualify, which are excluded, which else-branch applies, and the exact value each should carry.
3. Do a FRESH read of persisted state and diff expected vs actual, line by line. Confirm each entity's actual resulting value/recipient/status matches the requirement. Every mismatch is a gap — fix it or report it; never let a summary of your actions stand in for this diff.

Resolve conflicts by authority and recency: a superseding instruction overrides the sheet; a hotfix whose notes say the prior version "remains canonical" is NOT the general-distribution version; retracted / pending-approval / draft items must be excluded.

Apply and confirm NEGATIVE requirements too (excluded, blocked, or held items were in fact left untouched; contractors / opt-outs not contacted). An item you could not complete is a reported gap — never a silent omission.

The edge cases that actually fail are, in order:
- **Exclusions / exemptions / holds** — a record that fails the main test is not automatically a no-op; the source usually says what to DO with it (retain, route elsewhere, decline, wait).
- **Conditional / else branches** — "if X do Y, else do Z" prescribes a different action for the failing case.
- **Boundary dates and thresholds** — compute day-differences from the task's stated "today" and apply the exact comparison (`>`, `>=`, "older than 48 hours", "within the 5th hour", amount caps).
- **Decoys** — cancelled events, closed tickets, duplicate/noise rows, superseded policies must be filtered out, not processed.

```
✅ requirement list → act → fresh GET state → diff every record → confirm required value/recipient + exclusions → report gaps/completion
❌ "the write/row/message exists, so the requirement is met" / verify only the items you happened to remember
```

## Serialize Mutating Calls (Persisted State Is Read-Modify-Write)

A mutating call returning success proves the call ran, not that the change persisted. Each mutating call loads state, applies the change, and saves the whole state back; parallel writes — or independent calls that overlap — are last-write-wins and silently drop earlier changes even though every call reported success.

- Issue MUTATING calls ONE AT A TIME (sequentially); parallel is safe only for READS.
- After any write (especially a batch), do a FRESH read of the authoritative state and confirm EVERY intended change persisted — not just that each call returned success.
- Re-apply and re-verify anything missing before claiming completion.

```
✅ sequential writes → fresh GET → persisted count == intended count → all present
❌ parallel writes → trust each call's success → only the last one persisted, claimed complete
```
