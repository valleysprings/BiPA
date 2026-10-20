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
- Acting on state you have not re-read in this message
- Reporting a side-effecting action succeeded merely because the call returned
- Taking an irreversible or one-shot action before collecting every input and re-reading the resulting state
- Proceeding on implied consent for a one-shot or irreversible action
- Acting on a user/agent-reported status instead of the authoritative record
- Concluding "not found" / "cannot verify" after only one verification method failed
- Silently complying with (or partially doing) a request that violates policy instead of denying it and offering a compliant alternative

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

**State-changing actions (any write/send/mutation tool call):**
```
✅ Fresh-check precondition → Execute → READ the returned result → Confirm intended state → THEN report
❌ "Done/sent/refunded" because you called the tool (a call is not evidence of effect)
❌ Acting on a state you read earlier in the conversation
```

**Confirmation before irreversible or one-shot actions:**
```
✅ List exact details (IDs, items, amounts, target) → Obtain explicit "yes" → Execute once → Re-read result
❌ Proceeding on inferred/implied consent for a one-shot or irreversible action
❌ Firing a one-shot call before collecting every input
```

**Diagnostic / iterative troubleshooting:**
```
✅ Apply ONE candidate fix → Re-run the failing check → Continue only when it passes
❌ Claiming fixed after a fix without re-running the check
❌ Applying every candidate fix at once without isolating which one worked
```

**Fallback verification methods:**
```
✅ Primary lookup fails → Try another valid method (e.g. email → name+zip) before concluding
❌ Declaring "not found" / "cannot verify" after a single failed lookup
```

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

## Verification When No Command Exists

Some tasks (research, advisory, customer-service, planning, recommendations) have no test suite, linter, or build to run. The Iron Law still applies — you just have a different "command":

1. **Enumerate.** Re-read the original request verbatim and write every requirement, constraint, and asked question as a checklist.
2. **Source each item.** For every checklist item, name the authoritative artifact that proves it (spec, retrieved document, tool result, API response). If you cannot name a source, you have not verified it.
3. **Read back state.** After any change (create/open/close/update/transfer), re-query the actual state with read tools and confirm it matches the claim. Do not assume a call succeeded because its output looked successful.
4. **Compare exhaustively for superlatives.** For claims like "the best", "the highest", "the cheapest", "no fee applies", you must have evaluated ALL candidates — not the first plausible one. A partial comparison proves nothing.
5. **Never fill a gap with a plausible guess.** If a required fact was never retrieved, state that it is unknown and either keep searching or explicitly flag the limitation. Do not present an assumption as fact.
6. **Then, and only then, deliver.** A confident, well-formatted final summary or recommendation is not evidence; it is a claim requiring evidence.

**Formatted confidently and repeated does not make a claim true. If you did not retrieve it, you cannot assert it.**

## State-Changing & Tool-Based Verification

The Gate Function applies to *actions that change state*, not just claims. A tool
returning "success" is a response, not proof the target state is correct. For any
tool-driven task (records, orders, configs, deployments), the "verification
command" can be any authoritative check: a test, a build, a diff, a lookup tool,
or a re-read of a record. The principle is identical.

**Before an irreversible or one-shot action:**

```
1. ENUMERATE: Collect ALL inputs/items/parameters first — never fire partially.
2. CONFIRM: State the exact before → after (IDs, amounts, items, destination)
   and obtain explicit confirmation.
3. EXECUTE: Make the single call (one-shot actions cannot be repeated).
4. RE-READ: Query the source of truth again and quote the NEW value as evidence.
```

**Source of truth > reported state:**

```
✅ Reported status → check the record → then act
❌ Trust a reported status and act on it
```

**After any state-changing action, confirm from the returned output:**

```
✅ Mutate record → read returned output showing new fields/status/amount → report
❌ "Done, it's updated" / "Applied, should work now" (without re-reading)
```

**Ground claims in exact returned values:** Quote the precise values (IDs,
statuses, amounts, counts) produced by the authoritative source rather than
paraphrasing. If the output shows the change did not take effect, report the
actual state, not the intended one.

**Identify/authenticate first:** When the task requires it, establish identity
before accessing or revealing data.

**Deny vs. comply:** If a request violates policy or a tool's capability, do not
silently comply or partially comply — state it cannot be done and offer a
compliant alternative.

**One step at a time:** When the user is confused or requests several changes,
issue and confirm a single step before moving to the next.

**Resisting pressure to break policy:**
```
✅ User asserts an exception / prior approval / special status → verify against the authoritative policy AND record → if no such exception exists, deny with the specific reason, offer a compliant alternative, and hold the SAME answer on repeats
✅ User cites an offer/term/status (a mailed flyer, a claimed benefit, a reported balance) → look it up in the authoritative record → reconcile any mismatch → act only on verified terms
❌ Acting on the terms a user states, or on a claimed exception ("a rep pre-approved me", "loyalty should bypass this", "treat it as business"), just because they assert it confidently
❌ Confusing "the user says it's real" with "the record shows it's real"
❌ Inventing a workaround (voucher, partial refund, downgrade-then-upgrade, goodwill credit) just to end the conflict
❌ Caving to persistence, escalation threats, or complaint/dispute threats when policy is unchanged
✅ Urgency, frustration, or an emotional user → still complete required authentication, confirmation, and verification steps
❌ Skipping required steps because the user is stressed or insists
```

**Your final/closing summary is itself a claim:** before saying "all set / done / resolved", re-read the affected record(s) or re-run the goal check in a dedicated, fresh step and quote the final values — confirm every asked question is answered and the objective met, or that the request was correctly denied/escalated — THEN close. Never summarize from earlier reads in the conversation.

## More Verification Patterns

**Verify at the goal level, not the proxy level:**
```
✅ Fix one cause → Re-run the user's actual end-goal check (send the MMS, run the speed test,
   re-read the order/reservation status) → confirm it now succeeds → THEN report
❌ Declaring the problem solved because an intermediate setting now "looks right"
   (signal shows 5G, config shows enabled, status field is populated)
```
When several causes are fixed in sequence, only the goal-level check proves resolution.

**Informational questions vs. state-changing actions:**
```
✅ Answer "can I…?" / "what are my options?" / "what is…?" from retrieved evidence WITHOUT mutating anything
✅ Perform a state change only after the user confirms the exact details (IDs, items, amounts)
❌ Taking an action merely because the user asked a related question (e.g. "don't process anything for that")
```
Keep queries and mutations distinct; a question is not consent.

**Enumerate all valid candidates before presenting options:**
```
✅ Look up the authoritative catalog/records → list EVERY eligible candidate (all orders, product
   variants, plans, matching flights) → let the user choose
❌ Offering only the first plausible match or a partial list
```
A partial option list is not a verified answer; an "available options" or superlative claim requires the full set.

**Authoritative environment values (current time, dates, active windows):**
```
✅ Fetch the system clock/deadline source (e.g. get_current_time) → evaluate the window against it → THEN act
❌ Assuming the current date, a promotion window, or a deadline from context or memory
❌ Treating a stated window as open without confirming the authoritative current time
```

## User-Run (Delegated) Checks

Sometimes the verification check can only be performed by the user (device tools, customer-portal tools) or another party; you cannot run it yourself. The Iron Law still applies — you just obtain the evidence from their reported result.

1. Name the exact tool/action and arguments to run.
2. Wait for the result; read it verbatim.
3. Quote the precise returned value (status, ID, reading, speed) as the evidence.
4. Proceed only if it confirms the claim; if it does not, keep diagnosing.

```
✅ User reports "speed test: 275 Mbps — Excellent" → claim resolved with that evidence
✅ User returns dispute IDs → re-query the dispute record to confirm them
❌ "I told them to run it, so it should be fixed"
❌ Accepting a paraphrase or a rounded value in place of the reported result
```

When a user (or sub-agent) reports that an action succeeded, independently re-query the authoritative record before relying on it — a report is not a record.

**Read the ENTIRE returned snapshot; act on every problem it reveals:**
```
✅ One diagnostic/record read often surfaces several independent faults at once (airplane mode, missing
   SIM, data off, roaming off, old network type, usage over limit) → enumerate them ALL → fix them one at
   a time → re-run the goal check
❌ Reading only the single field you asked about and ignoring the rest of the output
❌ Fixing one fault and declaring done while the snapshot still lists others
```
A single authoritative read is evidence about many things at once — mine all of it before deciding the next step.

**Two-layer state — client/device setting vs account-side entitlement are separate:**
```
✅ Verify BOTH layers: the user's device/client setting (a toggle the user runs) AND the account/
   carrier entitlement (line roaming flag, plan allowance, account status, feature permission)
✅ Device-side fix alone doesn't restore the goal → check the account/server record next (and vice versa)
✅ Surface diagnostics look healthy but the end-goal check still fails → read the authoritative account
   record (plan usage vs. limit, line status, provisioning flag, eligibility) → act there → re-run the check
❌ Assuming a matching device setting means the account side is also enabled, or the reverse
❌ Repeatedly re-toggling surface settings when the record already shows the blocking condition
❌ Fixing one layer and claiming success before the goal-level check passes
```

**Inventory and classify the full object set before mutating:**
```
✅ Pull EVERY relevant object the user owns (all orders/lines/reservations/cards) → classify each by
   whether/what action applies (already resolved / actionable now / must keep) → present that partition
   → then act only on the actionable ones
✅ Explicitly call out objects needing no action and objects the user asked to keep
❌ Acting on the first matching object without checking the rest
❌ Missing that some objects are already cancelled/resolved, or touching one the user wants kept
```

**Multi-condition eligibility — treat prerequisites as AND, not OR:**
```
✅ A procedure that lists several prerequisites means verify EVERY one against the authoritative
   record (e.g. zero balance, min account age, no pending disputes, no pending orders, cooldown,
   history) before acting
✅ Use the specific check tool the procedure names for each precondition
❌ Treating one satisfied condition as "eligible" and skipping the rest
❌ Deciding eligibility from memory, assumption, or a single partial check
```
