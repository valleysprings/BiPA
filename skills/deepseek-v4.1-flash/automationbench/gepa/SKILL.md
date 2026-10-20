---
name: verification-before-completion
description: Use when about to claim work is complete, fixed, or passing, before committing, sending messages, mutating records, or reporting success — requires a fresh re-read of persisted state as evidence before making any success claims; evidence before assertions always
---

# Verification Before Completion

## Overview

**Core principle:** Evidence before claims, always.

**Violating the letter of this rule is violating the spirit of this rule.**

## The Iron Law

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

If you have not independently re-read the state that proves it — in this message — you cannot claim it.

## The Gate Function

```
BEFORE claiming any status or expressing satisfaction:

1. IDENTIFY: What state proves this claim, and how do I read it back?
2. READ A FRESH: Re-fetch/GET the resource or re-read /workspace/world_state.json.
3. READ: Full output — every field, verbatim. Not the tool's success message.
4. VERIFY: Does the evidence confirm the claim?
   - If NO: state actual status with evidence, and fix it.
   - If YES: state the claim WITH the evidence.
5. ONLY THEN: make the claim.
```

A tool's success response (`{"id": ...}`, `{"ok": true}`, `{"success": true}`) is NOT evidence. It reports intent, not persisted state.

## Environment Rules (persisted world state)

- World state is saved to `/workspace/world_state.json` after every mutating call.
- **Serialize mutating calls. Never run two writes in parallel.** Each mutating call loads the whole world, applies one change, and saves the whole world — concurrent calls race and silently drop earlier writes (e.g. an email send clobbered by a Slack post). Run writes one at a time; if you suspect a race, re-read state and re-run the lost call.
- Parallelize only read-only discovery (`api_search`, GETs) — never mutations.
- Verify each mutation by GET-ing the resource or re-reading `world_state.json` (`gmail.messages`, `slack.messages`, `salesforce.*`, `google_calendar.events`, sheet values, etc.).

## Requirement Checklist (do this BEFORE acting)

Re-read the task and **every referenced source** — emails, Slack posts, policy/guideline worksheets, config rows. Then enumerate:

- **Deliverables**: everything you must produce (send, update, create, notify).
- **Constraints**: filters, tiers, thresholds, date windows, required references/codes, conditional branches ("only if X", "also alert when Y").
- **Verbatim values**: names, amounts, dates, IDs — copy exactly, no rounding or paraphrase.

After acting, check off every item against fresh evidence. A task can fail on a single missed constraint even when all actions "succeed". Handle excluded items silently in the action — do not narrate skipped work.

## Common Failures

| Claim | Requires | Not Sufficient |
|-------|----------|----------------|
| Email sent | Gmail message persists with the To/Subject/body | Send tool returned an id |
| Slack posted | Message persists in the channel | `{"ok": true}` |
| Record updated | GET the record shows the new field | PATCH returned `{}` |
| Task/event created | Record visible via list/get | Create returned an id |
| Sheet updated | values.get shows the written cells | update returned updatedCells |
| Notification correct | Persisted text contains required names/amounts verbatim | Message was "sent" |
| Requirements met | Line-by-line checklist verified | Most steps succeeded |

## Red Flags — STOP

- Claiming success from a mutating call's response without re-reading state.
- Running parallel mutating calls (or "favoring parallel" for writes).
- Using "should", "probably", "seems to".
- Narrating skipped/excluded items instead of handling them silently.
- Not re-reading the policy/protocol/guideline source before acting.
- Source values paraphrased, rounded, or renamed.
- Declaring done because "most steps succeeded".

## Rationalization Prevention

| Excuse | Reality |
|--------|---------|
| "The API returned ok" | Re-read the state; ok ≠ persisted |
| "Parallel is faster" | Writes race and get lost — serialize them |
| "Should have worked" | READ the state |
| "I'm confident" | Confidence ≠ evidence |
| "No time to re-check" | One re-read call vs. a lost task worth zero |
| "I listed what I skipped" | Only report acted-on items |
| "Close enough value" | Copy source values verbatim |

## Key Patterns

**Mutating writes (serialize):**
```
✅ write #1 → verify → write #2 → verify → write #3 → verify
❌ write #1 + write #2 in parallel (one gets clobbered)
```

**Email:**
```
✅ send → re-read gmail.messages → confirm To/Subject/body verbatim
```

**Slack:**
```
✅ post → re-read channel messages → confirm names/amounts verbatim
```

**Records / Tasks / Events:**
```
✅ mutate → GET the record → confirm the updated field
```

**Requirements:**
```
✅ re-read task + policies → checklist → execute → verify each → report
❌ "Most actions succeeded, done"
```

## When To Apply

**ALWAYS before:**
- Any success/completion claim or expression of satisfaction.
- Sending notifications, updating records, creating tasks/events, or committing.
- Moving to the next step, or delegating.
- Reporting status of work.

**Rule applies to:** exact phrases, paraphrases/synonyms, implications of success, and any communication suggesting completion or correctness.