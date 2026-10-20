---
name: workflow-task-verification
description: Use for business-record workflows that apply policies, act in external systems, update trackers, and send notifications; complete required actions and verify persisted results
---

# Workflow Task Verification

## Core Rule

Complete the full requested workflow, not just a plausible subset. Base actions on live records and applicable procedures. Use the task’s stated date when evaluating deadlines, eligibility, and recency. Do not claim success without destination readback.

## Workflow

1. Initialize as instructed. Read `/workspace/tooling.md` and `/workspace/agent_config.json`; confirm available tools and `allowed_services`.
2. Read the authoritative source records, policies, thresholds, templates, and referenced exception, hold, cooldown, or action records. Use API parameter mechanisms for query parameters when available. Treat errors, partial responses, and empty reads cautiously; verify whether a response is genuinely empty before concluding data or an operation is unavailable.
3. Identify records in scope and apply every row-level rule. Distinguish eligible, acted-on, held, escalated, and skipped records. Apply exclusions silently where requested, but still perform any required routing, escalation, or notification for them.
4. Check destination state before mutating. Match each eligible record to its exact destination and operation. Do not substitute a tracker update for the actual requested action.
5. Calculate using the stated formula and thresholds. Preserve source values verbatim when requested. Keep source values distinct from derived values; do not round unless required. When a derived rounded value may be expected or useful, include it clearly labeled alongside the precise value.
6. Send required notifications to every policy-specified recipient and channel, including exception-routing recipients. Include affected entity names and relevant source amounts or metrics verbatim. Include an aggregate total when the requested report summarizes multiple acted-on amounts. Do not describe skipped or excluded records in messages or summaries when instructed not to.
7. Update only the intended tracker cells or rows. Preserve headers, schema, and unrelated data. Use the exact required status/classification labels. If a configured taxonomy has a broad and a specific label, include both accurately (for example, “At Risk - Critical”) rather than replacing the configured classification.
8. Verify every consequential mutation by reading back the destination. Check record identity, fields, values, status, recipients, message subject and body, and persisted tracker rows. For non-idempotent actions, check for an existing result before retrying.

## Handling Incomplete Configuration

- Do not stop the entire workflow merely because a discovery response omits nested details or returns a partial/empty representation. Retry relevant reads with correct parameters and inspect the actual action endpoint.
- If the task identifies an authoritative template, destination, recipient, or procedure and the intended operation is available, use those task-provided details and reasonable role defaults supported by the record or workflow. Do not invent a different destination or unauthorized service.
- For document-signing workflows, inspect template documents and recipient roles, but do not treat missing metadata alone as proof that sending is impossible. Use the supplied template and intended signer when the envelope operation supports it; update each in-scope tracker row to the required sent/deferred state and verify envelope and tracker readbacks.
- If an operation truly cannot be performed after reasonable discovery, state the specific blocker and do not mark it complete. Continue with other independent required actions where possible.

## Special Checks

- Finance: a journal, disbursement, write-off, or payment must be performed in the system that owns that transaction. Route bankruptcy, approval, and other policy exceptions to their designated recipients. Reports should include exact acted-on amounts and required totals.
- Approvals and expenses: honor all receipt and category requirements, approval limits, exceptions, and approver notifications. Do not bypass policy because a user suggests an exception.
- External briefings: obey current prep instructions, embargo dates, recipient rules, and confidentiality notes. Preserve source metrics verbatim and label derived figures and subsets accurately.
- Scoring and outreach: use the configured thresholds and exact classification vocabulary; include score, classification, and relevant evidence in CRM. Record all relevant discovered decision-maker profile details when updating a lead or contact, not only the minimum requested fields.
- Coaching or performance reports: apply tenure, volume, status, exemption, and threshold rules per agent. Include the exact observed rate and count in the tracker and manager message; if a rounded rate is useful or expected, show it separately as a labeled rounded value.

## Mutation Safety and Completion

- Prefer parallel calls for independent reads. Perform state-changing calls sequentially when they share an application or workbook.
- Inspect mutation responses, then read back the persisted destination state. An API success response alone is not proof of completion.
- Before finalizing, check records, eligibility and exceptions, required actions, tracker updates, all notifications, and persisted evidence.
- Report factual results only. Mention only actions taken and verified; state any genuine blocker plainly.