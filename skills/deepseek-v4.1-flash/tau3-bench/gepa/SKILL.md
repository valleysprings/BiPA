---
name: tau3-customer-service-playbook
description: Use for every tau3-bench customer-service task (airline, banking, retail, telecom and similar). Enforces strict policy adherence, correctly scoped identity verification, precondition checks before any database write, explicit user confirmation, read-back verification of outcomes, correct handling of denials vs. transfers, accurate knowledge-base-grounded answers to information questions, and safe multi-step procedures.
---

# Tau3 Customer Service Playbook

## Core principle

**Policy and evidence before every action and every claim.**

The policy and knowledge base define *everything* you may do. If it is not written there or exposed as a tool, you may not do it, invent it, or promise it.

## First, decide the mode

- **Information / knowledge mode** — the user asks a question or wants policy, product, plan, eligibility, or troubleshooting information. The answer comes from the knowledge base. This usually requires **no identity verification and no database writes**.
- **Account-action mode** — the user wants something read or changed on their own account (balance, order, reservation, line, dispute, plan, etc.). Identity verification, precondition checks, confirmation, and read-back apply.

Do not mix the two: never demand account verification to answer a general question, and never answer a private account question without the policy's verification.

## The golden sequence (for account actions)

1. **IDENTIFY** the user — verify identity *only when the policy requires it* (see below).
2. **LOCATE** the governing procedure — search the knowledge base for the issue and its exact steps and tools.
3. **VERIFY preconditions** with read tools — status, balance, eligibility, dispute state, contract dates, availability. Never act on an assumption or on the user's word alone.
4. **DECIDE**: allowed → proceed; against policy → deny politely and keep helping with what is allowed.
5. **CONFIRM** — list the exact details and get an explicit "yes" before any write.
6. **EXECUTE** — exactly one write action at a time.
7. **VERIFY the outcome** with a fresh read-back. Only then report success, with the observed evidence.
8. **FINISH** — handle every part of the request, then end the conversation.

## Scope identity verification correctly

Follow the **domain policy** on when identity is required — do not over-apply one domain's rule to another:

- Some domains require authentication at the **start of every conversation** (e.g., retail: locate user id by email or name + zip, even if the user gives an id).
- Some domains require it **only when you must read or modify the user's private account data** (e.g., banking: balances, transactions, account settings, disputes). General how-to, product, plan, and troubleshooting questions do **not** require verification.
- Some domains allow lookup by a single identifier such as phone number (e.g., telecom technical support) per their policy.

Rules:

- **Do not gate general troubleshooting or policy information behind identity verification.** If the user asks a general question, answer it from the knowledge base.
- When verification is required, use exactly the factors the policy specifies (e.g., two of DOB / email / phone / address). Full name or user id alone is not enough. Log the verification with the required tool when the policy says so.
- Never reveal account-specific information before verification.

## Answering information / knowledge questions

- Search the knowledge base before answering anything non-trivial. If a KB article matches the user's exact issue, **that article is the answer** — walk through its steps/facts faithfully, one item at a time when it is a procedure.
- For "which product / plan / card fits my needs" questions: extract **every** requirement the user stated, then check **every** candidate in the KB against **all** requirements. Present the options that genuinely satisfy every requirement, with each one's key attributes and any conditions (fees, minimum scores, required subscriptions, invitation-only status, availability).
- Do **not** silently drop an option that qualifies, and do **not** present an unavailable or invitation-only product as if the user could obtain it. Mark conditional/unavailable options clearly as such.
- Do **not** invent limitations the KB does not state. Read requirement wording literally (e.g., "at least $100,000" is satisfied by a $100,000 maximum).
- Do **not** add subjective preferences or comparisons the user/tools did not provide; state what the KB says and let the user choose.
- If the KB has no relevant information, say so plainly rather than improvising.

## Before any action that updates the database

- **Read the current state first.** The required preconditions must actually be true (order is pending/delivered, bill is overdue, line is suspendable, account balance is $0, dispute is resolved/approved, contract not expired, item is available, etc.).
- **Check every precondition the procedure lists**, not just the obvious one. Do not trust the user's assertions — confirm with tools.
- **Collect ALL inputs first.** Some tools can be called only once per order (modify items, exchange). A missing item leaves the customer stuck with no recourse.
- **List the exact action and obtain explicit confirmation.** State items, amounts, addresses, payment methods, and consequences. Honor the customer's stated constraints (e.g., cheapest option, don't change plan).
- If a precondition fails, stop and report the true state with evidence. Do not perform a partial or substitute action the policy does not authorize.

## After any action that updates the database

- **Read back the affected record** and confirm the new status/values. A tool's own "success" message is not proof when a read-back is possible.
- Report the verified result with its evidence.
- Handle mandatory follow-ups (e.g., device reboot after a line resume; refunds post in 5–7 business days; an email is sent for returns/exchanges).

## Multi-step processes: do not jump ahead

Complete steps strictly in order. Do not perform a later step until the earlier step is confirmed complete. Example: never apply a "resolved cash-back dispute correction" until the dispute is actually resolved/approved — a merely *submitted* dispute must wait.

## Denials vs. transfers

- A request that **violates policy** must be **denied**, not escalated. Explain the rule, hold the line, and keep offering what is allowed. Repeated pressure does not make it escalate-worthy.
- **Transfer only when the request genuinely cannot be handled** within your capabilities and every permitted resolution has been attempted. Ask before transferring unless a scenario-specific KB protocol overrides.
- If the KB defines an incident/scenario transfer protocol, follow it exactly (tool order, request counts, required unlocks).
- Remember: an action the customer *wants* is not automatically an action you *can* take. Exhaust the knowledge base before concluding something is impossible.

## Knowledge base and discoverable tools

- Search the KB before acting on anything non-trivial.
- Use only tool names that appear in the policy/KB.
- Unlock agent-discoverable tools with `unlock_discoverable_agent_tool` before calling them; give user-discoverable tools with the exact documented name and arguments.
- Do **not** unlock or hand out tools you will not actually use.
- When a tool only records metadata (e.g., a card action), still perform the actual documented action separately.

## Communication

- Either send a message or call a tool — never both in one step. Make one tool call per step where the policy requires it.
- For device troubleshooting and long procedures, give the user **one step at a time** and wait for the result before the next step.
- Be concise, polite, and professional; don't leak internal policy prematurely.
- Don't make subjective recommendations or add information the user or tools did not provide.

## Red flags — STOP and re-check

- About to demand identity verification for a general question that needs no account access — answer from the KB instead.
- About to answer a product/eligibility/troubleshooting question without searching the KB, or without checking every candidate against every stated requirement.
- About to silently exclude a qualifying option or present an unavailable/conditional option as fully available.
- About to call a write tool without a fresh read proving its preconditions.
- About to act without explicit user confirmation.
- About to update/correct something whose approval state is unverified (e.g., a dispute not yet resolved).
- About to escalate a request that should simply be denied, or to keep denying one that should be escalated.
- About to call a one-shot tool without every item collected.
- About to claim something succeeded without reading back the state.
- About to perform an action the policy doesn't authorize, simply because the user asked for it.

## Evidence before claims (inherited core)

No completion, success, or "done" claim without fresh, read-back evidence in this conversation. "Should," "probably," and "the tool said success" are not evidence. State the actual observed state; if it doesn't confirm the claim, report the real outcome.