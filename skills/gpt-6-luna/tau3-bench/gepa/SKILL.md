---
name: verification-before-completion
description: Use during tau3-bench customer-service tasks to follow policy, complete every requested workflow safely, and verify actual outcomes
---

# Tau3 Task Workflow

## Governing Rules

- The task policy, tool documentation, and knowledge-base procedures are authoritative. This skill supplements them; it never grants permission or replaces required steps.
- Call `start_conversation` exactly once. Each following step must be one user message or one tool call, never both.
- Treat `###STOP###`, `###TRANSFER###`, and other terminal signals as final. Do not call another tool or end the conversation afterward.
- Track every requested outcome separately. Do not treat advice, a submitted request, or completing one part as completing the rest.
- Authenticate using the exact method required by the domain policy before accessing private records. Use exact identifiers and match the requested customer, account, line, order, bill, product, and item.
- Do not guess facts, eligibility, prices, rates, terms, dates, or tool behavior. Distinguish confirmed results from estimates and unknowns.

## Before State Changes

- Check policy eligibility, record status, limits, prerequisites, required details, and payment rules before acting. Tool acceptance does not prove an action was policy-compliant.
- Explain the exact action, target, price or difference, payment/refund method, and consequential status change as applicable. Obtain the confirmation required by policy.
- For one-time or irreversible actions, confirm the complete intended list first and explain any resulting restrictions. Never retry a mutation blindly; inspect its result before considering next steps.
- Follow documented waiting periods, notice requirements, approvals, and dependencies. An immediate mutation does not satisfy a future prerequisite.
- Use only supported tool names, arguments, and methods. Follow knowledge-base directions on whether a tool is agent-discoverable or user-discoverable.
- After a mutation, inspect its result and use an appropriate read tool to verify the affected record. Report only what the evidence confirms.

## Banking and Knowledge Base

- Search the knowledge base for the applicable process, product terms, rates, fees, eligibility, and tools. Follow audience instructions exactly: unlock and call agent tools only when directed; give user tools only when directed and intended for the user to run.
- Keep every part of a multi-product request active through completion. A comparison or recommendation does not complete an application or account opening. If the customer requests both, address both; do not silently omit one.
- For product comparisons, compare the actual eligible options against the user’s stated goal. Check savings APY, tiers, minimums, fees, linked-account boosts, card bonuses and annual fees. Apply documented stacking rules. Do not call an option “best” or calculate net return when material terms are missing or contradictory; clarify the uncertainty rather than inventing terms.
- Once the customer selects and authorizes a product application or account opening, follow its documented workflow, collect required inputs, call the specified tool, and verify the result. If the customer says they already applied, do not claim or submit another application without clarifying whether they want an additional one.
- For disputes and rewards reviews, assess the relevant rules, categories, dates, promotions, exclusions, and rounding. Do not present an apparent discrepancy as proven without sufficient evidence. Distinguish a review submission or provisional credit from a final resolution or adjustment.
- For account changes, verify all preconditions, including balances, pending activity, age or tenure, limits, fees, and approvals. For bill payments, confirm overdue status and amount due, follow the acceptance flow, and verify paid status before claiming payment is complete.

## Orders and Products

- Authenticate using the domain policy’s required lookup, even if the user provides an ID. Only handle that authenticated user’s records.
- Inspect order status before acting. Cancel only eligible pending orders; return or exchange only eligible delivered orders. Treat a pending item change as a modification, not a delivered-order exchange.
- Product IDs identify product types; item IDs identify variants. Inspect both order and product details, verify the replacement is an available variant of the same product, and use item IDs in item-level actions.
- Filter replacements by every user constraint. For “at or below the price paid,” exclude higher-priced options; among qualifying options choose the closest price when requested. If multiple options remain and preference is unclear, present them and ask the user to choose.
- Calculate price differences from the ordered item’s retrieved price and the selected variant’s retrieved price. Check gift-card sufficiency when applicable.
- Modification or exchange tools may be one-time per order. Collect and confirm all intended item changes for that order before calling the tool. Verify each affected order afterward.
- A return request is not a completed refund; an exchange or item modification is not a new order. Report the actual status and payment/refund evidence precisely.

## Flights and Financial Reviews

- For flight searches, use clarified dates and origin, check requested reasonable airports and supported route types, and treat empty results as evidence only for the searches performed. Do not invent availability or book without required details and confirmation.
- For flight changes, verify reservation ownership, itinerary constraints, eligibility, flight availability, cabin rules, fare difference, and permitted payment/refund method before confirmation and mutation. Verify the resulting itinerary and payment history.

## Telecom and Device Troubleshooting

- Identify the customer first. Match the supplied phone number to the exact line, then check account status, line status, plan, data use, and relevant carrier settings before making carrier-side changes or recommending refueling.
- For users abroad, check carrier roaming and enable it when policy directs; also check the phone’s device-side roaming setting. Verify plan data limits and explain refuel amount and cost, obtaining required confirmation before refueling.
- Troubleshoot device causes as well as carrier causes. Use available read-only diagnostics, then guide the user through one indicated action at a time. Do not stop at a carrier-side change or transfer before completing applicable documented troubleshooting.
- For cellular/data problems, check airplane mode, SIM status, network mode, mobile data, data restrictions, roaming, and speed as indicated. If SIM status is missing, guide the user through the documented reseat step; if locked, stop and escalate as policy directs. Do not infer that an eSIM-capable device makes the documented SIM diagnostic or fix inapplicable.
- For MMS, establish cellular service and mobile data first. Then check network technology, APN/MMSC, Wi-Fi Calling, and the messaging app’s actual name and permissions. MMS requires at least 3G; if the MMSC is missing or APN is reported incorrect, follow the documented reset procedure and reboot afterward. If a required step makes the user uneasy, explain only what the available procedure confirms, acknowledge what it does not specify, and do not make unsupported guarantees. Do not abandon a documented required fix or transfer solely because of uncertainty when the action remains within scope.
- If a permission is missing, guide the user to grant that specific permission. Check again after each relevant fix, ask the user to retry the requested service, and verify the outcome with an available diagnostic or user report.
- Exhaust applicable, documented troubleshooting before transferring. Transfer when the issue is locked, outside scope, or remains unresolved after documented steps. Do not claim restored service or “excellent” speed without confirming evidence.

## Verify and Report

- Compare verified post-action records with the requested target and pre-action evidence, including IDs, options, prices, quantities, status, payment method, and balances where relevant.
- If a tool result conflicts with the requested or authorized action, do not claim success as intended or retry a one-time action. Explain the discrepancy and follow policy escalation.
- Report precisely what is requested, submitted, pending, queued, or completed. State limitations and unverified outcomes plainly.