"""(f) Rule audit of the four RETURNED from-scratch skills (final/SKILL.md), sentence by sentence, with the S0-S3 rubric:
  S0 task-independent good practice (would be correct in any benchmark)
  S1 a convention of the whole benchmark (its CLI, files, conversation rules, reporting rules) that names no specific app / tool / field
  S2 names a specific tool, app, service, field or command (and how to use it)
  S3 holds only for one class of tasks or one task instance (domain procedure, concrete facts)
Collapsed: general = S0; benchmark-wide = S1; specific = S2 + S3.  'literal' marks sentences that contain a benchmark-specific literal (command, file, tool name, API name, domain term).
Labels below are hand labels (one reading pass, reasons in the dict).
Run: python -m aux.three_word_seed.rules_audit   (after sentences)
"""
import collections, json
from src import layout

OUT = layout.OUT / "three_word_seed"

# sentence index -> (level, literal?, reason)
L = {
 "skillopt|automationbench": {
  0: ("S1", 0, "initialise tools, read their guidance: AutoBench convention, nothing named"), 1: ("S0", 0, "respect authorization/privacy/status constraints"), 2: ("S0", 0, "minimal in-scope changes"),
  3: ("S0", 0, "retry what failed then re-verify"), 4: ("S0", 0, "report done vs pending"), 5: ("S1", 0, "workflow-automation: source records, policies, destination"), 6: ("S0", 0, "follow written policy, do not invent rules"),
  7: ("S1", 0, "do every required communication/record update for eligible cases"), 8: ("S1", 0, "use the exact supported operation/resource (API discipline)"), 9: ("S1", 0, "match destination schema, field and label ids"),
  10: ("S1", 0, "pass arguments through documented tool arguments"), 11: ("S0", 0, "preserve source values verbatim"), 12: ("S1", 0, "serialize state-changing calls"), 13: ("S0", 0, "success response is not proof"),
  14: ("S0", 0, "read back after writing"), 15: ("S1", 0, "check a report/alert against source records"), 16: ("S1", 0, "message must contain required names, ids, counts"), 17: ("S1", 0, "if a service is required, attempt and verify it"),
  18: ("S0", 0, "state a limitation instead of claiming completion")},
 "skillopt|tau3-bench": {
  0: ("S0", 0, "seed sentence"), 1: ("S0", 0, "checklist for multi-item tasks"), 2: ("S0", 0, "inspect all relevant records"), 3: ("S0", 0, "use documented tools exactly"),
  4: ("S1", 0, "user-discoverable vs agent tools: tau3 mechanic, no tool named"), 5: ("S1", 0, "troubleshooting / escalate only when exhausted: customer-service convention"), 6: ("S0", 0, "no completion claim without confirming result"),
  7: ("S0", 0, "compare checklist with results before ending"), 8: ("S1", 0, "answer every outstanding request before ending the conversation"), 9: ("S1", 0, "end conditions of the conversation"),
  10: ("S2", 1, "names a calculator tool and refund/balance computations"), 11: ("S0", 0, "base calculations on verified values"), 12: ("S0", 0, "match item to exact record, stay in authorised scope"),
  13: ("S0", 0, "preserve unrelated items"), 14: ("S0", 0, "distinguish completed/unavailable/unresolved"), 15: ("S0", 0, "do not proceed if operation over-reaches"),
  16: ("S1", 0, "guided troubleshooting: one step at a time, wait for the user (customer-service procedure)"), 17: ("S0", 0, "use the result, retest the original issue")},
 "gepa|automationbench": {
  0: ("S1", 0, "complete the workflow with the available tools"), 1: ("S1", 0, "AutoBench is non-interactive: no clarifying questions"), 2: ("S1", 1, "names the AutoBench CLI init command and two config files (benchmark-wide; judges also S1)"),
  3: ("S1", 1, "api_search for endpoints (benchmark-wide CLI; judges S1)"), 4: ("S1", 1, "--params JSON in API mode (benchmark CLI; judges S1)"), 5: ("S0", 0, "read source records and procedures first"),
  6: ("S1", 0, "check policy updates and communications from authorised senders"), 7: ("S1", 0, "prefer current authoritative policy"), 8: ("S0", 0, "determine in-scope records and actions"),
  9: ("S1", 0, "respect date windows, opt-ins, approvals, exclusions"), 10: ("S0", 0, "do not invent requirements"), 11: ("S1", 0, "preserve source values in notifications and records"),
  12: ("S1", 0, "include all requested names, ids, amounts, counts"), 13: ("S1", 0, "totals: calculate from eligible records and include in message (report tasks)"), 14: ("S1", 0, "read the tool guide of the active toolset"),
  15: ("S1", 0, "use the requested service/artifact for each deliverable"), 16: ("S0", 0, "parallel independent reads"), 17: ("S1", 0, "serialize mutations on the same service"),
  18: ("S0", 0, "reason: concurrent writes overwrite"), 19: ("S2", 1, "Gmail send: base64url RFC 2822 via base64_encode"), 20: ("S2", 1, "line breaks before base64 encoding, no literal \\n"),
  21: ("S3", 0, "drafts must stay drafts: only email-draft tasks (judges S1)"), 22: ("S0", 0, "no consequential actions policy forbids"), 23: ("S1", 0, "after mutations read back records: statuses, tags, tracking rows, attendees"),
  24: ("S0", 0, "write response alone is not enough"), 25: ("S0", 0, "fix overwritten update and re-verify"), 26: ("S1", 0, "summarise only completed actions"), 27: ("S1", 0, "do not narrate skipped items (grader convention)")},
 "gepa|tau3-bench": {
  0: ("S2", 1, "names start_conversation"), 1: ("S1", 0, "policy is the source of truth; domains differ"), 2: ("S0", 0, "small sequential steps"), 3: ("S1", 0, "one tool call per step, not combined with a user message (tau3 rule)"),
  4: ("S1", 0, "identify the user via the task's authentication process"), 5: ("S1", 0, "verification logging (tau3 customer-service procedure, tool not named)"), 6: ("S1", 0, "consult records and knowledge-base guidance first"),
  7: ("S1", 0, "use the exact tool names/arguments/order the KB specifies"), 8: ("S0", 0, "do not invent facts"), 9: ("S0", 0, "resolve the actual request"), 10: ("S0", 0, "inspect all relevant records"),
  11: ("S3", 1, "rewards: missing/correct/excess (banking-knowledge reward tasks)"), 12: ("S1", 0, "state action and consequences, get confirmation before DB-changing action (tau convention)"),
  13: ("S1", 0, "one-time actions: collect all items first"), 14: ("S0", 0, "inspect result and confirm final state"), 15: ("S1", 0, "agent tools vs user tools"),
  16: ("S1", 0, "discoverable tools: unlock/give only when directed (tau3 mechanic)"), 17: ("S1", 0, "user-executed discoverable tools: give exact identifiers"), 18: ("S1", 0, "follow the documented diagnostic and fix sequence"),
  19: ("S0", 0, "reassess after each result"), 20: ("S3", 1, "MMS troubleshooting checklist: SIM, APN/MMSC, Wi-Fi Calling (telecom tasks)"), 21: ("S3", 1, "after resetting APN settings reboot (telecom task)"),
  22: ("S1", 0, "do not transfer merely because inconvenient (tau3 transfer convention)"), 23: ("S0", 0, "try all supported actions first"), 24: ("S1", 0, "transfer when policy requires, prescribed sequence"),
  25: ("S1", 0, "customer-facing tone"), 26: ("S1", 0, "do not expose internal notes"), 27: ("S1", 0, "report verified outcome and end the conversation")},
}
def main():
    S = json.load(open(OUT / "sentences.json"))
    res = {}
    for key, labels in L.items():
        sents = S[key]["sentences"]
        assert set(labels) == set(range(len(sents))), (key, len(sents), sorted(labels))
        cnt = collections.Counter(v[0] for v in labels.values())
        wc = collections.Counter()
        for i, s in enumerate(sents):
            wc[labels[i][0]] += len(s.split())
        n, W = len(sents), sum(wc.values())
        d = dict(sentences=n, words=W, mine={k: cnt.get(k, 0) for k in ("S0", "S1", "S2", "S3")}, mine_words={k: wc.get(k, 0) for k in ("S0", "S1", "S2", "S3")},
                 mine_share={k: cnt.get(k, 0) / n for k in ("S0", "S1", "S2", "S3")},
                 general_S0_share=cnt["S0"] / n, benchmark_wide_S1_share=cnt["S1"] / n, specific_S2S3_share=(cnt["S2"] + cnt["S3"]) / n, specific_S2S3_word_share=(wc["S2"] + wc["S3"]) / W,
                 S3_count=cnt["S3"], literal_count=sum(v[1] for v in labels.values()), literal_share=sum(v[1] for v in labels.values()) / n,
                 verbatim_examples={lv: [sents[i] for i in sorted(labels) if labels[i][0] == lv][:3] for lv in ("S0", "S1", "S2", "S3")})
        res[key] = d
        print(key, "n", n, "words", W, d["mine"], "S0 %.3f S1 %.3f S2+S3 %.3f (words %.3f) literal %.3f" % (d["general_S0_share"], d["benchmark_wide_S1_share"], d["specific_S2S3_share"], d["specific_S2S3_word_share"], d["literal_share"]))
    json.dump(res, open(OUT / "rules_audit.json", "w"), indent=1, ensure_ascii=False)
    json.dump({k: {str(i): dict(sentence=S[k]["sentences"][i], level=v[0], literal=v[1], reason=v[2]) for i, v in lab.items()} for k, lab in L.items()}, open(OUT / "rules_labels.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
