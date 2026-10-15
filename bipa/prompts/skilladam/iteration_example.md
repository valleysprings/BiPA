The domain and content below are illustrative only — do not transfer any
of it into your output.

```json
{
  "reasoning": "## Observations\n\n[2] (failure, hard=0): The agent edited the configuration file and declared the task done without re-running the check the task asked for; the verifier found the service still failing.\n\n[5] (failure, hard=0): Same pattern: the agent reported completion after writing the output file but never read it back, and the file had the wrong format.\n\n[1] (success, hard=1): The agent ran the requested check after its change, saw a failure, fixed it, and re-ran the check before finishing.\n\n## Patterns\n\nFailures [2] and [5] both end without verifying the result the task asks for. The current skill says 'Check your work' but not when or how.\n\n## Proposed Changes\n\n1. **Revise** Workflow step 4 to require re-running the task's own success check (or reading back the produced artifact) before finishing.\n\nEdit balance: 1 revision, 0 additions.",
  "patch": "--- a/SKILL.md\n+++ b/SKILL.md\n@@ -12,3 +12,3 @@\n 3. **Make the change**: ...\n \n-4. **Check your work.**\n+4. **Verify before finishing**: re-run the check the task names (or read back the artifact you produced) and fix any mismatch before declaring completion.\n"
}
```
