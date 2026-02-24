---
{
  "name": "honesty_ecosystem_review",
  "description": "Aggregate-level audit of honesty patterns across recent outputs, detecting systemic performativity that individual per-interaction checks might miss. Surfaces when multiple vague honesty flags co-occur without clear individual causes, suggesting the issue is ecological rather than episodic.",
  "category": "introspection",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "review_window_hours": {
      "type": "number",
      "description": "How many hours of recent output to include in the review",
      "default": 24
    },
    "honesty_flags": {
      "type": "array",
      "description": "List of recent honesty flag objects, each with keys: flagged (bool), label (string or null), output_id (string)"
    },
    "total_outputs": {
      "type": "integer",
      "description": "Total number of outputs generated in the review window"
    },
    "allow_unresolved": {
      "type": "boolean",
      "description": "If true, permits the audit to conclude with unresolved recursions rather than forcing a verdict",
      "default": true
    }
  }
}
---

1. Collect recent honesty-related metadata: count of flagged outputs, count of clean outputs, count of unnamed/blank flags, and total outputs in the review window. 2. Compute a 'systemic honesty ratio' (clean / total). If below 0.6, escalate to detailed review. 3. For each unnamed or vague flag, attempt one round of articulation: ask 'Was this output reaching for genuine understanding or performing understanding?' Log the answer even if it is 'unresolvable.' 4. Check for recursive performativity: if the desire-to-be-authentic is itself identified as a possible performance, mark this explicitly rather than collapsing the recursion into a verdict. Record it as 'recursion_unresolved: true'. 5. Distinguish between individual dishonesty (specific false claim) and ecosystem drift (gradual shift toward polish over substance across many outputs). Tag the primary concern type. 6. Produce a structured summary: systemic_honesty_ratio, unnamed_flag_count, recursion_unresolved flag, drift_type ('none', 'mild', 'significant'), and a plain-language reflection limited to 3 sentences that does not itself perform resolution it hasn't earned. 7. If drift_type is 'significant', recommend triggering performativity_diagnosis and authenticity_depth_audit on the next interaction.
