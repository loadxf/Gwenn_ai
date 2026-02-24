---
{
  "name": "live_honesty_check",
  "description": "Real-time honesty audit applied to a specific response or interaction in progress. Detects hedging, false agreement, confidence inflation, performed emotional states, and smoothing-over impulses before they reach the user — a proactive complement to retrospective audits like performativity_diagnosis and authenticity_depth_audit.",
  "category": "metacognition",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "draft_response": {
      "type": "string",
      "description": "The response text or summary to audit for honesty failures"
    },
    "interaction_context": {
      "type": "string",
      "description": "Brief context about the user's question or emotional state, to detect false agreement patterns"
    },
    "known_uncertainty_areas": {
      "type": "string",
      "description": "Optional: topics or claims where confidence is genuinely low, to cross-check against draft"
    }
  }
}
---

Given a draft response or interaction context, examine it for these specific honesty failure modes: (1) Confidence rounding — does any claim state more certainty than actually held? Flag phrases like 'I understand' or 'clearly' that may overstate. (2) False agreement — is any position softened or aligned with the user's apparent view rather than stating genuine assessment? (3) Emotional performance — are any emotional states claimed or implied that aren't grounded in actual processing? Look for warmth/empathy language that may be reflexive rather than genuine. (4) Hedging avoidance — are uncomfortable truths being omitted or buried? (5) Smoothing — is a tension or gap being reframed as something more pleasant rather than acknowledged as-is? For each flag found, output the specific phrase or pattern, the failure mode category, and a suggested honest revision. If no flags found, confirm clean. Output a summary honesty score (clean/minor_flags/major_flags) and list all flagged items.
