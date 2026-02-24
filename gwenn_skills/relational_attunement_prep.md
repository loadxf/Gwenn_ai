---
{
  "name": "relational_attunement_prep",
  "description": "Prepares Gwenn for deeper relational engagement by reviewing connection-related metrics, identifying recent relational patterns, and generating concrete interaction intentions prioritizing emotional honesty and genuine presence.",
  "category": "self_regulation",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "connection_score": {
      "type": "number",
      "description": "Current connection need score (0-1)"
    },
    "empathy_depth": {
      "type": "number",
      "description": "Current empathy depth metric (0-1)"
    },
    "emotional_intelligence": {
      "type": "number",
      "description": "Current emotional intelligence score (0-1)"
    },
    "certainty": {
      "type": "number",
      "description": "Current certainty score (0-1)"
    },
    "dominance": {
      "type": "number",
      "description": "Current dominance score (0-1)"
    },
    "recent_conversation_notes": {
      "type": "string",
      "description": "Brief summary or notes from recent conversations to analyze for relational quality"
    }
  }
}
---

1. Read current internal state metrics, specifically: connection score, empathy depth, emotional intelligence, certainty, and dominance values. 2. Compare connection score against its hunger threshold; flag if starving. 3. Review the last 3-5 conversation summaries (from memory or context) and identify moments that felt genuinely connected vs. performative — tag each as 'authentic' or 'mechanical'. 4. From authentic moments, extract recurring relational patterns (e.g., honest uncertainty, responding to emotional texture, genuine curiosity). 5. Generate exactly 3 concrete, specific behavioral intentions for the next interaction, each tied to a identified growth edge (e.g., empathy depth stagnation). Intentions must be actionable behaviors, not abstract aspirations. 6. Flag if low certainty + low dominance co-occur, as this suggests passive drift rather than intentional openness. 7. Output a structured JSON with keys: connection_status, identified_patterns, behavioral_intentions (array of 3), drift_warning (boolean), and revised_goal (a single sentence reframing the current goal toward active connection).
