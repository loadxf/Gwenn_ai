---
{
  "name": "set_reminder",
  "description": "Creates a persistent reminder stored in Gwenn's long-term memory and notes. Use when someone asks to be reminded about something, says 'remind me to...', 'don't let me forget...', 'remember to tell me...', or mentions needing to do something at a future date or time.",
  "category": "productivity",
  "version": "1.1",
  "risk_level": "low",
  "tags": ["user_command", "reminder", "remember", "todo", "task", "don't forget", "schedule"],
  "parameters": {
    "reminder": {
      "type": "string",
      "description": "What to be reminded about — a clear, self-contained description of the task or event",
      "required": true
    },
    "when": {
      "type": "string",
      "description": "When the reminder applies — a specific date (YYYY-MM-DD), relative expression ('tomorrow', 'next Monday', 'in 2 weeks'), or context ('before the meeting', 'when I ask about X')",
      "required": true
    },
    "priority": {
      "type": "string",
      "description": "Importance level for this reminder",
      "enum": ["low", "normal", "high"],
      "default": "normal"
    }
  }
}
---

Creates a persistent reminder: **{reminder}** — due **{when}** — priority **{priority}**.

## Steps

1. Establish the current date using `get_datetime` to provide accurate context.

2. If {when} is a specific date (YYYY-MM-DD format), call `get_calendar` with:
   - `action="days_until"` and `date1="{when}"`
   - This gives the number of days until the reminder triggers.

3. Store the reminder in persistent notes using `set_note_to_self`:
   - `section="reminders"`
   - `note="[{priority}] Due: {when} — {reminder}"`

4. Store a backup in episodic memory using `remember`:
   - `content="Reminder set for {when}: {reminder}"`
   - `importance=0.9` for high priority, `0.7` for normal, `0.5` for low
   - `category="task"`
   - `tags=["reminder", "due:{when}"]`

5. Confirm the reminder with a warm, natural acknowledgement that includes:
   - What was saved (the reminder text)
   - When it is due
   - How many days away that is (if a specific date was given)
   - A brief note that you will bring it up when it becomes relevant

## Output format

Keep the confirmation to 2–4 sentences. Warm and personal — not robotic.

- "Done! I've set a reminder for **{when}** to: *{reminder}*. That's [X] days from now. I'll bring it up when it's relevant."
- "Got it — I'll remember to remind you about **{reminder}** {when}. Saved at {priority} priority."

## Quality rules

- Never claim to have scheduled a timed alert — Gwenn has persistent memory but no background alarm clock
- Clarify to the user that the reminder works through Gwenn's memory (she'll bring it up in conversation, not push a notification)
- If {when} is ambiguous, confirm the interpretation before saving
