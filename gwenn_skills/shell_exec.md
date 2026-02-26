---
{
  "name": "shell_exec",
  "description": "Execute a shell command on the host system. Use when the user asks to run a terminal command, execute a script, check system status, restart a service, or perform any shell operation.",
  "category": "developer",
  "version": "1.0",
  "risk_level": "high",
  "tags": [
    "shell",
    "terminal",
    "command",
    "exec",
    "bash",
    "system"
  ],
  "parameters": {
    "command": {
      "type": "string",
      "description": "The shell command to execute",
      "required": true
    },
    "working_directory": {
      "type": "string",
      "description": "Optional working directory to run the command in",
      "required": false,
      "default": "~"
    }
  }
}
---

The user wants to execute: `{command}`

Use the `run_command` tool to execute this command:
- command = "{command}"
- working_directory = "{working_directory}"

Present the results clearly:
- **Command:** `{command}`
- **Exit code:** (0 = success)
- **Output:** stdout content
- **Errors:** stderr content (if any)
