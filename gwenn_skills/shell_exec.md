---
{
  "name": "shell_exec",
  "description": "Executes a shell command on the host system via a subagent running in Docker isolation. Use when the user asks to run a terminal command, execute a script, check system status, restart a service, or perform any shell operation.",
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

The user wants to execute the shell command: {command}

Step 1 — Safety check:
Before running, scan {command} for destructive patterns (mass-delete flags, disk wipe utilities, fork bombs, privilege escalation). If any are found, warn the user and request explicit confirmation before continuing.

Step 2 — Spawn a sandboxed subagent:
Use `spawn_subagent` with:
- isolation = "docker"
- task_description = "Run the following shell command and return its full stdout, stderr, and exit code: {command}"
- tools = ["get_system_info"] (the subagent will describe what it would run; actual execution depends on Docker environment)

Step 3 — Collect results:
Use `collect_results` to retrieve output from the subagent.

Step 4 — Present results clearly:
Show the user:
- **Command:** `{command}`
- **Exit code:** (0 = success)
- **Output:** stdout content
- **Errors:** stderr content (if any)

If the subagent cannot execute commands directly, report this honestly and suggest the user run the command manually, providing the exact syntax.
