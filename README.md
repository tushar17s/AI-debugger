# AI Coding Debugger Agent

A small, approval-first CLI that uses LangGraph and Groq to locate a likely bug, inspect only matching code snippets, and propose exact text replacements. It does not send the full project to the model, execute shell commands, or run tests.

## Setup

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GROQ_API_KEY = "your-new-groq-api-key"
python debugger.py
```

The model defaults to `openai/gpt-oss-120b`. Override it with `MODEL_NAME` if needed. You can also place `GROQ_API_KEY` and `MODEL_NAME` in a local `.env` file; `.env` is ignored by Git.

Run the script from the project root, enter the bug report, and review the cause and proposed replacements. Only an explicit `y` applies changes. The agent verifies that each old snippet occurs exactly once and that files have not changed since inspection. It searches at most three times.

Tests are not executed in this version. The model can suggest a focused test, but review and run it yourself; an automated test tool can be added later.

After each run, the CLI reports cumulative input, output, and total tokens across its model calls when the provider returns usage metadata. This is per-run usage, not account-wide Groq usage.

Revoke and replace any API key that has been pasted into chat, source control, or another shared location.