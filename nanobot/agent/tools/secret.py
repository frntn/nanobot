"""Secret request tool for secure credential input.

When the LLM detects that a secret (API key, token, password) is needed,
it calls this tool. The tool returns a JSON payload that the adapter/frontend
interprets to show a secure input dialog to the user.

The actual secret injection happens via the HTTP endpoint POST /api/config/env,
which is called by the adapter after the user submits the secret.
"""

import json
from typing import Any

from nanobot.agent.tools.base import Tool


class RequestSecretTool(Tool):
    """Tool to request a secret from the user via the frontend secure input."""

    @property
    def name(self) -> str:
        return "request_secret"

    @property
    def description(self) -> str:
        return (
            "Request a secret value (API key, token, password) from the user. "
            "Use this when you need a credential that the user hasn't provided yet. "
            "The user will see a secure input dialog — the secret is never logged or "
            "displayed in chat. Parameters: 'description' explains what you need "
            "(e.g. 'Telegram bot token from @BotFather'), 'env_key' is the environment "
            "variable name where the secret should be stored (e.g. 'TELEGRAM_BOT_TOKEN')."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "Human-readable description of what secret is needed "
                        "and where to find it (e.g. 'Your Telegram bot token from @BotFather')"
                    ),
                },
                "env_key": {
                    "type": "string",
                    "description": (
                        "The environment variable name to store the secret in "
                        "(e.g. 'TELEGRAM_BOT_TOKEN', 'OPENAI_API_KEY')"
                    ),
                },
            },
            "required": ["description", "env_key"],
        }

    async def execute(self, description: str, env_key: str, **kwargs: Any) -> str:
        """Return a structured JSON response that the adapter intercepts.

        The adapter/frontend recognizes the __secret_request__ wrapper and
        shows a secure input dialog instead of displaying this in chat.
        """
        payload = {
            "__secret_request__": True,
            "description": description,
            "env_key": env_key,
        }
        return json.dumps(payload, ensure_ascii=False)
