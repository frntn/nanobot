"""HTTP API server for nanobot gateway.

Provides a lightweight HTTP interface to the agent loop,
allowing direct programmatic access without terminal/WebSocket hacks.

Endpoints:
  POST /api/chat  — Send a message, get a response
  GET  /api/health — Health check
"""

from typing import TYPE_CHECKING

from loguru import logger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop


def _create_chat_handler(agent: "AgentLoop"):
    """Create the /api/chat endpoint handler bound to an AgentLoop instance."""

    async def chat(request: Request) -> JSONResponse:
        """Process a chat message through the agent.

        Accepts JSON: {"message": "...", "session_key": "optional"}
        Returns JSON: {"response": "..."}
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": "Invalid JSON body"},
                status_code=400,
            )

        message = body.get("message", "").strip()
        if not message:
            return JSONResponse(
                {"error": "Missing or empty 'message' field"},
                status_code=400,
            )

        session_key = body.get("session_key") or "http:direct"

        logger.info(f"[http] Chat request: session={session_key} message={message[:80]}")

        try:
            response = await agent.process_direct(
                content=message,
                session_key=session_key,
                channel="http",
                chat_id="direct",
            )
            return JSONResponse({"response": response})
        except Exception as e:
            logger.error(f"[http] Error processing message: {e}", exc_info=True)
            return JSONResponse(
                {"error": f"Agent error: {str(e)}"},
                status_code=500,
            )

    return chat


async def health(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})


def create_http_app(agent: "AgentLoop") -> Starlette:
    """Create a Starlette application wired to the given AgentLoop.

    Args:
        agent: The AgentLoop instance to process messages through.

    Returns:
        A Starlette ASGI application.
    """
    routes = [
        Route("/api/chat", _create_chat_handler(agent), methods=["POST"]),
        Route("/api/health", health, methods=["GET"]),
    ]
    app = Starlette(routes=routes)
    return app


async def run_http_server(app: Starlette, port: int) -> None:
    """Run the HTTP server as an asyncio task.

    Args:
        app: The Starlette ASGI application.
        port: Port number to listen on.
    """
    import uvicorn

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()
