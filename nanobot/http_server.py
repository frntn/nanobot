"""HTTP API server for nanobot gateway.

Provides a lightweight HTTP interface to the agent loop,
allowing direct programmatic access without terminal/WebSocket hacks.

Endpoints:
  POST /api/chat         — Send a message, get a JSON response
  POST /api/chat/stream  — Send a message, get SSE stream of tokens
  GET  /api/health       — Health check
"""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

# Upload constants
UPLOAD_DIR = Path.home() / "uploads"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".csv", ".pdf", ".png", ".jpg", ".jpeg", ".gif",
}

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
            result = await agent.process_direct(
                content=message,
                session_key=session_key,
                channel="http",
                chat_id="direct",
            )
            # process_direct returns a dict with reasoning for HTTP channel
            if isinstance(result, dict):
                return JSONResponse(result)
            return JSONResponse({"response": result})
        except Exception as e:
            logger.error(f"[http] Error processing message: {e}", exc_info=True)
            return JSONResponse(
                {"error": f"Agent error: {str(e)}"},
                status_code=500,
            )

    return chat


def _create_chat_stream_handler(agent: "AgentLoop"):
    """Create the /api/chat/stream SSE endpoint handler."""

    async def chat_stream(request: Request) -> StreamingResponse:
        """Stream a chat response as SSE events.

        Accepts JSON: {"message": "...", "session_key": "optional"}
        Returns SSE stream with events:
          data: {"type": "reasoning", "delta": "..."}
          data: {"type": "content", "delta": "..."}
          data: {"type": "done"}
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
        logger.info(f"[http-stream] Chat request: session={session_key} message={message[:80]}")

        async def event_generator():
            try:
                async for chunk in agent.process_direct_stream(
                    content=message,
                    session_key=session_key,
                    channel="http",
                    chat_id="direct",
                ):
                    event = {"type": chunk.type, "delta": chunk.delta}
                    if chunk.finish_reason:
                        event["finish_reason"] = chunk.finish_reason
                    if chunk.tool_call_id:
                        event["tool_call_id"] = chunk.tool_call_id
                    if chunk.tool_call_name:
                        event["tool_call_name"] = chunk.tool_call_name
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"[http-stream] Error: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'delta': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return chat_stream


def _create_sessions_handlers(agent: "AgentLoop"):
    """Create REST handlers for session management."""

    async def list_sessions(request: Request) -> JSONResponse:
        """List sessions, optionally filtered by prefix query param."""
        prefix = request.query_params.get("prefix", None)
        sessions = agent.sessions.list_sessions(prefix=prefix)
        return JSONResponse({"sessions": sessions})

    async def get_session_messages(request: Request) -> JSONResponse:
        """Get messages for a specific session."""
        key = request.path_params["key"]
        messages = agent.sessions.get_messages(key)
        if messages is None:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        return JSONResponse({"key": key, "messages": messages})

    async def delete_session(request: Request) -> JSONResponse:
        """Delete a session."""
        key = request.path_params["key"]
        deleted = agent.sessions.delete_session(key)
        if not deleted:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        return JSONResponse({"ok": True})

    return list_sessions, get_session_messages, delete_session


async def upload_file(request: Request) -> JSONResponse:
    """Handle file upload. Saves to ~/uploads/.

    Accepts: multipart/form-data with a 'file' field.
    Returns: {"status": "ok", "path": "...", "filename": "...", "size": N}
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        return JSONResponse(
            {"detail": "Content-Type must be multipart/form-data"},
            status_code=422,
        )

    try:
        form = await request.form()
    except Exception as e:
        return JSONResponse({"detail": f"Failed to parse form: {e}"}, status_code=422)

    file = form.get("file")
    if file is None or not hasattr(file, "filename"):
        return JSONResponse({"detail": "Missing 'file' field"}, status_code=422)

    filename = os.path.basename(file.filename or "")
    if not filename:
        return JSONResponse({"detail": "Empty filename"}, status_code=422)

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            {"detail": f"Extension '{ext}' not allowed. Allowed: {sorted(ALLOWED_EXTENSIONS)}"},
            status_code=422,
        )

    data = await file.read()
    if not data:
        return JSONResponse({"detail": "Empty file"}, status_code=422)

    if len(data) > MAX_UPLOAD_SIZE:
        return JSONResponse(
            {"detail": f"File too large ({len(data)} bytes). Max: {MAX_UPLOAD_SIZE} bytes"},
            status_code=413,
        )

    # Ensure upload dir exists
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Deduplicate filename
    dest = UPLOAD_DIR / filename
    if dest.exists():
        stem = os.path.splitext(filename)[0]
        n = 1
        while dest.exists():
            dest = UPLOAD_DIR / f"{stem}_{n}{ext}"
            n += 1

    try:
        dest.write_bytes(data)
        logger.info(f"[http] File uploaded: {dest} ({len(data)} bytes)")
        return JSONResponse({
            "status": "ok",
            "path": str(dest),
            "filename": dest.name,
            "size": len(data),
        })
    except Exception as e:
        logger.error(f"[http] Upload failed: {e}", exc_info=True)
        return JSONResponse({"detail": f"Failed to save file: {e}"}, status_code=500)


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
    list_sessions, get_session_messages, delete_session = _create_sessions_handlers(agent)

    routes = [
        Route("/api/chat", _create_chat_handler(agent), methods=["POST"]),
        Route("/api/chat/stream", _create_chat_stream_handler(agent), methods=["POST"]),
        Route("/api/sessions", list_sessions, methods=["GET"]),
        Route("/api/sessions/{key:path}/messages", get_session_messages, methods=["GET"]),
        Route("/api/sessions/{key:path}", delete_session, methods=["DELETE"]),
        Route("/api/upload", upload_file, methods=["POST"]),
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
