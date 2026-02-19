"""HTTP API server for nanobot gateway.

Provides a lightweight HTTP interface to the agent loop,
allowing direct programmatic access without terminal/WebSocket hacks.

Endpoints:
  POST /api/chat         — Send a message, get a JSON response
  POST /api/chat/stream  — Send a message, get SSE stream of tokens
  POST /api/config/env   — Inject a secret into os.environ (for secure credential input)
  GET  /api/files        — List files in workspace directory
  GET  /api/files/{path} — Download a file from workspace directory
  GET  /api/health       — Health check (unauthenticated — used by K8s probes)
"""

import json
import mimetypes
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

# ---------------------------------------------------------------------------
# Authentication middleware
# ---------------------------------------------------------------------------

# Endpoints that do NOT require authentication (K8s liveness/readiness probes)
_PUBLIC_PATHS = frozenset({"/api/health"})


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Validate ``Authorization: Bearer <secret>`` on all endpoints except health.

    If the env var ``NANOBOT_API_SECRET`` is unset or empty, the middleware is
    permissive (no-op) so that local development and tests keep working without
    having to set the variable.
    """

    async def dispatch(self, request: Request, call_next):
        expected_secret = os.environ.get("NANOBOT_API_SECRET", "")

        # If no secret is configured, skip authentication (dev/test mode)
        if not expected_secret:
            return await call_next(request)

        # Allow public paths through without auth
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Validate Authorization header
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401,
            )

        token = auth_header[7:]  # strip "Bearer "
        if token != expected_secret:
            return JSONResponse(
                {"error": "Invalid API secret"},
                status_code=401,
            )

        return await call_next(request)

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

    async def rename_session(request: Request) -> JSONResponse:
        """Rename a session (update its title)."""
        key = request.path_params["key"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        title = body.get("title")
        if not title or not isinstance(title, str):
            return JSONResponse({"error": "Missing or invalid 'title'"}, status_code=400)
        renamed = agent.sessions.rename_session(key, title.strip())
        if not renamed:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        return JSONResponse({"ok": True})

    return list_sessions, get_session_messages, delete_session, rename_session


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


# ---------------------------------------------------------------------------
# Workspace file browsing
# ---------------------------------------------------------------------------

WORKSPACE_DIR = Path.home() / ".nanobot" / "workspace"


async def list_workspace_files(request: Request) -> JSONResponse:
    """List files in the workspace directory.

    Returns JSON: {"files": [{"path": "relative/path", "name": "file.txt", "size": 1234, "is_dir": false}, ...]}
    """
    if not WORKSPACE_DIR.is_dir():
        return JSONResponse({"files": []})

    files = []
    for item in sorted(WORKSPACE_DIR.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(WORKSPACE_DIR)
            # Skip hidden files/directories
            if any(part.startswith(".") for part in rel_path.parts):
                continue
            try:
                size = item.stat().st_size
            except OSError:
                size = 0
            files.append({
                "path": str(rel_path),
                "name": item.name,
                "size": size,
                "is_dir": False,
            })

    return JSONResponse({"files": files})


async def download_workspace_file(request: Request) -> FileResponse | JSONResponse:
    """Download a file from the workspace directory.

    Security: strict path traversal protection — resolved path must be within WORKSPACE_DIR.
    """
    file_path = request.path_params.get("path", "")

    if not file_path:
        return JSONResponse({"error": "Missing file path"}, status_code=400)

    # Reject obvious traversal attempts early
    if ".." in file_path.split("/") or ".." in file_path.split("\\"):
        logger.warning(f"[http] Path traversal attempt blocked: {file_path}")
        return JSONResponse({"error": "Invalid path"}, status_code=400)

    # Resolve to absolute path and verify it's within workspace
    try:
        target = (WORKSPACE_DIR / file_path).resolve()
    except (ValueError, OSError):
        return JSONResponse({"error": "Invalid path"}, status_code=400)

    workspace_resolved = WORKSPACE_DIR.resolve()

    # Path traversal protection: ensure target is within workspace
    if not str(target).startswith(str(workspace_resolved) + os.sep) and target != workspace_resolved:
        logger.warning(f"[http] Path traversal blocked: {file_path} resolved to {target}")
        return JSONResponse({"error": "Access denied"}, status_code=403)

    if not target.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)

    # Determine Content-Type
    content_type, _ = mimetypes.guess_type(str(target))
    if not content_type:
        content_type = "application/octet-stream"

    logger.info(f"[http] File download: {file_path} ({target.stat().st_size} bytes)")

    return FileResponse(
        path=str(target),
        filename=target.name,
        media_type=content_type,
    )


async def config_env(request: Request) -> JSONResponse:
    """Inject a secret into the nanobot process environment.

    Accepts JSON: {"key": "ENV_VAR_NAME", "value": "secret_value"}
    Sets os.environ[key] = value so the nanobot can use it immediately.

    Security: The value is NEVER logged.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    key = body.get("key", "").strip()
    value = body.get("value")

    if not key:
        return JSONResponse({"error": "Missing or empty 'key' field"}, status_code=400)
    if value is None:
        return JSONResponse({"error": "Missing 'value' field"}, status_code=400)

    # Validate key format: only allow alphanumeric + underscore (env var convention)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return JSONResponse(
            {"error": "Invalid key format. Use only A-Z, a-z, 0-9, underscore. Must start with a letter or underscore."},
            status_code=400,
        )

    os.environ[key] = str(value)
    logger.info(f"[http] Environment variable set: {key}=***")

    return JSONResponse({"status": "ok", "key": key})


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
    list_sessions, get_session_messages, delete_session, rename_session = _create_sessions_handlers(agent)

    routes = [
        Route("/api/chat", _create_chat_handler(agent), methods=["POST"]),
        Route("/api/chat/stream", _create_chat_stream_handler(agent), methods=["POST"]),
        Route("/api/sessions", list_sessions, methods=["GET"]),
        Route("/api/sessions/{key:path}/messages", get_session_messages, methods=["GET"]),
        Route("/api/sessions/{key:path}", delete_session, methods=["DELETE"]),
        Route("/api/sessions/{key:path}", rename_session, methods=["PATCH"]),
        Route("/api/upload", upload_file, methods=["POST"]),
        Route("/api/files", list_workspace_files, methods=["GET"]),
        Route("/api/files/{path:path}", download_workspace_file, methods=["GET"]),
        Route("/api/config/env", config_env, methods=["POST"]),
        Route("/api/health", health, methods=["GET"]),
    ]
    app = Starlette(
        routes=routes,
        middleware=[Middleware(BearerAuthMiddleware)],
    )
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
