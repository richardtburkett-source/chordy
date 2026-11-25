"""
Chordy web server
===========================

Serves the web page and hands out LiveKit room tokens. Each token carries the
session setup and asks LiveKit to send the "chordy" agent into the room.

    uv run server.py            # then open http://localhost:8080

Keep `uv run agent.py dev` running in another window.
"""

from __future__ import annotations

import json
import os
import secrets
import time
import webbrowser
from datetime import timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv
from livekit import api
from livekit.protocol.agent_dispatch import RoomAgentDispatch
from livekit.protocol.room import RoomConfiguration

load_dotenv(".env")

AGENT_NAME = "chordy"
PORT = int(os.getenv("PORT", "8080"))
HOST = os.getenv("HOST", "127.0.0.1")         # set HOST=0.0.0.0 when hosting online
ACCESS_CODE = os.getenv("ACCESS_CODE", "").strip()  # set online so strangers can't spend your minutes
WEB_DIR = Path(__file__).parent / "web"
FIELDS = ("name",)


def make_session(setup: dict) -> dict:
    if ACCESS_CODE and not secrets.compare_digest(str(setup.get("code", "")).strip(), ACCESS_CODE):
        raise PermissionError("That access code isn't right.")
    url = os.getenv("LIVEKIT_URL", "").strip()
    key = os.getenv("LIVEKIT_API_KEY", "").strip()
    secret = os.getenv("LIVEKIT_API_SECRET", "").strip()
    if not (url and key and secret):
        raise ValueError("LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in .env")

    clean = {k: str(setup.get(k, ""))[: (4000 if k == "description" else 120)] for k in FIELDS}
    room = f"chordy-{time.strftime('%H%M%S')}-{secrets.token_hex(2)}"
    token = (
        api.AccessToken(key, secret)
        .with_identity(f"musician-{secrets.token_hex(3)}")
        .with_name(clean["name"] or "Musician")
        .with_ttl(timedelta(hours=1))
        .with_grants(api.VideoGrants(room_join=True, room=room, can_publish_data=True))
        .with_room_config(
            RoomConfiguration(
                agents=[RoomAgentDispatch(agent_name=AGENT_NAME, metadata=json.dumps(clean))]
            )
        )
        .to_jwt()
    )
    return {"url": url, "token": token, "room": room}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/config":
            data = json.dumps({"needsCode": bool(ACCESS_CODE)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/session":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            setup = json.loads(self.rfile.read(n) or b"{}") if n else {}
            body, status = make_session(setup if isinstance(setup, dict) else {}), 200
        except PermissionError as e:
            body, status = {"error": str(e)}, 403
        except Exception as e:
            body, status = {"error": str(e)}, 500
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self) -> None:
        if self.command == "GET":
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args) -> None:
        if "/api/" in str(args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    link = f"http://localhost:{PORT}"
    print(f"\nChordy is running at {link}")
    print("Keep `uv run agent.py dev` running in another window. Ctrl + C to stop.\n")
    if HOST == "127.0.0.1" and not os.getenv("NO_BROWSER"):
        webbrowser.open(link)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
