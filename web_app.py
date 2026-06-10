"""
Local web UI for Coach.

Run:
  python web_app.py
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import json
import mimetypes
import os
import signal
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_agent_module = None
_runtime = None
_ollama_process = None
_runtime_lock = threading.Lock()
_chat_lock = threading.Lock()
_ollama_lock = threading.Lock()


def _json_bytes(payload: dict, status: int = 200) -> tuple[int, bytes]:
    return status, json.dumps(payload).encode("utf-8")


def _load_agent_module():
    global _agent_module
    if _agent_module is None:
        _agent_module = importlib.import_module("agent")
    return _agent_module


def _ollama_binary_path() -> str | None:
    configured_path = os.getenv("OLLAMA_BIN")
    candidates = [
        configured_path,
        "/Applications/Ollama.app/Contents/Resources/ollama",
        shutil.which("ollama"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _ollama_model_names(timeout: float = 2.0) -> set[str]:
    agent = _load_agent_module()
    request = agent.Request(agent._ollama_api_url("/api/tags"), method="GET")
    with agent.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {model.get("name", "") for model in payload.get("models", [])}


def _ollama_running() -> bool:
    try:
        _ollama_model_names(timeout=1.0)
    except Exception:
        return False
    return True


def _agent_setup_status() -> dict:
    try:
        agent = _load_agent_module()
    except Exception as exc:
        return {
            "ok": False,
            "stage": "python",
            "error": str(exc),
            "hint": "Run pip install -r requirements.txt, then restart the UI.",
        }

    installed_models: set[str] = set()
    ollama_error = None
    try:
        installed_models = _ollama_model_names()
    except Exception as exc:
        ollama_error = str(exc)

    required_models = [agent.MODEL_NAME, agent.EMBEDDING_MODEL_NAME]
    missing_models = [
        model
        for model in required_models
        if ollama_error is None
        and not agent.is_ollama_model_installed(model, installed_models)
    ]
    ollama_running = ollama_error is None

    if ollama_running and not missing_models:
        stage = "ready"
    elif not ollama_running:
        stage = "stopped"
    else:
        stage = "ollama"

    return {
        "ok": stage == "ready",
        "stage": stage,
        "error": None if stage == "stopped" else ollama_error,
        "ollama_running": ollama_running,
        "ollama_binary": _ollama_binary_path(),
        "missing_models": missing_models,
        "pull_commands": [f"ollama pull {model}" for model in missing_models],
        "model": agent.MODEL_NAME,
        "embedding_model": agent.EMBEDDING_MODEL_NAME,
        "base_url": agent.OLLAMA_BASE_URL,
        "thread_id": agent.THREAD_ID,
    }


def _ensure_runtime():
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            agent = _load_agent_module()
            agent.ensure_local_models_available()
            agent.init_db()
            _runtime = agent.MentorRuntime()
            _runtime.__enter__()
        return _load_agent_module(), _runtime


def _close_runtime() -> None:
    global _runtime
    if _runtime is not None:
        _runtime.__exit__(None, None, None)
        _runtime = None


def _ollama_listener_pids() -> list[int]:
    result = subprocess.run(
        ["lsof", "-tiTCP:11434", "-sTCP:LISTEN"],
        capture_output=True,
        check=False,
        text=True,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            continue
    return pids


def _start_ollama_server() -> dict:
    global _ollama_process
    with _ollama_lock:
        if _ollama_running():
            return {"started": False, "status": _agent_setup_status()}

        binary_path = _ollama_binary_path()
        if binary_path is None:
            raise RuntimeError("Ollama binary was not found.")

        _ollama_process = subprocess.Popen(
            [binary_path, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if _ollama_running():
                return {"started": True, "status": _agent_setup_status()}
            if _ollama_process.poll() is not None:
                break
            time.sleep(0.5)

        raise RuntimeError("Ollama did not start within 20 seconds.")


def _stop_ollama_server() -> dict:
    global _ollama_process
    with _ollama_lock:
        _close_runtime()

        pids = _ollama_listener_pids()
        if _ollama_process is not None and _ollama_process.poll() is None:
            pids.append(_ollama_process.pid)

        for pid in sorted(set(pids)):
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _ollama_running():
            time.sleep(0.25)

        for pid in sorted(set(pids)):
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue

        _ollama_process = None
        return {"stopped": True, "status": _agent_setup_status()}


atexit.register(_close_runtime)


class CoachHandler(BaseHTTPRequestHandler):
    server_version = "CoachUI/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._send_json(_agent_setup_status())
            return
        if parsed.path == "/api/history":
            self._handle_history(parsed.query)
            return
        if parsed.path == "/api/memories":
            self._handle_memories(parsed.query)
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            self._handle_chat()
            return
        if parsed.path == "/api/ollama/start":
            self._handle_ollama_start()
            return
        if parsed.path == "/api/ollama/stop":
            self._handle_ollama_stop()
            return
        self.send_error(404, "Not found")

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/memories/"):
            self._handle_memory_update(parsed.path)
            return
        self.send_error(404, "Not found")

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/memories/"):
            self._handle_memory_delete(parsed.path)
            return
        self.send_error(404, "Not found")

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8"))

    def _send_json(self, payload: dict, status: int = 200) -> None:
        response = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def _memory_id_from_path(self, request_path: str) -> str:
        memory_id = unquote(request_path.removeprefix("/api/memories/")).strip()
        if not memory_id or "/" in memory_id:
            raise ValueError("Invalid memory id.")
        return memory_id

    def _handle_chat(self) -> None:
        try:
            body = self._read_json()
            message = str(body.get("message", "")).strip()
            thread_id = str(body.get("thread_id", "")).strip() or None
            if not message:
                self._send_json({"error": "Message is required."}, status=400)
                return

            agent, runtime = _ensure_runtime()
            with _chat_lock:
                reply = runtime.reply(message, thread_id=thread_id)
                history = runtime.history(thread_id=thread_id)
            self._send_json(
                {
                    "reply": reply,
                    "history": history,
                    "memories": agent.list_recent_memories(),
                }
            )
        except Exception as exc:
            self._send_json({"error": str(exc), "status": _agent_setup_status()}, 500)

    def _handle_history(self, query: str) -> None:
        try:
            params = parse_qs(query)
            thread_id = (params.get("thread_id") or [""])[0].strip() or None
            _agent, runtime = _ensure_runtime()
            with _chat_lock:
                history = runtime.history(thread_id=thread_id)
            self._send_json({"history": history})
        except Exception as exc:
            self._send_json({"error": str(exc), "status": _agent_setup_status()}, 500)

    def _handle_memories(self, query: str) -> None:
        try:
            params = parse_qs(query)
            limit = int((params.get("limit") or ["12"])[0])
            agent, _runtime = _ensure_runtime()
            self._send_json({"memories": agent.list_recent_memories(limit=limit)})
        except Exception as exc:
            self._send_json({"error": str(exc), "status": _agent_setup_status()}, 500)

    def _handle_memory_update(self, request_path: str) -> None:
        try:
            memory_id = self._memory_id_from_path(request_path)
            body = self._read_json()
            fact = str(body.get("fact", "")).strip()
            agent, _runtime = _ensure_runtime()
            with _chat_lock:
                memory = agent.update_memory(memory_id, fact)
                memories = agent.list_recent_memories()
            self._send_json({"memory": memory, "memories": memories})
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, 404)
        except Exception as exc:
            self._send_json({"error": str(exc), "status": _agent_setup_status()}, 500)

    def _handle_memory_delete(self, request_path: str) -> None:
        try:
            memory_id = self._memory_id_from_path(request_path)
            agent, _runtime = _ensure_runtime()
            with _chat_lock:
                deleted = agent.delete_memory(memory_id)
                memories = agent.list_recent_memories()
            if not deleted:
                self._send_json({"error": f"Memory not found: {memory_id}"}, 404)
                return
            self._send_json({"deleted": True, "memories": memories})
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self._send_json({"error": str(exc), "status": _agent_setup_status()}, 500)

    def _handle_ollama_start(self) -> None:
        try:
            self._send_json(_start_ollama_server())
        except Exception as exc:
            self._send_json({"error": str(exc), "status": _agent_setup_status()}, 500)

    def _handle_ollama_stop(self) -> None:
        try:
            self._send_json(_stop_ollama_server())
        except Exception as exc:
            self._send_json({"error": str(exc), "status": _agent_setup_status()}, 500)

    def _serve_static(self, request_path: str) -> None:
        relative_path = request_path.lstrip("/") or "index.html"
        static_path = (WEB_DIR / relative_path).resolve()
        web_root = WEB_DIR.resolve()

        if web_root not in static_path.parents and static_path != web_root:
            self.send_error(403, "Forbidden")
            return
        if static_path.is_dir():
            static_path = static_path / "index.html"
        if not static_path.exists():
            self.send_error(404, "Not found")
            return

        content_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
        content = static_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Coach local web UI")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), CoachHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Coach UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Coach UI.")
    finally:
        _close_runtime()
        server.server_close()


if __name__ == "__main__":
    main()
