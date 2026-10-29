# SPDX-License-Identifier: GPL-3.0-or-later
"""What the two browser tools (wall editor, camera editor) share: a small local JSON web server.

Each tool serves one HTML page (``tools/web/*.html``) and a few JSON routes, on 127.0.0.1 only, with Python's
standard library.  A tool defines its routes in a :class:`JsonHandler` subclass and calls :func:`serve`.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

WEB = pathlib.Path(__file__).parent / "web"

_ONE_THREAD = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def single_blas_thread(module: str):
    """Restart the current process once with one BLAS thread.

    numpy's BLAS starts an idle thread per CPU core when it is imported, and the tools compute next to nothing; the
    variables only take effect before numpy is loaded, hence the restart (``python -m module`` + the same args)."""
    if any(os.environ.get(k) != v for k, v in _ONE_THREAD.items()):
        os.execve(sys.executable, [sys.executable, "-m", module, *sys.argv[1:]], {**os.environ, **_ONE_THREAD})


class JsonHandler(BaseHTTPRequestHandler):
    """Serves ``page`` at ``/`` and dispatches ``/api/<name>`` to the methods ``get_<name>(query)`` (GET, query
    string as a dict) and ``post_<name>(body)`` (POST, JSON body).  A method returns what is sent back as JSON; an
    exception is sent as ``{"error": message}`` with status 400, for the page to show."""

    page: pathlib.Path = None       # set by the subclass
    state: dict = None              # set by serve(): shared by all requests

    def log_message(self, *args):
        """Quiet: no line per request in the terminal."""

    def _send(self, code: int, body, ctype: str = "application/json"):
        """Send ``body`` (bytes as they are, anything else as JSON) with status ``code``."""
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self, prefix: str, arg):
        """Answer the request with the page, or with the method ``<prefix>_<name>`` of the route ``/api/<name>``."""
        route = urlparse(self.path).path
        if prefix == "get" and route in ("/", "/index.html"):
            return self._send(200, self.page.read_bytes(), "text/html; charset=utf-8")
        method = getattr(self, f"{prefix}_{route[len('/api/'):]}", None) if route.startswith("/api/") else None
        if method is None:
            return self._send(404, {"error": f"no route {route}"})
        try:
            self._send(200, method(arg))
        except KeyError as e:
            self._send(400, {"error": f"missing field {e}"})
        except Exception as e:      # noqa: BLE001 -- reported in the page
            self._send(400, {"error": str(e)})

    def do_GET(self):
        """GET: the page, or ``get_<name>(query)``."""
        self._dispatch("get", {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()})

    def do_POST(self):
        """POST: ``post_<name>(body)``."""
        n = int(self.headers.get("Content-Length", 0))
        self._dispatch("post", json.loads(self.rfile.read(n) or b"{}"))

    def get_ping(self, query):
        """The page polls it to tell whether the server still runs."""
        return {"ok": True}


def serve(handler: type[JsonHandler], state: dict, name: str, port: int, open_browser: bool = True,
          threaded: bool = True):
    """Serve ``handler`` on 127.0.0.1 (the first free port from ``port``), open it in the browser, run until
    Ctrl+C.  ``threaded=False`` answers one request at a time, in this thread (needed when a request renders with
    MuJoCo: its OpenGL context belongs to the thread that made it)."""
    handler.state = state
    server_class = ThreadingHTTPServer if threaded else HTTPServer
    for p in range(port, port + 20):
        try:
            server = server_class(("127.0.0.1", p), handler)
            break
        except OSError:
            continue
    else:
        sys.exit(f"no free port in {port}-{port + 19}")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"{name}: {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
