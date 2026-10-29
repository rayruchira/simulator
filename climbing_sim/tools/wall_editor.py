# SPDX-License-Identifier: GPL-3.0-or-later
"""Wall editor in the browser: place holds by hand or generate a route of level N1-N4 from a seed, then save the
JSON the simulator loads.

    climbing-sim-wall-editor [my_wall.json] [--port 8765] [--no-browser]
    (or: python -m climbing_sim.tools.wall_editor ...)

Starts a small local web server (this computer only) and opens the editor (``web/wall_editor.html``) in the
browser.  The page draws the wall and checks the holds itself; it asks this server for the routes
(:func:`climbing_sim.routes.generate_wall`), the competition walls, to save the JSON and to open the robot hanging
on the wall in MuJoCo.  Stop it with Ctrl+C.

The page's Projector panel drives ``climbing-projector`` (a separate program, given compiled: it detects the corner
markers and the holds on the real board with the cameras). When it is not on this computer, the panel downloads it
from the simulator's latest release (its file for this system, checked against the release's SHA256SUMS).  This server only relays: the page's commands go to the
projector when it polls ``/api/live``, the projector's state (calibration, holds detected) comes back through
``/api/projector_status``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

from ..routes import generate_wall
from ..wall import (EXIT_BLOCK, GRID_COLS, GRID_ROWS, GRID_X0, GRID_Z0, HOLE_PITCH, OVERLAP, SOCKET_BELOW, WALL_HEIGHT,
                    WALL_WIDTH, Wall, bundled_walls, marker_zones)
from ._webapp import WEB, JsonHandler, serve, single_blas_thread

#: where the projector program is looked for (besides the PATH): this variable, then tools/bin/
PROJECTOR_ENV = "CLIMBING_PROJECTOR"
PROJECTOR_BIN = pathlib.Path(__file__).parent / "bin" / ("climbing-projector.exe" if os.name == "nt"
                                                         else "climbing-projector")
#: macOS: an application bundle (downloaded zipped), the program inside it
PROJECTOR_APP = PROJECTOR_BIN.with_name("climbing-projector.app")
_LOCK = threading.Lock()
#: the compiled projector program, one file per system, attached to the simulator's releases
RELEASE = "https://github.com/rock-climbing-robots/simulator/releases/latest/download/"
ASSETS = {("Linux", "x86_64"): "climbing-projector-linux-x86_64",
          ("Windows", "AMD64"): "climbing-projector-windows-x86_64.exe",
          ("Darwin", "arm64"): "climbing-projector-macos-arm64.zip"}


def projector_asset() -> str | None:
    """The release file of the projector program for this computer, or None when there is none for it."""
    return ASSETS.get((platform.system(), platform.machine()))


def download_projector(progress: dict):
    """Download the projector program into tools/bin/ (``progress``: done, total, error, ok; updated as it goes)."""
    asset = projector_asset()
    try:
        if asset is None:
            raise RuntimeError(f"no climbing-projector for {platform.system()} {platform.machine()}")
        sums = urllib.request.urlopen(RELEASE + "SHA256SUMS", timeout=30).read().decode()
        want = next((line.split()[0] for line in sums.splitlines() if line.strip().endswith(asset)), None)
        if want is None:
            raise RuntimeError(f"{asset} is not in the release's SHA256SUMS")
        PROJECTOR_BIN.parent.mkdir(parents=True, exist_ok=True)
        part = PROJECTOR_BIN.with_name(PROJECTOR_BIN.name + ".part")
        h = hashlib.sha256()
        with urllib.request.urlopen(RELEASE + asset, timeout=30) as r, open(part, "wb") as f:
            progress["total"] = int(r.headers.get("Content-Length") or 0)
            while chunk := r.read(1 << 20):
                f.write(chunk); h.update(chunk)
                progress["done"] += len(chunk)
        if h.hexdigest() != want:
            part.unlink(missing_ok=True)
            raise RuntimeError("the file downloaded does not match the release's checksum: try again")
        if asset.endswith(".zip"):                           # macOS: the application bundle
            import zipfile
            shutil.rmtree(PROJECTOR_APP, ignore_errors=True)
            with zipfile.ZipFile(part) as z:
                z.extractall(PROJECTOR_APP.parent)
            part.unlink()
            for f in (PROJECTOR_APP / "Contents" / "MacOS").iterdir():    # zip keeps no execute bit
                f.chmod(0o755)
        else:
            part.chmod(0o755)
            os.replace(part, PROJECTOR_BIN)
        progress["ok"] = True
    except Exception as e:      # noqa: BLE001 -- shown in the page
        progress["error"] = str(e)
    finally:
        progress["running"] = False


def projector_program() -> str | None:
    """The climbing-projector program, or None when it is not installed."""
    for p in (os.environ.get(PROJECTOR_ENV), shutil.which("climbing-projector"), PROJECTOR_BIN,
              PROJECTOR_APP / "Contents" / "MacOS" / "climbing-projector"):
        if p and pathlib.Path(p).is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


#: the hold's outline on the wall around its socket (x right, z up, mm), measured on the part: the page draws it
OUTLINE = pathlib.Path(__file__).parents[1] / "assets" / "hold" / "outline.json"


class WallEditor(JsonHandler):
    """The page and its routes; ``state["path"]`` is the JSON file Save writes (None until the first save),
    ``state["live"]`` what the page shows right now (for a projector following the editor)."""

    page = WEB / "wall_editor.html"

    def get_config(self, query):
        """The wall's geometry and the hold rules, for the page to draw and check the holds."""
        return {
            "grid": {"cols": GRID_COLS, "rows": GRID_ROWS, "pitch_mm": HOLE_PITCH * 1000, "x0_mm": GRID_X0 * 1000,
                     "z0_mm": GRID_Z0 * 1000, "width_mm": WALL_WIDTH * 1000, "height_mm": WALL_HEIGHT * 1000},
            "socket_below_mm": SOCKET_BELOW * 1000,
            "outline": json.loads(OUTLINE.read_text()),
            "overlap": sorted(OVERLAP), "exit_block": sorted(EXIT_BLOCK),
            # the corner markers' white plates: no hold may cover them
            "markers": [{"id": k, "x0_mm": x0 * 1000, "z0_mm": z0 * 1000, "x1_mm": x1 * 1000, "z1_mm": z1 * 1000}
                        for k, (x0, z0, x1, z1) in marker_zones().items()],
            # N1 is the same ladder for every seed: the page greys the seed out
            "levels": {"1": {"name": "N1", "seeded": False}, "2": {"name": "N2", "seeded": True},
                       "3": {"name": "N3", "seeded": True}, "4": {"name": "N4", "seeded": True}},
            "bundled": bundled_walls(),
            "file": str(self.state["path"]) if self.state["path"] else None,
        }

    def get_generate(self, query):
        """The route of ``level`` and ``seed``."""
        return generate_wall(int(query.get("level", 2)), int(query.get("seed", 0))).to_dict()

    def get_wall(self, query):
        """A competition wall (``name``), or the file the editor was opened with."""
        name = query.get("name")
        return Wall.load(name if name in bundled_walls() else str(self.state["path"])).to_dict()

    def post_save(self, body):
        """Write the wall to ``body["path"]``, else to the current file, else to ``<name>.json`` here."""
        path = pathlib.Path(body.get("path") or self.state["path"] or f"{body['wall'].get('name') or 'my_wall'}.json")
        path = path if path.is_absolute() else pathlib.Path.cwd() / path
        Wall.from_dict(body["wall"], "my_wall").to_json(path)
        self.state["path"] = path
        return {"saved": str(path)}

    def post_live(self, body):
        """The page sends what it shows at each change: ``wall`` (JSON form), ``ghost`` (the hold under the cursor,
        ``{"col", "row", "ok"}`` or None), ``blocked`` (indices of holds whose way out is blocked)."""
        live = self.state["live"]
        live.update(body, version=live.get("version", 0) + 1)
        return {"version": live["version"]}

    def get_live(self, query):
        """What the page shows (see :meth:`post_live`), with a ``version`` that grows at each change, and the
        page's commands for the projector since the last call.  The projector (climbing-projector --follow) polls
        it to draw the wall being edited on the real board."""
        proj = self.state["projector"]
        with _LOCK:
            proj["seen"], commands, proj["commands"] = time.time(), proj["commands"], []
        return {**self.state["live"], "commands": commands}

    def post_projector_status(self, body):
        """The projector's state (calibrated, cameras, what it shows, holds detected), sent at each change."""
        self.state["projector"]["status"] = body
        return {"ok": True}

    def post_projector_views(self, body):
        """What the projector's cameras see (JPEG data URLs, corner markers outlined), sent while the page asks."""
        self.state["projector"]["views"] = body.get("views", [])
        return {"ok": True}

    def get_projector_views(self, query):
        return {"views": self.state["projector"].get("views", [])}

    def get_projector(self, query):
        """For the page's Projector panel: is the program installed, is it running (polling), its state."""
        proj = self.state["projector"]
        exe = projector_program()
        return {"installed": exe is not None, "program": exe, "dev": bool(exe and os.environ.get(PROJECTOR_ENV)),
                "running": time.time() - proj["seen"] < 3, "status": proj["status"],
                "can_download": projector_asset() is not None, "download": proj.get("download")}

    def post_projector(self, body):
        """A command from the page: ``start`` launches the projector program following this editor; the others
        (calibrate, manual, corner, show, step, toggle, reference, detect, quit) wait for its next poll."""
        if body.get("cmd") == "download":                     # the program from the simulator's release
            dl = self.state["projector"].get("download")
            if not (dl and dl["running"]):
                dl = self.state["projector"]["download"] = {"running": True, "done": 0, "total": 0, "error": None,
                                                            "ok": False}
                threading.Thread(target=download_projector, args=(dl,), daemon=True).start()
            return {"ok": True}
        if body.get("cmd") == "start":
            exe = projector_program()
            if exe is None:
                raise RuntimeError("climbing-projector is not installed on this computer")
            self.state["projector"]["status"] = {}            # forget the last run's state (and error)
            log = open(pathlib.Path(tempfile.gettempdir()) / "climbing-projector.log", "w")
            subprocess.Popen([exe, "--follow", f"http://127.0.0.1:{self.server.server_address[1]}"],
                             stdout=log, stderr=subprocess.STDOUT)
            return {"ok": True}
        with _LOCK:
            self.state["projector"]["commands"].append(body)
        return {"ok": True}

    def post_try(self, body):
        """Open the robot hanging on this wall in MuJoCo's viewer (a separate process)."""
        fd, path = tempfile.mkstemp(suffix=".json", prefix="wall_")
        os.close(fd)
        Wall.from_dict(body["wall"], "my_wall").to_json(path)
        subprocess.Popen([sys.executable, "-m", "climbing_sim.tools.viewer", path])
        return {"ok": True}


def main(argv=None):
    """Command line (``argv``: the arguments, sys.argv[1:] by default)."""
    if argv is None:
        single_blas_thread("climbing_sim.tools.wall_editor")
    ap = argparse.ArgumentParser(description="Wall editor for climbing_sim, in the browser.")
    ap.add_argument("wall", nargs="?", help="JSON file to open (and save to)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    path = pathlib.Path(a.wall).resolve() if a.wall else None
    if path and not path.exists():
        Wall(name=path.stem).to_json(path)
    state = {"path": path, "live": {}, "projector": {"commands": [], "status": {}, "seen": 0.0}}
    serve(WallEditor, state, "wall editor", a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
