# SPDX-License-Identifier: GPL-3.0-or-later
"""Camera editor in the browser: enter each real camera's calibration (its intrinsics, and the extrinsics your
code computes from the corner markers), check it on a photo from that camera, then save the JSON the simulator
loads.

    climbing-sim-cameras [my_cameras.json] [--port 8766] [--no-browser]
    (or: python -m climbing_sim.tools.camera_editor ...)

Starts a small local web server (this computer only) and opens the editor (``web/camera_editor.html``) in the
browser.  The board and the marker plates are drawn over the photo from the parameters entered: if they fall on
the real board, the calibration is right.  Nothing is detected here: detecting the markers and estimating the
cameras' pose is the teams' job (rulebook, Perception).  Stop it with Ctrl+C.
"""
from __future__ import annotations

import argparse
import base64
import io
import pathlib
from dataclasses import asdict, replace

import numpy as np

from .. import aruco
from ..cameras import (D435I_HFOV, D435I_MODES, D435I_VFOV, Camera, d435i_intrinsics, default_cameras,
                       load_cameras, marker_visibility, save_cameras)
from ..wall import WALL_HEIGHT, WALL_WIDTH
from ._webapp import WEB, JsonHandler, serve, single_blas_thread


class Previewer:
    """Renders what a camera sees: the scene (the robot hanging on a wall) is built once with one camera, which is
    given each camera's pose and intrinsics in turn (no rebuild)."""

    def __init__(self, wall: str = "N2"):
        """Build the scene on ``wall``, the robot hanging on its start holds (a few seconds)."""
        import mujoco
        from ..sim import ClimbingSim
        self.mujoco = mujoco
        self.sim = ClimbingSim(wall, cameras=[replace(default_cameras()[0], name="probe")])
        self.sim.reset_hanging(settle_s=0.0)
        self.cam_id = self.sim.model.camera("probe").id
        self.renderers = {}                       # one per image size

    def render_jpeg(self, cam: Camera, max_width: int = 1280) -> bytes:
        """``cam``'s image as JPEG bytes (at most ``max_width`` pixels wide)."""
        m, i = self.sim.model, self.cam_id
        m.cam_pos[i], m.cam_quat[i] = cam.mujoco_pose()
        # the intrinsics as Camera.add_to_spec sets them, in pixel units
        m.cam_sensorsize[i] = [cam.width, cam.height]
        m.cam_intrinsic[i] = [cam.fx, cam.fy, cam.width / 2 - cam.cx, cam.height / 2 - cam.cy]
        m.cam_resolution[i] = [cam.width, cam.height]
        self.mujoco.mj_forward(m, self.sim.data)
        size = (cam.width, cam.height)
        if size not in self.renderers:
            self.renderers[size] = self.mujoco.Renderer(m, height=cam.height, width=cam.width)
        renderer = self.renderers[size]
        renderer.update_scene(self.sim.data, camera=i)
        from PIL import Image
        img = Image.fromarray(renderer.render())
        if img.width > max_width:
            img = img.resize((max_width, round(img.height * max_width / img.width)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue()


def camera_from_dict(d: dict) -> Camera:
    """A camera from its JSON form (as sent by the page; other keys are ignored)."""
    return Camera(name=str(d.get("name") or "cam"), width=int(d["width"]), height=int(d["height"]),
                  fx=float(d["fx"]), fy=float(d["fy"]), cx=float(d["cx"]), cy=float(d["cy"]),
                  rvec=[float(v) for v in d["rvec"]], tvec=[float(v) for v in d["tvec"]])


def describe(cam: Camera) -> dict:
    """What the page draws for a camera, computed from its parameters only (no image processing):

    * ``plates``: the four corner markers' white plates in the image (pixels), and ``visible``: whether each
      whole plate is in the frame;
    * ``board``: the board's four edges in the image, sampled, ``None`` where a point is behind the camera;
    * ``position``, ``axis``, ``look_at``: the camera's position, its optical axis and where that axis meets the
      wall (wall frame, m; ``look_at`` is None when it looks away from the wall)."""
    R, t = cam.R_wall_to_cam(), np.asarray(cam.tvec, float)
    plates = {}
    for mid in aruco.CORNERS:
        x0, y0 = aruco.corner_plate_origin(mid, WALL_WIDTH, WALL_HEIGHT)
        corners = [[x0, y0, 0], [x0 + aruco.PLATE, y0, 0], [x0 + aruco.PLATE, y0 + aruco.PLATE, 0],
                   [x0, y0 + aruco.PLATE, 0]]
        plates[str(mid)] = cam.project_wall(corners).round(1).tolist()
    board = [[0, 0, 0], [WALL_WIDTH, 0, 0], [WALL_WIDTH, WALL_HEIGHT, 0], [0, WALL_HEIGHT, 0]]
    edges = []
    for a, b in zip(board, board[1:] + board[:1]):
        pts = np.linspace(a, b, 24)
        depth = (pts @ R.T + t)[:, 2]
        uv = cam.project_wall(pts).round(1)
        edges.append([uv[k].tolist() if depth[k] > 0.05 else None for k in range(len(pts))])
    p, axis = cam.position_wall(), R[2]
    look = p + axis * (-p[2] / axis[2]) if axis[2] < -1e-6 else None
    return {"visible": {str(k): v for k, v in marker_visibility(cam).items()}, "plates": plates, "board": edges,
            "position": p.round(4).tolist(), "axis": axis.round(4).tolist(),
            "look_at": None if look is None else look.round(4).tolist()}


class CameraEditor(JsonHandler):
    """The page and its routes; ``state["path"]`` is the JSON file Save writes, ``state["previewer"]`` renders."""

    page = WEB / "camera_editor.html"

    def get_config(self, query):
        """The wall's size, the D435i's fields of view and colour modes, the file opened."""
        modes = sorted(D435I_MODES, reverse=True)
        return {"wall": {"width": WALL_WIDTH, "height": WALL_HEIGHT},
                "d435i": {"hfov": D435I_HFOV, "vfov": D435I_VFOV, "modes": [d435i_intrinsics(w, h) for w, h in modes]},
                "file": str(self.state["path"]) if self.state["path"] else None}

    def get_cameras(self, query):
        """The cameras of the file opened, or the two default cameras (also when ``defaults`` is given)."""
        path = self.state["path"]
        cams = load_cameras(path) if path and path.exists() and "defaults" not in query else default_cameras()
        return {"cameras": [asdict(c) for c in cams]}

    def post_preview(self, body):
        """The camera's simulated image (a data URL) and what :func:`describe` gives."""
        cam = camera_from_dict(body["camera"])
        jpeg = self.state["previewer"].render_jpeg(cam)
        return {"image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(), **describe(cam)}

    def post_describe(self, body):
        """What :func:`describe` gives (no image: for the cameras not shown)."""
        return describe(camera_from_dict(body["camera"]))

    def post_look_at(self, body):
        """rvec, tvec of the camera placed at ``position`` and looking at ``target`` (wall frame, m)."""
        c = camera_from_dict(body["camera"])
        placed = Camera.looking_at(c.name, body["position"], body["target"], width=c.width, height=c.height,
                                   fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy)
        return {"rvec": [float(v) for v in placed.rvec], "tvec": [float(v) for v in placed.tvec]}

    def post_save(self, body):
        """Write the cameras to ``body["path"]``, else to the current file, else to ``cameras.json`` here."""
        path = pathlib.Path(body.get("path") or self.state["path"] or "cameras.json")
        path = path if path.is_absolute() else pathlib.Path.cwd() / path
        save_cameras([camera_from_dict(c) for c in body["cameras"]], path)
        self.state["path"] = path
        return {"saved": str(path)}


def main(argv=None):
    """Command line (``argv``: the arguments, sys.argv[1:] by default)."""
    if argv is None:
        single_blas_thread("climbing_sim.tools.camera_editor")
    ap = argparse.ArgumentParser(description="Camera editor for climbing_sim, in the browser.")
    ap.add_argument("cameras", nargs="?", help="JSON file to open (and save to)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    print("building the scene...", flush=True)
    state = {"path": pathlib.Path(a.cameras).resolve() if a.cameras else None, "previewer": Previewer()}
    # one request at a time, in this thread: the renderer's OpenGL context belongs to it
    serve(CameraEditor, state, "camera editor", a.port, open_browser=not a.no_browser, threaded=False)


if __name__ == "__main__":
    main()
