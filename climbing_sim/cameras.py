# SPDX-License-Identifier: GPL-3.0-or-later
"""External cameras with the real cameras' calibration (OpenCV conventions).

A camera is given by
* its intrinsics: image size (width, height) in pixels, focal lengths fx, fy and principal point cx, cy
  in pixels (OpenCV's camera matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]); lens distortion is not
  simulated (render undistorted images);
* its extrinsics in the WALL frame (origin at the wall's bottom-left corner seen from the climbing side,
  x right, y up, z out of the wall towards the climber; metres): OpenCV's rvec, tvec such that a wall
  point P is seen at P_cam = R(rvec) P + tvec (what cv2.solvePnP / cv2.aruco.estimatePoseBoard give with
  the wall markers), or equivalently the camera position and a point it looks at.

Configuration files (JSON):  {"cameras": [{"name": "cam_left", "width": 1280, "height": 720,
"fx": 900, "fy": 900, "cx": 640, "cy": 360, "rvec": [...], "tvec": [...]}, ...]}
``climbing-sim-cameras`` (tools/camera_editor.py) edits them in the browser with a live preview.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, asdict, field

import numpy as np

from . import aruco
from .wall import WALL_HEIGHT, WALL_WIDTH, wall_to_world


def rodrigues(rvec) -> np.ndarray:
    """Rotation matrix of an OpenCV rotation vector (axis * angle)."""
    r = np.asarray(rvec, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rvec_of(R) -> np.ndarray:
    """OpenCV rotation vector of a rotation matrix."""
    R = np.asarray(R, float)
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-6:
        # 180 deg: axis from the symmetric part
        w, v = np.linalg.eigh((R + np.eye(3)) / 2)
        return v[:, np.argmax(w)] * np.pi
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


@dataclass
class Camera:
    """One camera: its name, intrinsics (pixels) and extrinsics (rvec, tvec: wall frame -> camera, OpenCV).

    Build it from your calibration (``Camera("cam_left", 1280, 720, fx, fy, cx, cy, rvec, tvec)``), or place it
    by hand with :meth:`looking_at`.  The default values are a D435i at 1280 x 720 at the wall frame's origin.
    """
    name: str = "cam"
    width: int = 1280                 # defaults: a RealSense D435i colour stream at 1280 x 720 (nominal)
    height: int = 720
    fx: float = 924.28
    fy: float = 925.74
    cx: float = 640.0
    cy: float = 360.0
    rvec: list = field(default_factory=lambda: [0.0, 0.0, 0.0])     # wall -> camera (OpenCV)
    tvec: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # ---- poses --------------------------------------------------------------------------------
    @classmethod
    def looking_at(cls, name, position_wall, target_wall, up_wall=(0.0, 1.0, 0.0), **intrinsics) -> "Camera":
        """A camera at position_wall looking at target_wall (wall frame, metres), image 'up' towards up_wall."""
        p, t, u = (np.asarray(v, float) for v in (position_wall, target_wall, up_wall))
        z = t - p; z /= np.linalg.norm(z)                     # OpenCV: z forward
        x = np.cross(z, u); x /= np.linalg.norm(x)            # x right (OpenCV: y down = -up)
        y = np.cross(z, x)
        R = np.stack([x, y, z])                               # rows: camera axes in the wall frame
        cam = cls(name=name, **intrinsics)        # missing intrinsics: the D435i defaults
        cam.rvec = rvec_of(R).tolist()
        cam.tvec = (-R @ p).tolist()
        return cam

    def R_wall_to_cam(self) -> np.ndarray:
        """Rotation matrix R of rvec: a wall point P is at R P + tvec in the camera frame."""
        return rodrigues(self.rvec)

    def position_wall(self) -> np.ndarray:
        """Optical centre in the wall frame."""
        R = self.R_wall_to_cam()
        return -R.T @ np.asarray(self.tvec, float)

    def K(self) -> np.ndarray:
        """OpenCV's camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def project_wall(self, p_wall) -> np.ndarray:
        """Pixel coordinates (u, v), shape (N, 2), of wall-frame points (N, 3): OpenCV's pinhole model (no
        distortion).  Points behind the camera give meaningless pixels: check their depth first."""
        P = np.asarray(p_wall, float).reshape(-1, 3) @ self.R_wall_to_cam().T + np.asarray(self.tvec, float)
        uv = P[:, :2] / P[:, 2:3]
        return uv * [self.fx, self.fy] + [self.cx, self.cy]

    # ---- MuJoCo -------------------------------------------------------------------------------
    def mujoco_pose(self):
        """(pos, quat) of the MuJoCo camera in the world frame (MuJoCo cameras look along their -z, y up)."""
        R = self.R_wall_to_cam()
        # camera axes in the wall frame (rows of R), then in the world frame
        x_w, y_w, z_w = (wall_to_world(R[i]) for i in range(3))
        Rm = np.stack([x_w, -y_w, -z_w], axis=1)              # columns: MuJoCo camera x, y, z in the world
        pos = wall_to_world(self.position_wall())
        return pos, _quat_of(Rm)

    def add_to_spec(self, spec):
        """Add this camera to a mujoco.MjSpec world body, with its intrinsics."""
        pos, quat = self.mujoco_pose()
        c = spec.worldbody.add_camera()
        c.name = self.name
        c.pos = pos.tolist()
        c.quat = quat.tolist()
        # intrinsics in pixel units: sensor of (width x height) "pixels", focal and principal point in pixels
        c.resolution = [int(self.width), int(self.height)]
        c.sensor_size = [float(self.width), float(self.height)]
        c.focal_length = [float(self.fx), float(self.fy)]
        # MuJoCo's principal "length": offset of the sensor centre from the optical axis (checked against the
        # OpenCV projection with segmentation renders: both components are centre - principal point)
        c.principal_length = [float(self.width / 2 - self.cx), float(self.height / 2 - self.cy)]
        return c


def _quat_of(R) -> np.ndarray:
    """(w, x, y, z) quaternion of a rotation matrix."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


# Intel RealSense D435i colour camera (the hackathon's cameras): 69.4 x 42.5 deg field of view (datasheet), i.e.
# focal lengths from the half-image over tan(half-FOV); replace them by the calibrated values of each camera
D435I_HFOV, D435I_VFOV = 69.4, 42.5
D435I_MODES = {(1920, 1080), (1280, 720), (960, 540), (848, 480), (640, 480), (640, 360), (424, 240), (320, 240),
               (320, 180)}


def d435i_intrinsics(width: int = 1280, height: int = 720) -> dict:
    """Nominal intrinsics of the D435i colour stream at a given resolution (principal point at the centre)."""
    if (width, height) not in D435I_MODES:
        raise ValueError(f"{width}x{height} is not a D435i colour mode: {sorted(D435I_MODES)}")
    fx = (width / 2) / np.tan(np.radians(D435I_HFOV / 2))
    fy = (height / 2) / np.tan(np.radians(D435I_VFOV / 2))
    if width / height < 1.7:              # 4:3 modes crop the sensor horizontally: same focal length
        fx = fy
    return dict(width=width, height=height, fx=float(fx), fy=float(fy), cx=width / 2, cy=height / 2)


def default_cameras(width: int = 1280, height: int = 720) -> list[Camera]:
    """Two D435i, 2.4 m in front of the wall, 0.55 m to each side, 0.9 m high, aimed at the wall's centre:
    both see the four corner markers."""
    centre = [0.4, 0.7, 0.0]
    kw = d435i_intrinsics(width, height)
    return [Camera.looking_at("cam_left", [-0.15, 0.9, 2.4], centre, **kw),
            Camera.looking_at("cam_right", [0.95, 0.9, 2.4], centre, **kw)]


def marker_visibility(camera: Camera, margin_px: float = 0.0) -> dict:
    """Corner marker ID -> True when its whole plate (the 4 corners of the white border) is in front of the
    camera and inside the image (with a margin in pixels).

    Pure geometry: the plates' known corners are projected with the camera's parameters, no image is looked
    at.  Occlusions (robot, holds) are not taken into account: render the scene to check them."""
    out = {}
    R, t = camera.R_wall_to_cam(), np.asarray(camera.tvec, float)
    for mid in aruco.CORNERS:
        x0, y0 = aruco.corner_plate_origin(mid, WALL_WIDTH, WALL_HEIGHT)
        C = np.array([[x0 + a, y0 + b, 0.0] for a in (0, aruco.PLATE) for b in (0, aruco.PLATE)])
        z = (C @ R.T + t)[:, 2]
        uv = camera.project_wall(C)
        out[mid] = bool((z > 0.05).all() and (uv[:, 0] >= margin_px).all() and (uv[:, 1] >= margin_px).all()
                        and (uv[:, 0] <= camera.width - 1 - margin_px).all()
                        and (uv[:, 1] <= camera.height - 1 - margin_px).all())
    return out


def load_cameras(path) -> list[Camera]:
    """Read a camera file (``{"cameras": [...]}``, see the top of this module)."""
    d = json.loads(pathlib.Path(path).read_text())
    return [Camera(**c) for c in d["cameras"]]


def save_cameras(cameras, path):
    """Write cameras (a list of Camera) to a camera file."""
    pathlib.Path(path).write_text(json.dumps({"cameras": [asdict(c) for c in cameras]}, indent=1))
