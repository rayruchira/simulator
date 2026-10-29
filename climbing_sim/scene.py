# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the MuJoCo scene: the robot on its floor (Rhoban's microban_sim, climbing version), and around it the wall
with its holds, the ArUco markers and the cameras.

:class:`climbing_sim.ClimbingSim` does it for you; use :func:`build_spec` to add your own things to the scene
before compiling it.  Geom groups (MuJoCo viewer, keys 0-5): 1 visual meshes, 3 collision shapes, 4 sockets.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import tempfile

import microban_sim
import mujoco
import numpy as np

from . import aruco
from .cameras import Camera, default_cameras
from .wall import WALL_WIDTH, WALL_HEIGHT, WALL_THICKNESS, GRID_COLS, GRID_ROWS, Wall, hole_position, wall_to_world

ASSETS = pathlib.Path(__file__).parent / "assets"
HOLD_VISUAL = ASSETS / "hold" / "hold_visual.stl"
# hold collisions (the hold frame: origin at the socket centre): the socket and its lip as spheres, all inside the real
# hold (30, cup_spheres.json), the back plate, gussets and rear block as 4 hole-free hulls, and a 1 mm skin of 10 convex
# patches on the cup's outside below the lip (nothing gets into the hold from below); the same as the training model
HOLD_PARTS = sorted(glob.glob(str(ASSETS / "hold" / "hull_*.stl"))) + sorted(glob.glob(str(ASSETS / "hold" / "skin_*.stl")))
HOLD_SPHERES = json.loads((ASSETS / "hold" / "cup_spheres.json").read_text())

#: joints driven by servos, in the order of the robot's actuators (the head is a servo too): microban_model's
JOINTS = microban_sim.JOINTS
#: the four limbs, each ending in a ball that hooks into the holds' sockets
LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]
#: limb -> name of the MuJoCo site at the centre of its ball
BALL_SITES = {l: f"{l}_ball" for l in LIMBS}


def _wall_texture(cache: pathlib.Path, px_per_mm: int = 2) -> str:
    """The board seen from the front: plywood colour with its hole grid (image top = top of the wall)."""
    from PIL import Image
    path = cache / f"wall_front_{px_per_mm}.png"
    if not path.exists():
        W, H = int(WALL_WIDTH * 1000 * px_per_mm), int(WALL_HEIGHT * 1000 * px_per_mm)
        img = np.empty((H, W, 3), np.uint8); img[:] = (219, 191, 143)
        yy, xx = np.mgrid[0:H, 0:W]
        r = 1.6 * px_per_mm
        for i in range(GRID_COLS):
            x = hole_position(i, 0)[0] * 1000 * px_per_mm
            for j in range(GRID_ROWS):
                z = hole_position(0, j)[2] * 1000 * px_per_mm
                y = H - z
                x0, x1, y0, y1 = int(x - r - 1), int(x + r + 2), int(y - r - 1), int(y + r + 2)
                m = (xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2 <= r * r
                img[y0:y1, x0:x1][m] = (90, 70, 45)
        Image.fromarray(img).save(path)
    return str(path)


def _marker_texture(marker_id: int, cache: pathlib.Path) -> str:
    """PNG file of an ArUco marker with its white plate (made once in ``cache``)."""
    from PIL import Image
    path = cache / f"aruco_{marker_id}.png"
    if not path.exists():
        Image.fromarray(aruco.image(marker_id, px_per_cell=64)).save(path)
    return str(path)


def build_spec(wall: Wall | None = None, cameras: list[Camera] | None = None, robot: bool = True,
               markers: bool = True, floor_friction: float = 1.0,
               hold_friction: float = 0.5) -> mujoco.MjSpec:
    """The scene as a ``mujoco.MjSpec``: add things to it, then ``.compile()`` it.

    wall: the holds to bolt (default: the N2 wall); cameras: :func:`default_cameras` when None, [] for none;
    robot / markers: include the robot / the four corner markers; floor_friction, hold_friction: sliding friction
    coefficients.
    """
    wall = Wall.load("N2") if wall is None else wall
    cameras = default_cameras() if cameras is None else cameras
    if robot:           # the climbing robot on its floor, its servos simulated by BAM (microban_sim)
        spec = microban_sim.build_spec("climber", floor_friction=floor_friction)
    else:
        spec = mujoco.MjSpec()
        spec.option.timestep = 0.002
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3.0, 3.0, 0.05],
                                friction=[floor_friction, 0.005, 0.0001])
    spec.modelname = "climbing_sim"
    spec.compiler.texturedir = ""
    w = spec.worldbody

    # ---- the board --------------------------------------------------------------------------------
    spec.add_texture(name="wood", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                           rgb1=[0.84, 0.72, 0.52], rgb2=[0.84, 0.72, 0.52], width=8, height=8)
    spec.add_material(name="wood", rgba=[0.86, 0.75, 0.56, 1]).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "wood"
    w.add_geom(name="wall", type=mujoco.mjtGeom.mjGEOM_BOX, material="wood",
               size=[WALL_WIDTH / 2, WALL_THICKNESS / 2, WALL_HEIGHT / 2],
               pos=[WALL_WIDTH / 2, WALL_THICKNESS / 2, WALL_HEIGHT / 2], condim=3)
    # the front face with its hole grid: a textured plate just in front of the board (visual only)
    cache = pathlib.Path(tempfile.gettempdir()) / "climbing_sim_textures"
    cache.mkdir(exist_ok=True)
    spec.add_texture(name="wall_front", type=mujoco.mjtTexture.mjTEXTURE_2D, file=_wall_texture(cache))
    spec.add_material(name="wall_front").textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "wall_front"
    w.add_geom(name="wall_front", type=mujoco.mjtGeom.mjGEOM_BOX, material="wall_front",
               size=[WALL_WIDTH / 2, WALL_HEIGHT / 2, 0.0002], pos=[WALL_WIDTH / 2, -0.0002, WALL_HEIGHT / 2],
               quat=[0.7071068, 0.7071068, 0, 0], contype=0, conaffinity=0, group=1)

    # ---- holds --------------------------------------------------------------------------------------
    spec.add_mesh(name="hold_visual", file=str(HOLD_VISUAL))
    for f in HOLD_PARTS:
        spec.add_mesh(name="hold_" + os.path.basename(f)[:-4], file=f)
    for k, h in enumerate(wall.holds):
        b = w.add_body(name=f"hold{k}_c{h.col}_r{h.row}", pos=h.socket.tolist())
        b.add_site(name=f"hold{k}_socket", size=[0.002, 0, 0], rgba=[0.1, 0.9, 0.2, 1], group=4)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hold_visual", contype=0, conaffinity=0, group=1,
                   rgba=[0.85, 0.35, 0.15, 1])
        for f in HOLD_PARTS:
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hold_" + os.path.basename(f)[:-4], group=3,
                       rgba=[0.25, 0.55, 0.95, 1], friction=[hold_friction, 0.005, 0.0001], condim=3)
        for sp in HOLD_SPHERES:
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[sp["r"], 0, 0], pos=sp["pos"], group=3,
                       rgba=[1.0, 0.5, 0.0, 1], friction=[hold_friction, 0.005, 0.0001], condim=3)

    # ---- ArUco markers ------------------------------------------------------------------------------
    if markers:
        for mid in aruco.CORNERS:
            spec.add_texture(name=f"aruco{mid}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                             file=_marker_texture(mid, cache))
            spec.add_material(name=f"aruco{mid}").textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"aruco{mid}"
            x0, y0 = aruco.corner_plate_origin(mid, WALL_WIDTH, WALL_HEIGHT)
            c = wall_to_world([x0 + aruco.PLATE / 2, y0 + aruco.PLATE / 2, 0.0005])
            # a thin box whose front face (-y) carries the texture, image up = world +z
            w.add_geom(name=f"aruco{mid}", type=mujoco.mjtGeom.mjGEOM_BOX, material=f"aruco{mid}",
                       size=[aruco.PLATE / 2, aruco.PLATE / 2, 0.0005], pos=c.tolist(),
                       quat=[0.7071068, 0.7071068, 0, 0], contype=0, conaffinity=0, group=1)

    # ---- cameras ------------------------------------------------------------------------------------
    for cam in cameras:
        cam.add_to_spec(spec)
    w.add_camera(name="overview", pos=[0.4, -2.2, 0.9], xyaxes=[1, 0, 0, 0, 0.2, 1])
    return spec


def build_model(**kw) -> mujoco.MjModel:
    """The compiled scene (``build_spec(**kw).compile()``)."""
    return build_spec(**kw).compile()
