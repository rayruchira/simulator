# SPDX-License-Identifier: GPL-3.0-or-later
"""Visual-only 42 mm DICT_4X4_50 ID 5 on the existing trunk's rear face.

The PNG is OpenCV's generateImageMarker(DICT_4X4_50, 5, 600), without
any extra white margin. The entire texture, including its black border,
occupies exactly 42 x 42 mm. No printable sticker or backing plate is added.
"""
from pathlib import Path

import mujoco
import numpy as np

MARKER_SIZE = 0.042
MARKER_CENTRE = np.array([-0.033, 0.0, 0.05475])
R_TRUNK_MARKER = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
VISUAL_OFFSET = 50e-6  # render-only separation: avoids GPU depth-buffer z-fighting
VISUAL_HALF_THICKNESS = 0.5e-6
GEOM_NAME = "robot_aruco5"


def add_robot_marker(spec: mujoco.MjSpec) -> None:
    """Attach a massless, non-colliding textured surface to the existing trunk.

    The model's trunk frame already is x forward, y left, z up. Local marker
    x points trunk -y, y points trunk +z, and the outward normal is trunk -x.
    Its front face is 50 micrometres outside the supplied marker plane. The
    earlier 1 micrometre separation was below depth precision on some GPUs,
    causing the torso to show through the black code. The nominal mounting
    transform and physical robot model are unchanged.
    """
    trunk = spec.body("trunk")
    texture = Path(__file__).parent / "assets" / "aruco" / "dict_4x4_50_id5.png"
    spec.add_texture(name=GEOM_NAME, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(texture))
    material = spec.add_material(name=GEOM_NAME, texuniform=False, rgba=[1, 1, 1, 1], specular=0, shininess=0, emission=0.25)
    
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = GEOM_NAME
    
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, R_TRUNK_MARKER.ravel())
    
    centre = MARKER_CENTRE + R_TRUNK_MARKER[:, 2] * (VISUAL_OFFSET - VISUAL_HALF_THICKNESS)
    
    trunk.add_geom(name=GEOM_NAME, type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[MARKER_SIZE / 2, MARKER_SIZE / 2, VISUAL_HALF_THICKNESS],
                   pos=centre.tolist(), quat=quat.tolist(), material=GEOM_NAME,
                   contype=0, conaffinity=0, mass=0, density=0, group=1)
