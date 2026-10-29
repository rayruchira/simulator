# SPDX-License-Identifier: GPL-3.0-or-later
"""The robot hanging on the start holds of a wall, in MuJoCo's interactive viewer.

    python -m climbing_sim.tools.viewer [N1|N2|N3|my_wall.json]

The servos hold the start posture (drag the robot with Ctrl + right mouse to feel them).  Key 3 (or Rendering >
Geom groups) shows the collision shapes: robot convex parts (blue), servo boxes (yellow), ball ends (cyan), hold
spheres (orange), hold hulls and skin (light blue).  The wall editor's "Open in MuJoCo" button opens it.
"""
from __future__ import annotations

import sys
import time

import mujoco
import mujoco.viewer

from ..scene import LIMBS
from ..sim import ClimbingSim


def show(wall: str = "N2"):
    """Open the viewer on ``wall`` (a bundled wall name or a JSON file), the robot hanging on its start holds (standing
    in front of the wall when the wall has no start holds for the four limbs)."""
    sim = ClimbingSim(wall, cameras=[])
    if set(sim.wall.start) == set(LIMBS):
        sim.reset_hanging()
    else:
        sim.reset()
    target = sim.joint_positions()
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # collision meshes drawn as the convex hulls MuJoCo collides with
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1
        while viewer.is_running():
            t0 = time.time()
            sim.step(target)
            viewer.sync()
            time.sleep(max(0.0, sim.control_dt - (time.time() - t0)))     # real time


def main():
    """Command line: the wall's name or file as the first argument (default N2)."""
    show(sys.argv[1] if len(sys.argv) > 1 else "N2")


if __name__ == "__main__":
    main()
