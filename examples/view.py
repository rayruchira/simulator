# SPDX-License-Identifier: GPL-3.0-or-later
"""Open the simulator in MuJoCo's viewer: the robot hanging on the start holds of a wall, its servos holding
the start posture (drag the robot with Ctrl + right mouse to feel the servos).  Key 3 (or Rendering > Geom groups)
shows the collision shapes: robot convex parts (blue), servo boxes (yellow), ball ends (cyan), hold spheres (orange),
hold hulls and skin (light blue).

    python examples/view.py [N1|N2|N3|my_wall.json]
"""
import sys
import time

import mujoco.viewer

from climbing_sim import ClimbingSim

sim = ClimbingSim(sys.argv[1] if len(sys.argv) > 1 else "N2")
sim.reset_hanging()
target = sim.joint_positions()
with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1     # collision meshes drawn as the hulls MuJoCo collides with
    while v.is_running():
        t0 = time.time()
        sim.step(target)
        v.sync()
        time.sleep(max(0.0, sim.control_dt - (time.time() - t0)))
