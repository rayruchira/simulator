# SPDX-License-Identifier: GPL-3.0-or-later
"""The planning tools, on one example: forward and inverse kinematics, and trying a motion from a saved state.

The robot hangs on its four start holds.  We ask for a posture (the trunk 2 cm higher, the four balls where they
are), play it, look at the result, and come back to where we started.

    python examples/planning.py
"""
from climbing_sim import ClimbingSim

sim = ClimbingSim("N2", cameras=[])
obs = sim.reset_hanging()
trunk_pos, trunk_quat = sim.trunk_pose()
balls = sim.ball_positions()

# forward kinematics: where the balls would be for given joint angles (nothing is simulated)
print("right foot ball at", sim.ball_positions_at(obs["q"])["right_foot"].round(3))

# inverse kinematics: the joint angles of a posture (here: the trunk 2 cm higher, the balls where they are)
q, error = sim.joint_angles_for(balls, trunk_pos + [0.0, 0.0, 0.02], trunk_quat)
print(f"inverse kinematics: balls within {error * 1000:.1f} mm of the targets")

# try it from a saved state, then come back
start = sim.get_state()
for _ in range(100):                                      # 100 steps of 20 ms: 2 s
    obs = sim.step(q)
print(f"trunk moved {(sim.trunk_pose()[0][2] - trunk_pos[2]) * 1000:+.1f} mm, balls hooked: {sim.hooked()}")
sim.set_state(start)
print(f"back to the start: trunk at {(sim.trunk_pose()[0][2] - trunk_pos[2]) * 1000:+.1f} mm")
