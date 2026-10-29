# SPDX-License-Identifier: GPL-3.0-or-later
"""The control loop of the API: bend the right knee while the robot hangs on its four start holds.

Every step() holds the 19 joint targets (rad, order climbing_sim.JOINTS) for 20 ms while BAM simulates the
servos; the observation is what the real robot measures (joint angles and velocities, IMU, motor currents).
The knee does not reach its target: the leg is in a closed chain with the other three limbs, which hold the
robot.
"""
import numpy as np

from climbing_sim import JOINTS, ClimbingSim

sim = ClimbingSim("N2")
obs = sim.reset_hanging()
q0 = obs["q"].copy()
k = JOINTS.index("right_knee")
for t in range(100):                                   # 2 s
    target = q0.copy()
    target[k] = q0[k] + 0.6 * min(1.0, t / 25)         # ramp the knee target by 0.6 rad
    obs = sim.step(target)
    if t % 20 == 19:
        print(f"t={obs['time']:.2f}s  knee target {target[k]:+.2f}  measured {obs['q'][k]:+.2f} rad  "
              f"pack current {obs['current'].sum():.2f} A  gyro {np.round(obs['gyro'], 2)}  hooked {sim.hooked()}")
