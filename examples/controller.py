"""A controller that runs both here and on the real robot: the shape to give yours.

    python examples/controller.py                         # in the simulator (hanging on the wall, 10 s)
    make run CONTROLLER=.../examples/controller.py        # on the real robot, from the microban repository (dev_v2):
                                                          # keyboard [c] / gamepad [Y] starts and stops it

It only holds the pose it starts in and turns the head slowly: replace step() with yours. Use only what the robot
measures (the observation); targets are the 19 joint angles in the order of JOINTS (rad, every zero at the motor's
zero), or None to keep the last ones. A trained policy goes here too: load it in __init__ (onnxruntime, ...), build
its input from obs in step().
"""
import math

import numpy as np

from microban_model import JOINTS          # the same as climbing_sim.JOINTS; installed on the robot too

HEAD = JOINTS.index("head")


class Controller:
    kp_fw = 125            # the servos' gain while it runs (the robot runs learned policies at 125)

    def reset(self, obs):
        """Called when it starts: obs is the first observation."""
        self.start = obs["q"].copy()
        self.t0 = obs["time"]

    def step(self, obs):
        """Every 20 ms: the observation in, the 19 joint targets out."""
        target = self.start.copy()
        target[HEAD] = self.start[HEAD] + 0.3 * math.sin(0.5 * (obs["time"] - self.t0))
        return target


if __name__ == "__main__":
    from climbing_sim import ClimbingSim, run_controller
    sim = ClimbingSim("N2", cameras=[], kp_fw=125)
    sim.reset_hanging()
    obs = run_controller(Controller(), sim, seconds=10.0)
    print("after 10 s: hooked", sim.hooked(), "| head", round(float(obs["q"][HEAD]), 3), "rad")
