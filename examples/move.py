# SPDX-License-Identifier: GPL-3.0-or-later
"""A move that runs both here and on the real robot: the shape to give yours.

    python examples/move.py         # in the simulator: hanging on wall N2, 10 s

On the robot: copy the file into microban/src/moves/ (microban, branch dev_v2). The robot's program finds it by
itself: it appears in the dashboard (Start / Stop, its options and sliders). `make sim WALL=N2` runs it in the
robot's program on this simulator, `make deploy` sends it to the robot.

It only holds the pose it starts in and turns the head slowly: replace step() with yours. Use only what the robot
measures (the observation); targets are the 19 joint angles in the order of JOINTS (rad, every zero at the motor's
zero), or {joint: angle} for some joints (the others keep their last target); None ends it. A trained policy goes here too: load it in __init__ (onnxruntime, ...), build
its input from obs in step().
"""
import math

from microban_model import JOINTS          # the same as climbing_sim.JOINTS; installed on the robot too

HEAD = JOINTS.index("head")


class HoldMove:                             # no base class: the same file runs here and on the robot
    description = "Holds the pose it starts in, turns the head slowly"
    kp = 125                                # the servos' gain while it runs (the robot runs learned policies at 125)
    options = {"head_amplitude": 0.3}       # edited live in the robot's dashboard (here: this default)
    option_docs = {"head_amplitude": "how far the head turns each way (rad)"}
    sliders = {"head_speed": (0.0, 2.0, 0.5)}   # name -> (min, max, start), shown while it runs

    def start(self, obs):
        """Called when it starts: obs is the first observation."""
        self.q0 = obs["q"].copy()
        self.phase = 0.0

    def step(self, obs):
        """Every 20 ms: the observation in, the 19 joint targets out."""
        self.phase += 0.02 * self.slider["head_speed"]     # rad of the head's sine per step
        target = self.q0.copy()
        target[HEAD] = self.q0[HEAD] + self.options["head_amplitude"] * math.sin(self.phase)
        return target


if __name__ == "__main__":
    from climbing_sim import ClimbingSim, run_move
    sim = ClimbingSim("N2", cameras=[], kp_fw=125)
    sim.reset_hanging()
    obs = run_move(HoldMove(), sim, seconds=10.0)
    print("after 10 s: hooked", sim.hooked(), "| head", round(float(obs["q"][HEAD]), 3), "rad")
