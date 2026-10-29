# SPDX-License-Identifier: GPL-3.0-or-later
"""Run your moves, the same files as on the real robot (microban, branch dev_v2: ``src/moves/``), from Rhoban's robot
simulator (microban_sim): ``run_move(move, sim, seconds)``, and ``load_moves(path)`` -> ``{name: move}``.

A move is a class with ``step(obs) -> targets`` (the 19 joint targets, rad, order ``JOINTS``; None keeps the last
ones), called every 20 ms with ``sim.observation()``, and optionally ``reset(obs)``, ``kp_fw`` (the servos' gain while
it runs, 125 by default), ``options`` and ``sliders`` (edited live on the robot's dashboard; here they keep their
defaults). ``run_controller`` is the old name of ``run_move``.
"""
from microban_sim import load_moves, run_controller, run_move

__all__ = ["run_move", "load_moves", "run_controller"]
