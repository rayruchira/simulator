# SPDX-License-Identifier: GPL-3.0-or-later
"""Run your own controller, the same file as on the real robot (microban: ``make run CONTROLLER=my_controller.py``):
``run_controller(controller, sim, seconds)``, from Rhoban's robot simulator (microban_sim).

A controller is a class with ``step(obs) -> targets`` (the 19 joint targets, rad, order ``JOINTS``; None keeps the
last ones), called every 20 ms with ``sim.observation()``, and optionally ``reset(obs)`` (called when it starts) and
``kp_fw`` (the servos' gain while it runs; 125 by default, as the robot runs learned policies).
"""
from microban_sim import run_controller

__all__ = ["run_controller"]
