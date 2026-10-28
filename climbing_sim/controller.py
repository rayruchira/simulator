"""Run your own controller, the same file as on the real robot (microban: ``make run CONTROLLER=my_controller.py``).

A controller is a class with ``step(obs) -> targets`` (the 19 joint targets, rad, order ``JOINTS``; None keeps the
last ones), called every 20 ms with ``sim.observation()``, and optionally ``reset(obs)`` (called when it starts) and
``kp_fw`` (the servos' gain while it runs; 125 by default, as the robot runs learned policies).
"""
from .sim import ClimbingSim

KP_CONTROLLER = 125.0


def run_controller(controller, sim: ClimbingSim, seconds: float, on_step=None) -> dict:
    """Run ``controller`` on ``sim`` for ``seconds`` of simulated time, from where the robot is (as on the robot, where
    the torque goes on where the robot is when it starts). ``on_step(sim, obs)``, if given, is called after every step
    (to watch, log or stop: return True to stop). Returns the last observation."""
    sim.servos.ctrl.model.actuator.kp = float(getattr(controller, "kp_fw", KP_CONTROLLER))
    obs = sim.observation()
    if hasattr(controller, "reset"):
        controller.reset(obs)
    for _ in range(int(round(seconds / sim.control_dt))):
        obs = sim.step(controller.step(obs))
        if on_step is not None and on_step(sim, obs):
            break
    return obs
