# SPDX-License-Identifier: GPL-3.0-or-later
"""The hackathon's simulator: the climbing robot on the competition wall, with its holds, markers and cameras.

    from climbing_sim import ClimbingSim
    sim = ClimbingSim(wall="N2")            # wall, holds, markers, two default cameras
    obs = sim.reset_hanging()               # the robot hanging on the wall's four start holds
    for _ in range(500):
        obs = sim.step(target)              # joint position targets (rad), one call = one 20 ms control step
    img = sim.render("cam_left")

``ClimbingSim`` is Rhoban's robot simulator (:class:`microban_sim.MicrobanSim`: the robot, its servos simulated by
BAM, its sensors as the robot reads them, ``step`` / ``observation`` / ``get_state`` / ``set_state`` / ``view``) in
the climbing version, with the wall around it.  It adds what is about the wall: ``reset()`` in front of it,
``reset_hanging()`` on its start holds, the four ball ends (``ball_positions``, ``ball_positions_at``,
``joint_angles_for`` on the balls), which ball is in which hold (``hooked``), the calibrated cameras (``render``).
For RL: ``examples/rl_env.py`` wraps this class as a Gymnasium environment.
"""
from __future__ import annotations

import json
import warnings

import mujoco
import numpy as np
from microban_sim import LATENCY_MS, MicrobanSim

from .cameras import default_cameras, load_cameras, marker_visibility
from .scene import ASSETS, BALL_SITES, JOINTS, LIMBS, build_spec
from .wall import Wall

STAND = json.loads((ASSETS / "poses" / "stand.json").read_text())     # standing in front of the wall
HANG = json.loads((ASSETS / "poses" / "hang.json").read_text())       # hanging on four holds


class ClimbingSim(MicrobanSim):
    """Robot + wall + holds + corner markers + cameras.

    wall: a :class:`Wall`, a bundled wall name ("N1", "N2", "N3") or a wall JSON file.
    cameras: a list of :class:`Camera`, a camera JSON file (``climbing-sim-cameras`` edits them) or None for the
        two default cameras; a warning tells when a camera does not see the four corner markers.
    control_dt: duration of one :meth:`step` (s); the physics runs at 2 ms.
    latency_ms: delay between sending a target and the servos acting on it (10 ms, measured on the robot).
    kp_fw: the servos' firmware position P gain: 400, the XL330's factory value, by default; 125 (softer) is
        recommended for reinforcement learning only (the real robot runs its learned policies at 125).
    vin: the servos' supply voltage (V).
    ideal_sensors: True gives every measurement of the current step (no sensor delay), for debugging.
    seed: seed of ``self.rng`` (numpy random generator), for your own randomisation.

    Attributes you may use: ``wall`` (the :class:`Wall`), ``joint_range`` (19 x 2, min and max angle of each joint,
    rad, order ``JOINTS``), ``control_dt`` (s), ``servos`` (gain, torque), ``rng``, ``model`` / ``data`` (the plain
    MuJoCo objects).
    """

    def __init__(self, wall="N2", cameras=None, control_dt=0.02, latency_ms=LATENCY_MS, kp_fw=400.0, vin=7.5,
                 ideal_sensors=False, seed=None):
        self.wall = wall if isinstance(wall, Wall) else Wall.load(str(wall))
        if isinstance(cameras, (str, bytes)) or hasattr(cameras, "__fspath__"):
            cameras = load_cameras(cameras)
        if cameras is None:
            cameras = default_cameras()
        self.cameras = {c.name: c for c in cameras}
        for c in self.cameras.values():
            missing = [m for m, ok in marker_visibility(c).items() if not ok]
            if missing:
                warnings.warn(f"camera {c.name} does not see the whole corner marker(s) {missing}: the real cameras "
                              f"must see all four")
        super().__init__("climber", spec=build_spec(wall=self.wall, cameras=cameras), control_dt=control_dt,
                         latency_ms=latency_ms, kp_fw=kp_fw, vin=vin, ideal_sensors=ideal_sensors, seed=seed)

    # ---- starts ----------------------------------------------------------------------------------------
    def reset(self, trunk_pos=None, trunk_quat=None, joints: dict | None = None):
        """Robot standing on the floor in front of the wall (the mount start spot: feet 74.5 mm from the wall, centred
        under holds 13 and 14 of N2) unless a pose is given (world frame).  The joints are in the standing pose of
        Rhoban's walking robot, the one the real robot stands in by itself: arms along the body (shoulder pitches
        pi/2, rolls +-10 deg, elbows -20 deg), hips pitched -10 deg, hip and ankle rolls +-5 deg, the rest 0;
        ``joints`` ({name: angle}) changes some.  At kp_fw 400 the robot stands still (it settles 2 cm back in
        the first seconds); with soft servos (kp_fw 125) it falls unless balanced actively.  For climbing, start
        with reset_hanging()."""
        return super().reset(STAND["trunk_pos"] if trunk_pos is None else trunk_pos,
                             STAND["trunk_quat"] if trunk_quat is None else trunk_quat,
                             {**STAND["joints"], **(joints or {})})

    def reset_hanging(self, start: dict | None = None, settle_s: float = 1.0):
        """Robot hanging on its four start holds (the wall's "start" holds, or start = {limb: Hold}): from a
        hanging posture, the trunk pose then the joints are fitted so that the four balls sit in the sockets (1 mm
        above them), then the robot settles for settle_s with the servos holding that posture."""
        start = start or self.wall.start
        assert set(start) == set(LIMBS), f"start holds for {LIMBS}"
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        d.qpos[0:3] = 0.0; d.qpos[3:7] = [1, 0, 0, 0]
        for n, a in zip(JOINTS, self._qadr):
            d.qpos[a] = HANG["joints"].get(n, 0.0)
        mujoco.mj_kinematics(self.model, d)
        P = np.array([d.site(BALL_SITES[l]).xpos for l in LIMBS])
        S = np.array([start[l].socket for l in LIMBS]) + [0.0, 0.0, 0.001]      # 1 mm above: no initial overlap
        cp, cs = P.mean(0), S.mean(0)
        U, _, Vt = np.linalg.svd((P - cp).T @ (S - cs))
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        d.qpos[0:3] = cs - R @ cp
        q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten()); d.qpos[3:7] = q
        # the start holds of another wall are not where this posture's are: fit the joints to them
        d.qpos[self._qadr], _ = self.joint_angles_for(dict(zip(LIMBS, S)), d.qpos[0:3], d.qpos[3:7], iters=200)
        self._start()
        target = self.joint_positions()
        for _ in range(int(round(settle_s / self.control_dt))):
            self.step(target)
        return self.observation()

    # ---- the balls (ground truth: not measured on the real robot) ---------------------------------------
    def ball_positions(self) -> dict:
        """World positions of the four ball ends (limb -> xyz)."""
        return {l: self.data.site(s).xpos.copy() for l, s in BALL_SITES.items()}

    def ball_positions_at(self, q, trunk_pos=None, trunk_quat=None) -> dict:
        """Forward kinematics: world positions of the four balls (limb -> xyz) for the joint angles ``q`` (19, order
        ``JOINTS``) and a trunk pose (default: the current one).  The simulation is not touched."""
        at = self.site_positions_at(q, list(BALL_SITES.values()), trunk_pos, trunk_quat)
        return {l: at[s] for l, s in BALL_SITES.items()}

    def joint_angles_for(self, balls: dict, trunk_pos, trunk_quat, q_start=None, iters: int = 100,
                         tol: float = 1e-4):
        """Inverse kinematics: joint angles (19, order ``JOINTS``) that put each ball of ``balls`` (limb -> world xyz;
        the limbs not given keep their angles) where asked, the trunk at ``trunk_pos`` / ``trunk_quat``.  Damped least
        squares from ``q_start`` (default: the current angles), joints kept within their ranges.  Returns
        ``(q, error)``, error = the largest ball distance left (m); check it: a target out of reach stays off."""
        assert set(balls) <= set(LIMBS), f"balls: limb -> xyz, limbs among {LIMBS}"
        return super().joint_angles_for({BALL_SITES[l]: p for l, p in balls.items()}, trunk_pos, trunk_quat,
                                        q_start, iters, tol)

    def hooked(self, tol=0.006) -> dict:
        """limb -> index (in ``wall.holds``) of the hold whose socket holds its ball (within tol m), or None."""
        socks = self.wall.sockets()
        out = {}
        for l, p in self.ball_positions().items():
            d = np.linalg.norm(socks - p, axis=1) if len(socks) else np.array([np.inf])
            out[l] = int(np.argmin(d)) if d.min() < tol else None
        return out

    # ---- images -------------------------------------------------------------------------------------
    def render(self, camera="overview", width=None, height=None) -> np.ndarray:
        """RGB image (H, W, 3), uint8, from a scene camera: one of ``self.cameras`` (at its calibrated size by
        default) or ``"overview"``, a camera that sees the whole wall."""
        cam = self.cameras.get(camera)
        if width is None:
            width, height = (cam.width, cam.height) if cam else (960, 720)
        return super().render(camera, width, height)
