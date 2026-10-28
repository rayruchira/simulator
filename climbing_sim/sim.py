"""The simulator: physics at 500 Hz, the 19 XL330 servos simulated by BAM, the robot's sensors, camera images.

    from climbing_sim import ClimbingSim
    sim = ClimbingSim(wall="N2")            # wall, holds, markers, two default cameras
    obs = sim.reset_hanging()               # the robot hanging on the wall's four start holds
    for _ in range(500):
        obs = sim.step(target)              # joint position targets (rad), one call = one 20 ms control step
    img = sim.render("cam_left")

What the real robot measures is in the observation (``sim.observation()``, also returned by ``step``): joint
positions and velocities, IMU (in the trunk frame), motor currents.  The rest (``ball_positions``, ``trunk_pose``,
``hooked``) is ground truth, for debugging and evaluation only.

For planning: ``ball_positions_at(q, trunk_pos, trunk_quat)`` (forward kinematics), ``joint_angles_for(balls,
trunk_pos, trunk_quat)`` (inverse kinematics), both without touching the simulation, and ``get_state()`` /
``set_state(state)`` (try a motion, then come back).  For RL: ``examples/rl_env.py`` wraps this
class as a Gymnasium environment.
"""
from __future__ import annotations

import copy
import json
import warnings
from collections import deque

import mujoco
import numpy as np

from .cameras import default_cameras, load_cameras, marker_visibility
from .scene import ASSETS, BALL_SITES, JOINTS, LIMBS, build_spec
from .wall import Wall

STAND = json.loads((ASSETS / "poses" / "stand.json").read_text())     # standing in front of the wall
HANG = json.loads((ASSETS / "poses" / "hang.json").read_text())       # hanging on four holds


class Servos:
    """The Dynamixel XL330 servos as identified on the bench (BAM, Rhoban's better-actuator-models, model m6):
    position control by the firmware (P gain kp_fw), PWM voltage, DC motor with back-EMF, current limit,
    load-dependent friction; a supply voltage that drops under load; a command latency."""

    def __init__(self, model, data, kp_fw=400.0, vin=7.5, vin_drop_resistance=0.0366, vin_min=6.0, max_current=1.75,
                 latency_steps=10):
        from bam.model import load_model, _resolve_json_path
        from bam.mujoco import MujocoController
        bam_model = load_model(_resolve_json_path(None, "xl330", "m6"))
        bam_model.actuator.kp = kp_fw
        bam_model.actuator.vin = vin
        bam_model.actuator.max_current = max_current          # the firmware's current limit
        # supply + wires resistance (ohm): the voltage drops by R * sum(|torque|) / kt
        self.ctrl = MujocoController(bam_model, JOINTS, model, data, vin_drop_resistance=vin_drop_resistance,
                                     vin_min=vin_min)
        self.latency = deque(maxlen=max(1, latency_steps + 1))
        self.kt = bam_model.kt.value
        self.data = data
        self.torque = np.ones(len(JOINTS), bool)              # torque enabled, per servo (order JOINTS)

    def reset(self, qpos):
        """Start from rest at ``qpos`` (the full MuJoCo qpos), with an empty latency queue, every torque on."""
        self.ctrl.reset()
        self.torque[:] = True
        self.ctrl.q_target = np.asarray(qpos)[self.ctrl.qpos_indexes].copy()     # hold where the joints are
        self.latency.clear()

    def set_target(self, target):
        """Send joint targets (rad); the firmware gets the one sent ``latency_steps`` physics steps ago."""
        self.latency.append(np.asarray(target, float).copy())
        self.ctrl.q_target = self.latency[0].copy()

    def update(self):
        """Set the motor torques for the next physics step (none for a servo whose torque is off: it goes limp, only
        its friction left)."""
        self.ctrl.update()
        if not self.torque.all():
            self.data.ctrl[: len(JOINTS)][~self.torque] = 0.0

    def currents(self, data) -> np.ndarray:
        """Motor currents (A), what the real servos report."""
        return np.abs(data.ctrl[: len(JOINTS)]) / self.kt

    def get_state(self, model) -> dict:
        """The servos' memory: the commands on their way, the current target, BAM's internal state, and the joint
        friction BAM writes into ``model`` at every step (it depends on the load)."""
        c, act = self.ctrl, self.ctrl.model.actuator
        return {"latency": [t.copy() for t in self.latency], "q_target": c.q_target.copy(), "last_ts": c.last_ts,
                "torque": self.torque.copy(),
                # BAM's memory: its control law's state, the last control, the PWM duty cycle (the supply drop)
                "actuator_state": copy.deepcopy(c.actuator_state), "control": copy.deepcopy(c.control),
                "duty_cycle": copy.deepcopy(getattr(act, "duty_cycle", None)), "kp": act.kp,
                "frictionloss": model.dof_frictionloss.copy(), "damping": model.dof_damping.copy()}

    def set_state(self, model, state: dict):
        c = self.ctrl
        self.latency.clear()
        self.latency.extend(t.copy() for t in state["latency"])
        c.q_target, c.last_ts = state["q_target"].copy(), state["last_ts"]
        self.torque[:] = state["torque"]
        c.actuator_state, c.control = copy.deepcopy(state["actuator_state"]), copy.deepcopy(state["control"])
        if state["duty_cycle"] is not None:
            c.model.actuator.duty_cycle = copy.deepcopy(state["duty_cycle"])
        c.model.actuator.kp = state["kp"]
        model.dof_frictionloss[:] = state["frictionloss"]
        model.dof_damping[:] = state["damping"]


class ClimbingSim:
    """Robot + wall + holds + corner markers + cameras.

    wall: a :class:`Wall`, a bundled wall name ("N1", "N2", "N3") or a wall JSON file.
    cameras: a list of :class:`Camera`, a camera JSON file (``climbing-sim-cameras`` edits them) or None for the
        two default cameras; a warning tells when a camera does not see the four corner markers.
    control_dt: duration of one :meth:`step` (s); the physics runs at 2 ms.
    latency_ms: delay between sending a target and the servos acting on it.
    kp_fw: the servos' firmware position P gain: 400, the XL330's factory value, by default; 125 (softer) is
        recommended for reinforcement learning only (the real robot runs its learned policies at 125).
    vin: the servos' supply voltage (V).
    seed: seed of ``self.rng`` (numpy random generator), for your own randomisation.

    Attributes you may use: ``wall`` (the :class:`Wall`), ``joint_range`` (19 x 2, min and max angle of each joint,
    rad, order ``JOINTS``), ``control_dt`` (s), ``rng``, ``model`` / ``data`` (the plain MuJoCo objects).
    """

    def __init__(self, wall="N2", cameras=None, control_dt=0.02, latency_ms=20.0, kp_fw=400.0, vin=7.5, seed=None):
        self.wall = wall if isinstance(wall, Wall) else Wall.load(str(wall))
        if isinstance(cameras, (str, bytes)) or hasattr(cameras, "__fspath__"):
            cameras = load_cameras(cameras)
        if cameras is None:
            cameras = default_cameras()
        self.spec = build_spec(wall=self.wall, cameras=cameras)
        self.cameras = {c.name: c for c in cameras}
        for c in self.cameras.values():
            missing = [m for m, ok in marker_visibility(c).items() if not ok]
            if missing:
                warnings.warn(f"camera {c.name} does not see the whole corner marker(s) {missing}: the real cameras "
                              f"must see all four")
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.n_sub = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.control_dt = self.n_sub * self.model.opt.timestep
        self.servos = Servos(self.model, self.data, kp_fw=kp_fw, vin=vin,
                             latency_steps=int(round(latency_ms / 1000 / self.model.opt.timestep)))
        self.rng = np.random.default_rng(seed)
        m = self.model
        self._qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINTS])
        self._vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in JOINTS])
        self.joint_range = np.array([m.jnt_range[m.joint(n).id] for n in JOINTS])
        self._renderers = {}                  # one per image size
        self._fk_data = mujoco.MjData(m)       # scratch data for ball_positions_at
        R = np.zeros(9)                        # the IMU site's orientation in the trunk (fixed)
        mujoco.mju_quat2Mat(R, m.site_quat[m.site("imu").id])
        self._R_trunk_imu = R.reshape(3, 3)

    # ---- state ---------------------------------------------------------------------------------------
    def reset(self, trunk_pos=None, trunk_quat=None, joints: dict | None = None):
        """Robot standing on the floor in front of the wall (the mount start spot: feet 74.5 mm from the wall, centred
        under holds 13 and 14 of N2) unless a pose is given (world frame).  The joints are in the standing pose of
        Rhoban's walking robot, the one the real robot stands in by itself: arms along the body (shoulder pitches
        pi/2, rolls +-10 deg, elbows -20 deg), hips pitched -10 deg, hip and ankle rolls +-5 deg, the rest 0;
        ``joints`` ({name: angle}) changes some.  At kp_fw 400 the robot stands still (it settles 2 cm back in
        the first seconds); with soft servos (kp_fw 125) it falls unless balanced actively.  For climbing, start
        with reset_hanging()."""
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        d.qpos[0:3] = STAND["trunk_pos"] if trunk_pos is None else trunk_pos
        d.qpos[3:7] = STAND["trunk_quat"] if trunk_quat is None else trunk_quat
        j = dict(STAND["joints"]); j.update(joints or {})
        for n, a in zip(JOINTS, self._qadr):
            d.qpos[a] = j.get(n, 0.0)
        mujoco.mj_forward(self.model, d)
        self.servos.reset(d.qpos)
        self.servos.set_target(self.joint_positions())
        return self.observation()

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
        mujoco.mj_forward(self.model, d)
        self.servos.reset(d.qpos)
        target = self.joint_positions()
        self.servos.set_target(target)
        for _ in range(int(round(settle_s / self.control_dt))):
            self.step(target)
        return self.observation()

    def step(self, target=None):
        """One control step: send ``target`` (19 joint angles, rad, in the order of ``JOINTS``; None: keep the last
        one) and simulate ``control_dt`` seconds, the servos simulated by BAM at every physics step.  Returns the
        observation (:meth:`observation`)."""
        if target is not None:
            target = np.asarray(target, float)
            assert target.shape == (len(JOINTS),), f"target: {len(JOINTS)} joint angles (order JOINTS)"
        for _ in range(self.n_sub):
            if target is not None:
                self.servos.set_target(target)
            self.servos.update()
            mujoco.mj_step(self.model, self.data)
        return self.observation()

    def set_torque(self, on=True):
        """Servos' torque on or off, as on the real robot (Dynamixel Torque Enable): ``on`` is one bool for every
        servo, or 19 (order ``JOINTS``).  A servo whose torque is off goes limp (only its friction left)."""
        self.servos.torque[:] = np.broadcast_to(np.asarray(on, bool), self.servos.torque.shape)

    # ---- what the real robot measures ------------------------------------------------------------------
    def joint_positions(self) -> np.ndarray:
        """The 19 joint angles (rad, order ``JOINTS``), what the servos' encoders read."""
        return self.data.qpos[self._qadr].copy()

    def joint_velocities(self) -> np.ndarray:
        """The 19 joint velocities (rad/s, order ``JOINTS``)."""
        return self.data.qvel[self._vadr].copy()

    def imu(self) -> dict:
        """What the IMU gives, in the TRUNK frame (the IMU chip is mounted turned in the trunk; converted here):
        ``gyro`` (rad/s), ``accel`` (m/s^2, includes gravity: about +9.81 upwards at rest), ``gravity`` (unit vector
        of gravity, pointing down), ``quat`` (the trunk's orientation, w x y z, world frame)."""
        s = lambda n: self.data.sensor(n).data.copy()
        R_world_imu = np.zeros(9)
        mujoco.mju_quat2Mat(R_world_imu, s("orientation"))
        R_world_trunk = R_world_imu.reshape(3, 3) @ self._R_trunk_imu.T
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_world_trunk.flatten())
        return {"gyro": self._R_trunk_imu @ s("imu_ang_vel"), "accel": self._R_trunk_imu @ s("imu_accel"),
                "gravity": R_world_trunk.T @ np.array([0.0, 0.0, -1.0]), "quat": quat}

    def currents(self) -> np.ndarray:
        """The 19 motor currents (A, order ``JOINTS``), what the real servos report."""
        return self.servos.currents(self.data)

    def observation(self) -> dict:
        """What the real robot measures: ``q``, ``dq`` (joints), ``gyro``, ``accel``, ``gravity``, ``quat`` (IMU,
        trunk frame), ``current`` (motors), plus the simulation ``time`` (s)."""
        return {"q": self.joint_positions(), "dq": self.joint_velocities(), **self.imu(), "current": self.currents(),
                "time": self.data.time}

    # ---- ground truth (not measured on the real robot: for debugging, rewards, evaluation) ------------
    def ball_positions(self) -> dict:
        """World positions of the four ball ends (limb -> xyz)."""
        return {l: self.data.site(s).xpos.copy() for l, s in BALL_SITES.items()}

    def ball_positions_at(self, q, trunk_pos=None, trunk_quat=None) -> dict:
        """Forward kinematics: world positions of the four balls (limb -> xyz) for the joint angles ``q`` (19, order
        ``JOINTS``) and a trunk pose (default: the current one).  The simulation is not touched."""
        d = self._fk_data
        d.qpos[:] = self.data.qpos
        if trunk_pos is not None:
            d.qpos[0:3] = trunk_pos
        if trunk_quat is not None:
            d.qpos[3:7] = trunk_quat
        d.qpos[self._qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return {l: d.site(s).xpos.copy() for l, s in BALL_SITES.items()}

    def joint_angles_for(self, balls: dict, trunk_pos, trunk_quat, q_start=None, iters: int = 100,
                         tol: float = 1e-4):
        """Inverse kinematics: joint angles (19, order ``JOINTS``) that put each ball of ``balls`` (limb -> world xyz;
        the limbs not given keep their angles) where asked, the trunk at ``trunk_pos`` / ``trunk_quat``.  Damped least
        squares from ``q_start`` (default: the current angles), joints kept within their ranges.  Returns
        ``(q, error)``, error = the largest ball distance left (m); check it: a target out of reach stays off."""
        assert set(balls) <= set(LIMBS), f"balls: limb -> xyz, limbs among {LIMBS}"
        m, d = self.model, self._fk_data
        low, high = self.joint_range.T
        d.qpos[:] = self.data.qpos
        d.qpos[0:3], d.qpos[3:7] = trunk_pos, trunk_quat
        d.qpos[self._qadr] = self.joint_positions() if q_start is None else q_start
        sites = [m.site(BALL_SITES[l]).id for l in balls]
        goal = np.concatenate([np.asarray(balls[l], float) for l in balls])
        jac = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = goal - np.concatenate([d.site_xpos[s] for s in sites])
            if np.abs(err).max() < tol:
                break
            J = []
            for s in sites:
                mujoco.mj_jacSite(m, d, jac, None, s)
                J.append(jac[:, self._vadr].copy())             # the 19 joints only: the trunk stays put
            J = np.vstack(J)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(len(err)), err)
            d.qpos[self._qadr] = np.clip(d.qpos[self._qadr] + dq, low, high)
        mujoco.mj_kinematics(m, d)
        err = goal - np.concatenate([d.site_xpos[s] for s in sites])
        return d.qpos[self._qadr].copy(), float(np.linalg.norm(err.reshape(-1, 3), axis=1).max())

    # ---- save / restore (planning: try a motion, then come back) ------------------------------------
    def get_state(self) -> dict:
        """A copy of everything that evolves: the MuJoCo data and the servos' memory."""
        data = mujoco.MjData(self.model)
        mujoco.mj_copyData(data, self.model, self.data)
        return {"data": data, "servos": self.servos.get_state(self.model)}

    def set_state(self, state: dict):
        """Go back to a state from :meth:`get_state` (the same steps then give exactly the same motion)."""
        mujoco.mj_copyData(self.data, self.model, state["data"])
        self.servos.set_state(self.model, state["servos"])

    def trunk_pose(self):
        """World position and orientation quaternion (w, x, y, z) of the trunk."""
        return self.data.qpos[0:3].copy(), self.data.qpos[3:7].copy()

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
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        r = self._renderers[key]
        r.update_scene(self.data, camera=camera)
        return r.render()

    def close(self):
        """Free the renderers (their OpenGL contexts)."""
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
