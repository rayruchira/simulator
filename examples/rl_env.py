# SPDX-License-Identifier: GPL-3.0-or-later
"""A minimal Gymnasium environment around ClimbingSim, a starting point for reinforcement learning.

    uv pip install gymnasium          # not a dependency of climbing_sim
    python examples/rl_env.py         # runs a few random actions

* observation: what the real robot measures (joint angles and velocities, gravity and gyro in the trunk frame), so a
  policy trained on it can run on the robot;
* action: one change of the 19 joint targets per 20 ms step, in [-1, 1] times ``max_step`` rad;
* servos at ``kp_fw=125`` (softer than the factory 400): recommended for RL, the gain the real robot runs learned
  policies at;
* reward and end of episode: yours to design (``reward`` and ``terminated`` below).  Ground truth (trunk pose,
  balls in holds: ``sim.trunk_pose()``, ``sim.hooked()``...) may be used there, not in the observation.

ClimbingSim runs one robot on the CPU: for many robots at once, run several processes (gymnasium.vector).
"""
import gymnasium as gym
import numpy as np

from climbing_sim import JOINTS, ClimbingSim


class ClimbEnv(gym.Env):
    def __init__(self, wall="N2", episode_s=10.0, max_step=0.05):
        self.sim = ClimbingSim(wall, cameras=[], kp_fw=125.0)
        self.max_step = max_step
        self.max_steps = int(episode_s / self.sim.control_dt)
        n = len(JOINTS)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2 * n + 6,), np.float32)

    def _observe(self, obs):
        return np.concatenate([obs["q"], obs["dq"], obs["gravity"], obs["gyro"]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.sim.reset_hanging()
        self.target, self.steps = obs["q"].copy(), 0
        return self._observe(obs), {}

    def step(self, action):
        low, high = self.sim.joint_range.T
        self.target = np.clip(self.target + self.max_step * np.asarray(action), low, high)
        obs = self.sim.step(self.target)
        self.steps += 1
        reward = 0.0                                            # your reward
        terminated = False                                      # your end of episode (a fall, the top...)
        return self._observe(obs), reward, terminated, self.steps >= self.max_steps, {}


if __name__ == "__main__":
    env = ClimbEnv()
    obs, _ = env.reset(seed=0)
    total = 0.0
    for t in range(100):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample() * 0.2)
        total += reward
        if terminated or truncated:
            break
    print(f"{t + 1} random steps, return {total:.2f}, terminated {terminated}")
