import numpy as np
import gymnasium as gym


def make_env(env_name: str, render: bool = False):
    """Create a MuJoCo environment with optional rendering."""
    render_mode = "rgb_array" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    return env


class EnergyInfoWrapper(gym.Wrapper):
    """Wrapper that adds energy consumption and root x‑position to the info dict."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def _compute_energy(self) -> float:
        """Compute the energy consumed in the current step using actuator forces and velocities."""
        data = self.unwrapped.data
        tau = np.asarray(data.actuator_force, dtype=np.float64)
        qvel = np.asarray(data.qvel, dtype=np.float64)
        act_dim = self.action_space.shape[0]
        # Extract the joint velocities corresponding to the actuators
        qd = qvel[6: 6 + act_dim] if len(qvel) >= 6 + act_dim else qvel[-act_dim:]
        n = min(len(tau), len(qd))
        dt = float(getattr(self.unwrapped, "dt", 0.015))
        return float(np.sum(np.abs(tau[:n] * qd[:n])) * dt)

    def _root_x(self) -> float:
        """Return the current x‑coordinate of the root body."""
        return float(self.unwrapped.data.qpos[0])

    def reset(self, **kwargs):
        """Reset the environment and add energy/root_x/alive to the info dict."""
        obs, info = self.env.reset(**kwargs)
        info = info or {}
        info["energy"] = 0.0
        info["root_x"] = self._root_x()
        info["alive"] = True
        return obs, info

    def step(self, action):
        """Step the environment and augment the info dict with energy, root_x, and alive."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = info or {}
        info["energy"] = self._compute_energy()
        info["root_x"] = self._root_x()
        info["alive"] = not (terminated or truncated)
        return obs, reward, terminated, truncated, info


def make_energy_env(env_name: str, render: bool = False) -> EnergyInfoWrapper:
    """Convenience function to create a wrapped energy environment."""
    return EnergyInfoWrapper(make_env(env_name, render=render))


def get_env_dt(env: gym.Env) -> float:
    """Retrieve the simulation timestep (dt) from the environment."""
    return float(getattr(env.unwrapped, "dt", 0.015))