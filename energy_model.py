from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnergyModelConfig:
    """Configuration for the energy‑based objective."""
    w_energy:   float = 1.0
    w_smooth:   float = 0.05
    w_progress: float = 3.0
    w_pose:     float = 15.0
    w_alive:    float = 2.0
    height_idx: int   = 0
    pitch_idx:  int   = 3
    roll_idx:   Optional[int] = 2
    height_min: float = 1.0
    height_max: float = 1.6
    pitch_min:  float = -0.3
    pitch_max:  float = 0.3
    roll_min:   float = -0.3
    roll_max:   float = 0.3


@dataclass
class StepCost:
    """Per‑step cost components."""
    energy:   float = 0.0
    smooth:   float = 0.0
    progress: float = 0.0
    pose:     float = 0.0
    total:    float = 0.0


class EnergyObjective:
    """Implements a cost function for a trajectory based on energy,
    action smoothness, forward progress, posture, and survival."""

    def __init__(self, cfg: EnergyModelConfig):
        self.cfg = cfg

    def step_cost(self, energy, delta_u, delta_x, obs_next, alive) -> StepCost:
        """Compute the cost for a single time step."""
        c = self.cfg
        c_energy   = c.w_energy * energy
        c_smooth   = c.w_smooth * float(np.dot(delta_u, delta_u))
        c_progress = -c.w_progress * max(delta_x, 0.0)
        c_pose     = self._pose_cost(obs_next)
        c_alive    = -c.w_alive if alive else c.w_alive * 10.0
        total = c_energy + c_smooth + c_progress + c_pose + c_alive
        return StepCost(c_energy, c_smooth, c_progress, c_pose, total)

    def _pose_cost(self, obs: np.ndarray) -> float:
        """Penalize deviations from allowed posture ranges."""
        c = self.cfg
        obs_dim = len(obs)
        cost = 0.0
        if c.height_idx < obs_dim:
            h = obs[c.height_idx]
            if h < c.height_min:
                cost += c.w_pose * (c.height_min - h) ** 2
            elif h > c.height_max:
                cost += c.w_pose * (h - c.height_max) ** 2
        if c.pitch_idx is not None and c.pitch_idx < obs_dim:
            p = obs[c.pitch_idx]
            if p < c.pitch_min:
                cost += c.w_pose * (c.pitch_min - p) ** 2
            elif p > c.pitch_max:
                cost += c.w_pose * (p - c.pitch_max) ** 2
        if c.roll_idx is not None and c.roll_idx < obs_dim:
            r = obs[c.roll_idx]
            if r < c.roll_min:
                cost += c.w_pose * (c.roll_min - r) ** 2
            elif r > c.roll_max:
                cost += c.w_pose * (r - c.roll_max) ** 2
        return cost

    def trajectory_cost(self, energies, delta_us, root_xs, obs_seq, alives) -> float:
        """Compute total cost for a full trajectory."""
        T = len(energies)
        total = 0.0
        for t in range(T):
            dx   = root_xs[t+1] - root_xs[t]
            sc   = self.step_cost(energies[t], delta_us[t], dx,
                                  obs_seq[t+1], bool(alives[t]))
            total += sc.total
        return total

    def summary(self, energies, delta_us, root_xs, obs_seq, alives) -> dict:
        """Return a summary dictionary of trajectory statistics."""
        T = len(energies)
        return {
            "total_energy": float(np.sum(energies)),
            "total_smooth": float(np.sum([np.dot(delta_us[t], delta_us[t]) for t in range(T)])),
            "progress_m":   float(root_xs[-1] - root_xs[0]),
            "alive_steps":  int(np.sum(alives)),
            "total_cost":   self.trajectory_cost(energies, delta_us, root_xs, obs_seq, alives),
        }