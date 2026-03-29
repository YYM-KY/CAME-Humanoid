from __future__ import annotations
import numpy as np
import mujoco

try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False
    print("[WARNING] Please install cma: pip install cma")

from energy_model import EnergyObjective
from env_wrapper import get_env_dt


class CMAESOptimizer:
    """
    CMA-ES based trajectory optimizer for a given window of actions.
    It optimizes residual perturbations around a baseline action sequence.
    """

    def __init__(self, env, energy_obj: EnergyObjective,
                 window=40, stride=20, sigma0=0.05,
                 maxiter=80, popsize=16):
        if not CMA_AVAILABLE:
            raise ImportError("Please install cma: pip install cma")
        self.env = env
        self.obj = energy_obj
        self.window = window
        self.stride = stride
        self.sigma0 = sigma0
        self.maxiter = maxiter
        self.popsize = popsize
        self.act_dim = env.action_space.shape[0]

    def _save_state(self):
        """Save a full snapshot of the physical state (qpos, qvel)."""
        data = self.env.unwrapped.data
        return {
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
        }

    def _restore_state(self, snap: dict):
        """Restore the physical state from a snapshot."""
        data = self.env.unwrapped.data
        data.qpos[:] = snap["qpos"]
        data.qvel[:] = snap["qvel"]
        mujoco.mj_forward(self.env.unwrapped.model, data)

    def _get_obs(self):
        """Retrieve current observation from the environment."""
        try:
            return self.env.unwrapped._get_obs()
        except AttributeError:
            return np.zeros(self.env.observation_space.shape[0])

    def _simulate_window(self, snap: dict,
                         u_base_w: np.ndarray,
                         delta_u_flat: np.ndarray) -> float:
        """
        Simulate a window of steps starting from a saved state, using
        baseline actions + a delta perturbation. Compute the cost.
        """
        self._restore_state(snap)
        W = len(u_base_w)
        delta_u = delta_u_flat.reshape(W, self.act_dim)
        u_seq = np.clip(u_base_w + delta_u, -1.0, 1.0)

        energies = []
        root_xs = [float(self.env.unwrapped.data.qpos[0])]
        obs_seq = [self._get_obs()]
        alives = []

        for t in range(W):
            obs, _, terminated, truncated, info = self.env.step(u_seq[t])
            energies.append(info.get("energy", 0.0))
            root_xs.append(info.get("root_x", root_xs[-1]))
            obs_seq.append(obs.copy())
            alive = not (terminated or truncated)
            alives.append(alive)
            if not alive:
                # Fallen: fill remaining steps with zeros and heavy penalty
                remaining = W - t - 1
                energies.extend([0.0] * remaining)
                root_xs.extend([root_xs[-1]] * remaining)
                obs_seq.extend([obs.copy()] * remaining)
                alives.extend([False] * remaining)
                break

        return self.obj.trajectory_cost(
            np.array(energies),
            delta_u,
            np.array(root_xs),
            np.array(obs_seq),
            np.array(alives, dtype=bool),
        )

    def _optimize_window(self, snap: dict,
                         u_base_w: np.ndarray) -> np.ndarray:
        """
        Run CMA-ES to find optimal perturbations for a given window.
        Returns the delta array (W x act_dim).
        """
        W = len(u_base_w)
        dim = W * self.act_dim
        x0 = np.zeros(dim)

        es = cma.CMAEvolutionStrategy(x0, self.sigma0, {
            "maxiter": self.maxiter,
            "popsize": self.popsize,
            "verbose": -9,
            "bounds": [-0.3, 0.3],    # keep perturbations moderate
            "tolx": 1e-4,
            "tolfun": 1e-4,
            "seed": 42,
        })

        best_cost = float("inf")
        best_delta = x0.copy()

        while not es.stop():
            solutions = es.ask()
            costs = [self._simulate_window(snap, u_base_w, sol)
                     for sol in solutions]
            es.tell(solutions, costs)
            for sol, c in zip(solutions, costs):
                if c < best_cost:
                    best_cost = c
                    best_delta = sol.copy()

        return best_delta.reshape(W, self.act_dim)

    def optimize(self, init_states: np.ndarray,
                 init_actions: np.ndarray):
        """
        Optimize the whole action sequence window‑by‑window.
        Returns optimized actions, states, and energies.
        """
        T = len(init_actions)
        opt_actions = init_actions.copy()

        # Pre‑record snapshots at each window start position
        print("[CMA-ES] Pre‑recording window snapshots...")
        self.env.reset()
        window_snaps = {}
        start_positions = list(range(0, T, self.stride))

        for t in range(T):
            if t in start_positions:
                window_snaps[t] = self._save_state()
            _, _, terminated, truncated, _ = self.env.step(init_actions[t])
            if terminated or truncated:
                print(f"  Baseline terminated at step {t+1}, reusing last snapshot.")
                last_snap = window_snaps.get(t, window_snaps[
                    max(k for k in window_snaps if k <= t)])
                for s in start_positions:
                    if s > t and s not in window_snaps:
                        window_snaps[s] = last_snap
                break

        n_windows = len([s for s in start_positions if s < T])
        print(f"[CMA-ES] Total steps={T}, window={self.window}, "
              f"stride={self.stride}, windows={n_windows}")

        for win_idx, start in enumerate(start_positions):
            if start >= T:
                break
            end = min(start + self.window, T)
            W = end - start
            u_base_w = init_actions[start:end]

            snap = window_snaps.get(start)
            if snap is None:
                print(f"  [Warning] No snapshot for window start {start}, skipping.")
                continue

            print(f"[CMA-ES] Window {win_idx+1}/{n_windows} "
                  f"[{start}:{end}] optimizing...", flush=True)

            delta_u_w = self._optimize_window(snap, u_base_w)
            opt_actions[start:end] = np.clip(
                u_base_w + delta_u_w, -1.0, 1.0)

        # Final full rollout to obtain states and energies
        print("[CMA-ES] Final full rollout...")
        opt_states, opt_energies = self._final_rollout(opt_actions)
        return opt_actions, opt_states, opt_energies

    def _final_rollout(self, actions: np.ndarray):
        """Simulate the optimized action sequence from scratch."""
        self.env.reset()
        states = [self._get_obs()]
        energies = []
        for a in actions:
            obs, _, terminated, truncated, info = self.env.step(a)
            energies.append(info.get("energy", 0.0))
            states.append(obs.copy())
            if terminated or truncated:
                break
        return states, energies