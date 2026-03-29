from __future__ import annotations
import argparse
import os
import numpy as np
import torch

from config              import Config
from env_wrapper         import make_energy_env
from energy_model        import EnergyObjective, EnergyModelConfig
from cmaes_optimizer     import CMAESOptimizer
from trajectory_executor import TrajectoryExecutor
from utils               import (set_seed, create_output_dirs,
                                 save_summary, save_comparison_plot)

try:
    from lib.agent_ppo import PPOAgent
except ImportError:
    class PPOAgent:
        def __init__(self, obs_dim, act_dim):
            self.act_dim = act_dim
        def get_action_and_value(self, obs):
            return torch.randn(1, self.act_dim), None, None, None
        def load_state_dict(self, _): pass
        def to(self, _): return self


def collect_baseline(env, agent, steps: int, device) -> dict:
    """Collect a baseline trajectory using the given agent."""
    obs, info = env.reset()
    states    = [obs.copy()]
    actions   = []
    energies  = []
    root_xs   = [info.get("root_x", 0.0)]

    for t in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32,
                             device=device).unsqueeze(0)
        with torch.no_grad():
            action, *_ = agent.get_action_and_value(obs_t)
        action = action.squeeze(0).cpu().numpy()
        obs, _, terminated, truncated, info = env.step(action)
        actions.append(action.copy())
        energies.append(info.get("energy", 0.0))
        root_xs.append(info.get("root_x", root_xs[-1]))
        states.append(obs.copy())
        if terminated or truncated:
            print(f"  [Collect] Terminated at step {t+1}, padding with zero actions.")
            for _ in range(steps - t - 1):
                za = np.zeros(env.action_space.shape[0])
                obs, _, t2, tr2, info = env.step(za)
                actions.append(za)
                energies.append(info.get("energy", 0.0))
                root_xs.append(info.get("root_x", root_xs[-1]))
                states.append(obs.copy())
                if t2 or tr2:
                    break
            break

    T = min(len(actions), steps)
    return {
        "states":   np.array(states[:T+1]),
        "actions":  np.array(actions[:T]),
        "energies": np.array(energies[:T]),
        "root_xs":  np.array(root_xs[:T+1]),
    }


def record_video_baseline(env_name: str, agent, n_steps: int,
                          device, video_path: str) -> dict:
    """
    Record a video of the baseline policy before optimization.
    The environment is created in rgb_array mode.
    """
    print(f"  Recording baseline video ({n_steps} steps)...")
    env = make_energy_env(env_name, render=True)
    obs, info = env.reset()
    first_x = info.get("root_x", 0.0)
    frames = []
    energies = []
    root_xs = [first_x]

    for t in range(n_steps):
        frame = env.render()
        if isinstance(frame, np.ndarray) and frame.ndim == 3:
            frames.append(frame)

        obs_t = torch.tensor(obs, dtype=torch.float32,
                             device=device).unsqueeze(0)
        with torch.no_grad():
            action, *_ = agent.get_action_and_value(obs_t)
        action = action.squeeze(0).cpu().numpy()
        obs, _, terminated, truncated, info = env.step(action)
        energies.append(info.get("energy", 0.0))
        root_xs.append(info.get("root_x", root_xs[-1]))
        if terminated or truncated:
            print(f"  Baseline fell at step {t+1}.")
            frame = env.render()
            if isinstance(frame, np.ndarray):
                frames.append(frame)
            break

    env.close()

    if frames:
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, frames, fps=30)
        print(f"  Baseline video saved ({len(frames)} frames): {video_path}")

    return {
        "total_energy": float(np.sum(energies)),
        "progress":     root_xs[-1] - first_x,
        "energy_curve": np.array(energies),
        "root_x_curve": np.array(root_xs),
        "alive_steps":  len(energies),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str,
                        default=Config.pretrained_model)
    parser.add_argument("--collect_steps", type=int,
                        default=Config.collect_steps)
    parser.add_argument("--render", action="store_true",
                        help="Record comparison videos before and after optimization")
    parser.add_argument("--sigma0", type=float, default=Config.cma_sigma0)
    parser.add_argument("--window", type=int,   default=Config.cma_window)
    args = parser.parse_args()

    set_seed(Config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Config.get_output_dir()
    dirs = create_output_dirs(output_dir)

    print("=" * 55)
    print("  Humanoid Energy Optimization — CMA‑ES Residual Search")
    print("=" * 55)

    # Initialize environment and policy
    env = make_energy_env(Config.env_name)
    obs_sample, _ = env.reset()
    obs_dim = obs_sample.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, device={device}")

    agent = PPOAgent(obs_dim, act_dim).to(device)
    if os.path.isfile(args.pretrained_model):
        agent.load_state_dict(
            torch.load(args.pretrained_model, map_location=device))
        print(f"  Loaded model: {args.pretrained_model}")
    else:
        print(f"  [WARNING] Model not found at {args.pretrained_model}, using random policy.")

    # Step 1: Record baseline video
    baseline_video = os.path.join(dirs["videos"], "01_before_optimization.mp4")
    if args.render:
        print("\n[Step 1] Recording baseline video...")
        bl_metrics = record_video_baseline(
            Config.env_name, agent, args.collect_steps,
            device, baseline_video)
    else:
        print("\n[Step 1] Skipping video recording (use --render to enable).")
        bl_metrics = None

    # Step 2: Collect baseline trajectory (used for optimization)
    print(f"\n[Step 2] Collecting baseline trajectory ({args.collect_steps} steps)...")
    baseline_traj = collect_baseline(env, agent, args.collect_steps, device)
    env.close()
    T_actual = len(baseline_traj["actions"])
    print(f"  Collected {T_actual} steps, "
          f"total energy={baseline_traj['energies'].sum():.3f} J, "
          f"progress={baseline_traj['root_xs'][-1]-baseline_traj['root_xs'][0]:.3f} m")
    np.savez(os.path.join(dirs["trajectories"], "baseline_traj.npz"),
             **baseline_traj)

    if bl_metrics is None:
        bl_metrics = {
            "total_energy": float(baseline_traj["energies"].sum()),
            "progress":     float(baseline_traj["root_xs"][-1]
                                  - baseline_traj["root_xs"][0]),
            "energy_curve": baseline_traj["energies"],
            "root_x_curve": baseline_traj["root_xs"],
            "alive_steps":  T_actual,
        }

    # Step 3: CMA‑ES optimization
    print(f"\n[Step 3] CMA‑ES optimization "
          f"(sigma0={args.sigma0}, window={args.window})...")

    em_cfg = EnergyModelConfig(
        w_energy   = Config.w_energy,
        w_smooth   = Config.w_smooth,
        w_progress = Config.w_progress,
        w_pose     = Config.w_pose,
        w_alive    = Config.w_alive,
        height_idx = Config.height_idx,
        pitch_idx  = Config.pitch_idx,
        roll_idx   = Config.roll_idx,
        height_min = Config.height_min,
        height_max = Config.height_max,
        pitch_min  = Config.pitch_min,
        pitch_max  = Config.pitch_max,
        roll_min   = Config.roll_min,
        roll_max   = Config.roll_max,
    )
    energy_obj = EnergyObjective(em_cfg)

    opt_env = make_energy_env(Config.env_name)
    optimizer = CMAESOptimizer(
        opt_env, energy_obj,
        window  = args.window,
        stride  = Config.cma_stride,
        sigma0  = args.sigma0,
        maxiter = Config.cma_maxiter,
        popsize = Config.cma_popsize,
    )
    opt_actions, opt_states, opt_energies = optimizer.optimize(
        baseline_traj["states"], baseline_traj["actions"])
    opt_env.close()

    total_opt_energy = float(np.sum(opt_energies))
    print(f"  Optimization completed, total energy={total_opt_energy:.3f} J")
    np.savez(os.path.join(dirs["trajectories"], "optimized_traj.npz"),
             actions=opt_actions, energies=np.array(opt_energies))

    # Step 4: Execute and record optimized video
    print("\n[Step 4] Executing and recording optimized trajectory...")
    opt_video = os.path.join(dirs["videos"], "02_after_optimization.mp4")

    exec_env = make_energy_env(Config.env_name, render=args.render)
    executor = TrajectoryExecutor(exec_env, opt_actions)
    opt_metrics = executor.run(
        render          = args.render,
        save_video_path = opt_video if args.render else None,
    )
    exec_env.close()

    print(f"  Optimized: energy={opt_metrics['total_energy']:.3f} J, "
          f"progress={opt_metrics['progress']:.3f} m, "
          f"alive steps={opt_metrics['alive_steps']}")

    # Step 5: Comparison
    reduction = ((bl_metrics["total_energy"] - opt_metrics["total_energy"])
                 / (bl_metrics["total_energy"] + 1e-8) * 100)

    print("\n" + "=" * 55)
    print(f"  Baseline energy:  {bl_metrics['total_energy']:.3f} J")
    print(f"  Optimized energy: {opt_metrics['total_energy']:.3f} J")
    print(f"  Energy reduction: {reduction:.1f}%")
    print(f"  Baseline progress:  {bl_metrics['progress']:.3f} m")
    print(f"  Optimized progress: {opt_metrics['progress']:.3f} m")
    print(f"  Alive steps:         {opt_metrics['alive_steps']} / {T_actual}")
    if args.render:
        print(f"\n  Video files:")
        print(f"    Before: {baseline_video}")
        print(f"    After:  {opt_video}")
    print("=" * 55)

    save_comparison_plot(bl_metrics, opt_metrics, output_dir)
    save_summary({
        "config": {
            "sigma0": args.sigma0,
            "window": args.window,
            "collect_steps": args.collect_steps,
            "weights": {
                "w_energy":   Config.w_energy,
                "w_smooth":   Config.w_smooth,
                "w_progress": Config.w_progress,
                "w_pose":     Config.w_pose,
                "w_alive":    Config.w_alive,
            }
        },
        "baseline":  {k: v for k, v in bl_metrics.items()
                      if not isinstance(v, np.ndarray)},
        "optimized": {k: v for k, v in opt_metrics.items()
                      if not isinstance(v, np.ndarray)},
        "energy_reduction_pct": reduction,
    }, output_dir)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()