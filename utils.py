import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_output_dirs(base_dir: str) -> dict:
    dirs = {
        "trajectories": os.path.join(base_dir, "trajectories"),
        "videos":       os.path.join(base_dir, "videos"),
        "plots":        os.path.join(base_dir, "plots"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    return dirs


def save_summary(summary: dict, output_dir: str):
    def _convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict):       return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):       return [_convert(i) for i in obj]
        return obj
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(_convert(summary), f, indent=2)
    print(f"[Utils] Summary saved to: {path}")


def save_comparison_plot(baseline: dict, optimized: dict, output_dir: str):
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    if baseline:
        ax.plot(np.cumsum(baseline["energy_curve"]), label="Baseline (PPO)", lw=2)
    ax.plot(np.cumsum(optimized["energy_curve"]), label="CMA-ES optimized", lw=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative energy (J)")
    ax.set_title("Energy consumption")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    if baseline:
        ax.plot(baseline["root_x_curve"], label="Baseline", lw=2)
    ax.plot(optimized["root_x_curve"], label="CMA-ES", lw=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Root X (m)")
    ax.set_title("Forward progress")
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    if baseline and len(baseline["energy_curve"]) > 0:
        n = min(len(baseline["energy_curve"]), len(optimized["energy_curve"]))
        ax.plot(baseline["energy_curve"][:n], label="Baseline", lw=1, alpha=0.7)
        ax.plot(optimized["energy_curve"][:n], label="CMA-ES", lw=1, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Step energy (J)")
    ax.set_title("Per-step energy")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    path = os.path.join(plots_dir, "comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Utils] Comparison plot saved to: {path}")