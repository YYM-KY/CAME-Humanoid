import numpy as np
import imageio.v2 as imageio
import os


class TrajectoryExecutor:
    """Executes a given action sequence and optionally records a video."""

    def __init__(self, env, actions: np.ndarray):
        self.env = env
        self.actions = actions

    def run(self,
            render: bool = False,
            save_video_path: str = None) -> dict:
        """
        Run the trajectory.

        Args:
            render: If True, render frames for video.
            save_video_path: Path to save the video (if render=True).

        Returns:
            Dictionary containing metrics: total_energy, progress, alive_steps,
            energy_curve, root_x_curve, n_frames.
        """
        frames = []
        obs, info = self.env.reset()
        first_x = info.get("root_x", 0.0)
        energy_curve = []
        root_x_curve = [first_x]
        alive_steps = 0

        for t, action in enumerate(self.actions):
            if render:
                frame = self.env.render()
                if isinstance(frame, np.ndarray) and frame.ndim == 3:
                    frames.append(frame)

            obs, reward, terminated, truncated, info = self.env.step(action)
            energy_curve.append(info.get("energy", 0.0))
            root_x_curve.append(info.get("root_x", root_x_curve[-1]))

            if not (terminated or truncated):
                alive_steps += 1
            else:
                print(f"  [Executor] Terminated at step {t+1}.")
                # Capture one extra frame if rendering
                if render:
                    frame = self.env.render()
                    if isinstance(frame, np.ndarray) and frame.ndim == 3:
                        frames.append(frame)
                break

        if save_video_path and frames:
            os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
            imageio.mimsave(save_video_path, frames, fps=30)
            print(f"  [Executor] Video saved ({len(frames)} frames): {save_video_path}")
        elif render and not frames:
            print("  [Executor] Warning: render mode enabled but no frames captured. "
                  "Ensure environment was created with render=True.")

        return {
            "total_energy": float(np.sum(energy_curve)),
            "progress":     root_x_curve[-1] - first_x,
            "alive_steps":  alive_steps,
            "energy_curve": np.array(energy_curve),
            "root_x_curve": np.array(root_x_curve),
            "n_frames":     len(frames),
        }