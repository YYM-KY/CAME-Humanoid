import os
from datetime import datetime

class Config:
    env_name         = "Humanoid-v5"
    seed             = 42
    video_fps        = 30
    render           = False
    pretrained_model = "model.pt"
    collect_steps    = 500

    # CMA-ES parameters
    cma_sigma0   = 0.05    # Smaller search step, more conservative
    cma_maxiter  = 80
    cma_popsize  = 16
    cma_window   = 40
    cma_stride   = 20

    # Cost weights
    w_energy   = 2.0    # Increased: more emphasis on energy saving
    w_smooth   = 0.1    # Increased: more emphasis on smoothness
    w_progress = 1.5    # Reduced: ensure not falling
    w_pose     = 20.0   # Increased: stricter posture constraint
    w_alive    = 3.0    # Increased: survival

    # Pose indices in the observation vector
    height_idx = 0
    pitch_idx  = 3
    roll_idx   = 2

    # Pose bounds
    height_min = 1.0
    height_max = 1.6
    pitch_min  = -0.3
    pitch_max  = 0.3
    roll_min   = -0.3
    roll_max   = 0.3

    output_dir = None

    @classmethod
    def get_output_dir(cls):
        if cls.output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_cmaes_opt")
            return os.path.join(script_dir, "cmaes_runs", folder)
        return cls.output_dir