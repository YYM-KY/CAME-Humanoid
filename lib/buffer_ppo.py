import torch


class PPOBuffer:
    """
    Buffer for storing trajectories
    """

    def __init__(self, obs_dim, act_dim, size, num_envs, device, gamma=0.99, gae_lambda=0.95):
        self.capacity = size
        self.device = device   # 🔥 必须加

        self.obs_buf = torch.zeros((size, num_envs, *obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size, num_envs, *act_dim), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.val_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.term_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.trunc_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.logprob_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)

        # 🔥 energy buffer（核心新增）
        self.energies = torch.zeros((size, num_envs), dtype=torch.float32, device=device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ptr = 0

    def store(self, obs, act, rew, val, term, trunc, logprob, energies):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.term_buf[self.ptr] = term
        self.trunc_buf[self.ptr] = trunc
        self.logprob_buf[self.ptr] = logprob

        # 🔥 存 energy
        self.energies[self.ptr] = torch.as_tensor(
            energies, dtype=torch.float32, device=self.device
        ).view(-1)

        self.ptr += 1

    def calculate_advantages(self, last_vals, last_terminateds, last_truncateds):
        assert self.ptr == self.capacity, "Buffer not full"

        with torch.no_grad():
            adv_buf = torch.zeros_like(self.rew_buf)
            last_gae = 0.0

            for t in reversed(range(self.capacity)):
                next_vals = last_vals if t == self.capacity - 1 else self.val_buf[t + 1]
                term_mask = 1.0 - (last_terminateds if t == self.capacity - 1 else self.term_buf[t + 1])
                trunc_mask = 1.0 - (last_truncateds if t == self.capacity - 1 else self.trunc_buf[t + 1])

                delta = self.rew_buf[t] + self.gamma * next_vals * term_mask - self.val_buf[t]
                last_gae = delta + self.gamma * self.gae_lambda * term_mask * trunc_mask * last_gae
                adv_buf[t] = last_gae

            ret_buf = adv_buf + self.val_buf
            return adv_buf, ret_buf

    def get(self):
        assert self.ptr == self.capacity
        self.ptr = 0
        return self.obs_buf, self.act_buf, self.logprob_buf