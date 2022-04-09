import torch
import numpy as np


DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, num_envs, obs_dim, act_dim, size):
        self.num_envs = num_envs
        
        self.obs_buf = np.zeros(
            (size, num_envs, obs_dim), dtype=np.float32
        )  
        self.obs2_buf = np.zeros(
            (size, num_envs, obs_dim), dtype=np.float32
        )  
        self.act_buf = np.zeros(
            (size, num_envs, act_dim), dtype=np.float32
        )  # core.combined_shape(size, act_dim)
        self.rew_buf = np.zeros((size, num_envs), dtype=np.float32)
        self.done_buf = np.zeros((size, num_envs), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.weights = None

    def store(self, obs, act, rew, next_obs, done):
        # pdb.set_trace()
        self.obs_buf[self.ptr] = obs.detach().cpu().numpy()
        self.obs2_buf[self.ptr] = next_obs.detach().cpu().numpy()
        self.act_buf[self.ptr] = act  # .detach().cpu().numpy()
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):

        idxs = np.random.choice(
            self.size, size=min(batch_size, self.size), replace=False
        )
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )

        return {
            k: torch.tensor(v, dtype=torch.float32, device=DEVICE)
            for k, v in batch.items()
        }
