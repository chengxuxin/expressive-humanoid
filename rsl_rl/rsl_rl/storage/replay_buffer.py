import torch
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, buffer_size, device):
        """Initialize a ReplayBuffer object.
        Arguments:
            buffer_size (int): maximum size of buffer
        """
        self.amp_obs = torch.zeros(buffer_size, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0
    
    def insert(self, amp_obs):
        """Add new states to memory."""
        
        num_obs = amp_obs.shape[0]
        start_idx = self.step
        end_idx = self.step + num_obs
        if end_idx > self.buffer_size:
            self.amp_obs[self.step:self.buffer_size] = amp_obs[:self.buffer_size - self.step]
            self.amp_obs[:end_idx - self.buffer_size] = amp_obs[self.buffer_size - self.step:]  # put the rest at the beginning
        else:
            self.amp_obs[start_idx:end_idx] = amp_obs

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_obs) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield self.amp_obs[sample_idxs].to(self.device)
