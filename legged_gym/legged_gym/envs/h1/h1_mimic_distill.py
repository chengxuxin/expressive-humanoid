from isaacgym.torch_utils import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import torch
from legged_gym.utils.math import *
from legged_gym.envs.h1.h1_mimic import H1Mimic, global_to_local, local_to_global
from isaacgym import gymtorch, gymapi, gymutil

import torch_utils

class H1MimicDistill(H1Mimic):
    def check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        ref_keybody_dev = self._reward_tracking_demo_key_body() < 0.2
        self.reset_buf |= ref_keybody_dev
        
        self.reset_buf |= self.time_out_buf
    
    def compute_observations(self):
        super().compute_observations()
        self.extras["decoder_demo_obs"] = self.get_decoder_demo_obs()  # root vel

    def get_decoder_demo_obs(self):
        return self._curr_demo_obs_buf[:, 19:19+3].clone()

    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _resample_commands(self, env_ids):
        self.commands[env_ids, :] = 0
    
            