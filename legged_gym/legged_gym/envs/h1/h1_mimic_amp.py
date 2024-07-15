from isaacgym.torch_utils import *
import torch
from legged_gym.utils.math import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.h1.h1_mimic import H1Mimic, global_to_local, local_to_global
from legged_gym.envs.base.legged_robot import euler_from_quaternion

import torch_utils

class H1MimicAMP(H1Mimic):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        
        self._num_amp_obs_per_step = cfg.amp.num_obs_per_step
        self._num_amp_obs_steps = cfg.amp.num_obs_steps
        self._amp_obs_buf = torch.zeros((cfg.env.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=sim_device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        self._amp_obs_demo_buf = None

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    
    def reset_idx(self, env_ids, init=False):
        super().reset_idx(env_ids, init)
        if len(env_ids) != 0:
            self._compute_amp_observations(env_ids)
            self._init_amp_obs_default(env_ids)
        return
    
    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations() # latest on the left

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat
        return
    
    def _compute_amp_observations(self, env_ids=None):
        cur_key_body_pos_local = global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim, :3], self.root_states[:, :3])#.view(self.num_envs, -1)

        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations_curr(self.root_states[:, :3], self.base_quat, self.base_lin_vel, self.base_ang_vel,
                                                                self.dof_pos, self.dof_vel, cur_key_body_pos_local)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations_curr(self.root_states[env_ids, :3], self.base_quat[env_ids], self.base_lin_vel[env_ids], self.base_ang_vel[env_ids],
                                                                     self.dof_pos[env_ids], self.dof_vel[env_ids],
                                                                     cur_key_body_pos_local[env_ids] )
        return
    
    ######### demonstrations #########
    def fetch_amp_obs_demo(self, num_samples):

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = self._motion_lib.sample_motions(num_samples)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat
    
    def build_amp_obs_demo(self, motion_ids, motion_times0):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, local_key_body_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times, get_lbp=True)
        
        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos)
        return amp_obs_demo
    
    ######### utils #########
    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
    
    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return
    
    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return
    
    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step


#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos):
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)
    local_root_vel = quat_rotate(heading_rot, root_vel)

    roll, pitch, yaw = euler_from_quaternion(root_rot)
    return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)

# @torch.jit.script
def build_amp_observations_curr(root_pos, root_rot, local_root_vel, local_root_ang_vel, dof_pos, dof_vel, local_key_body_pos):
    roll, pitch, yaw = euler_from_quaternion(root_rot)
    return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)