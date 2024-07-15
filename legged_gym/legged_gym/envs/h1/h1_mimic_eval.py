from isaacgym.torch_utils import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import torch
from legged_gym.utils.math import *
from legged_gym.envs.h1.h1_mimic import H1Mimic, global_to_local, local_to_global
from isaacgym import gymtorch, gymapi, gymutil

import torch_utils

class H1MimicEval(H1Mimic):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        height_cutoff = self.root_states[:, 2] < 0.5

        # motion_end = self.episode_length_buf * self.dt >= self._motion_lengths
        # self.reset_buf |= motion_end

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # self.time_out_buf |= motion_end

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= height_cutoff
    
    # def resample_motion_times(self, env_ids):
    #     return 0*self._motion_lib.sample_time(self._motion_ids[env_ids])

    def render_record(self, mode="rgb_array"):
        if self.global_counter % 2 == 0:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            imgs = []
            for i in range(self.num_envs):
                cam = self._rendering_camera_handles[i]
                root_pos = self.root_states[i, :3].cpu().numpy()
                cam_pos = root_pos + np.array([0, -2, 0.3])
                self.gym.set_camera_location(cam, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
                
                img = self.gym.get_camera_image(self.sim, self.envs[i], cam, gymapi.IMAGE_COLOR)
                w, h = img.shape
                imgs.append(img.reshape([w, h // 4, 4]))
            return imgs
        return None
    
    def _create_envs(self):
        super()._create_envs()
        if self.cfg.env.record_video or self.cfg.env.record_frame:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 720
            camera_props.height = 480
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                # root_pos = self.root_states[i, :3].cpu().numpy()
                # cam_pos = root_pos + np.array([0, 1, 0.5])
                cam_pos = np.array([2, 0, 0.3])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0*cam_pos))
    
    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reward_eval_ang_vel(self):
        return torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=-1))
            