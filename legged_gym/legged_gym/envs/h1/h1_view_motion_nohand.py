from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from scipy.spatial.transform import Rotation as R
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/cxx/humanoid_yd/ASE/ase")
sys.path.append("/home/cxx/humanoid_yd/ASE/ase/utils")

from motion_lib import MotionLib

class H1ViewMotionNoHand(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        self._dof_body_ids = [1, 2, 3, # Hip, Knee, Ankle
                              4, 5, 6,
                              7,       # Torso
                              8, 9, # Shoulder, Elbow, Hand
                              10, 11]  # 11
        self._valid_dof_body_ids = torch.ones(len(self._dof_body_ids)+2*4, device=self.device, dtype=torch.bool)
        self._valid_dof_body_ids[-1] = 0
        self._valid_dof_body_ids[-6] = 0
        self._dof_offsets = [0, 3, 4, 5, 8, 9, 10, 
                             11, 
                             14, 15, 18, 19]  # 12
        key_bodies = ["pelvis", "torso_link"]
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)

        motion_file = "/home/cxx/humanoid_yd/ASE/ase/poselib/data/07_01_cmu_h1.npy"
        self._load_motion(motion_file)

        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)
        self._motion_dt = sim_params.dt

        self.post_physics_step()


    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.actor_handles[0]
        body_ids = []

        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids
    
    
    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)  # +2 for hand dof not used
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return
    
    def step(self, actions):
        actions = self.reindex(actions)

        actions.to(self.device)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape) * 0
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.extras["delta_yaw_ok"] = self.delta_yaw < 0.6
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def post_physics_step(self):
        super().post_physics_step()
        self._motion_sync()
        return
    
    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids
        motion_times = self.episode_length_buf * self._motion_dt

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
           = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        
        dof_pos_hip = dof_pos[:, :3].clone()  # order in urdf zyx, order in motion xyz
        dof_pos[:, :3] = 0
        dof_pos[:, 0] = dof_pos_hip[:, 2]
        dof_pos[:, 1] = dof_pos_hip[:, 0]
        dof_pos[:, 2] = dof_pos_hip[:, 1]

        dof_pos_hip = dof_pos[:, 5:8].clone()  # order in urdf zyx, order in motion xyz
        dof_pos[:, 5:8] = 0
        dof_pos[:, 5] = dof_pos_hip[:, 2]
        dof_pos[:, 6] = dof_pos_hip[:, 0]
        dof_pos[:, 7] = dof_pos_hip[:, 1]


        dof_pos_shoulder = dof_pos[:, 11:14].clone()  # order in urdf yxz, order in motion xyz
        dof_pos[:, 11:14] = 0
        dof_pos[:, 11] = dof_pos_shoulder[:, 1]
        dof_pos[:, 12] = dof_pos_shoulder[:, 0]
        dof_pos[:, 13] = dof_pos_shoulder[:, 2]

        dof_pos_shoulder = dof_pos[:, 15:18].clone()  # order in urdf yxz, order in motion xyz
        dof_pos[:, 15:18] = 0
        dof_pos[:, 15] = dof_pos_shoulder[:, 1]
        dof_pos[:, 16] = dof_pos_shoulder[:, 0]
        dof_pos[:, 17] = dof_pos_shoulder[:, 2]

        # dof_pos = dof_pos[:, self._valid_dof_body_ids]
        # dof_vel = 0*dof_vel[:, self._valid_dof_body_ids]
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:10] = root_vel
        self.root_states[env_ids, 10:13] = root_ang_vel
        
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel
        return
    
    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
