from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.h1.h1_mimic import *
from legged_gym.envs.h1.h1_mimic_eval import H1MimicEval
import os
from legged_gym import LEGGED_GYM_ROOT_DIR, ASE_DIR

class H1MimicViewMotion(H1MimicEval):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.save = True
        # cfg.motion.motion_type = "single"
        # cfg.motion.motion_name = "13_20"  # comes from cmd line arg
        cfg.motion.num_envs_as_motions = True
        cfg.motion.no_keybody = True
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # assert only one env and one motion
        self.motion_frame = torch.zeros_like(self._motion_times, dtype=torch.long, device=self.device)
        self.saved_flags = torch.zeros_like(self._motion_times, dtype=torch.bool, device=self.device)
        self.total_frames = self._motion_lib.get_motion_num_frames(self._motion_ids)
        self.motor_strength *= 0.0

        self.motion_names = self._motion_lib.get_motion_files(self._motion_ids)
        self.motion_names = [name.split("/")[-1].split(".")[0] for name in self.motion_names]
        self.to_save_list = []
        for i in range(self.num_envs):
            self.to_save_list.append(torch.zeros((self.total_frames[i], len(self._key_body_ids_sim), 3), dtype=torch.float32, device=self.device))
        # self.local_rigid_body_pos_path = os.path.join(ASE_DIR, f"ase/poselib/data/retarget_npy/{cfg.motion.motion_name}_key_bodies.npy")

        
    def post_physics_step(self):
        super().post_physics_step()
        if hasattr(self, 'motion_frame'):
            # print(f"frame {self.motion_frame.item()}/{self.total_frames}")
            self._motion_sync()
            done_percentage = (self.motion_frame.float().sum() / self.total_frames.float().sum()).item()
            print(f"done percentage: {done_percentage}")
            if self.save:
                save_ids = torch.where(self.motion_frame == self.total_frames - 1)[0]
                if len(save_ids) > 0:
                    for i in save_ids:
                        if not self.saved_flags[i]:
                            assert self.motion_frame[i] == self.to_save_list[i].shape[0] - 1
                            np.save(os.path.join(ASE_DIR, f"ase/poselib/data/retarget_npy/{self.motion_names[i]}_key_bodies.npy"), self.to_save_list[i].cpu().numpy())
                            print(f"saved {self.motion_names[i]}")
                    self.saved_flags[save_ids] = True
                    
                if torch.all(self.saved_flags):
                    print("all saved")
                    exit()
            self.motion_frame[~self.saved_flags] += 1

        return
    
    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    def compute_reward(self):
        return
    
    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids
        # print(self._motion_times[self.lookat_id])
        # motion_times = self.episode_length_buf * self._motion_dt
        motion_fps = self._motion_lib.get_motion_fps(self._motion_ids)
        
        motion_times = 1.0 / motion_fps * self.motion_frame

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
           = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)

        # if not self.save:
        #     root_pos[:, :2] = (self._curr_demo_root_pos - self.init_root_pos_global_demo + self.init_root_pos_global)[:, :2]
        #     dof_pos=self._curr_demo_obs_buf[:, :self.num_dof]
        #     root_rot = self._curr_demo_quat.clone()
        if not self.save:
            root_pos[:, 2] = 30
        # root_rot[:, :3] = 0
        # root_rot[:, 3] = 1
        # root_pos[:, 2] -= 0.3
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
        if self.save:
            local_end_pos = global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim, :3], self.root_states[:, :3])
            for i in range(self.num_envs):
                self.to_save_list[i][self.motion_frame[i], :, :] = local_end_pos[i, :]
        
        return
    
    def update_demo_obs(self):
        return
        if not self.save:
            super().update_demo_obs()
        else:
            demo_motion_times = self._motion_demo_offsets + self._motion_times[:, None]  # [num_envs, demo_dim]
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos\
                = self._motion_lib.get_motion_state(self._motion_ids.repeat_interleave(self._motion_num_future_steps), demo_motion_times.flatten())
            local_key_body_pos = torch.zeros(self.num_envs*self._motion_num_future_steps, self._num_key_bodies*3, device=self.device)  # Fake key body pos
            dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)
            
            self._curr_demo_root_pos[:] = root_pos.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]
            self._curr_demo_quat[:] = root_rot.view(self.num_envs, self._motion_num_future_steps, 4)[:, 0, :]
            self._curr_demo_root_vel[:] = root_vel.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]
            
            demo_obs = build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, local_key_body_pos, self._dof_offsets)
            self._demo_obs_buf[:] = demo_obs.view(self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo)[:]

