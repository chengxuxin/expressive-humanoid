# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
from PIL import Image
from legged_gym.utils.helpers import get_load_path as get_load_path_auto
from tqdm import tqdm

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="jit"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    args.task = "h1_mimic_eval" if args.task == "h1_mimic" or args.task == "h1_mimic_amp" else args.task
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.motion.motion_curriculum = True
    env_cfg.env.num_envs = 2#2 if not args.num_envs else args.num_envs
    env_cfg.env.episode_length_s = 30
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.,
                                    "parkour_hurdle": 0.,
                                    "parkour_flat": 0.1,
                                    "parkour_step": 0.,
                                    "parkour_gap": 0., 
                                    "demo": 0.}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 5
    env_cfg.domain_rand.max_push_vel_xy = 2.5
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    if_distill = ppo_runner.if_distill

    if args.task == "h1_mimic_merge":
        # upper policy
        log_pth_upper = "../../logs/{}/".format(args.proj_name) + "070-20"
        log_pth_upper = get_load_path_auto(log_pth_upper, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
        log_pth_upper = os.path.dirname(log_pth_upper)
        path = os.path.join(log_pth_upper, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_upper = torch.jit.load(path, map_location=env.device)
    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
        estimator = ppo_runner.get_estimator_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)
    infos = {}
    infos["decoder_demo_obs"] = env.get_decoder_demo_obs() if if_distill else None
    

    if args.record_video:
        mp4_writers = []
        import imageio
        env.enable_viewer_sync = False
        for i in range(env.num_envs):
            motion_id = env._motion_lib.get_motion_files([env._motion_ids[i]])[0].split("/")[-1].split(".")[0]
            motion_description = env._motion_lib.get_motion_description(env._motion_ids[i])
            video_name = motion_id + "-" + motion_description +".mp4"
            run_name = log_pth.split("/")[-1]
            path = f"../../logs/videos_retarget/{run_name}"
            if not os.path.exists(path):
                os.makedirs(path)
            video_name = os.path.join(path, video_name)
            mp4_writer = imageio.get_writer(video_name, fps=25)
            mp4_writers.append(mp4_writer)
    
    if args.record_frame:
        env.enable_viewer_sync = False
        paths = []
        for i in range(env.num_envs):
            motion_id = env._motion_lib.get_motion_files([env._motion_ids[i]])[0].split("/")[-1].split(".")[0]
            motion_description = env._motion_lib.get_motion_description(env._motion_ids[i])
            frame_name = motion_id + "-" + motion_description
            run_name = log_pth.split("/")[-1]
            path = f"../../logs/videos_retarget/debug-{env.cfg.motion.motion_name}/{frame_name}"
            if not os.path.exists(path):
                os.makedirs(path)
            paths.append(path)
    
    if not args.record_video and not args.record_frame and not args.record_data:
        traj_length = 100*int(env.max_episode_length)
    else:
        traj_length = int(env.max_episode_length)
    if args.record_data:
        data_buf = torch.zeros(env.num_envs, traj_length, 15)
    for i in tqdm(range(traj_length)):
        if args.use_jit:
            obs = obs[:, env.cfg.env.n_feature:]
            actions = policy_jit(obs.detach())
        elif if_distill:
            obs_student = infos["decoder_demo_obs"].clone()
            obs_student[:, :2] = env.commands[:, :2]
            
            delta_yaw = env.commands[:, 3]
            obs[:, 5] = torch.sin(delta_yaw)
            obs[:, 6] = torch.cos(delta_yaw)
            actions = ppo_runner.alg.student_actor(obs.detach(), obs_student, hist_encoding=True)
        elif args.task == "h1_mimic_merge":
            actions_lower = policy(obs.detach(), hist_encoding=True)
            # upper, featuren comes from history so remove here
            obs_upper = env.compute_observations_upper().detach()
            actions_upper = policy_upper(obs_upper[:, H1MimicUpperCfg.env.n_feature:])
            actions = torch.cat((actions_lower, actions_upper), dim=1)
        else:
            est_states = estimator(obs.detach()[:, train_cfg.estimator.prop_start:train_cfg.estimator.prop_start+train_cfg.estimator.prop_dim])
            # obs[:, train_cfg.estimator.priv_start:train_cfg.estimator.priv_start+train_cfg.estimator.priv_states_dim] = est_states
            actions = policy(obs.detach(), hist_encoding=True)
            
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        if args.record_video:
            imgs = env.render_record(mode='rgb_array')
            if imgs is not None:
                for i in range(env.num_envs):
                    mp4_writers[i].append_data(imgs[i])
        if args.record_frame:
            imgs = env.render_record(mode='rgb_array')
            if imgs is not None:
                transparent = np.array(imgs)[..., 0] != 0
                for i in range(env.num_envs):
                    data = imgs[i]
                    data[..., -1] = 255 * transparent[i]
                    image = Image.fromarray(data, 'RGBA')
                    # Save the image
                    image.save(os.path.join(paths[i], f"{env.global_counter}.png"))
                    # cv2.imwrite(os.path.join(paths[i], f"{env.global_counter}.png"), imgs[i])
        if args.record_data:
            data_buf[:, i, 0] = env._motion_times[:] 
            data_buf[:, i, 1:4] = env.base_lin_vel[:] 
            data_buf[:, i, 4:6] = env.contact_filt.float()[:]
            # data_buf[:, i, 6] = env.rand_vx_cmd
            data_buf[:, i, 7] = env._motion_ids[:]
            data_buf[:, i, 8:11] = env._curr_demo_keybody[:, -1]
            data_buf[:, i, 11:14] = env._curr_demo_keybody[:, 2]
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        
        # Interaction
        if env.button_pressed:
            print(f"env_id: {env.lookat_id:<{5}}"
                  f"motion file: {env._motion_lib.get_motion_files([env._motion_ids[env.lookat_id]])[0].split('/')[-1].split('.')[0]:<{10}}"
                  f"vx: {env.commands[env.lookat_id, 0]:<{8}.2f}"
                  f"vy: {env.commands[env.lookat_id, 1]:<{8}.2f}"
                  f"d_yaw: {env.commands[env.lookat_id, 3]:<{8}.2f}"
                  f"description: {env._motion_lib.get_motion_description(env._motion_ids[env.lookat_id]):<{30}}")
            print(env._motion_lib.get_motion_length(env._motion_ids[env.lookat_id]))
    
    if args.record_video:
        for mp4_writer in mp4_writers:
            mp4_writer.close()
    if args.record_data:
        data_buf = data_buf.cpu().numpy()
        np.save(os.path.join("../tests", "contact_vs_semantic.npy"), data_buf)
    

if __name__ == '__main__':
    args = get_args()
    play(args)
