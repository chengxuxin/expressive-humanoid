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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1Cfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.1] # x,y,z [m]
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #     'left_hip_yaw_joint': 0.0,
        #     'left_hip_roll_joint': 0.,
        #     'left_hip_pitch_joint': -0.4,

        #     'left_knee_joint': 0.8,
        #     'left_ankle_joint': -0.45,

        #     'right_hip_yaw_joint': -0.0,
        #     'right_hip_roll_joint': -0.,
        #     'right_hip_pitch_joint': -0.4,

        #     'right_knee_joint': 0.8,
        #     'right_ankle_joint': -0.45,

        #     'torso_joint': -0.0,

        #     'left_shoulder_pitch_joint': 0.0,
        #     'left_shoulder_roll_joint': 0.0,
        #     'left_shoulder_yaw_joint': 0.0,
        #     'left_elbow_joint': 0.5,

        #     'right_shoulder_pitch_joint':0.0,
        #     'right_shoulder_roll_joint':0.0,
        #     'right_shoulder_yaw_joint':0.0,
        #     'right_elbow_joint':0.5
        # }

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.,
            'left_hip_pitch_joint': 0.,

            'left_knee_joint': 0.2,
            'left_ankle_joint': -0.2,

            'right_hip_yaw_joint': -0.0,
            'right_hip_roll_joint': -0.,
            'right_hip_pitch_joint': 0,

            'right_knee_joint': 0.2,
            'right_ankle_joint': -0.2,

            'torso_joint': -0.0,

            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.1,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.2,

            'right_shoulder_pitch_joint':0.0,
            'right_shoulder_roll_joint':-0.1,
            'right_shoulder_yaw_joint':0.0,
            'right_elbow_joint':1.2
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 80.}  # [N*m/rad]
        damping = {'joint': 1}     # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_custom_collision.urdf'
        torso_name = "torso_link"
        foot_name = "ankle"
        penalize_contacts_on = ["shoulder", "elbow", "knee", "hip"]
        terminate_after_contacts_on = ["torso_link", "hip_pitch_link", "hip_yaw_link", "knee_link"]#, "thigh", "calf"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
    
    class amp():
        num_obs_steps = 10
        num_obs_per_step = 19 + 3 # 19 joint angles + 3 base ang vel

class H1CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005

    class amp():
        amp_input_dim = H1Cfg.amp.num_obs_steps * H1Cfg.amp.num_obs_per_step
        amp_disc_hidden_dims = [512, 256]

        amp_replay_buffer_size = 10000
        amp_demo_buffer_size = 10000
        amp_demo_fetch_batch_size = 512
        amp_learning_rate = 1.e-4

        amp_reward_coef = 2.0