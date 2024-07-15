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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .a1.a1_parkour_config import A1ParkourCfg, A1ParkourCfgPPO
from .go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO
from .h1.h1_config import H1Cfg, H1CfgPPO
from .h1.h1_mimic_config import H1MimicCfg, H1MimicCfgPPO, H1MimicDistillCfgPPO
from .h1.h1_mimic_amp_config import H1MimicAMPCfg, H1MimicAMPCfgPPO
from .h1.h1_view_motion import H1ViewMotion
from .h1.h1_amp import H1AMP
from .h1.h1_mimic import H1Mimic
from .h1.h1_mimic_amp import H1MimicAMP
from .h1.h1_mimic_view_motion import H1MimicViewMotion
from .h1.h1_mimic_eval import H1MimicEval
from .h1.h1_mimic_distill import H1MimicDistill

import os
import ipdb

from legged_gym.utils.task_registry import task_registry

# task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
# task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
# task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
# task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
# task_registry.register( "a1", LeggedRobot, A1ParkourCfg(), A1ParkourCfgPPO() )
# task_registry.register( "go1", LeggedRobot, Go1RoughCfg(), Go1RoughCfgPPO() )
task_registry.register( "h1", LeggedRobot, H1Cfg(), H1CfgPPO() )
# task_registry.register( "h1_view", H1ViewMotion, H1Cfg(), H1CfgPPO() )
task_registry.register( "h1_amp", H1AMP, H1Cfg(), H1CfgPPO() )
task_registry.register( "h1_mimic", H1Mimic, H1MimicCfg(), H1MimicCfgPPO() )
task_registry.register( "h1_view", H1MimicViewMotion, H1MimicCfg(), H1MimicCfgPPO() )
task_registry.register( "h1_mimic_eval", H1MimicEval, H1MimicCfg(), H1MimicCfgPPO() )
task_registry.register( "h1_mimic_amp", H1MimicAMP, H1MimicAMPCfg(), H1MimicAMPCfgPPO() )
task_registry.register( "h1_mimic_distill", H1MimicDistill, H1MimicCfg(), H1MimicDistillCfgPPO() )




