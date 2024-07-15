# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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


# import os
# import json

# from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
# from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
# import sys
# from tqdm import tqdm

# source fbx file path
# all_fbx_path = "data/cmu_fbx_all/"
# all_fbx_files = os.listdir(all_fbx_path)
# all_fbx_files.sort()
# for fbx_file in tqdm(all_fbx_files):
#     if fbx_file.endswith(".fbx"):
#         print(fbx_file)
#         motion = SkeletonMotion.from_fbx(
#             fbx_file_path=all_fbx_path + fbx_file,
#             root_joint="Hips",
#             fps=60
#         )
#         motion.to_file(f"data/npy/{fbx_file[:-4]}.npy")

import os
import multiprocessing
from tqdm import tqdm
from poselib.skeleton.skeleton3d import SkeletonMotion

def process_file(i, fbx_file, all_fbx_path):
    try:
        if fbx_file.endswith(".fbx"):
            print(i, fbx_file)
            motion = SkeletonMotion.from_fbx(
                fbx_file_path=all_fbx_path + fbx_file,
                root_joint="Hips",
                fps=60
            )
            motion.to_file(f"data/npy/{fbx_file[:-4]}.npy")
    except:
        print(f"Error in {fbx_file}")

def main():
    all_fbx_path = "data/cmu_fbx_all/"
    all_fbx_files = os.listdir(all_fbx_path)
    all_fbx_files.sort()

    all_fbx_filtered = []
    for fbx in all_fbx_files:
        npy = fbx.split(".")[0] + ".npy"
        target_motion_file = os.path.join(all_fbx_path, "../npy/" + npy)
        if os.path.exists(target_motion_file):
            print("Already exists, skip: ", fbx)
            continue
        all_fbx_filtered.append(fbx)
    all_fbx_filtered.sort()
    print(len(all_fbx_filtered))
    
    # Number of processes
    n_workers = multiprocessing.cpu_count()

    # Create a pool of worker processes
    with multiprocessing.Pool(n_workers) as pool:
        # Using starmap to pass multiple arguments to the process_file function
        list(tqdm(pool.starmap(process_file, [(i, fbx_file, all_fbx_path) for i, fbx_file in enumerate(all_fbx_filtered)]), total=len(all_fbx_files)))

if __name__ == "__main__":
    main()