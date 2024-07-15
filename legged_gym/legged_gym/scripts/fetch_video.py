import subprocess, os
import argparse
import socket
import yaml

def fetch_videos(args, config):
    host_name = socket.gethostname()
    # video_path_src = os.path.join(config[args.server]["Path"], config["video"]["path"], args.exptid + "*")
    video_path_src = os.path.join(config[args.server]["Path"], config["video"]["path"])
    video_path_dst = os.path.join(config[host_name]["Path"], config["video"]["path"])
    src_host = config[args.server]["HostName"]


    print(f"Copying from: {args.server:}{video_path_src}\nCopying to: {host_name:}{video_path_dst}")
    if args.dry:
        rsync_arg = "-avnz"
    else:
        rsync_arg = "-avz"
    p = subprocess.Popen(["rsync", rsync_arg, "--ignore-times", "--size-only", "--progress",
                         src_host + ":" + video_path_src, 
                         video_path_dst])
    sts = os.waitpid(p.pid, 0)

parser = argparse.ArgumentParser()
parser.add_argument('server', type=str)
# parser.add_argument('exptid', type=str)
parser.add_argument("--dry", action="store_true")



# parse server_config.yaml

server_config = yaml.load(open("server_config.yaml", "r"), Loader=yaml.FullLoader)

args = parser.parse_args()
fetch_videos(args, server_config)

# sts = os.waitpid(p.pid, 0)