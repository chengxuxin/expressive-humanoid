import subprocess, os
import argparse
import socket
import yaml

def main(args, config):
    host_name = socket.gethostname()
    logs_path_src = os.path.join(config[args.server]["Path"], config["common"]["path"])
    logs_path_dst = os.path.join(config[host_name]["Path"], config["common"]["path"])
    src_host = config[args.server]["HostName"]

    folders = subprocess.check_output(["ssh", src_host, "cd", logs_path_src, "&&", "find", "*", "-maxdepth", "0", "-type", "d"]).decode("utf-8")
    folder_list = folders.split("\n")
    for name in folder_list:
        if len(name) >= 6:
            if name[:6] == args.exptid:
                exp_path_src = os.path.join(logs_path_src, name)
                break
    models = subprocess.check_output(["ssh", src_host, "cd", exp_path_src, "&&", "find", "*", "-maxdepth", "0"]).decode("utf-8")
    models = models.split("\n")
    models.sort(key=lambda m: '{0:0>15}'.format(m))
    model = models[-1]
    if args.ckpt:
        model = f"model_{args.ckpt}.pt"
    model_path_src = os.path.join(exp_path_src, model)
    model_path_dst = os.path.join(logs_path_dst, name, model)
    os.makedirs(os.path.dirname(model_path_dst), exist_ok=True)

    print(f"Copying from: {args.server:}{model_path_src}\nCopying to: {host_name:}{model_path_dst}")
    p = subprocess.Popen(["rsync", "-avz",
                         src_host + ":" + model_path_src, 
                         model_path_dst])
    sts = os.waitpid(p.pid, 0)

parser = argparse.ArgumentParser()
parser.add_argument('exptid', type=str)
parser.add_argument('server', type=str)
parser.add_argument('--ckpt', type=str, required=False, default='')


# parse server_config.yaml

server_config = yaml.load(open("server_config.yaml", "r"), Loader=yaml.FullLoader)

args = parser.parse_args()
main(args, server_config)

# sts = os.waitpid(p.pid, 0)