import os
import time
import re
import sys

work_dir = os.getenv("HOME")
data_path = os.path.join(os.getenv("WORKDIR"), 'imagenet_tar')
save_dir = os.path.join(os.getenv("WORKDIR"), 'checkpoints')
config_path = os.path.join(os.getenv("CURRENT_DIR"), 'baseline.yaml')
platform = 'itp'
branch = 'retrain'

def ompi_rank():
    return int(os.environ.get('OMPI_COMM_WORLD_RANK'))


def get_master_ip():
    regexp = "[\s\S]*export[\s]*DLTS_SD_worker0_IP=([0-9.]+)[\s|s]*"

    with open("/dlts-runtime/env/init.env", 'r') as f:
        line = f.read()
    print("^^^^^^^^", line)
    match = re.match(regexp, line)
    if match:
        ip = str(match.group(1))
        print("master node ip is " + ip)
        return ip

# os.system("ifconfig")
if os.environ.get('AZ_BATCH_MASTER_NODE', None) is None:
    os.environ.setdefault("AZ_BATCH_MASTER_NODE", get_master_ip())

print("---------------------------", os.environ['AZ_BATCH_MASTER_NODE'])


def ompi_local_size():
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE'))


os.chdir(work_dir)
print(work_dir)
# print("current dir is {}:".format(os.getcwd()))
print("ompi_rank: {}, ompi_local_size: {}".format(ompi_rank(), ompi_local_size()))
os.system("ls -l")
print("*************os.environ: {}*****************".format(os.environ))
if ompi_rank() % ompi_local_size() == 0:
    os.system("nvidia-smi")
    print("*************os.enviro: {}\n*****************".format(os.environ))
    if os.path.isfile(os.path.join(os.environ['HOME'], 'gitdone.txt')):
        print("gitdone file exists!!!!!!!!")
        os.system("rm $HOME/gitdone.txt")

    if os.path.isfile(os.path.join(os.environ['HOME'], 'done.txt')):
        print("done file exists!!!!!!!!")
        os.system("rm $HOME/done.txt")

    if os.path.isdir('FastBaseline'):
        print("FastBaseline exist!!!!!!")
        print("current dir {} !!!!!!!!!".format(os.getcwd()))
        os.system("ls -l")
        os.system("rm -rf FastBaseline")
    # os.system("git clone https://github.com/DingXiaoH/ACNet.git")

    os.system("git clone https://b05eca24df070e0eb172481346d0fe9d8a962639@github.com/hwpengms/FastBaseline.git")
    os.system('echo done > $HOME/gitdone.txt')
    print("********* finish git clone {}**************".format(ompi_rank()))
else:
    while not os.path.exists(os.path.join(os.environ['HOME'], 'gitdone.txt')):
        print("wait for git clone")
        time.sleep(10)

# os.chdir('./ACNet')
os.chdir('./FastBaseline')
os.system(f"git checkout {branch}")
print("current dir {} (after cd ./FastBaseline)".format(os.getcwd()))
# os.chdir('./FastBaseline')


time.sleep(30)

if ompi_rank() % ompi_local_size() == 0:
    os.system("sh ./install.sh")
    print("finish build%d" % ompi_rank())
    time.sleep(10)
    os.system('echo done > $HOME/done.txt')
    print("********* finish buiild {}**************".format(ompi_rank()))
else:
    while not os.path.exists(os.path.join(os.environ['HOME'], 'done.txt')):
        time.sleep(20)

os.system("pwd")
os.system("ls -l")
time.sleep(30)

# os.system("/opt/miniconda/bin/python ./tools/train_base.py --model ours --model_selection 285 --ensemble_method boosting --temp 2\
#  --data {} --output {} --platform {} --config {}".format(data_path, save_dir, platform, config_path))

os.system("/opt/miniconda/bin/python ./tools/train_base.py --model ours --num_head 2 --arch-list 4 2 0 3 3 4 2 3 3 1 3 3 2 0 4 1 2 4 3 3 5 5 1 0 3 5 5 1 0 \
 --data {} --output {} --platform {} --config {}".format(data_path, save_dir, platform, config_path))
# os.system("/opt/miniconda/bin/python ./tools/supernet.py --sched spos_linear \
# --pool_size 10 --meta_sta_epoch 20 --update_2nd --update_iter 200 \
# --epochs 120  --batch-size 128 --warmup-epochs 0 \
# --lr 0.5  --opt-eps 0.001 --block ghost \
# --color-jitter 0.06 --drop 0.  -j 8 --num-classes 1000 --flops_minimum 0 --flops_maximum 600 \
# --data {} --output {} --platform {}".format(data_path, save_dir, platform))

if ompi_rank() % ompi_local_size() == 0:
    os.chdir(os.environ['HOME'])
    os.system("rm -rf FastBaseline")
    os.system("rm done.txt")
    os.system("rm gitdone.txt")

