import os
import time
import re
import sys

work_dir = os.getenv("HOME")
data_path = os.path.join(os.getenv("WORKDIR"), 'imagenet_tar')
save_dir = os.path.join(os.getenv("WORKDIR"), 'checkpoints', 'deit', 'deit_small_batch_size_128_8_gpus_lr_1e-3_warmup_epochs_20')
platform = 'itp'

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

    if os.path.isdir('deit'):
        print("deit exist!!!!!!")
        print("current dir {} !!!!!!!!!".format(os.getcwd()))
        os.system("ls -l")
        os.system("rm -rf deit")
    # os.system("git clone https://github.com/DingXiaoH/ACNet.git")

    os.system("git clone https://b05eca24df070e0eb172481346d0fe9d8a962639@github.com/silent-chen/deit.git")
    os.system('echo done > $HOME/gitdone.txt')
    print("********* finish git clone {}**************".format(ompi_rank()))
else:
    while not os.path.exists(os.path.join(os.environ['HOME'], 'gitdone.txt')):
        print("wait for git clone")
        time.sleep(10)

os.chdir('./deit')
print("current dir {} (after cd ./deit)".format(os.getcwd()))


time.sleep(30)

if ompi_rank() % ompi_local_size() == 0:
    os.system("pip install -r ./requirements.txt")
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

os.system("/opt/miniconda/bin/python ./main.py --model ours --model deit_small_patch16_224 --batch-size 128 --lr 1e-3 --warmup-epochs 20 \
 --data-path {} --output_dir {} --platform {}".format(data_path, save_dir, platform))

if ompi_rank() % ompi_local_size() == 0:
    os.chdir(os.environ['HOME'])
    os.system("rm -rf deit")
    os.system("rm done.txt")
    os.system("rm gitdone.txt")

