import sys
import argparse
import azureml.core 
from azureml.core import Experiment, Workspace, Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import MpiConfiguration, RunConfiguration, DEFAULT_GPU_IMAGE
from azureml.train.estimator import Estimator
from azureml.widgets import RunDetails
from azureml.core import Environment
from azureml.contrib.core.compute.k8scompute import AksCompute
from azureml.contrib.core.k8srunconfig import K8sComputeConfiguration
import signal
from azureml.core import Keyvault

run=None

Resources = {
    "itplabrr1cl1": {"subscription_id": "46da6261-2167-4e71-8b0d-f4a45215ce61", "resource_group": "researchvc",
                     "workspace_name": "resrchvc", "workspace_region": "westus2", 'cluster_name': "itplabrr1cl1",
                     'vm_size': "STANDARD_ND40rs_V2"},
    "itpeastusv100cl": {"subscription_id": "46da6261-2167-4e71-8b0d-f4a45215ce61", "resource_group": "researchvc-eus",
                        "workspace_name": "resrchvc-eus", "workspace_region": "eastus",
                        'cluster_name': "itpeastusv100cl", 'vm_size': "STANDARD_ND40S_V2"},
    "itpseasiav100cl": {"subscription_id": "46da6261-2167-4e71-8b0d-f4a45215ce61", "resource_group": "researchvc-sea",
                        "workspace_name": "resrchvc-sea", "workspace_region": "southeastasia",
                        'cluster_name': "itpseasiav100cl", 'vm_size': "STANDARD_ND40S_V2"},
    "itpscusv100cl": {"subscription_id": "46da6261-2167-4e71-8b0d-f4a45215ce61", "resource_group": "researchvc-sc",
                      "workspace_name": "resrchvc-sc", "workspace_region": "southcentralus",
                      'cluster_name': "itpscusv100cl", 'vm_size': "STANDARD_ND40S_V2"},
    "itpeusp100cl": {"subscription_id": "46da6261-2167-4e71-8b0d-f4a45215ce61", "resource_group": "researchvc-eus",
                      "workspace_name": "resrchvc-eus", "workspace_region": "eastus",
                      'cluster_name': "itpeusp100cl", 'vm_size': "STANDARD_NC24RS_V2"},
    "itpeusp40cl": {"subscription_id": "46da6261-2167-4e71-8b0d-f4a45215ce61", "resource_group": "researchvc-eus",
                     "workspace_name": "resrchvc-eus", "workspace_region": "eastus",
                     'cluster_name': "itpeusp40cl", 'vm_size': "STANDARD_ND24RS_V1"},
}
# data_blobs={
#     "westus2":{"blob_container_name":"t-zhihua","blob_account_name":"vqawus","blob_account_key":"4heuSDtANaeyzsl3wN63tyH7AU+Bu/grZVcdRwxXJ1ibTLpNS1Ge1KXfadEKHzR5H1EiBz8DLbPFedCQCi7u6Q=="},
#     "southcentralus":{"blob_container_name":"t-zhihua","blob_account_name":"vqasc","blob_account_key":"0Fgx22TON1LhpyZ6DEJY5pDmmdGl2YgkEruRVkpSNbIp6nN1IXcV2FbgSGGs/67WB+CRvw0lulaVsErxvjFkWA=="},
#     "westeurope":{"blob_container_name":"t-zhihua","blob_account_name":"sharedws0645472567","blob_account_key":"j5uifC7gQ/YdesG1PyWiYmItJT1vDFQLcTK1oZgBLydpnDZgWjhElO0NobeD9cMtisiF/rpGnNvJdhLCRVuQmQ=="},
#     "japaneast":{"blob_container_name":"t-zhihua","blob_account_name":"msramsmjp","blob_account_key":"MK8Jb85zfZwyOsP6HW/TDq58M76WRZQpATYlyPtEzV+9vuDKUrl6UWPZ5SMr4gsTE4sGCnUPAO56QwJddUJTHw=="},
# }
data_blobs={
    "southcentralus": {"blob_container_name":"v-miche2","blob_account_name":"azsussc","blob_account_key":"arfNxv4j360fnxgipU0zYpoyCwVQANS8Pws+592T5qfSiiFfBVs1JNZhE5Vovw45SwdpdYqZeEAY+Z7jzn7/gg=="},
    "eastus": {"blob_container_name": "v-miche2", "blob_account_name": "azsuse2",
                "blob_account_key": "HFYUvAX9qBIYkGRZXLlWEIJcxdKQgPetFrRnrmH0hPjMFNakzv7+0W+i5bASaY5SMxE1Ho7cNiQODCs2XSVfyg=="},
    "westus2": {"blob_container_name": "v-miche2", "blob_account_name": "azsusw2",
                   "blob_account_key": "9YGymX2HFamDpX7T9nQYrnNkkRKfhRq1rNpjNyCH/cPxu4mY0gBEi1AaI3h7s4aDWIXitE2BTxnUgVbevHy0Rw=="},
    "southeastasia": {"blob_container_name": "v-miche2", "blob_account_name": "azssea",
                "blob_account_key": "KH10qX86FrKCU16FJ45fVFHp8TsmsHgnKFYMVEF+RRrf/RS1YbdZzdhJ2GokrIpkhNSq1Q55PrRzsQAkul95bA=="}
}

GPU_NUM={"STANDARD_NC24RS_V3":4,"STANDARD_NC24S_V3":4,"STANDARD_NC24S_V2":4,"STANDARD_NC24RS_V2":4,"STANDARD_NC6S_V3":1,'STANDARD_ND40s_V2':8,'STANDARD_ND40rs_V2':8,'STANDARD_ND40S_V2':8, 'STANDARD_ND24RS_V2':4, 'STANDARD_ND24RS_V1': 4}
PHILLY_SKUS={'G1':1,'G2':2,'G4':4,'G8':8,"G16":16}

def parse_args():
    parser = argparse.ArgumentParser("Azureml Script")
    parser.add_argument("--isPrepare",action="store_true",help="create the workspace and data storage")
    parser.add_argument("--name",type=str,help="experiment name")
    parser.add_argument("--cluster",type=str,help="cluster name", default="itpeastusv100cl")
    parser.add_argument("--gpus",type=int,help="total gpus", default=8)
    parser.add_argument("--philly_sku",type=str,help="philly sku")
    parser.add_argument("--command",type=str,help="command",default='python deit_begin.py')
    parser.add_argument("--branch",type=str,help="github branch", default="None")
    parser.add_argument("--entry_script",type=str,help="remote begin file",default="entry-script.py")
    parser.add_argument("--preemption",action='store_true',help="use promete job")
    parser.add_argument("--debug",action='store_true',help="use debug mode")
    parser.add_argument("--datastore_name", type=str, default="minghao")


    return parser


def create_source(args):
    # Configuration
    
    subscription_id = Resources[args.cluster]['subscription_id']
    resource_group = Resources[args.cluster]['resource_group']
    workspace_name = Resources[args.cluster]['workspace_name']
    workspace_region = Resources[args.cluster]['workspace_region']

    
    vm_size = Resources[args.cluster]['vm_size']
    
    

    # your blob_container
    blob_container_name = data_blobs[workspace_region]["blob_container_name"]
    blob_account_name = data_blobs[workspace_region]["blob_account_name"]
    blob_account_key = data_blobs[workspace_region]["blob_account_key"]
    # datastore_name="flickr_frames"+"_"+Resources[args.cluster]['workspace_region']

    datastore_name = args.datastore_name + '_' + workspace_region
    #prepare the workspace

    ws = None
    try:
        print("Connecting to workspace '%s'..." % workspace_name)
        ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    except:
        print("Workspace not accessible. Creating a new one...")
        try:
            ws = Workspace.create(
                name = workspace_name,
                subscription_id = subscription_id,
                resource_group = resource_group, 
                location = workspace_region,
                create_resource_group = False,
                exist_ok = True)
        except:
            print("Failed to connect to workspace. Quit with error.")
    print(ws.get_details())
    ws.write_config()

    #prepare the compute in the workspace
    try:
        ct = ComputeTarget(workspace=ws, name=args.cluster)
        print("Found existing cluster '%s'. Skip." % args.cluster)
    except ComputeTargetException:
        print("Creating new cluster '%s'..." % args.cluster)
        compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size, min_nodes=1, max_nodes=5)
        ct = ComputeTarget.create(ws, args.cluster, compute_config)
        ct.wait_for_completion(show_output=True)
    print(ct.get_status())
    
    if datastore_name not in ws.datastores:
        Datastore.register_azure_blob_container(
        workspace=ws, 
        datastore_name=datastore_name,
        container_name=blob_container_name,
        account_name=blob_account_name,
        account_key=blob_account_key
        )
        print("Datastore '%s' registered." % datastore_name)
    else:
        print("Datastore '%s' has already been regsitered." % datastore_name)

def submit_job(args):
    experiment_name = args.name
    subscription_id = Resources[args.cluster]['subscription_id']
    resource_group = Resources[args.cluster]['resource_group']
    workspace_name = Resources[args.cluster]['workspace_name']
    workspace_region = Resources[args.cluster]['workspace_region']

    vm_size = Resources[args.cluster]['vm_size']
    gpu_per_node=4
    if vm_size in GPU_NUM.keys() and args.philly_sku is None:
        gpu_per_node = GPU_NUM[vm_size]
    elif args.philly_sku is not None:
        gpu_per_node = PHILLY_SKUS[args.philly_sku]
    else:
        print("----!! sku error----")
    
    

    # your blob_container
    # blob_container_name = data_blobs[workspace_region]["blob_container_name"]
    # blob_account_name = data_blobs[workspace_region]["blob_account_name"]
    # blob_account_key = data_blobs[workspace_region]["blob_account_key"]
    blob_container_name = data_blobs[workspace_region]["blob_container_name"]
    blob_account_name = data_blobs[workspace_region]["blob_account_name"]
    blob_account_key = data_blobs[workspace_region]["blob_account_key"]
    # datastore_name="flickr_frames"+"_"+Resources[args.cluster]['workspace_region']

    datastore_name = args.datastore_name + '_' + workspace_region

    # ws = Workspace.from_config()
    ws = Workspace(subscription_id = subscription_id,
                resource_group = resource_group,  
                workspace_name = workspace_name) 

    if datastore_name not in ws.datastores:
        Datastore.register_azure_blob_container(
        workspace=ws, 
        datastore_name=datastore_name,
        container_name=blob_container_name,
        account_name=blob_account_name,
        account_key=blob_account_key
        )
        print("Datastore '%s' registered." % datastore_name)
    else:
        print("Datastore '%s' has already been regsitered." % datastore_name)

    target_name_list=[]
    for key, target in ws.compute_targets.items():
        target_name_list.append(target.name)
        if type(target) is AksCompute:
            print('Found compute target:{}\ttype:{}\tprovisioning_state:{}\tlocation:{}'.format(target.name, target.type, target.provisioning_state, target.location))
    assert args.cluster in target_name_list
    ct = ComputeTarget(workspace=ws, name=args.cluster)
    ds = Datastore(workspace=ws, name=datastore_name)

    myenv = Environment(name="myenv")
    myenv.docker.enabled=True
    myenv.docker.base_image="silentchen/nas:stable"
    myenv.docker.shm_size="16G"
    myenv.docker.gpu_support=True
    myenv.python.user_managed_dependencies=True
    # myenv.environment_variables={"NCCL_IB_DISABLE":1,}

    mpi = MpiConfiguration()
    mpi.process_count_per_node = gpu_per_node

    # keyvault = ws.get_default_keyvault()
    # keyvault.set_secret(name="honxueKey", value = "")

    est =Estimator(
        compute_target=ct,
        node_count = args.gpus//gpu_per_node,
        distributed_training=mpi,
        source_directory="./src-remote",
        entry_script=args.entry_script,
        script_params={
        "--workdir": ds.as_mount(),
        "--command": args.command,
        },
        environment_definition=myenv
    )
    global run
    exp = Experiment(workspace=ws, name=experiment_name)
    
    if ws.compute_targets[args.cluster].type in ['Cmk8s']:
        k8sconfig = K8sComputeConfiguration()
        k8s = dict()
        k8s['gpu_count'] = 8
        if args.debug:
            k8s['enable_ssh']=True
            k8s['ssh_public_key']=''
        if args.preemption:
            k8s['preemption_allowed'] = True
            k8s['node_count_min'] = 1
        k8sconfig.configuration = k8s
        est.run_config.cmk8scompute = k8sconfig

    run = exp.submit(est)
    if 'ipykernel' in sys.modules:
        RunDetails(run).show()
    else:
        run.wait_for_completion(show_output=True)
    # print(run.get_details())

if __name__=="__main__":
    parse = parse_args()
    args = parse.parse_args()
    if args.isPrepare:
        create_source(args)
    # signal.signal(signal.SIGINT,cancel_job)
    submit_job(args)