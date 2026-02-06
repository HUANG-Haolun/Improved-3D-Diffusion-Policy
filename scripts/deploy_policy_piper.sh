# Examples:

#   bash scripts/deploy_policy.sh idp3 gr1_dex-3d 0913_example
#   bash scripts/deploy_policy.sh dp_224x224_r3m gr1_dex-image 0913_example

dataset_path=/home/mihawk/Improved-3D-Diffusion-Policy/imitation_data4


save_ckpt=True

DEFAULT_ALG_NAME="idp3"
DEFAULT_TASK_NAME="piper"
DEFAULT_ADDITION_INFO="default7.2"
DEBUG=False
wandb_mode=offline

alg_name=${1:-$DEFAULT_ALG_NAME}     
task_name=${2:-$DEFAULT_TASK_NAME}
addition_info=${3:-$DEFAULT_ADDITION_INFO}
config_name=${alg_name}
seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=0
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


cd Improved-3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python deploy_piper.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.dataset.zarr_path=$dataset_path 



                                