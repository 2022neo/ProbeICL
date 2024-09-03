set -e
gpu_ids=$1
exps_dir=$2
task_name=$3
finder_L=$4

CUDA_VISIBLE_DEVICES=$gpu_ids  /home/jiyuwen/micromamba/envs/icl/bin/python3 init1_prepare_dataset.py --exps_dir $exps_dir --task_name $task_name --finder_L $finder_L
CUDA_VISIBLE_DEVICES=$gpu_ids  /home/jiyuwen/micromamba/envs/icl/bin/python3 init2_scoring_dataset.py --exps_dir $exps_dir --task_name $task_name