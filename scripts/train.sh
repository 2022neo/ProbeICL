set -e
gpu_ids=$1
exps_dir=$2
task_name=$3

CUDA_VISIBLE_DEVICES=$gpu_ids python training_retriever.py \
--exps_dir $exps_dir --task_name $task_name --learning_rate 0.00001 --epoches 9 \
--temperature 1 --hard_mask 0 --mask_type 1 --dropout 0.2 --train_ds -1 \
--preference_penalty 0.1 --ortho_loss_penalty 100 \
--top_k 80 --multi_ctrs 1 --rand_ctx 0 --filter_positive 1 --batch_size 8 --reward_type 6 --gamma 0.1 --norm_option 0
