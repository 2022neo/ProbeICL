set -e
gpu_ids=$1
exps_dir=$2
task_name=$3

for ctrs_loss_penalty in 1 0.1; do
    for label_loss_penalty in 0.001 1; do
        for ortho_loss_penalty in 1 10; do
            for dropout in 0.2 0.1; do
                for rand_neg in 0 1; do
                    for multi_ctrs in 0 1; do
                        for top_k in 80 190 220; do
                            filter_positive=1
                            CUDA_VISIBLE_DEVICES=$gpu_ids python training_retriever.py \
                            --exps_dir $exps_dir --task_name $task_name --learning_rate 0.00001 --epoches 6 \
                            --temperature 1 --hard_mask 1 --mask_type 3 --dropout $dropout \
                            --ctrs_loss_penalty $ctrs_loss_penalty --label_loss_penalty $label_loss_penalty --ortho_loss_penalty $ortho_loss_penalty \
                            --top_k $top_k --multi_ctrs $multi_ctrs --rand_neg $rand_neg --filter_positive $filter_positive
                        done
                    done
                done
            done
        done
    done
done
