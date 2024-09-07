set -e
gpu_ids=$1
exps_dir=$2
task_name=$3

sh scripts/inference.sh $gpu_ids $exps_dir $task_name

for ctrs_loss_penalty in 1 0.1; do
    for label_loss_penalty in 0.001 1; do
        for ortho_loss_penalty in 1 10; do
            for filter_positive in 1 0; do
                for rand_neg in 1 0; do
                    for multi_ctrs in 0 1; do
                        for top_k in 80 160 190; do
                            CUDA_VISIBLE_DEVICES=$gpu_ids python training_retriever.py \
                            --exps_dir $exps_dir --task_name $task_name --learning_rate 0.00001 --epoches 6 \
                            --temperature 1 --hard_mask 1 --mask_type 3 --dropout 0.2 \
                            --ctrs_loss_penalty $ctrs_loss_penalty --label_loss_penalty $label_loss_penalty --ortho_loss_penalty $ortho_loss_penalty \
                            --top_k $top_k --multi_ctrs $multi_ctrs --rand_neg $rand_neg --filter_positive $filter_positive
                            # inference
                            sh scripts/inference.sh $gpu_ids $exps_dir $task_name
                        done
                    done
                done
            done
        done
    done
done
