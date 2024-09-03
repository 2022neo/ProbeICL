set -e
gpu_ids=$1
exps_dir=$2
task_name=$3

sh scripts/inference.sh $gpu_ids $exps_dir $task_name

for ortho_loss_penalty in 1; do
    for ctrs_loss_penalty in 1 0.001; do
        for label_loss_penalty in 0.001 1; do
            for neg_step in 1 -1; do
                for pos_step in -1 1; do
                    for lr_flag in 1 0; do
                        CUDA_VISIBLE_DEVICES=$gpu_ids /home/jiyuwen/micromamba/envs/icl/bin/python training_retriever.py \
                        --exps_dir $exps_dir --task_name $task_name --epoches 10 --learning_rate 0.00001 \
                        --ortho_loss_penalty $ortho_loss_penalty --ctrs_loss_penalty $ctrs_loss_penalty --label_loss_penalty $label_loss_penalty \
                        --temperature 1 --multi_ctrs 1 --hard_mask 1 --rand_neg 0 \
                        --dropout 0.1 --mask_type 3 --top_k 80 --lr_flag $lr_flag --pos_step $pos_step --neg_step $neg_step 
                        # inference
                        sh scripts/inference.sh $gpu_ids $exps_dir $task_name

                        CUDA_VISIBLE_DEVICES=$gpu_ids /home/jiyuwen/micromamba/envs/icl/bin/python training_retriever.py \
                        --exps_dir $exps_dir --task_name $task_name --epoches 10 --learning_rate 0.00001  \
                        --ortho_loss_penalty $ortho_loss_penalty --ctrs_loss_penalty $ctrs_loss_penalty --label_loss_penalty $label_loss_penalty\
                        --temperature 1 --multi_ctrs 1 --hard_mask 1 --rand_neg 0 \
                        --dropout 0.2 --mask_type 3 --top_k 60 --lr_flag $lr_flag --pos_step $pos_step --neg_step $neg_step 
                        # inference
                        sh scripts/inference.sh $gpu_ids $exps_dir $task_name

                        CUDA_VISIBLE_DEVICES=$gpu_ids /home/jiyuwen/micromamba/envs/icl/bin/python training_retriever.py \
                        --exps_dir $exps_dir --task_name $task_name --epoches 10 --learning_rate 0.00001 \
                        --ortho_loss_penalty $ortho_loss_penalty --ctrs_loss_penalty $ctrs_loss_penalty --label_loss_penalty $label_loss_penalty \
                        --temperature 1 --multi_ctrs 1 --hard_mask 1 --rand_neg 0 \
                        --dropout 0.3 --mask_type 3 --top_k 80 --lr_flag $lr_flag --pos_step $pos_step --neg_step $neg_step 
                        # inference
                        sh scripts/inference.sh $gpu_ids $exps_dir $task_name
                    done
                done
            done
        done
    done
done
# {"ckpt": "e01_a60.pt", "acc": 0.77, "aug_acc": 0.76, "train_loss": 774.2, "avg_shot": 3.0, "epoches": 10, "lr": 1e-05, "k_shot": 3, "use_choosen": 0, "label_penalty": 0.001, "ortho_penalty": 1.0, "ctrs_penalty": 1.0, "tau": 1.0, "hard_mask": 1, "hard_neg": 1, "mask_type": 3, "multi_ctrs": 1, "lm_name": "EleutherAI/gpt-neo-2.7B", "top_k": 80, "dropout": 0.3}
# {"ckpt": "e02_a60.pt", "acc": 0.76, "aug_acc": 0.75, "train_loss": 723.7, "avg_shot": 3.0, "epoches": 10, "lr": 1e-05, "k_shot": 3, "use_choosen": 0, "label_penalty": 0.001, "ortho_penalty": 1.0, "ctrs_penalty": 1.0, "tau": 1.0, "hard_mask": 1, "hard_neg": 1, "mask_type": 3, "multi_ctrs": 1, "lm_name": "EleutherAI/gpt-neo-2.7B", "top_k": 60, "dropout": 0.2}
# {"ckpt": "e06_a62.pt", "acc": 0.76, "aug_acc": 0.76, "train_loss": 679.5, "avg_shot": 3.0, "epoches": 10, "lr": 1e-05, "k_shot": 3, "use_choosen": 0, "label_penalty": 0.001, "ortho_penalty": 1.0, "ctrs_penalty": 1.0, "tau": 1.0, "hard_mask": 1, "hard_neg": 1, "mask_type": 3, "multi_ctrs": 1, "lm_name": "EleutherAI/gpt-neo-2.7B", "top_k": 80, "dropout": 0.1}

