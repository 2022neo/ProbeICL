set -e
gpu_ids=$1
exps_dir=$2
task_name=$3
for file in "$exps_dir/$task_name/inference/saved_retriever"/*.pt; do
    if [ -f "$file" ]; then
        ckptname=$(basename "$file")
        echo "$ckptname"
        CUDA_VISIBLE_DEVICES=$gpu_ids python evaluating_tretriever.py --exps_dir $exps_dir --task_name $task_name --ckptname $ckptname
    else
        echo "No .pt files found."
    fi
done
# for file in "$exps_dir/$task_name/inference/saved_retriever"/*.pt; do
#     if [ -f "$file" ]; then
#         name=$(basename "$file")
#         rm "$file"
#         echo "remove $name"
#     else
#         echo "No .pt files found."
#     fi
# done