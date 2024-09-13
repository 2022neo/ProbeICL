set -e
gpu_ids=$1
exps_dir=$2
task_name=$3
finder_L=$4
GPUS_PER_CHUNK=$5
llm_model="EleutherAI/gpt-neo-2.7B" #You can replace it with a local path, otherwise download from HuggingFace by default.

CUDA_VISIBLE_DEVICES=$gpu_ids  python init1_prepare_dataset.py --exps_dir $exps_dir --task_name $task_name --finder_L $finder_L --llm_model $llm_model

IFS=',' read -r -a GPULIST <<< "$gpu_ids"
NUM_GPUS=${#GPULIST[@]}
CHUNKS=$((NUM_GPUS / GPUS_PER_CHUNK))

if [ "$CHUNKS" -le 0 ]; then
    echo "CHUNKS $CHUNKS is not valid."
    exit  # Exit the script early as we found the CHUNKS that is not valid
fi
echo "Using $CHUNKS GPUs"
for IDX in $(seq 1 $CHUNKS); do
    START=$(((IDX-1) * GPUS_PER_CHUNK))
    LENGTH=$GPUS_PER_CHUNK # Length for slicing, not the end index
    CHUNK_GPUS=(${GPULIST[@]:$START:$LENGTH})
    # Convert the chunk GPUs array to a comma-separated string
    CHUNK_GPUS_STR=$(IFS=,; echo "${CHUNK_GPUS[*]}")
    ALL_GPUS_FREE=0
    while [ $ALL_GPUS_FREE -eq 0 ]; do
        ALL_GPUS_FREE=1  # Assume all GPUs are free initially
        for GPU_ID in $CHUNK_GPUS; do
            MEM_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID | tr -d '[:space:]')
            # Assuming a GPU is considered free if its memory usage is less than 100 MiB
            if [ "$MEM_USAGE" -ge 1000 ]; then
                ALL_GPUS_FREE=0
                echo "GPU $GPU_ID is in use. Memory used: ${MEM_USAGE}MiB."
                break  # Exit the loop early as we found a GPU that is not free
            fi
        done
        if [ $ALL_GPUS_FREE -eq 0 ]; then
            echo "Not all GPUs in chunk are free. Checking again in 10 seconds..."
            sleep 10
        fi
    done
    echo "CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR"
    CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR  python init2_scoring_dataset.py \
    --exps_dir $exps_dir --task_name $task_name --num_chunks $CHUNKS --chunk_id $(($IDX - 1)) \
    > $exps_dir/_cache/run_$IDX.log 2>&1 &
done

wait
CUDA_VISIBLE_DEVICES=$gpu_ids  python init3_merge_scored_dataset.py --exps_dir $exps_dir --task_name $task_name --num_chunks $num_chunks