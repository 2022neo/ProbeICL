# $ProbeICL$: Probe-Based Combinatorial Example Selection for In-Context Learning

# Setup <a name="setup"></a>
Our code is borrowed from [Se2](https://github.com/microsoft/LMOps/tree/main/se2) and [DPR](https://github.com/facebookresearch/DPR). 
Please configure an environment identical to [Se2](https://github.com/microsoft/LMOps/tree/main/se2).

**Pay attention !** Do not change the version of [en-core-web-sm](https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz) in [Se2](https://github.com/microsoft/LMOps/tree/main/se2).

# Quick Start <a name="quickstart"></a>
Our pipeline contains 3 main stages: Scoring, Training, and Inference. You can start from any stages once you provide intermedia data (for training) and checkpoints(for inference). We recommend that you start with the COPA task as it has a small amount of data.

We now have following tasks to run:
```
tasklist=(copa mrpc arc_c arc_e openbookqa qqp paws mnli qnli snli rte sst2 sst5 sentiment140 hellaswag ag_news roc_story roc_ending gigaword aeslc common_gen e2e_nlg)
```


## Scoring <a name="Scoring"></a>
```bash
sh scripts/score.sh ${gpu_ids} ${exps_dir} ${task_name} ${finder_L} ${GPUS_PER_CHUNK}
#such as:
sh scripts/score.sh "0,7" "/mnt/16t_3/jiyuwen/projects/DPR/exps" "copa" 600 1
```

You can load the model using a local path by modifying the script of ```scripts/score.sh```, otherwise it will be downloaded into ```${exps_dir}/_cache```.

It is recommended to set `${exps_dir}` to an absolute path.
The scored training data for ```${task_name}``` in ```${tasklist}```  will be saved to ```${exps_dir}/${task_name}/```. Randomly sampling ```${finder_L}``` examples for each query. Larger ```${finder_L}``` is expected for better performance.
The runtime of scoring process increases as the value of ```${finder_L}``` increases, but each dataset only requiring a single run of scoring. ```${GPUS_PER_CHUNK}``` specifies the number of GPUs in ```${gpu_ids}``` allocated for each subprocess. Increase ```${GPUS_PER_CHUNK}``` if the GPU memory is too small to run.


## Training <a name="Training"></a>
```bash
bash scripts/train.sh ${gpu_ids} ${exps_dir} ${task_name}
#such as:
sh scripts/train.sh "0,7" "/mnt/16t_3/jiyuwen/projects/DPR/exps" "mrpc"
```

We recommend writing unique training scripts for each task to try different parameters.
The results for ```${task_name}``` will be saved to ```${exps_dir}/${task_name}/inference```.

We also provide ray-based scripts for **Automatic Hyperparameter Search**:
```bash
CUDA_VISIBLE_DEVICES="0,1,4,5,6,7,8,9" python training_retriever_opt.py --exps_dir "/mnt/16t_3/jiyuwen/projects/DPR/exps" --task_name "mrpc" --num_samples 3000 --gpus_per_trial 2 --cpus_per_trial 10 --train_ds -1
```

```${num_samples}``` represents the maximum number of parameter search iterations.
When CPU resource utilization is too low, you can decrease ```${cpus_per_trial}```; when individual GPU memory is less than 40GB, you can increase ```${gpus_per_trial}```. If ```${train_ds}>0```, the training set will be cut down to the size of ```${train_ds}``` for quick testing of algorithm performance (we set ```${train_ds}=400``` for COPA).



## Inference <a name="Inference"></a>
**This step can be skipped !** 
We have already included the execution of the ```inference.sh``` script by default in the ```train.sh``` script.

```bash
bash scripts/inference.sh ${gpu_ids} ${exps_dir} ${task_name}
#such as:
sh scripts/inference.sh "0,7" "/mnt/16t_3/jiyuwen/projects/DPR/exps" "copa"
```

The results for ```${task_name}``` will be saved to ```${exps_dir}/${task_name}/inference```. You can customize `scripts/analyse.ipynb` to analyze experimental results and determine the optimal parameter.