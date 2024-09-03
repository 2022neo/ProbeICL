# $ProbeICL$: PROBEICL: PROBE-BASED COMBINATORIAL EXAMPLE SELECTION FOR IN-CONTEXT LEARNING

# Setup <a name="setup"></a>
Our code is borrowed from [Se2](https://github.com/microsoft/LMOps/tree/main/se2) and [DPR](https://github.com/facebookresearch/DPR). 
Please configure an environment identical to [Se2](https://github.com/microsoft/LMOps/tree/main/se2).

**Pay attention !** Do not change the version of [en-core-web-sm](https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz) in [Se2](https://github.com/microsoft/LMOps/tree/main/se2).

# Quick Start <a name="quickstart"></a>
Our pipeline contains 3 main stages: Scoring, Training, and Inference. You can start from any stages once you provide intermedia data (for training) and checkpoints(for inference). We recommend that you start with the COPA task as it has a small amount of data.

We now have following tasks to run:
```
tasklist=(copa arc_c arc_e openbookqa mrpc qqp paws mnli qnli snli rte sst2 sst5 sentiment140 hellaswag ag_news roc_story roc_ending gigaword aeslc common_gen e2e_nlg)
```


## Scoring <a name="Scoring"></a>
```bash
sh scripts/score.sh ${gpu_ids} ${exps_dir} ${task_name} ${finder_L}
#such as:
sh scripts/score.sh "0,1,2,3,4,5,6,7,8,9" "/mnt/16t_3/jiyuwen/projects/DPR/exps" "copa" 600
```

The scored training data for ```${task_name}``` in ```${tasklist}```  will be saved to ```${exps_dir}/${task_name}/```. Randomly sampling ```${finder_L}``` examples for each query. Larger ```${finder_L}``` is expected for better performance.
The runtime of scoring process increases as the value of ```${finder_L}```$ increases, yet each dataset only requiring a single run of scoring.

## Training <a name="Training"></a>
```bash
bash scripts/train.sh ${gpu_ids} ${exps_dir} ${task_name}
#such as:
sh scripts/train.sh "0,7" "/mnt/16t_3/jiyuwen/projects/DPR/exps" "copa"
```

We recommend writing unique training scripts for each task to try different parameters.
The results for ```${task_name}``` will be saved to ```${exps_dir}/${task_name}/inference```

## Inference <a name="Inference"></a>
**This step can be skipped !** 
We have already included the execution of the ```inference.sh``` script by default in the ```train.sh``` script.

```bash
bash scripts/inference.sh ${gpu_ids} ${exps_dir} ${task_name}
#such as:
sh scripts/inference.sh "0,1,2,3,4,5,6,7,8,9" "/mnt/16t_3/jiyuwen/projects/DPR/exps" "copa"
```

The results for ```${task_name}``` will be saved to ```${exps_dir}/${task_name}/inference```

