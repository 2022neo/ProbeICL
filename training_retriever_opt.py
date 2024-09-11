import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray import train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from pathlib import Path
from utils.tools import getConfig
from functools import partial
import argparse
from training_retriever import train,valid,evaluate,prepare_trial
import os
from utils.tools import load_ckpt_cfg, load_retriever_ckpt, load_optimizer_ckpt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', 
                        type=str, help='we use single task in our experience, so it will be a task name', 
                        required=True)
    parser.add_argument('--exps_dir', 
                        type=str, help='Directory for saving all the intermediate and final outputs.', 
                        required=True)
    parser.add_argument('--gpus_per_trial', 
                        type=int, help='', 
                        default=2)
    parser.add_argument('--cpus_per_trial', 
                        type=int, help='', 
                        default=32)
    parser.add_argument('--num_samples', 
                        type=int, help='', 
                        default=30)
    args = parser.parse_args()
    args.config_file = str(Path(args.exps_dir)/args.task_name/'config.json')
    args.taskpath = str(Path(args.exps_dir)/args.task_name)
    return args


def trial(param, taskpath):
    cfgpath = str(Path(taskpath)/"config.json")
    config = getConfig(cfgpath,param)

    train_dataset,valid_dataset,test_dataset,llm,retriever,tensorizer,optimizer,scheduler,scaler,prompt_parser,task = prepare_trial(config)
    for epc in range(1,config.epoches+1):
        train_loss,acc = train(train_dataset, llm, retriever, tensorizer, optimizer, scheduler, scaler, prompt_parser, config, epc)
        valid_info = valid(valid_dataset, llm, retriever, tensorizer, config, task, epc)
        result_info = evaluate(test_dataset,train_dataset.prompt_pool,retriever,tensorizer,prompt_parser,task,config,llm,epc)
        score=result_info["test_info"]["score"]

        with tune.checkpoint_dir(epc) as checkpoint_dir:
            save_content = {
                'epoch':epc,
                'train_loss':train_loss,
                'train_acc':acc,
                'state_dict':retriever.state_dict(),
                'optimizer_state':optimizer.state_dict(),
                'config':config,
            }
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(save_content, path)
        tune.report(score=score,metric=result_info["test_info"])



def main(max_num_epochs=10):
    cmd_args = parse_args()
    taskpath = cmd_args.taskpath
    cfgpath = str(Path(taskpath)/"config.json")
    config = getConfig(cfgpath,cmd_args)

    # setup space of hyperparameters
    space = {
        "lr": tune.loguniform(1e-6, 1e-4),
        "ctrs_loss_penalty": tune.loguniform(1e-2, 1),
        "label_loss_penalty": tune.loguniform(1e-4, 10),
        "ortho_loss_penalty": tune.loguniform(1e-2, 100),
        "dropout": tune.choice([0.2, 0.1, 0.3]),
        "top_k": tune.choice([80, 190]),
        "rand_neg": tune.choice([0, 1]),
        "multi_ctrs": tune.choice([0, 1]),
        "filter_positive": tune.choice([0, 1]),
        "mask_type": tune.choice([0, 1, 2, 3]),
        "batch_size": tune.choice([8, 32]),
        "epoches": 6,
        "temperature": 1,
        "hard_mask":1,
    }
    # ASHAScheduler for early stopping
    scheduler = ASHAScheduler(
        metric="score",
        mode="max",
        max_t=max_num_epochs,
        grace_period=3,
        reduction_factor=2)
    # report in cmd
    reporter = CLIReporter(
        metric_columns=["score", "accuracy"])
    # start to train
    checkpoint_dir = Path(config.taskpath)/'inference'/'saved_retriever'
    result = tune.run(
        partial(trial, taskpath=config.taskpath),
        # allocate resource for training
        resources_per_trial={"cpu": config.cpus_per_trial, "gpu": config.gpus_per_trial},
        config=space,
        num_samples=config.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=checkpoint_dir
        )
 
    # find the best trial
    best_trial = result.get_best_trial("score", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final score: {}".format(best_trial.last_result["score"]))
    print("Best trial final metric: {}".format(best_trial.last_result["metric"]))
    print("Best trial final ckpt: {}".format(best_trial.checkpoint.value))