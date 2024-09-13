import torch
from ray import tune
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint, get_checkpoint
import tempfile
from pathlib import Path
from utils.tools import getConfig
from functools import partial
import argparse
from training_retriever import train,valid,evaluate,prepare_trial
from utils.tools import load_ckpt_cfg, load_module_ckpt
import logging
import os
import numpy as np
import random
from ray.tune.search.hyperopt import HyperOptSearch

os.environ['TQDM_DISABLE'] = 'True'

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
    parser.add_argument('--train_ds', 
                        type=int, help='', 
                        default=-1)
    args = parser.parse_args()
    args.config_file = str(Path(args.exps_dir)/args.task_name/'config.json')
    args.taskpath = str(Path(args.exps_dir)/args.task_name)
    return args


def trial(param, cmd_args):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    cfgpath = str(Path(cmd_args.taskpath)/"config.json")
    config = getConfig(cfgpath,cmd_args)
    for k,v in param.items():
        config[k]=v

    start_epoch = 1
    train_dataset,valid_dataset,test_dataset,llm,retriever,tensorizer,optimizer,scheduler,scaler,prompt_parser,task = prepare_trial(config)
    
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_paths = list(Path(checkpoint_dir).rglob('*.pt'))
            if len(data_paths)>0:
                ckptfn = data_paths[0]
                config = load_ckpt_cfg(ckptfn)
                device = retriever.device
                load_module_ckpt(retriever,ckptfn,"state_dict",device)
                load_module_ckpt(optimizer,ckptfn,"optimizer_state",device)
                load_module_ckpt(scheduler,ckptfn,"scheduler_state",device)
            start_epoch = config["epoch"]+1


    for epc in range(start_epoch,config.epoches+1):
        train_loss,acc = train(train_dataset, llm, retriever, tensorizer, optimizer, scheduler, scaler, prompt_parser, config, epc)
        valid_info = valid(valid_dataset, llm, retriever, tensorizer, config, task, epc)

        with tempfile.TemporaryDirectory(dir=config.cache_dir) as checkpoint_dir:
            savepath = Path(checkpoint_dir)/config.ckptname
            save_content = {
                'epoch':epc,
                'train_loss':train_loss,
                'train_acc':acc,
                'state_dict':retriever.state_dict(),
                'optimizer_state':optimizer.state_dict(),
                'scheduler_state':scheduler.state_dict(),
                'config':config,
            }
            torch.save(save_content, str(savepath))
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            result_info = evaluate(test_dataset,train_dataset.prompt_pool,retriever,tensorizer,prompt_parser,task,config,llm,epc)
            score=result_info["test_info"]["score"]
            test_loss=result_info["test_info"]["loss"]
            valid_loss=valid_info["loss"]
            ray.train.report(
                {"score":score,"train_loss":train_loss,"test_loss":test_loss,"valid_loss":valid_loss},
                checkpoint=checkpoint,
            )


def main(max_num_epochs=10):
    cmd_args = parse_args()
    ray.init()
    # setup space of hyperparameters
    space = {
        "learning_rate": tune.choice([1e-7,1e-6,1e-5,1e-4,1e-3]),
        "ctrs_loss_penalty": tune.choice([1e-4,1e-3,1e-2,1e-1,1]),
        "label_loss_penalty": tune.choice([1e-4,1e-3,1e-2,1e-1,1,10,100]),
        "ortho_loss_penalty": tune.choice([1e-2,1e-1,1,10,100]),
        "dropout": tune.choice([0.2, 0.1, 0.3]),
        "top_k": tune.choice([40, 80, 190, 220, 400]),
        "rand_neg": tune.choice([0, 1]),
        "multi_ctrs": tune.choice([0, 1]),
        "filter_positive": tune.choice([0, 1]),
        "mask_type": tune.choice([0, 1, 2, 3]),
        "batch_size": tune.choice([8, 32, 128]),
        "epoches": tune.choice(list(range(1,7))),
        "temperature": tune.choice([10, 1, 0.1, 0.01]),
        "hard_mask":tune.choice([1, 0]),
    }
    current_best_params = [{
        "learning_rate": 1e-5,
        "ctrs_loss_penalty": 1,
        "label_loss_penalty": 0.001,
        "ortho_loss_penalty": 1,
        "dropout": 0.2,
        "top_k": 190,
        "rand_neg": 0,
        "multi_ctrs": 0,
        "filter_positive": 1,
        "mask_type": 3,
        "batch_size": 8,
        "epoches": 6,
        "temperature": 1,
        "hard_mask":1,
    }]
    # for more search alg, refer to https://docs.ray.io/en/latest/tune/api/suggestion.html
    searcher = HyperOptSearch(
        metric="score", mode="max",points_to_evaluate=current_best_params
    )
    # ASHAScheduler for early stopping
    scheduler = ASHAScheduler(
        metric="train_loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2)
    # report in cmd
    reporter = CLIReporter(
        metric_columns=["score", "train_loss", "test_loss","valid_loss"])
    # start to train
    ray_dir = Path(cmd_args.taskpath)/'inference'/'_ray'
    ray_dir.mkdir(exist_ok=True,parents=True)
    result = tune.run(
        partial(trial, cmd_args=cmd_args),
        resources_per_trial={"cpu": cmd_args.cpus_per_trial, "gpu": cmd_args.gpus_per_trial},
        config=space,
        name=cmd_args.task_name,
        num_samples=cmd_args.num_samples,
        max_concurrent_trials = torch.cuda.device_count()//cmd_args.gpus_per_trial,
        search_alg=searcher,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=str(ray_dir)
        )
 
    # find the best trial
    best_trial = result.get_best_trial("score", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final score: {}".format(best_trial.last_result["score"]))
    print("Best trial final metric: {}".format(best_trial.last_result["metric"]))
    print("Best trial final ckpt: {}".format(best_trial.checkpoint.value))

if __name__=='__main__':
    main()