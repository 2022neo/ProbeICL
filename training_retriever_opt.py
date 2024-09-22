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
import shutil
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
                        default=3000)
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
                print(f"Start from epoch {start_epoch}! (Resume from {ckptfn})")

    for epc in range(start_epoch,config.epoches+1):
        train_loss = train(train_dataset, llm, retriever, tensorizer, optimizer, scheduler, scaler, prompt_parser, config, epc)
        valid_info = valid(valid_dataset, llm, retriever, tensorizer, config, task, epc)
        result_info = evaluate(test_dataset,train_dataset.prompt_pool,retriever,tensorizer,prompt_parser,task,config,llm,epc)

        with tempfile.TemporaryDirectory(dir=config.cache_dir) as checkpoint_dir:
            savepath = Path(checkpoint_dir)/config.ckptname
            save_content = {
                'epoch':epc,
                'train_loss':train_loss,
                'state_dict':retriever.state_dict(),
                'optimizer_state':optimizer.state_dict(),
                'scheduler_state':scheduler.state_dict(),
                'config':config,
            }
            torch.save(save_content, str(savepath))
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            score=result_info["test_info"]["score"]
            test_loss=result_info["test_info"]["loss"]
            valid_loss=valid_info["loss"]
            ray.train.report(
                {"score":score,"train_loss":train_loss,"test_loss":test_loss,"valid_loss":valid_loss},
                checkpoint=checkpoint,
            )

def reproduce_trial(best_trial):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    best_checkpoint_dir = best_trial.checkpoint.value
    ckptfn = str(list(Path(best_checkpoint_dir).rglob('*.pt'))[0])
    Path(config.checkpoint_dir).mkdir(exist_ok=True,parents=True)
    shutil.copy2(ckptfn, config.checkpoint_dir)
    config = load_ckpt_cfg(ckptfn)
    train_dataset,valid_dataset,test_dataset,llm,retriever,tensorizer,optimizer,scheduler,scaler,prompt_parser,task = prepare_trial(config)
    device = retriever.device
    load_module_ckpt(retriever,ckptfn,"state_dict",device)
    result_info = evaluate(test_dataset,train_dataset.prompt_pool,retriever,tensorizer,prompt_parser,task,config,llm,config["epoch"])
    return result_info

def main(max_num_epochs=10):
    cmd_args = parse_args()
    ray.init()
    # setup space of hyperparameters
    space = {
        "learning_rate": tune.choice([1e-7,1e-6,1e-5,1e-4,1e-3]),
        "preference_penalty": tune.choice([1e-4,1e-3,1e-2,1e-1,1,10,100]),
        "ortho_loss_penalty": tune.choice([1e-2,1e-1,1,10,100]),
        "dropout": tune.choice([0.2, 0.1, 0.3]),
        "gamma": tune.choice([0.01, 0.1, 0.5, 1]),
        "top_k": tune.choice([80, 160, 320]),
        "rand_ctx": tune.choice([0, 1]),
        "multi_ctrs": tune.choice([0, 1]),
        "filter_positive": tune.choice([0, 1]),
        "mask_type": tune.choice([0, 1]),
        "batch_size": tune.choice([4, 8]),
        "norm_option": tune.choice([0, 1]),
        "reward_type": tune.choice([0, 1, 2, 3, 4, 5, 6]),
        "epoches": tune.choice([1, 3, 6, 9, 12]),
        "temperature": tune.choice([1, 10]),
        "hard_mask":tune.choice([0, 1]),
    }
    current_best_params = [{
        "learning_rate": 1e-5,
        "preference_penalty": 0.1,
        "ortho_loss_penalty": 100,
        "dropout": 0.2,
        "gamma":0.1,
        "top_k": 80,
        "rand_ctx": 0,
        "multi_ctrs": 1,
        "filter_positive": 1,
        "mask_type": 1,
        "batch_size": 8,
        "norm_option": 0,
        "reward_type": 6,
        "epoches": 6,
        "temperature": 1,
        "hard_mask":0,
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
    resume = "AUTO+ERRORED"
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
        local_dir=str(ray_dir),
        resume=resume,
        )
 
    # find the best trial
    best_trial = result.get_best_trial("score", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final score: {}".format(best_trial.last_result["score"]))
    with best_trial.checkpoint.as_directory() as checkpoint_dir:
        print("Best trial final ckpt: {}".format(checkpoint_dir))

if __name__=='__main__':
    main()