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
    args = parser.parse_args()
    args.config_file = str(Path(args.exps_dir)/args.task_name/'config.json')
    args.taskpath = str(Path(args.exps_dir)/args.task_name)
    return args


def trial(param, cmd_args):
    cfgpath = str(Path(cmd_args.taskpath)/"config.json")
    config = getConfig(cfgpath,cmd_args)
    for k,v in param.items():
        config[k]=v

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
    else:
        start_epoch = 1


    for epc in range(start_epoch,config.epoches+1):
        train_loss,acc = train(train_dataset, llm, retriever, tensorizer, optimizer, scheduler, scaler, prompt_parser, config, epc)
        valid_info = valid(valid_dataset, llm, retriever, tensorizer, config, task, epc)

        with tempfile.TemporaryDirectory(dir=config.checkpoint_dir) as tmp_ckpt_dir:
            savepath = Path(tmp_ckpt_dir)/config.ckptname
            # savepath.parent.mkdir(exist_ok=True,parents=True)
            config.ckptname=str(Path(savepath).relative_to(config.checkpoint_dir))
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
            checkpoint = Checkpoint.from_directory(tmp_ckpt_dir)
            result_info = evaluate(test_dataset,train_dataset.prompt_pool,retriever,tensorizer,prompt_parser,task,config,llm,epc)
            score=result_info["test_info"]["score"]
            ray.train.report(
                {"score":score,"metric":result_info["test_info"]},
                checkpoint=checkpoint,
            )



def main(max_num_epochs=10):
    cmd_args = parse_args()

    # setup space of hyperparameters
    space = {
        "lr": tune.qloguniform(1e-6, 1e-3, 1e-6),
        "ctrs_loss_penalty": tune.quniform(1e-2, 1, 1e-2),
        "label_loss_penalty": tune.quniform(1e-4, 10, 1e-4),
        "ortho_loss_penalty": tune.quniform(1e-2, 100, 1e-2),
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
        metric_columns=["score", "loss"])
    # start to train
    ray_dir = Path(cmd_args.taskpath)/'inference'/'_ray'
    ray_dir.mkdir(exist_ok=True,parents=True)
    result = tune.run(
        partial(trial, cmd_args=cmd_args),
        # allocate resource for training
        resources_per_trial={"cpu": cmd_args.cpus_per_trial, "gpu": cmd_args.gpus_per_trial},
        config=space,
        num_samples=cmd_args.num_samples,
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