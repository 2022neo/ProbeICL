from easydict import EasyDict as edict
import json
from pathlib import Path
import torch

def getConfig(cfg_path,cmd_args):
    config_path = cfg_path # './exps/copa/config.json'
    with open(config_path,'r',encoding='utf-8') as f:
        config = edict(json.load(f))
    for k,v in vars(cmd_args).items():
        config[k]=v

    config.cache_dir = str(Path(cmd_args.exps_dir)/config.cache_dir) 


    config.train_files = [str(Path(cmd_args.exps_dir)/p) for p in config.train_files]
    config.valid_files = [str(Path(cmd_args.exps_dir)/p) for p in config.valid_files]
    config.prompt_pool_paths = [str(Path(cmd_args.exps_dir)/p) for p in config.prompt_pool_paths]
    config.output_unscored_files = [str(Path(cmd_args.exps_dir)/p) for p in config.output_unscored_files]
    
    return config

def load_ckpt_cfg(ckptfn):
    ckpt = torch.load(ckptfn,map_location='cpu')
    ckpt_cfg = ckpt['config']
    ckpt_cfg['train_loss']=ckpt['train_loss']
    ckpt_cfg['train_acc']=ckpt['train_acc']
    return ckpt_cfg

def load_retriever_ckpt(retriever,ckptfn):
    ckpt = torch.load(ckptfn,map_location=retriever.device)
    retriever.load_state_dict(
        ckpt['state_dict']
    )

def load_optimizer_ckpt(optimizer,ckptfn):
    ckpt = torch.load(ckptfn,map_location=optimizer.device)
    optimizer.load_state_dict(
        ckpt['optimizer_state']
    )

def format_metric(values):
    return float(f"{values*100:.1f}")
