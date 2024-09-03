from dpr.utils.tasks import task_map
from easydict import EasyDict as edict
import json
from retriever import init_retriever
import random
from llm import CausalLM
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
import random
import pickle
import torch
from utils.tools import load_ckpt_cfg, load_retriever_ckpt

import logging
from dpr.options import setup_logger
# make logger
logger = logging.getLogger()
setup_logger(logger)



def gen_ctx_vectors(dataset, retriever, tensorizer, prompt_parser, task):
    retriever.eval()
    results = []
    size = len(dataset)
    batch = 32
    for start_ind in tqdm(range(0,size,batch)):
        end_ind = min(start_ind+batch,size)
        ctx_entries = [
            {
                "demonstration": prompt_parser.process(dataset[a],setup_type='qa',task=task)[0], 
                "title":dataset[a]["title"] if "title" in dataset[a] else None,
                "idx":dataset[a]["idx"],
            }
            for a in range(start_ind,end_ind)
        ] 
        ctx_tensors = retriever.encode_ctxs(ctx_entries,tensorizer)
        ctx_tensors = ctx_tensors.cpu()
        results.extend(
            [
                (ctx_entries[i]["idx"], ctx_tensors[i].view(-1).numpy())
                for i in range(ctx_tensors.size(0))
            ]
        )
    return results


import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', 
                        type=str, help='we use single task in our experience, so it will be a task name', 
                        required=True)
    parser.add_argument('--exps_dir', 
                        type=str, help='Directory for saving all the intermediate and final outputs.', 
                        required=True)
    parser.add_argument('--ckptname', 
                        type=str, help='ckptname for saved retriever.', 
                        required=True)
    args = parser.parse_args()
    args.config_file = str(Path(args.exps_dir)/args.task_name/'config.json')
    args.taskpath = str(Path(args.exps_dir)/args.task_name)
    return args

def main():
    args = parse_args()

    taskpath = args.taskpath
    ckptname = args.ckptname
    ckptfn = Path(taskpath)/'inference'/'saved_retriever'/ckptname
    config = load_ckpt_cfg(ckptfn)
    logger.info(config)
    task = task_map.cls_dic[config.task_name]()


    # DATA
    corpus = task.get_dataset(
            split="train",
            ds_size=config.ds_size,
            cache_dir=config.cache_dir,
        )
    logger.info('corpus: %d' % len(corpus))


    # MODEL
    retriever, tensorizer, prompt_parser = init_retriever(config,inference_only=True)
    load_retriever_ckpt(retriever,ckptfn)
    retriever.eval()
    with torch.no_grad():
        data = gen_ctx_vectors(corpus, retriever, tensorizer, prompt_parser, task)
    outfile = Path(taskpath)/'inference'/'corpus_emb.pkl'
    outfile.parent.mkdir(exist_ok=True,parents=True)
    with open(outfile, mode="wb") as f:
        pickle.dump(data, f)
    logger.info('generate corpus embeddings done.')

if __name__ == "__main__":
    main()

