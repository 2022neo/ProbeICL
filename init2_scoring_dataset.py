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
import logging
from dpr.options import setup_logger
import random
from collections import defaultdict
import time
from dpr.data.biencoder_data import ProbeIclDataset
import torch
import argparse
from utils.metric import metric_dict
from utils.tools import getConfig

random.seed(123)
logger = logging.getLogger()
setup_logger(logger)


def get_chunk_raw_data(inputfile, dataset, chunk_id, chunks):
    raw_data = dataset.get_raw_data(str(inputfile))
    assert chunk_id>0
    n_split = (chunk_id*len(raw_data))//chunks
    return raw_data[:n_split]


def scoring(raw_data, dataset, llm, outputfile, task):
    if Path(outputfile).exists():
        processed_data = [json.loads(line) for line in open(outputfile)]
        processed_idxs = set([d['id'] for d in processed_data])
        raw_data = [d for d in raw_data if d['id'] not in processed_idxs]
        print(f"### Resume from {len(processed_idxs)} samples. {len(raw_data)} samples to run.")
    ans_file = open(outputfile, "a")
    with open(outputfile, "a") as ans_file:
        pbar = tqdm(total=len(raw_data), desc=f"START SCORING...")
        for i, entry in enumerate(raw_data):
            pbar.update(1)
            question = ""
            have_choosen = entry["choosen"]
            for id in have_choosen:
                example = dataset.prompt_pool[id]
                question += dataset.format_example(example, dataset.prompt_setup_type) + " \n "
            question += dataset.format_example(entry, dataset.task_setup_type)
            query_entry = {
                "demonstration":question,
                "title":entry["title"] if "title" in entry else None,
            }
            answer_list,label = dataset.get_example_answers(entry)
            for i,ctx_entry in enumerate(entry['ctxs']):
                ctx_entries = [{
                    "demonstration":dataset.format_example(ctx_entry, dataset.prompt_setup_type),
                    "title":ctx_entry["title"] if "title" in ctx_entry else None,
                }]
                with torch.no_grad():
                    label_loss,pred = llm.inference(query_entry,ctx_entries,answer_list,label,force_pred=True)
                    if llm.option_num==1:
                        if pred is not None:
                            compute_metric=metric_dict[task.metric]
                            score=compute_metric(preds=[pred], labels=[label], return_list=True)[0]
                            ctx_entry['loss'] = 1-score
                            ctx_entry['one_shot_acc']=score
                        else:
                            ctx_entry['loss'] = label_loss.item()
                            ctx_entry['one_shot_acc']=1-label_loss.item()
                        ctx_entry['pred'] = pred
                    elif llm.option_num>1:
                        ctx_entry['pred']=pred
                        ctx_entry['loss']=label_loss.item()
                        ctx_entry['one_shot_acc']=int(pred == label)
                    else:
                        raise NotImplementedError
            entry['ctxs'] = sorted(entry['ctxs'],key = lambda x: x['loss']) 
            ans_file.write(json.dumps(entry) + "\n")
    pbar.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', 
                        type=str, help='we use single task in our experience, so it will be a task name', 
                        required=True)
    parser.add_argument('--exps_dir', 
                        type=str, help='Directory for saving all the intermediate and final outputs.', 
                        required=True)
    args = parser.parse_args()
    args.config_file = str(Path(args.exps_dir)/args.task_name/'config.json')
    args.taskpath = str(Path(args.exps_dir)/args.task_name)
    return args

def main():
    args = parse_args()
    cfgpath = args.config_file
    config = getConfig(cfgpath,args)
    logger.info(config)
    task = task_map.cls_dic[config.task_name]()
    config.option_num=task.class_num

    dataset = ProbeIclDataset(
        data_files=[],
        top_k=1,
        loss_type="dpr",
        multi_task=False,
        hard_neg=True,
        prompt_pool_paths = config.prompt_pool_paths,
        prompt_setup_type = config.prompt_setup_type,
        task_setup_type= config.task_setup_type,
    )
    # MODEL
    llm = CausalLM(config)

    # METRIC

    # SCORING
    for inputfile in config.output_unscored_files:
        name = Path(inputfile).stem+'.jsonl'
        outputfile = Path(args.taskpath)/"scored"/name
        outputfile.parent.mkdir(exist_ok=True,parents=True)
        raw_data = dataset.get_raw_data(str(inputfile))
        scoring(raw_data, dataset, llm, outputfile, task)
        
        for line in open(outputfile):
            json.loads(line) 
        ans_name = Path(inputfile).stem+'.json'
        ans_fn = Path(args.taskpath)/"scored"/ans_name
        with ans_fn.open('w') as f:
            json.dump([json.loads(line) for line in open(outputfile)],f,indent=2)

    
if __name__ == "__main__":
    main()

