from dpr.utils.tasks import task_map
import json
import random
from llm import CausalLM
from tqdm import tqdm
from pathlib import Path
import logging
from dpr.options import setup_logger
import random
from dpr.data.biencoder_data import get_raw_data, format_example, get_example_answers
import torch
import argparse
from utils.metric import metric_dict
from utils.tools import getConfig
from more_itertools import divide

random.seed(123)
logger = logging.getLogger()
setup_logger(logger)


def get_chunk_raw_data(inputfile, chunk_id, num_chunks):
    assert chunk_id in list(range(num_chunks))
    raw_data = get_raw_data(str(inputfile))
    children = divide(num_chunks,raw_data)
    return list(children[chunk_id])


def scoring(raw_data, llm, outputfile, task, config):
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
            question += format_example(entry, config.task_setup_type)
            query_entry = {
                "demonstration":question,
                "title":entry["title"] if "title" in entry else None,
            }
            answer_list,label = get_example_answers(entry)
            for i,ctx_entry in enumerate(entry['ctxs']):
                ctx_entries = [{
                    "demonstration":format_example(ctx_entry, config.prompt_setup_type),
                    "title":ctx_entry["title"] if "title" in ctx_entry else None,
                }]
                with torch.no_grad():
                    label_loss,pred = llm.inference(query_entry,ctx_entries,answer_list,label,force_pred=True)
                    ctx_entry['loss'] = label_loss.item()
                    ctx_entry['pred'] = pred
                    if llm.option_num==1:
                        if pred is not None:
                            compute_metric=metric_dict[task.metric]
                            score=compute_metric(preds=[pred], labels=[label], return_list=True)[0]
                            ctx_entry['one_shot_acc']=score
                        else:
                            ctx_entry['one_shot_acc']=1-label_loss.item()
                    elif llm.option_num>1:
                        ctx_entry['one_shot_acc']=int(pred == label)
                    else:
                        raise NotImplementedError
            if llm.option_num>1:
                entry['ctxs'] = sorted(entry['ctxs'],key = lambda x: x['loss']) 
            elif llm.option_num==1:
                entry['ctxs'] = sorted(entry['ctxs'],key = lambda x: -x['one_shot_acc']) 
            else:
                raise NotImplementedError
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
    parser.add_argument('--num_chunks', 
                        type=int, help='Number of chunks for Parallelism Scoring', 
                        default=1)
    parser.add_argument('--chunk_id', 
                        type=int, help='ID of chunk for Parallelism Scoring', 
                        default=0)
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


    # MODEL
    llm = CausalLM(config)

    # SCORING
    for inputfile in config.output_unscored_files:
        name = Path(inputfile).stem+f'_{config.num_chunks}_{config.chunk_id}.jsonl'
        outputfile = Path(args.taskpath)/"scored"/name
        outputfile.parent.mkdir(exist_ok=True,parents=True)
        if config.num_chunks>1:
            raw_data = get_chunk_raw_data(inputfile, config.chunk_id, config.num_chunks)
        else:
            raw_data = get_raw_data(inputfile)
        scoring(raw_data, llm, outputfile, task, config)
        
    
if __name__ == "__main__":
    main()

