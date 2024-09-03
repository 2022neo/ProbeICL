from dpr.utils.tasks import task_map
import json
from retriever import PrompParser
from llm import CausalLM
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from dpr.options import setup_logger
import torch
from utils.metric import metric_dict
from sklearn.metrics import f1_score, accuracy_score
from utils.tools import load_ckpt_cfg

# make logger
logger = logging.getLogger()
setup_logger(logger)



def test(testset, corpus, qid_to_ctx_rids, llm, prompt_parser, task):
    preds,aug_preds,all_labels = [],[],[]
    shot_num = []
    for i, jline in tqdm(enumerate(testset)):
        qid = jline["idx"]
        query_prompt,answer_list,label = prompt_parser.process(jline,setup_type='q',task=task)
        query_entry = {
            "demonstration":query_prompt,
            "title":jline["title"] if "title" in jline else None,
        }
        ctx_entries = [
            {
                "demonstration": prompt_parser.process(corpus[a],setup_type='qa',task=task)[0], 
                "title":corpus[a]["title"] if "title" in corpus[a] else None
            }
            for a in qid_to_ctx_rids[qid]
        ] 
        shot_num+=[len(ctx_entries)]
        with torch.no_grad():
            label_loss,pred = llm.inference(query_entry,ctx_entries,answer_list,label,force_pred=True)
            preds.append(pred)
            label_loss,aug_pred = llm.aug_inference(query_entry,ctx_entries,answer_list,label,force_pred=True)
            aug_preds.append(aug_pred)
            all_labels.append(label)

    metric_info = calculate_metric(preds,all_labels,task)
    aug_metric_info = calculate_metric(aug_preds,all_labels,task)
    avg_shot_num = np.mean(shot_num).item()
    return metric_info, aug_metric_info, avg_shot_num


def calculate_metric(preds,labels,task):
    if task.class_num>1:
        metric_info = {
            'F1_Macro':float(f"{f1_score(labels, preds, average='macro')*100:.1f}"),
            'F1_Micro':float(f"{f1_score(labels, preds, average='micro')*100:.1f}"),
            'F1_Weighted':float(f"{f1_score(labels, preds, average='weighted')*100:.1f}"),
            'ACC':float(f"{accuracy_score(labels, preds)*100:.1f}"),
        }
    else:
        compute_metric=metric_dict[task.metric]
        scores=compute_metric(preds=preds, labels=labels, return_list=False)
        metric_info = {
            f"{task.metric}":scores,
        }
    return metric_info

def get_qid_to_rids(dataset,qid_to_cids_path):
    with qid_to_cids_path.open('r') as f:
        qid_to_cids = json.load(f)
    cid_to_rid,qid_to_rids = {},{}
    for i,jline in enumerate(dataset):
        cid_to_rid[int(jline['idx'])] = i
    for qid,cids in qid_to_cids.items():
        qid_to_rids[int(qid)]=[cid_to_rid[int(cid)] for cid,score in cids]
    return qid_to_rids

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
    ckpt_cfg = load_ckpt_cfg(ckptfn)
    logger.info(ckpt_cfg)
    task = task_map.cls_dic[ckpt_cfg.task_name]()
    ckpt_cfg.option_num=task.class_num


    # DATA
    train_dataset = task.get_dataset(
            split="train",
            ds_size=ckpt_cfg.ds_size,
            cache_dir=ckpt_cfg.cache_dir,
        )
    logger.info('train_dataset: %d' % len(train_dataset))

    test_dataset = task.get_dataset(
            split="test",
            cache_dir=ckpt_cfg.cache_dir,
        )
    logger.info('test_dataset: %d' % len(test_dataset))


    # CTXS MAPPING
    print('START CTXS MAPPING !')
    qid_to_cids_path = Path(taskpath)/'inference'/'qid_to_cids.json'
    qid_to_ctx_rids = get_qid_to_rids(train_dataset,qid_to_cids_path)
    print('FINISH CTXS MAPPING !')
    
    # MODEL
    prompt_parser = PrompParser(ckpt_cfg)
    llm = CausalLM(ckpt_cfg)

    # VAL & TEST
    print('START TESTING !')
    test_info, aug_test_info, avg_shot_num = test(test_dataset, train_dataset, qid_to_ctx_rids, llm, prompt_parser, task)

    result_info = {
        'ckpt':ckptname,
        'test_info':test_info,
        'aug_test_info':aug_test_info,
        'valid_info':ckpt_cfg['valid_info'],
        'avg_shot':avg_shot_num,
        'epoches':ckpt_cfg['epoches'],
        'lr':ckpt_cfg['learning_rate'],
        'k_shot':ckpt_cfg['k_shot'],
        'label_penalty':ckpt_cfg['label_loss_penalty'],
        'ortho_penalty':ckpt_cfg['ortho_loss_penalty'],
        'ctrs_penalty':ckpt_cfg['ctrs_loss_penalty'],
        'tau':ckpt_cfg['temperature'],
        'hard_mask':ckpt_cfg['hard_mask'],
        'rand_neg':ckpt_cfg['rand_neg'],
        'mask_type':ckpt_cfg['mask_type'],
        'multi_ctrs':ckpt_cfg['multi_ctrs'],
        'lm_name':ckpt_cfg['lm_name'],
        'top_k':ckpt_cfg['top_k'],
        'dropout':ckpt_cfg['dropout'],
    }
    outfile = Path(taskpath)/'inference'/'infer_res.log'
    logger.info(f"test_info: {test_info}")
    with outfile.open('a') as f:
        f.write(json.dumps(result_info))
        f.write('\n')


if __name__ == "__main__":
    main()

