from dpr.indexer.faiss_indexers import (
    DenseFlatIndexer,
)
from typing import List, Tuple, Dict, Iterator
from torch import nn
from dpr.utils.tasks import task_map
from dpr.utils.data_utils import Tensorizer
from tqdm import tqdm
import logging
from dpr.options import setup_logger
import pickle
import torch
import numpy as np
import time
from collections import defaultdict
from pathlib import Path
from retriever import init_retriever
from utils.tools import load_ckpt_cfg, load_module_ckpt, calculate_metric
import json
from sklearn.metrics import f1_score, accuracy_score
logger = logging.getLogger()
setup_logger(logger)

import logging
from dpr.options import setup_logger
# make logger
logger = logging.getLogger()
setup_logger(logger)

def test(testset, corpus, qid_to_ctx_rids, prompt_parser, task, llm):
    preds,aug_preds,all_labels = [],[],[]
    losses,aug_losses = [],[]
    shot_num = []
    for jline in tqdm(testset):
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
        all_labels.append(label)

        label_loss,pred = llm.inference(query_entry,ctx_entries,answer_list,label,force_pred=True)
        preds.append(pred)
        losses += [label_loss.item()] if isinstance(label_loss,torch.Tensor) else []

        aug_label_loss,aug_pred = llm.aug_inference(query_entry,ctx_entries,answer_list,label,force_pred=True)
        aug_preds.append(aug_pred)
        aug_losses += [aug_label_loss.item()] if isinstance(aug_label_loss,torch.Tensor) else []
        
    metric_info = calculate_metric(preds,all_labels,task)
    if len(losses)>0:
        metric_info['loss']=float(f"{np.mean(losses):.6f}")

    aug_metric_info=None
    aug_metric_info = calculate_metric(aug_preds,all_labels,task)
    if len(aug_losses)>0:
        aug_metric_info['loss']=float(f"{np.mean(aug_losses):.6f}")

    avg_shot_num = np.mean(shot_num).item()
    return metric_info, aug_metric_info, avg_shot_num


def gen_ctx_vectors(corpus, retriever, tensorizer, prompt_parser, task):
    results = []
    size = len(corpus)
    batch = 32
    for start_ind in tqdm(range(0,size,batch)):
        end_ind = min(start_ind+batch,size)
        ctx_entries = [
            {
                "demonstration": prompt_parser.process(corpus[a],setup_type='qa',task=task)[0], 
                "title":corpus[a]["title"] if "title" in corpus[a] else None,
                "idx":corpus[a]["idx"],
                "id":a,
                "valid":a==corpus[a]["id"]
            }
            for a in range(start_ind,end_ind)
        ] 
        assert all([a["valid"] for a in ctx_entries])
        ctx_tensors = retriever.encode_ctxs(ctx_entries,tensorizer)
        ctx_tensors = ctx_tensors.detach().cpu()
        results.extend(
            [
                (ctx_entries[i]["id"], ctx_tensors[i].view(-1).numpy())
                for i in range(ctx_tensors.size(0))
            ]
        )
    assert end_ind==size
    return results

def generate_question_vectors(jline, retriever, tensorizer, prompt_parser, task):
    query_prompt,answer_list,label = prompt_parser.process(jline,setup_type='q',task=task)
    query_entry = {
        "demonstration":query_prompt,
        "title":jline["title"] if "title" in jline else None,
    }
    probes_ts = retriever.query_to_probes(query_entry,tensorizer)
    probes_ts = probes_ts.detach().cpu().numpy()
    return probes_ts

def iterate_encoded_files(doc_vectors) -> Iterator[Tuple]:
    for doc in doc_vectors:
        doc = list(doc)
        yield doc

class LocalFaissSearcher(object):
    def __init__(self,buffer_size,vector_dim):
        self.buffer_size=buffer_size
        self.vector_dim=vector_dim
        self.index = DenseFlatIndexer(buffer_size)
        self.index.init_index(vector_dim)

    def index_encoded_data(
        self,
        doc_vectors,
    ):  
        buffer_size = self.buffer_size
        buffer = []
        for i, item in enumerate(
            iterate_encoded_files(doc_vectors)
        ):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        # logger.info("index search time: %f sec.", time.time() - time0)
        return results

def retrieve_ctxs(test_dataset,corpus,retriever,tensorizer,prompt_parser,task,config):
    doc_vectors = gen_ctx_vectors(corpus, retriever, tensorizer, prompt_parser, task)
    buffer_size = 50000
    vector_dim = retriever.hidden_dim
    faiss_searcher = LocalFaissSearcher(buffer_size=buffer_size,vector_dim=vector_dim)
    faiss_searcher.index_encoded_data(doc_vectors)

    qid_to_ctx_rids = defaultdict(list)
    for jline in tqdm(test_dataset):
        qid = jline["idx"]
        assert qid not in qid_to_ctx_rids
        q_vector = generate_question_vectors(jline, retriever, tensorizer, prompt_parser, task) #[k-shot,dim]
        top_ids_and_scores = faiss_searcher.get_top_docs(q_vector,config.k_shot)
        for num_shot,(rid_lst,scores_lst) in enumerate(top_ids_and_scores):
            for id,score in zip(rid_lst,scores_lst):
                score = score.item()
                if id not in qid_to_ctx_rids[qid]:
                    qid_to_ctx_rids[qid].append(id)
                    break
    return qid_to_ctx_rids

def evaluate(test_dataset,corpus,retriever,tensorizer,prompt_parser,task,config,llm,epoch):
    retriever.eval()
    with torch.no_grad():
        qid_to_ctx_rids = retrieve_ctxs(test_dataset,corpus,retriever,tensorizer,prompt_parser,task,config)
        test_info, aug_test_info, avg_shot_num = test(test_dataset,corpus,qid_to_ctx_rids,prompt_parser,task,llm)
    result_info = {
        'test_info':test_info,
        'aug_test_info':aug_test_info,
        'valid_info':config.valid_info,
        'avg_shot':avg_shot_num,
        'epoches':config.epoches,
        'lr':config.learning_rate,
        'k_shot':config.k_shot,
        'label_penalty':config.label_loss_penalty,
        'ortho_penalty':config.ortho_loss_penalty,
        'ctrs_penalty':config.ctrs_loss_penalty,
        'tau':config.temperature,
        'hard_mask':config.hard_mask,
        'filter_positive':config.filter_positive,
        'rand_neg':config.rand_neg,
        'mask_type':config.mask_type,
        'multi_ctrs':config.multi_ctrs,
        'lm_name':config.lm_name,
        'top_k':config.top_k,
        'dropout':config.dropout,
        'ckpt':config.ckptname,
    }
    outfile = Path(config.taskpath)/'inference'/'infer_res.log'
    outfile.parent.mkdir(exist_ok=True,parents=True)
    logger.info(f"test_info: {test_info}")
    with outfile.open('a') as f:
        f.write(json.dumps(result_info))
        f.write('\n')
    return result_info

import argparse
from llm import CausalLM
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

    corpus = task.get_dataset(
            split="train",
            ds_size=ckpt_cfg.ds_size,
            cache_dir=ckpt_cfg.cache_dir,
        )
    logger.info('corpus: %d' % len(corpus))

    test_dataset = task.get_dataset(
            split="test",
            cache_dir=ckpt_cfg.cache_dir,
        )
    logger.info('test_dataset: %d' % len(test_dataset))
    llm = CausalLM(ckpt_cfg)
    retriever, tensorizer, prompt_parser = init_retriever(ckpt_cfg,inference_only=True)
    load_module_ckpt(retriever,ckptfn,"state_dict",retriever.device)
    res = evaluate(test_dataset,corpus,retriever,tensorizer,prompt_parser,task,ckpt_cfg,llm,ckptname)
    print(res)

if __name__ == "__main__":
    main()