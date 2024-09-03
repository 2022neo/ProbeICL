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
from utils.tools import load_ckpt_cfg, load_retriever_ckpt
import json
logger = logging.getLogger()
setup_logger(logger)


def generate_question_vectors(jline, retriever, tensorizer, prompt_parser, task):
    retriever.eval()
    query_prompt,answer_list,label = prompt_parser.process(jline,setup_type='q',task=task)
    query_entry = {
        "demonstration":query_prompt,
        "title":jline["title"] if "title" in jline else None,
    }
    probes_ts = retriever.query_to_probes(query_entry,tensorizer)
    probes_ts = probes_ts.cpu().numpy()
    return probes_ts

def iterate_encoded_files(vector_files: list) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
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
        vector_files: List[str],
    ):  
        buffer_size = self.buffer_size
        buffer = []
        for i, item in enumerate(
            iterate_encoded_files(vector_files)
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
    query_dataset = task.get_dataset(
            split="test",
            ds_size=config.ds_size,
            cache_dir=config.cache_dir,
        )
    logger.info('query_dataset: %d' % len(query_dataset))


    # MODEL
    retriever, tensorizer, prompt_parser = init_retriever(config,inference_only=True)

    # SEARCHER
    buffer_size = 50000
    vector_dim = retriever.hidden_dim
    corpus_files = [str(a) for a in (Path(taskpath)/'inference').rglob('*corpus_emb*.pkl')]
    faiss_searcher = LocalFaissSearcher(buffer_size=buffer_size,vector_dim=vector_dim)
    faiss_searcher.index_encoded_data(corpus_files)
    load_retriever_ckpt(retriever,ckptfn)
    retriever.eval()
    qid_to_cids = defaultdict(list)
    for jline in tqdm(query_dataset):
        qid = jline["idx"]
        assert qid not in qid_to_cids
        with torch.no_grad():
            q_vector = generate_question_vectors(jline, retriever, tensorizer, prompt_parser, task) #[k-shot,dim]
            top_ids_and_scores = faiss_searcher.get_top_docs(q_vector,config.k_shot+3)
            for num_shot,(idx_lst,scores_lst) in enumerate(top_ids_and_scores):
                for idx,score in zip(idx_lst,scores_lst):
                    score = score.item()
                    if idx not in qid_to_cids[qid]:
                        qid_to_cids[qid].append((idx,score))
                        break
                    # elif score<0:
                    #     break
    logger.info('retrieve ctx for testset done.')
    # SAVE
    outfile = Path(taskpath)/'inference'/'qid_to_cids.json'
    with outfile.open('w') as f:
        json.dump(qid_to_cids,f,indent=4)

if __name__ == "__main__":
    main()
