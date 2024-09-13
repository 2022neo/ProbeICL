import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple
from easydict import EasyDict as edict
import hydra
import jsonlines
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor as T
from dpr.data.tables import Table
from dpr.utils.data_utils import read_data_from_json_files, Tensorizer
from dpr.utils.tasks import (
    task_map,
    get_prompt_files,
)

logger = logging.getLogger(__name__)
BiEncoderPassage = collections.namedtuple(
    "BiEncoderPassage", ["text", "title", "meta_data"]
)


class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


class RepTokenSelector(object):
    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        raise NotImplementedError


class RepStaticPosTokenSelector(RepTokenSelector):
    def __init__(self, static_position: int = 0):
        self.static_position = static_position

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        return self.static_position


class RepSpecificTokenSelector(RepTokenSelector):
    def __init__(self, token: str = "[CLS]"):
        self.token = token
        self.token_id = None

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        if not self.token_id:
            self.token_id = tenzorizer.get_token_id(self.token)
        token_indexes = (input_ids == self.token_id).nonzero()
        # check if all samples in input_ids has index presence and out a default value otherwise
        bsz = input_ids.size(0)
        if bsz == token_indexes.size(0):
            return token_indexes

        token_indexes_result = []
        found_idx_cnt = 0
        for i in range(bsz):
            if (
                found_idx_cnt < token_indexes.size(0)
                and token_indexes[found_idx_cnt][0] == i
            ):
                # this samples has the special token
                token_indexes_result.append(token_indexes[found_idx_cnt])
                found_idx_cnt += 1
            else:
                logger.warning("missing special token %s", input_ids[i])

                token_indexes_result.append(
                    torch.tensor([i, 0]).to(input_ids.device)
                )  # setting 0-th token, i.e. CLS for BERT as the special one
        token_indexes_result = torch.stack(token_indexes_result, dim=0)
        return token_indexes_result


DEFAULT_SELECTOR = RepStaticPosTokenSelector()


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        selector: DictConfig = None,
        special_token: str = None,
        shuffle_positives: bool = False,
        query_special_suffix: str = None,
        encoder_type: str = None,
    ):
        if selector:
            self.selector = hydra.utils.instantiate(selector)
        else:
            self.selector = DEFAULT_SELECTOR
        self.special_token = special_token
        self.encoder_type = encoder_type
        self.shuffle_positives = shuffle_positives
        self.query_special_suffix = query_special_suffix

    def load_data(self):
        raise NotImplementedError

    def __getitem__(self, index) -> BiEncoderSample:
        raise NotImplementedError

    def _process_query(self, query: str):
        # as of now, always normalize query
        query = normalize_question(query)
        if self.query_special_suffix and not query.endswith(self.query_special_suffix):
            query += self.query_special_suffix

        return query
    
def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    return ctx_text


def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question

def get_raw_data(file_path):
    data_files = get_dpr_files(file_path)
    logger.info("dpr files: %s", data_files)
    raw_data = read_data_from_json_files(data_files)
    return raw_data

def format_example(entry, setup_type):
    task = task_map.cls_dic[entry["task_name"]]()
    if setup_type == "qa":
        sent = (
            task.get_question(entry)
            + task.get_answer(entry)
        )
    elif setup_type == "q":
        sent = task.get_question(entry)
    elif setup_type == "a":
        sent = task.get_answer(entry).strip()
    return remove_double_space(sent)

def get_example_answers(entry):
    task = task_map.cls_dic[entry["task_name"]]()
    answers = [remove_double_space(a) for a in task.get_answers(entry)]
    label = task.get_label(entry)
    return answers,label

def get_dpr_files(sources) -> List[str]:
    if isinstance(sources, str):
        sources = [sources] 
    res = []
    for source_name in sources:
        if os.path.exists(source_name) or glob.glob(source_name):
            res.extend(glob.glob(source_name))
        else:
            # try to use data downloader
            from dpr.data.download_data import download

            res.extend(download(source_name))
    logger.info("Toal files num %d" % len(res))
    return res


def reformat(text):
    return " ".join([f"{i+1}#) {x.strip()}" for i, x in enumerate(text.split(";"))])


import re
def remove_double_space(string):
    return re.sub("[ ]{2,}", " ", string)

def create_dict(raw_data: List[dict]) -> dict:
    TaskandId_to_data = {}
    for data in raw_data:
        task = data["task_name"]
        id = str(data["id"])
        TaskandId_to_data[task+id] = data
    return TaskandId_to_data

class ProbeIclDataset(Dataset):
    def __init__(
        self,
        data_files,
        top_k,
        cfg = None,
        loss_type: str = "dpr",
        multi_task: bool = False,
        selector: DictConfig = None,
        special_token: str = None,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        query_special_suffix: str = None,
        prompt_pool_paths: list = None,
        prompt_setup_type: str = "q",
        task_setup_type: str = "q",
    ):
        super().__init__(
            selector,
            special_token=special_token,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
            query_special_suffix=query_special_suffix,
        )
        assert loss_type in ['dpr']
        logger.info("loss_type: %s", loss_type)
        self.top_k = top_k
        self.data_files = data_files
        self.data = []
        self.loss_type = loss_type
        self.cfg=cfg

        logger.info("prompt files: %s", prompt_pool_paths)
        self.prompt_pool = read_data_from_json_files(prompt_pool_paths)
        logger.info("prompt passages num : %d", len(self.prompt_pool))
        self.TaskandId_to_data = create_dict(self.prompt_pool)
        self.multi_task = multi_task
        self.prompt_setup_type = prompt_setup_type
        self.task_setup_type = task_setup_type

    def get_entry(self, entry):
        if self.loss_type == 'dpr':
            task_name = entry["task_name"]
            task = task_map.cls_dic[task_name]()

            scored_cntxs = [
                {
                    "demonstration": format_example(p_example, self.prompt_setup_type),
                    "title":p_example["title"] if "title" in p_example else None,
                }
                for p_example in entry["ctxs"][:self.top_k]
            ] 
            scored_ids = [ctx_entry['id'] for ctx_entry in entry["ctxs"][:self.top_k]] 
            positive_idx_list = [
                i for i,ctx_entry in enumerate(entry["ctxs"][:self.top_k]) 
                if (entry["ctxs"][0]["one_shot_acc"]==True or entry["ctxs"][0]["one_shot_acc"] > 0)
            ] if self.cfg.option_num>1 else [0]

            # random example
            filtered_prompt_pool = [prompt for prompt in self.prompt_pool if prompt['id'] not in scored_ids]
            rand_cntx = [
                {
                    "demonstration": format_example(n_example, self.prompt_setup_type),
                    "title":n_example["title"] if "title" in n_example else None,
                }
                for n_example in random.choices(filtered_prompt_pool, k=self.top_k)
            ]

            # use task_name + id to finde example，for multi task data
            question = ""
            question += format_example(entry, self.task_setup_type)
            query_ctx = {
                "demonstration":question,
                "title":entry["title"] if "title" in entry else None,
            }
            answers,label = get_example_answers(entry)
            entry = edict({
                "query_ctx": query_ctx,
                "answers": answers,
                "label":label,
                "scored_cntxs": scored_cntxs,
                "positive_idx_list": positive_idx_list,
                "random_ctxs": rand_cntx,
            })
            return entry
        else:
            raise NotImplementedError

    def load_data(self,training=True):
        logger.info("dpr files: %s", self.data_files)
        raw_data = read_data_from_json_files(self.data_files)
        logger.info("********len(raw_data): %d", len(raw_data))
        self.data = []
        for entry in raw_data:
            self.data.append(self.get_entry(entry))
        # filter out those without positive ctx
        if self.loss_type == 'dpr':
            self.data = [r for r in self.data if len(r["positive_idx_list"])>0] if (training and self.cfg.filter_positive) else self.data
            logger.info("filter out data for : {}".format(len(raw_data) - len(self.data)))
            logger.info("Total filtered data size: {}".format(len(self.data)))
        else:
            print("loss type error")
        
    def __getitem__(self, index) -> BiEncoderSample:
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_qas(self) -> Tuple[List[str], List[str]]:
        return [s["question"] for s in self.data], [s["answers"] for s in self.data]

    def get_qas_range(
        self, start_idx: int, end_idx: int
    ) -> Tuple[List[str], List[str]]:
        return (
            [s["question"] for s in self.data[start_idx:end_idx]],
            [s["answers"] for s in self.data[start_idx:end_idx]],
        )


