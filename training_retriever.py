from dpr.utils.tasks import task_map
from easydict import EasyDict as edict
import json
from retriever import init_retriever
import random
from llm import CausalLM
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from dpr.options import setup_logger
import random
from dpr.data.biencoder_data import ProbeIclDataset
import torch
import argparse
from utils.tools import getConfig
from utils.metric import metric_dict
from evaluating_retriever import evaluate

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
logger = logging.getLogger()
setup_logger(logger)


def valid(dataset, llm, retriever, tensorizer, config, task, epoch):
    retriever.eval()
    idx_list = list(range(len(dataset)))
    all_score,all_loss,all_skew = [],[],[]
    pbar = tqdm(total=len(idx_list), desc=f"Valid on Epoch {epoch} ...")
    for i, ind in enumerate(idx_list):
        pbar.update(1)
        data = dataset[ind]
        query_entry,answer_list,label = data.query_ctx,data.answers,data.label
        with torch.no_grad():
            ctx_entries = data.scored_cntxs

            similarity = retriever.calculate_similarity(query_entry,ctx_entries,tensorizer)
            _, selected_ctxs = retriever.get_training_mask(similarity,ctx_entries,epoch,mask_type=1)
            label_loss,pred = llm.inference(query_entry,selected_ctxs,answer_list,label,force_pred=True)
            all_loss+=[label_loss.item()] if isinstance(label_loss,torch.Tensor) else []
            _, ranks = torch.sort(similarity.max(dim=0)[0], descending=True)
            ranks = ranks.tolist()
            n = len(ranks)-1
            skew = [abs(rank-idx)/n for idx,rank in enumerate(ranks)]
            # if llm.option_num>1:
            #     all_score+=[int(pred == label)]
            compute_metric=metric_dict[task.metric]
            score=compute_metric(preds=[pred], labels=[label], return_list=True)[0]
            all_score+=[score]
            all_skew.append(np.mean(skew).item())
    valid_info = {
        'score': float(f"{np.mean(all_score).item()*100:.1f}") if len(all_score)>0 else None,
        'loss': float(f"{np.mean(all_loss).item():.6f}") if len(all_loss)>0 else None,
        'skew': float(f"{np.mean(all_skew).item()*100:.3f}") if len(all_skew)>0 else None,
    }
    pbar.close()
    config['valid_info']=valid_info
    return valid_info


def train(train_dataset, llm, retriever, tensorizer, optimizer, scheduler, scaler, prompt_parser, config, epoch):
    retriever.train()
    config.update_cnt = config.update_cnt if 'update_cnt' in config else 0
    idx_list = list(range(len(train_dataset)))
    random.shuffle(idx_list)
    if "train_ds" in config and config.train_ds>0:
        idx_list=idx_list[:config.train_ds]
    acc = []
    train_loss = 0.0
    total_ctrs_loss,total_label_loss,total_ortho_loss = 0.0,0.0,0.0
    pbar = tqdm(total=len(idx_list), desc=f"Train on Epoch {epoch} ...")

    for i, ind in enumerate(idx_list):
        pbar.update(1)
        data = train_dataset[ind]
        query_entry,answer_list,label = data.query_ctx,data.answers,data.label

        scored_ctxs = data.scored_cntxs
        rand_ctxs = data.random_ctxs if config.rand_neg==1 else []
        ctx_entries = scored_ctxs+rand_ctxs

        # caculate similarity matrix
        similarity = retriever.calculate_similarity(query_entry,ctx_entries,tensorizer)
        ortho_loss = retriever.calculate_ortho_loss(similarity)
        total_ortho_loss += ortho_loss.item()
        loss = config.ortho_loss_penalty*ortho_loss

        # caculate contrastive loss
        if len(data.positive_idx_list)>0:
            if config.multi_ctrs:
                pos_idx_list_per_question = data.positive_idx_list
                ctrs_loss = retriever.get_multi_ctrs_loss(similarity,pos_idx_list_per_question)
            else:
                pos_idx_list_per_question = [0]
                ctrs_loss = retriever.get_ctrs_loss(similarity,pos_idx_list_per_question)
            total_ctrs_loss+=ctrs_loss.item()
            loss+=config.ctrs_loss_penalty*ctrs_loss

        # get k-shot mask
        mask, selected_ctxs = retriever.get_training_mask(similarity,ctx_entries,epoch,config.mask_type)
        label_loss,pred = llm(query_entry,selected_ctxs,answer_list,label,mask)
        total_label_loss+=label_loss.item()
        loss += config.label_loss_penalty*label_loss


        train_loss+=loss.item()
        loss.backward()

        if llm.option_num>1:
            acc+=[int(pred == label)]
        
        # update
        if (i+1)%config.batch_size==0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            config.update_cnt+=1
            info_text = f"CTRS: {total_ctrs_loss:.1f}, LABEL: {total_label_loss:.1f}, ORTHO: {total_ortho_loss:.1f}"
            if llm.option_num>1: 
                info_text+=f", ACC: {np.mean(acc).item()*100:.1f}"
            pbar.set_postfix_str(
                info_text
            )
    if (i+1)%config.batch_size!=0:
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        config.update_cnt+=1
    accurarcy = np.mean(acc).item() if llm.option_num>1 else None
    pbar.close()
    config.ckptname=f"e{epoch:02d}_l{int(train_loss)}.pt"
    return train_loss,accurarcy

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', 
                        type=str, help='we use single task in our experience, so it will be a task name', 
                        required=True)
    parser.add_argument('--exps_dir', 
                        type=str, help='Directory for saving all the intermediate and final outputs.', 
                        required=True)
    parser.add_argument("--lm_name", type=str, default="EleutherAI/gpt-neo-2.7B", help="EleutherAI/gpt-neo-2.7B or /mnt/16t_3/jiyuwen/projects/LLM-Research/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--label_loss_penalty", type=float, default=0.001, help="")
    parser.add_argument("--ctrs_loss_penalty", type=float, default=1.0, help="")
    parser.add_argument("--ortho_loss_penalty", type=float, default=0.0001, help="")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="")
    parser.add_argument("--epoches", type=int, default=6, help="")
    parser.add_argument("--k_shot", type=int, default=3, help="")
    parser.add_argument("--top_k", type=int, default=20, help="")
    parser.add_argument("--train_ds", type=int, default=-1, help="")
    parser.add_argument("--multi_ctrs", type=int, choices=[0, 1], default=0, help="")
    parser.add_argument("--rand_neg", type=int, choices=[0, 1], default=0, help="")
    parser.add_argument("--hard_mask", type=int, choices=[0, 1], default=0, help="")
    parser.add_argument("--mask_type", type=int, choices=[0, 1, 2, 3], default=0, help="")
    parser.add_argument("--norm_mask", type=int, choices=[0, 1], default=0, help="")
    parser.add_argument("--temperature", type=float, default=1, help="")
    parser.add_argument("--dropout", type=float, default=0.1, help="")
    parser.add_argument("--filter_positive", type=int, choices=[0, 1], default=1, help="1 or 0")
    parser.add_argument("--save_model", type=int, choices=[0, 1], default=0, help="1 or 0")

    args = parser.parse_args()
    args.taskpath = str(Path(args.exps_dir)/args.task_name)
    print("### Selected taskpath:", args.taskpath)
    return args


def prepare_trial(config):
    task = task_map.cls_dic[config.task_name]()
    config.option_num=task.class_num
    checkpoint_dir=Path(config.taskpath)/'inference'/'saved_retriever'
    checkpoint_dir.mkdir(exist_ok=True,parents=True)
    config.checkpoint_dir=str(checkpoint_dir)
    train_dataset = ProbeIclDataset(
        data_files=config.train_files,
        top_k=config.top_k,
        cfg=config,
        loss_type="dpr",
        multi_task=False,
        prompt_pool_paths = config.prompt_pool_paths,
        prompt_setup_type = config.prompt_setup_type,
        task_setup_type= config.task_setup_type,
    )
    train_dataset.load_data()

    valid_dataset = ProbeIclDataset(
        data_files=config.valid_files,
        top_k=config.top_k,
        cfg=config,
        loss_type="dpr",
        multi_task=False,
        prompt_pool_paths = config.prompt_pool_paths,
        prompt_setup_type = config.prompt_setup_type,
        task_setup_type= config.task_setup_type,
    )
    valid_dataset.load_data(training=False)

    test_dataset = task.get_dataset(
            split="test",
            cache_dir=config.cache_dir,
        )
    logger.info('test_dataset: %d' % len(test_dataset))

    # MODEL
    config.total_updates = config.epoches*len(train_dataset)//config.batch_size
    config.warmup_steps = config.total_updates//3
    retriever, tensorizer, optimizer, scheduler, scaler, prompt_parser = init_retriever(config)
    llm = CausalLM(config)
    return train_dataset,valid_dataset,test_dataset,llm, retriever, tensorizer, optimizer, scheduler, scaler, prompt_parser, task


 

def main():
    cmd_args = parse_args()
    taskpath = cmd_args.taskpath
    cfgpath = str(Path(taskpath)/"config.json")
    config = getConfig(cfgpath,cmd_args)
    logger.info(config)
    train_dataset,valid_dataset,test_dataset,llm,retriever,tensorizer,optimizer,scheduler,scaler,prompt_parser,task = prepare_trial(config)
    
    # TRAIN
    for epc in range(1,config.epoches+1):
        # train_loss,acc = train(train_dataset, llm, retriever, tensorizer, optimizer, scheduler, scaler, prompt_parser, config, epc)
        valid_info = valid(valid_dataset, llm, retriever, tensorizer, config, task, epc)
        result_info = evaluate(test_dataset,train_dataset.prompt_pool,retriever,tensorizer,prompt_parser,task,config,llm,epc)
        # valid test infomation will be saved to ${config.taskpath}/inference/infer_res.log
        if config.save_model:
            save_content = {
                'epoch':epc,
                'train_loss':train_loss,
                'train_acc':acc,
                'state_dict':retriever.state_dict(),
                'config':config,
            }
            output_file = Path(config.checkpoint_dir)/config.ckptname
            output_file.parent.mkdir(exist_ok=True,parents=True)
            torch.save(save_content,output_file)
    logger.info(f'Train done. Update Count: {config.update_cnt}/{config.total_updates}')

if __name__ == "__main__":
    main()

