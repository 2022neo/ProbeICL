import tqdm
import json
import random
from dpr.utils.tasks import task_map
import argparse
from pathlib import Path
from copy import deepcopy
class RandomFinder:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        Path(cfg.prompt_pool_path).parent.mkdir(exist_ok=True,parents=True)
        Path(cfg.output_train_file).parent.mkdir(exist_ok=True,parents=True)
        Path(cfg.output_valid_file).parent.mkdir(exist_ok=True,parents=True)
        task = task_map.cls_dic[self.cfg.task_name]()
        self.train_dataset = task.get_dataset(
            split="train",
            ds_size=None if "ds_size" not in cfg else cfg.ds_size, #cutoff to ds_size if len(dataset)>ds_size
            cache_dir=cfg.cache_dir,
        )

        print("started creating the prompt pool!")
        self.get_prompt_pool()
        print("finished creating the prompt pool!")

    # sample and save prompt pool
    def get_prompt_pool(self):
        self.prompt_pool = self.train_dataset
        for i, entry in enumerate(self.prompt_pool):
            entry["id"] = i
            entry["task_name"] = self.cfg.task_name
        print("prompt_pool size", len(self.prompt_pool))
        with open(self.cfg.prompt_pool_path, "w") as f:
            json.dump(self.prompt_pool, f)

# for each task input, sample L prompts for scoring from the prompt pool (i.e., task training data)
def find(cfg):
    random_finder = RandomFinder(cfg)
    data_list = random_finder.train_dataset
    idx_list = list(range(len(random_finder.prompt_pool)))
    prompt_pool = deepcopy(random_finder.prompt_pool)

    for i, element in tqdm.tqdm(enumerate(data_list)):
        assert element["id"] == i == prompt_pool[i]["id"]
        assert "choosen" not in element and "ctxs" not in element
        random.seed(i)
        # `ctxs` stores the sampled prompt ids 
        element["ctxs"] = [
            prompt_pool[a]
            for a in random.sample([idx for idx in idx_list if idx != i], k=min(cfg.finder_L, len(data_list)-1)) # avoid selecting the task input itself
        ] # randomly sample finder_L examples for each QA

    random.Random(42).shuffle(data_list)
    split_ratio=0.9
    n_train=int(len(data_list)*split_ratio)
    trainset,validset = data_list[:n_train],data_list[n_train:]
    return trainset,validset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', 
                        type=str, help='we use single task in our experience, so it will be a task name', 
                        required=True)
    parser.add_argument('--exps_dir', 
                        type=str, help='Directory for saving all the intermediate and final outputs.', 
                        required=True)
    parser.add_argument('--finder_L', 
                        type=int, help='', 
                        default=300)
    parser.add_argument('--ds_size', 
                        type=int,
                        help='number of maximum data examples sampled from each training dataset',
                        default=10000)
    parser.add_argument('--llm_model', 
                        type=str, help='pretrained model name or path.', 
                        required=True)

    args = parser.parse_args()
    args.cache_dir = str(Path(args.exps_dir)/'_cache')
    args.prompt_pool_path = str(Path(args.exps_dir)/args.task_name/'prompt_pool'/f'{args.task_name}_prompts.json')
    args.output_train_file = str(Path(args.exps_dir)/args.task_name/'unscored/train_split.json')
    args.output_valid_file = str(Path(args.exps_dir)/args.task_name/'unscored/valid_split.json')
    args.output_config_file = str(Path(args.exps_dir)/args.task_name/'config.json')
    return args


def main():
    cfg = parse_args()
    task_cls = task_map.cls_dic[cfg.task_name]()
    cfg.finder_L = task_cls.finder_L if task_cls.finder_L >= cfg.finder_L else cfg.finder_L
    print(type(task_cls),task_cls.finder_L,task_cls.run_scorer_bsz,task_cls.learning_rate,task_cls.balance_class)
    print(f"### class_num is {task_cls.class_num}! finder_L is {cfg.finder_L}")

    trainset,validset = find(cfg)
    with open(cfg.output_train_file,"w") as writer:
        writer.write(json.dumps(trainset, indent=4) + "\n")
    with open(cfg.output_valid_file,"w") as writer:
        writer.write(json.dumps(validset, indent=4) + "\n")
    print("finished find the samples")

    with open(cfg.output_config_file,"w") as writer:
        writer.write(json.dumps({
            "task_name":cfg.task_name,
            "ds_size":cfg.ds_size,
            "retriever_device":"cuda:0",
            "top_k":80,
            "cache_dir": str(Path(cfg.cache_dir).relative_to(Path(cfg.exps_dir))),
            "pretrained_model_name": "bert-base-uncased",
            "prompt_pool_paths":[
               str(Path(cfg.prompt_pool_path).relative_to(Path(cfg.exps_dir))),
            ],
            "output_unscored_files":[
                str(Path(cfg.output_train_file).relative_to(Path(cfg.exps_dir))),
                str(Path(cfg.output_valid_file).relative_to(Path(cfg.exps_dir))),
            ],
            "train_files":[
                str(Path(cfg.output_train_file.replace('/unscored/','/scored/')).relative_to(Path(cfg.exps_dir))),
            ],
            "valid_files":[
                str(Path(cfg.output_valid_file.replace('/unscored/','/scored/')).relative_to(Path(cfg.exps_dir))),
            ],
            "prompt_setup_type":"qa",
            "task_setup_type":"q",
            "sequence_length": 512,
            "epoches": 10,
            "learning_rate": task_cls.learning_rate,
            "balance_class": task_cls.balance_class,
            "finder_L": cfg.finder_L,
            "adam_eps": 1e-08,
            "weight_decay": 0.0,
            "dropout": 0.2,
            "multi_ctrs":False,
            "label_loss_penalty":0.001,
            "ortho_loss_penalty":1,
            "ctrs_loss_penalty":1,
            "representation_token_pos":0,
            "norm_mask":False,
            "projection_dim": 0,
            "batch_size": 8,
            "k_shot": 3,
            "temperature":1,
            "hard_mask":False,
            "generate_max_len":100,
            "lm_name": cfg.llm_model,
        }, indent=2) + "\n")

if __name__ == "__main__":
    main()