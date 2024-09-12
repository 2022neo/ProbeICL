from utils.tools import getConfig
import argparse
from pathlib import Path
import json
import os
from dpr.data.biencoder_data import get_raw_data

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
    for inputfile in config.output_unscored_files:
        name = f'*{Path(inputfile).stem}*.jsonl'
        outputpath = Path(args.taskpath)/"scored"
        data = []
        processed_ids = set([])
        for p in outputpath.rglob(name):
            # data += [json.loads(line) for line in p.open('r')]
            for line in p.open('r'):
                entry = json.loads(line)
                if entry['id'] not in processed_ids:
                    processed_ids.add(entry['id'])
                    data.append(entry)
        raw_data = get_raw_data(inputfile)
        assert len(raw_data)==len(data)
 
        ans_name = Path(inputfile).stem+'.json'
        ans_fn = Path(args.taskpath)/"scored"/ans_name
        with ans_fn.open('w') as f:
            json.dump(data,f,indent=2)
        
        for p in outputpath.rglob(name):
            os.remove(str(p))

if __name__=='__main__':
    main()