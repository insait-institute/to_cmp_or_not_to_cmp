import ijson
import json
from pathlib import Path
from collections import defaultdict
import argparse

from create_bench import set_run_params
from src.configs.global_run_settings import to_string, to_string_extra


MAP_ISO = {"yes" : 0, "no" : 1, "y" : 0, "n" : 1, "skip" : 2, "s" : 2}
MAP_CMP = {"a" : 0, "b" : 1, "c" : 2}


def load_single_result(file_path):
    with open(file_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        return data
    
    return None

def get_res_path(result_folder):
    root_dir = Path(result_folder)
    return [str(p) for p in root_dir.rglob("final_result.json")]

def exp_name_to_path(result_folder):
    dict_res = defaultdict(lambda: defaultdict(dict))

    list_path = sorted(get_res_path(result_folder)) # Sorting ensures the most recent result per dataset, experiment setting, and model is processed last (overwriting older ones).

    for path_results in list_path:
        path_config = path_results.replace("final_result.json","config.json")
            
        with open(path_results, "rb") as f:
            exp_name = next(ijson.items(f, "FinalResult.exp_name"), None)
        with open(path_config, "rb") as f:
            model_name = next(ijson.items(f, "model.name"), None)
            model_name = model_name.split('/')[-1] if model_name else None

            f.seek(0)
            dataset_name = next(ijson.items(f, "benchmark_configs.item.name"), None)
            
            if exp_name:
                dict_res[dataset_name][model_name][exp_name] = path_results

    return {k: {sk: dict(sv) for sk, sv in v.items()} for k, v in dict_res.items()}

def get_exp_name(params):
    parser = argparse.ArgumentParser()

    args = []
    for k, v in params.items():
            args.extend([k, str(v)])

    set_run_params(parser, init_bench=False, args_list=args)
    return to_string() + to_string_extra()


def get_data(dict_path, list_models, list_exps, list_datasets, dict_res=None):
    if not dict_res:
        dict_res = {}
    for exp in list_exps:
        for bench in list_datasets:
            for model in list_models:
                if not dict_res or not exp in dict_res or not bench in dict_res[exp] or not model in dict_res[exp][bench]:                    
                    if bench in dict_path and model in dict_path[bench] and exp in dict_path[bench][model]:
                        data = load_single_result(dict_path[bench][model][exp]) 
                        dict_res.setdefault(exp, {}).setdefault(bench, {}).setdefault(model, {})
                        dict_res[exp][bench][model] = data

    return dict_res