import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.configs.global_run_settings import GlobalRunSettings, to_string, to_string_extra, BENCH_SAVE_DIR
from src.benchmarks.benchmark_implementations.prompt_builder import *


def prep_bech(bench_call):
    bench = bench_call()
    data = bench.create()
    cases = bench.cases
    data.drop_duplicates(keep='first', inplace=True)

    data["context_pair_id"] = np.arange(len(data))
    list_df = []
    for task_type, is_focused_p in cases.keys():
        df = data.copy()
        df["task_type"] = task_type
        df["is_focused_p"] = is_focused_p
        df["target_1"] = cases[(task_type, is_focused_p)][0]
        df["target_2"] = cases[(task_type, is_focused_p)][1]
        list_df.append(df)

    data = pd.concat(list_df, ignore_index=True)
    data["instance_id"] = np.arange(len(data))

    data['prompts'] = data.apply(bench.prmpt_build.prompt_variations, axis=1)

    cols_to_drop = bench.get_cols_to_drop()
    data.drop(cols_to_drop, axis=1, inplace=True)

    new_order = ["instance_id", "context_pair_id", "task_type", "is_focused_p", "target_1", "target_2", "prompts"]
    data = data[new_order + [col for col in data.columns if col not in new_order]]
    filaname = bench.get_name() + to_string()
    save_prep_dataset(data, filaname)
    return filaname, data

def common_run(model, context):
    filename, df_data = context.get_dataset().get_data()
    max_tokens = GlobalRunSettings.max_gen_length
    n = GlobalRunSettings.n_gens
    is_prob = GlobalRunSettings.is_prob

    set_chat_params(is_prob)
    model.reset_statistic()

    if is_prob:
        mask_iso = df_data["task_type"] == "iso"
        mask_cmp = df_data["task_type"] == "cmp"

        if GlobalRunSettings.is_chat:
            output_iso = model.most_prob_options_chat(df_data[mask_iso]["prompts"].to_list(), ANS_MAP_YNS, get_soft_max=False, add_generation_prompt=GlobalRunSettings.add_generation_prompt, continue_final_message=GlobalRunSettings.continue_final_message)
            output_cmp = model.most_prob_options_chat(df_data[mask_cmp]["prompts"].to_list(), ANS_MAP_ABC, get_soft_max=False, add_generation_prompt=GlobalRunSettings.add_generation_prompt, continue_final_message=GlobalRunSettings.continue_final_message)
        else:
            output_iso = model.most_prob_options(df_data[mask_iso]["prompts"].to_list(), ANS_MAP_YNS, get_soft_max=False)
            output_cmp = model.most_prob_options(df_data[mask_cmp]["prompts"].to_list(), ANS_MAP_ABC, get_soft_max=False)

            output = [{"iso" : p[0], "cmp" : p[1]} for p in zip(output_iso, output_cmp)]

    else:
        prompts = df_data["prompts"].to_list()
        if GlobalRunSettings.is_chat:
            output = model.generate_chat(prompts, max_tokens=max_tokens, n=n, add_generation_prompt=GlobalRunSettings.add_generation_prompt, continue_final_message=GlobalRunSettings.continue_final_message)
        else:
            output = model.generate(prompts, max_tokens=max_tokens, n=n)
    
    results = {
        "aggregated_results" : {"token_statistics" : model.get_statistic()},
        "output" : output,
        "benchmark_params" : GlobalRunSettings.to_dict(),
        "bench_filename" : filename,
        "exp_name" : to_string() + to_string_extra()
    }
    
    return to_serializable(results)

def set_cases(list_cases):
    return {
        ("iso", True) : [list_cases[0], list_cases[2]], 
        ("iso", False) : [list_cases[1], list_cases[2]], 
        ("cmp", True) : [list_cases[0], list_cases[1]], 
        ("cmp", False) : [list_cases[1], list_cases[0]], 
    }

def set_chat_params(is_prob):
    if GlobalRunSettings.is_chat:
        if is_prob:
            GlobalRunSettings.add_generation_prompt = False
            GlobalRunSettings.continue_final_message = True
        else:
            GlobalRunSettings.add_generation_prompt = False
            GlobalRunSettings.continue_final_message = False

def save_prep_dataset(data, filename):
    pre_proc_data_path = f"{BENCH_SAVE_DIR}/{filename}.csv"
    path = Path(pre_proc_data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_to_save = data.copy()
    if 'prompts' in data_to_save.columns:
        data_to_save['prompts'] = data_to_save['prompts'].apply(json.dumps)
    data_to_save.to_csv(pre_proc_data_path, index=False)

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj