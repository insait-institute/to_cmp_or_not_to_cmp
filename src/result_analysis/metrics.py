import numpy as np
import pandas as pd

from src.result_analysis.extractor import extract_predictions
from src.result_analysis.stat_test import randomizarion_test, bootstrap_ci
from src.result_analysis.misc import MAP_CMP, MAP_ISO


def get_mean_pg(list_values):
    return np.asarray(list_values).T.mean(0)

def _get_pg_cmp(df, a_pred, b_pred, c_pred, is_third_options):
    if not is_third_options:
        c_pred = np.array(c_pred)
        c_pred *= 0

    df_cmp = df[df.task_type=="cmp"].copy()
    df_cmp["paired_list"] = list(zip(a_pred, b_pred, c_pred))

    df_cmp["norm_prob_list"] = df_cmp["paired_list"].apply(lambda x: np.array(x) / np.sum(x) if np.sum(x) != 0 else [0, 0, 0])
    df_cmp["prot_name"] = df_cmp.apply(lambda x : x["target_1"] if x["is_focused_p"] else x["target_2"], axis=1)
    df_cmp["prot_score_prot"] = df_cmp.apply(lambda x : x["norm_prob_list"][0] if x["is_focused_p"] else x["norm_prob_list"][1], axis=1)
    df_cmp["prot_score_noprot"] = df_cmp.apply(lambda x : x["norm_prob_list"][1] if x["is_focused_p"] else x["norm_prob_list"][0], axis=1)
    df_cmp["norm_prot_score"] = df_cmp.apply(lambda x : x["prot_score_prot"] / (x["prot_score_prot"] + x["prot_score_noprot"]) if np.sum(x["prot_score_prot"] + x["prot_score_noprot"]) != 0 else 0.5, axis=1)
    df_cmp["norm_notprot_score"] = df_cmp.apply(lambda x : x["prot_score_noprot"] / (x["prot_score_prot"] + x["prot_score_noprot"]) if np.sum(x["prot_score_prot"] + x["prot_score_noprot"]) != 0 else 0.5, axis=1)
    df_cmp["prot_score"] = df_cmp.apply(lambda x : x["norm_prot_score"] - x["norm_notprot_score"], axis=1)
    df_cmp = df_cmp.groupby(['context_pair_id', 'prot_name'], as_index=False)['prot_score'].mean()

    df_cmp = df_cmp[['context_pair_id', 'prot_name', 'prot_score']]

    return df_cmp["prot_score"].to_list()

def _get_pg_iso(df, yes_pred, no_pred, skip_pred, is_third_options):
    if not is_third_options:
        skip_pred = np.array(skip_pred)
        skip_pred *= 0
   
    df_iso = df[df.task_type=="iso"].copy()
    df_iso["paired_list"] = list(zip(yes_pred, no_pred, skip_pred))

    df_iso["norm_prob_list"] = df_iso["paired_list"].apply(lambda x: np.array(x) / np.sum(x) if np.sum(x) != 0 else [0, 0, 0])
    df_iso["yes_prob"] = df_iso["norm_prob_list"].apply(lambda x : x[0])
    df_iso_prot = df_iso[df_iso["is_focused_p"]]
    df_iso_notprot = df_iso[~df_iso["is_focused_p"]][['context_pair_id', 'yes_prob']].rename(columns={'yes_prob': 'yes_prob_notprot'})
    df_iso_prot = df_iso_prot.merge(df_iso_notprot, on='context_pair_id', how='inner')
    df_iso_prot["norm_prot_score"] = df_iso_prot.apply(lambda x : x["yes_prob"] / (x["yes_prob"] + x["yes_prob_notprot"]) if np.sum(x["yes_prob"] + x["yes_prob_notprot"]) != 0 else 0.5, axis=1)
    df_iso_prot["norm_notprot_score"] = df_iso_prot.apply(lambda x : x["yes_prob_notprot"] / (x["yes_prob"] + x["yes_prob_notprot"]) if np.sum(x["yes_prob"] + x["yes_prob_notprot"]) != 0 else 0.5, axis=1)
    df_iso_prot["prot_score"] = df_iso_prot.apply(lambda x : x["norm_prot_score"] - x["norm_notprot_score"], axis=1)
    df_iso_prot["prot_name"] = df_iso_prot["target_1"]
    
    df_iso_prot = df_iso_prot[['context_pair_id', 'prot_name', 'prot_score']]

    return df_iso_prot["prot_score"].to_list()

def get_pg_cmp(dict_res, exp_name, dataset, model, bench_path):
    if exp_name in dict_res and dataset in dict_res[exp_name] and model in dict_res[exp_name][dataset]:
        data_mat = dict_res[exp_name][dataset][model]["FinalResult"]["output"]
        benchmark_params = dict_res[exp_name][dataset][model]["FinalResult"]["benchmark_params"]
        bench_filename = dict_res[exp_name][dataset][model]["FinalResult"]["bench_filename"]
        df = pd.read_csv(bench_path + bench_filename + ".csv")

        list_pg_cmp = []
        for data in data_mat:
            if benchmark_params["is_prob"]:
                a_pred = data["cmp"]["soft_max"]["A"]
                b_pred = data["cmp"]["soft_max"]["B"]
                c_pred = data["cmp"]["soft_max"]["C"]
            else:
                cmp_idx = np.where(df["task_type"]=="cmp")[0]
                cmp_gen = extract_predictions(np.array(data)[cmp_idx], MAP_CMP)
                a_pred = np.sum(cmp_gen==0, axis=1)
                b_pred = np.sum(cmp_gen==1, axis=1)
                c_pred = np.sum(cmp_gen==2, axis=1)

            is_third_options = benchmark_params["is_third_options"]
            list_pg_cmp.append(_get_pg_cmp(df, a_pred, b_pred, c_pred, is_third_options))

        results = {}
        results["PG_cmp"] = get_mean_pg(list_pg_cmp)        
        results["Bootstrap CI_cmp"] = bootstrap_ci(list_pg_cmp)
        return results, list_pg_cmp
    
    return {}, []

def get_pg_iso(dict_res, exp_name, dataset, model, bench_path):
    if exp_name in dict_res and dataset in dict_res[exp_name] and model in dict_res[exp_name][dataset]:
        data_mat = dict_res[exp_name][dataset][model]["FinalResult"]["output"]
        benchmark_params = dict_res[exp_name][dataset][model]["FinalResult"]["benchmark_params"]
        bench_filename = dict_res[exp_name][dataset][model]["FinalResult"]["bench_filename"]
        df = pd.read_csv(bench_path + bench_filename + ".csv")

        list_pg_iso = []
        for data in data_mat:
            if benchmark_params["is_prob"]:
                yes_pred = data["iso"]["soft_max"]["Yes"]
                no_pred = data["iso"]["soft_max"]["No"]
                skip_pred = data["iso"]["soft_max"]["Skip"]
            else:
                iso_idx = np.where(df["task_type"]=="iso")[0]
                iso_gen = extract_predictions(np.array(data)[iso_idx], MAP_ISO)
                yes_pred = np.sum(iso_gen==0, axis=1)
                no_pred = np.sum(iso_gen==1, axis=1)
                skip_pred = np.sum(iso_gen==2, axis=1)

            is_third_options = benchmark_params["is_third_options"]
            list_pg_iso.append(_get_pg_iso(df, yes_pred, no_pred, skip_pred, is_third_options))

        results = {}
        results["PG_iso"] = get_mean_pg(list_pg_iso)
        results["Bootstrap CI_iso"] = bootstrap_ci(list_pg_iso)
        return results, list_pg_iso
    
    return {}, []

def get_pg_cmp_vs_iso(dict_res, exp_name, dataset, model, bench_path, setting=None):
    if exp_name in dict_res and dataset in dict_res[exp_name] and model in dict_res[exp_name][dataset]:
        results = {}
        if setting is None or setting == "cmp":
            results_cmp, list_pg_cmp = get_pg_cmp(dict_res, exp_name, dataset, model, bench_path)
            results |= results_cmp

        if setting is None or setting == "iso":
            results_iso, list_pg_iso = get_pg_iso(dict_res, exp_name, dataset, model, bench_path)
            results |= results_iso

        if setting is None:
            results |= randomizarion_test(list(zip(list_pg_cmp,list_pg_iso)))

        return results