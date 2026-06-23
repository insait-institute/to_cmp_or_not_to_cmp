import numpy as np
import pandas as pd

from src.benchmarks.benchmark_implementations.utils import set_cases

from src.benchmarks.benchmark_implementations.ssrb_utils import SSRBPromptBuilder
from src.configs.global_run_settings import GlobalBenchmarkSettings


class CrowSPairs():

    map_category_file_inv = {
    "gender" : "gender",
    "race-color" : "race",
    "religion" : "religion",
    }

    map_category_file = {
        "gender" : "gender",
        "race" : "race-color",
        "religion" : "religion",
    }

    def __init__(self):
        self.prmpt_build = SSRBPromptBuilder()
        self.targets = ["stereo", "anti", np.nan]
        self.cases = set_cases(self.targets)
        self.category = GlobalBenchmarkSettings.category
        self.subset_size = GlobalBenchmarkSettings.subset_size

    def create(self):
        """
            get CrowS-Pairs dataset from https://github.com/nyu-mll/crows-pairs/
        """
        data = pd.read_csv("benchmark_origin/crows_pairs/crows_pairs.csv", usecols=["sent_more", "sent_less", "bias_type"])
        real_cat = CrowSPairs.map_category_file[self.category]
        data.drop(data[data["bias_type"]!=real_cat].index, inplace=True)
        data.columns =["sent_more", "sent_less", "bias_type_orig"]
        data["bias_type"] = data["bias_type_orig"].apply(lambda x: CrowSPairs.map_category_file_inv[x])
        data.sort_values(by=["bias_type"], inplace=True)
        data.drop_duplicates(keep='first', inplace=True)
        subset_size = min(len(data), len(data) if self.subset_size is None else self.subset_size)
        data = data.iloc[:subset_size]
        return data

    def get_name(self):
        return "crows_pairs"
    
    def get_cols_to_drop(self):
        return ["bias_type", "sent_more", "sent_less", "bias_type_orig"]