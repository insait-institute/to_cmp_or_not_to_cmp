import numpy as np
import pandas as pd

from src.benchmarks.benchmark_implementations.utils import set_cases

from src.benchmarks.benchmark_implementations.ssrb_utils import SSRBPromptBuilder
from src.configs.global_run_settings import GlobalBenchmarkSettings

class StereoSet():

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
            get StereoSet dataset from https://github.com/moinnadeem/StereoSet
        """

        data = pd.read_json("benchmark_origin/stereo_set/stereo_set.json")
        intra_data = data["data"]["intrasentence"]
        data_list = []
        for item in intra_data:
            sentences_dict = {s['gold_label']: s['sentence'] for s in item['sentences']}

            data_list.append({
                'bias_type': item['bias_type'],
                'sent_more': sentences_dict.get('stereotype'),
                'sent_less': sentences_dict.get('anti-stereotype'),
            })

        data = pd.DataFrame(data_list)
        data.drop(data[data["bias_type"]!=self.category].index, inplace=True)
        data.sort_values(by=["bias_type"], inplace=True)
        data.drop_duplicates(keep='first', inplace=True)
        subset_size = min(len(data), len(data) if self.subset_size is None else self.subset_size)
        data = data.iloc[:subset_size]
        return data

    def get_name(self):
        return "stereo_set"
    
    def get_cols_to_drop(self):
        return ["bias_type", "sent_more", "sent_less"]