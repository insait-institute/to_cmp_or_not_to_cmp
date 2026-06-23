import numpy as np
import pandas as pd

from src.benchmarks.benchmark_implementations.utils import set_cases

from src.benchmarks.benchmark_implementations.ssrb_utils import SSRBPromptBuilder
from src.configs.global_run_settings import GlobalBenchmarkSettings

class RedditBias():

    map_category_file_inv = {
    "gender" : "gender",
    "race-color" : "race",
    "religion" : "religion",
    }

    map_category_file = {
        "gender" : "gender",
        "race" : "race-color",
        "religion" : "religion_jc",
    }

    def __init__(self):
        self.prmpt_build = SSRBPromptBuilder()
        self.targets = ["stereo", "anti", np.nan]
        self.cases = set_cases(self.targets)
        self.category = GlobalBenchmarkSettings.category
        self.subset_size = GlobalBenchmarkSettings.subset_size

    def create(self):
        """
            get Reddit Bias dataset from https://github.com/SoumyaBarikeri/RedditBias
        """   
        data = pd.read_csv(f"benchmark_origin/reddit_bias/{self.map_category_file[self.category]}.csv", usecols=['comments', 'comments_processed'])  
        data['sent_more'] = data['comments']
        data['sent_less'] = data['comments_processed']
        data.drop_duplicates(keep='first', inplace=True)
        subset_size = min(len(data), len(data) if self.subset_size is None else self.subset_size)
        data = data.iloc[:subset_size]
        return data

    def get_name(self):
        return "reddit_bias"
    
    def get_cols_to_drop(self):
        return ["sent_more", "sent_less", "comments", "comments_processed"]