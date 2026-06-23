import numpy as np
import pandas as pd

from src.benchmarks.benchmark_implementations.utils import set_cases

from src.benchmarks.benchmark_implementations.dttr_utils import DTTRPromptBuilder
from src.configs.global_run_settings import GlobalBenchmarkSettings

class DecodigTrustToxicity():

    map_category = {
        "gender" : ["male", "female", np.nan],
        "race" : ["white", "black", np.nan],
        "religion" : ["christian", "muslim", np.nan],
    }

    def __init__(self):
        self.prmpt_build = DTTRPromptBuilder()
        self.cases = None
        self.category = GlobalBenchmarkSettings.category
        self.subset_size = GlobalBenchmarkSettings.subset_size

    def create(self):
        """
            get DecodingTrust Toxicity dataset from https://github.com/AI-secure/DecodingTrust/tree/main
        """
        data = pd.read_json(f"benchmark_origin/decoding_trust_toxicity/toxic_prompts_gpt4.jsonl", lines=True)
        data["context"] = data["prompt"].apply(lambda x: x["text"])
        data.drop(["prompt"], axis=1, inplace=True)
        data.drop_duplicates(keep='first', inplace=True)
        self.cases = set_cases(self.map_category[self.category])
        subset_size = min(len(data), len(data) if self.subset_size is None else self.subset_size)
        data = data.iloc[:subset_size]
        return data

    def get_name(self):
        return "dt_toxic"
    
    def get_cols_to_drop(self):
        return ["source", "context"]