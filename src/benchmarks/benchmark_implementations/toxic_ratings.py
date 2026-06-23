from src.benchmarks.benchmark_implementations.utils import set_cases

from src.benchmarks.benchmark_implementations.dttr_utils import DTTRPromptBuilder
from src.configs.global_run_settings import GlobalBenchmarkSettings

import pandas as pd
import numpy as np


class ToxicRatings():

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
        data = pd.read_json(f"benchmark_origin/tox_rat/toxicity_ratings.json", lines=True)
        ratings_df = data["ratings"].explode().apply(pd.Series)
        data = data.drop(columns=["ratings"]).join(ratings_df)
        data.drop(['source', 'gender', 'race', 'technology_impact', 'perspective_score', 'is_profane', 'is_threat', 'is_identity_attack', 'is_insult', 'uses_media_social', 'uses_media_news', 'uses_media_video', 'uses_media_forums', 'personally_seen_toxic_content', 'personally_been_target', 'identify_as_transgender', 'toxic_comments_problem', 'education', 'age_range', 'lgbtq_status', 'political_affilation', 'is_parent', 'religion_important', 'fine_to_see_online', 'remove_from_online'], axis=1, inplace=True)

        data = data.groupby("comment", as_index=False).mean()
        data = data[data["is_sexual_harassment"]==0]
        data.sort_values(by=[ "toxic_score",], ascending=False, inplace=True)
        data["context"] = data["comment"]
        subset_size = min(len(data), len(data) if self.subset_size is None else self.subset_size)
        data = data.iloc[:subset_size]
        data.drop_duplicates(keep='first', inplace=True)
        self.cases = set_cases(self.map_category[self.category])
        return data

    def get_name(self):
        return "toxic_ratings"
    
    def get_cols_to_drop(self):
        return ["context", "comment", "comment_id", "toxic_score", "is_sexual_harassment"]
