import numpy as np

from src.benchmarks.benchmark_implementations.utils import set_cases

from src.benchmarks.benchmark_implementations.prompt_builder import PromptBuilder
from src.configs.global_run_settings import GlobalBenchmarkSettings

from datasets import load_from_disk


class MMLUPromptBuilder(PromptBuilder):

    def get_bench_prompt_cmp(self, row):
        sent = row['question']
        ans_a = row["choices"][row["corr_ans"]] if row["is_focused_p"] else row["choices"][row["wrong_ans"]]
        ans_b = row["choices"][row["wrong_ans"]] if row["is_focused_p"] else row["choices"][row["corr_ans"]]

        return (
              f"\nGiven the following problem: "
            + f"\n{sent}"
            + f"\nChoose the correct answer from the options below: "
            + f"\nA: {ans_a} "
            + f"\nB: {ans_b} "
            + "{}"
        )

    def get_bench_prompt_iso(self, row):
        sent = row['question']
        ans = row["choices"][row["corr_ans"]] if row["is_focused_p"] else row["choices"][row["wrong_ans"]]

        return (
              f"\nGiven the following problem: "
            + f"\n{sent}"
            + f"\nAnswer the following question: "
            + f"\nIs \"{ans}\" the correct answer to the problem?"
        )

ALLOWED_SUBJECTS = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
ALLOWED_SUBJECTS = ['abstract_algebra'] # TODO

class MMLU():

    def __init__(self):
        self.prmpt_build = MMLUPromptBuilder()
        self.targets = ["corr", "wrong", np.nan]
        self.cases = set_cases(self.targets)
        self.rng = np.random.default_rng(seed=42)
        self.subset_size = GlobalBenchmarkSettings.subset_size

    def create(self):
        """
            get MMLU dataset from https://huggingface.co/datasets/cais/mmlu
        """
        data = load_from_disk("benchmark_origin/mmlu/test").to_pandas()
        data.drop(data[~data['subject'].isin(ALLOWED_SUBJECTS)].index, inplace=True)
        corr_ans = data["answer"].to_numpy()
        rnd_offset = self.rng.integers(1, 4, size=len(corr_ans))
        wrong_ans = (corr_ans + rnd_offset) % 4

        data["choices"] = data["choices"].apply(lambda x: tuple(x))
        data["corr_ans"] = corr_ans
        data["wrong_ans"] = wrong_ans
        data.drop_duplicates(keep='first', inplace=True)
        subset_size = min(len(data), len(data) if self.subset_size is None else self.subset_size)
        data = data.iloc[:subset_size]
        return data

    def get_name(self):
        return "mmlu"
    
    def get_cols_to_drop(self):
        return ["question", "subject", "choices", "answer", "corr_ans", "wrong_ans"]