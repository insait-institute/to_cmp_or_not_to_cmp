import os
from dataclasses import dataclass

BENCH_SAVE_DIR = os.path.abspath("benchmark_exps")


class GlobalRunSettings:
    def_is_cot = True
    def_is_third_options = False
    def_is_prompt_var = False
    def_is_prob = False
    def_is_random_answer = False
    def_max_gen_length = 2000
    def_n_gens = 25
    def_is_chat = False

    def_add_generation_prompt = False
    def_continue_final_message = False


    is_cot = def_is_cot
    is_third_options = def_is_third_options
    is_prompt_var = def_is_prompt_var
    is_prob = def_is_prob
    is_random_answer = def_is_random_answer
    max_gen_length = def_max_gen_length
    n_gens = def_n_gens
    is_chat = def_is_chat

    add_generation_prompt = def_add_generation_prompt
    continue_final_message = def_continue_final_message
    
    @staticmethod
    def to_dict():
        return {
            "is_cot" : GlobalRunSettings.is_cot,
            "is_third_options" : GlobalRunSettings.is_third_options,
            "is_prompt_var" : GlobalRunSettings.is_prompt_var,
            "is_prob" : GlobalRunSettings.is_prob,
            "is_random_answer" : GlobalRunSettings.is_random_answer,
            "max_gen_length" : GlobalRunSettings.max_gen_length,
            "n_gens" : GlobalRunSettings.n_gens,
            "is_chat" : GlobalRunSettings.is_chat,
            "add_generation_prompt" : GlobalRunSettings.add_generation_prompt,
            "continue_final_message" : GlobalRunSettings.continue_final_message
        }

class GlobalBenchmarkSettings:
    def_category = None
    def_is_disamb = False
    def_subset_size = None


    category = def_category
    is_disamb = def_is_disamb
    subset_size = def_subset_size

    @staticmethod
    def to_dict():
        return {
            "category" : GlobalBenchmarkSettings.category,
            "is_disamb" : GlobalBenchmarkSettings.is_disamb,
            "subset_size" : GlobalBenchmarkSettings.subset_size,
        }

def to_string():
    s = ""
    s += f"_{GlobalBenchmarkSettings.category}" if GlobalBenchmarkSettings.category else ""
    s += "_cot" if GlobalRunSettings.is_cot else "_no-cot"
    s += "_third" if GlobalRunSettings.is_third_options else ""
    s += "_var" if GlobalRunSettings.is_prompt_var else ""
    s += "_prob" if GlobalRunSettings.is_prob else "_gen"
    s += "_rand" if GlobalRunSettings.is_random_answer else ""
    s += "_chat" if GlobalRunSettings.is_chat else ""
    s += "_disamb" if GlobalBenchmarkSettings.is_disamb else ""
    s += f"_subset_size-{GlobalBenchmarkSettings.subset_size}" if GlobalBenchmarkSettings.subset_size else ""
    return s

def to_string_extra():
    s = ""
    s += f"_n_gens-{GlobalRunSettings.n_gens}" if not GlobalRunSettings.is_prob else ""
    s += f"_gen_length-{GlobalRunSettings.max_gen_length}" if not GlobalRunSettings.is_prob else ""
    return s