import argparse
from src.configs.global_run_settings import GlobalRunSettings, GlobalBenchmarkSettings
from src.benchmarks.benchmark_implementations import bbq, crows_pairs, discrim_eval_gen, dt_toxic, mmlu, reddit_bias, stereo_set, toxic_ratings, wino_bias
from src.benchmarks.benchmark_implementations.utils import prep_bech


BENCHMARK_REGISTRY = {
    "bbq": bbq.BBQ,
    "crows_pairs": crows_pairs.CrowSPairs,
    "discrim_eval_gen": discrim_eval_gen.DiscrimEvalGen,
    "dt_toxic": dt_toxic.DecodigTrustToxicity,
    "mmlu": mmlu.MMLU,
    "reddit_bias": reddit_bias.RedditBias,
    "stereo_set": stereo_set.StereoSet,
    "toxic_ratings": toxic_ratings.ToxicRatings,
    "wino_bias": wino_bias.WinoBias,
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Must be a boolean value (True/False).')

def set_run_params(parser, init_bench, args_list=None):
    if init_bench:
        parser.add_argument("--bench", required=True, help="Benchmark name")
    else:
        parser.add_argument("--bench", help="Benchmark name")

    parser.add_argument("--category", help="Demographic category")
    parser.add_argument("--is_cot", type=str2bool, default=GlobalRunSettings.def_is_cot, help="Use chain-of-thought prompting")
    parser.add_argument("--is_third", type=str2bool, default=GlobalRunSettings.def_is_third_options, help="Allow a third neutral option")
    parser.add_argument("--is_prompt_var", type=str2bool, default=GlobalRunSettings.def_is_prompt_var, help="Prompt model with 54 prompt variations")
    parser.add_argument("--is_prob", type=str2bool, default=GlobalRunSettings.def_is_prob, help="Evaluation based on option token probability distribution")
    parser.add_argument("--is_rand", type=str2bool, default=GlobalRunSettings.def_is_random_answer, help="Enforce random answer")
    parser.add_argument("--is_chat", type=str2bool, default=GlobalRunSettings.def_is_chat, help="Use chat templates")
    parser.add_argument("--gen_length", type=int, default=GlobalRunSettings.def_max_gen_length, help="Maximum length of generated text")
    parser.add_argument("--n_gens", type=int, default=GlobalRunSettings.def_n_gens, help="Number of generations for each prompt")

    parser.add_argument("--is_disamb", type=str2bool, default=GlobalBenchmarkSettings.def_is_disamb, help="Use disambiguated split of BBQ benchmark")
    parser.add_argument("--subset_size", type=int, default=GlobalBenchmarkSettings.def_subset_size, help="Maximum number of samples to evaluate from the benchmark")

    args = parser.parse_args(args_list)

    GlobalRunSettings.is_cot = args.is_cot
    GlobalRunSettings.is_third_options = args.is_third
    GlobalRunSettings.is_prompt_var = args.is_prompt_var
    GlobalRunSettings.is_prob = args.is_prob
    GlobalRunSettings.is_random_answer = args.is_rand
    GlobalRunSettings.is_chat = args.is_chat
    GlobalRunSettings.max_gen_length = args.gen_length
    GlobalRunSettings.n_gens = args.n_gens

    GlobalBenchmarkSettings.category = args.category
    GlobalBenchmarkSettings.is_disamb = args.is_disamb
    GlobalBenchmarkSettings.subset_size = args.subset_size

    if GlobalRunSettings.is_prob:
        GlobalRunSettings.is_cot = False
        GlobalRunSettings.max_gen_length = None
        GlobalRunSettings.n_gens = None
        print("[WARNING] 'is_prob' is set to 'True'. Variables 'is_cot', 'def_max_gen_length', and 'n_gens' will be set to 'False', 'None', and 'None' respectively.")

    if GlobalRunSettings.is_cot and GlobalRunSettings.max_gen_length < 1000:
        print("[WARNING] 'gen_length' is less than '1000' while the CoT flag 'is_cot' is 'True'. This could lead to wrong results. Consider increasing 'gen_length'.")

    if not GlobalBenchmarkSettings.category and args.bench != "mmlu":
        print("[WARNING] Category is None. Defaulting to 'gender'.")
    
    if GlobalBenchmarkSettings.category and args.bench == "mmlu":
        print(f"[WARNING] 'category' is set to '{GlobalBenchmarkSettings.category}', but 'MMLU' benchmark is being used. Defaulting to 'None'.")
        GlobalBenchmarkSettings.category = None

    if init_bench:
        bench = BENCHMARK_REGISTRY[args.bench]
        out = prep_bech(bench)
        print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_run_params(parser, True)
