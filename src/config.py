import json
import os
from pathlib import Path

from src.benchmarks.base_benchmark import BaseBenchmark, BaseData
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_data_config import DataConfig
from src.registry import BENCHMARK_PROCESSORS, ComponentRegistry, registry

CONFIG_DIR = os.path.abspath("../configs/")
RESULTS_DIR = os.path.abspath("../results/")
CODE_ROOT_PATH = Path(os.path.abspath(__file__)) / "../src"

dataset_registry = ComponentRegistry(BaseData, DataConfig)
benchmark_registry = ComponentRegistry(BaseBenchmark, BenchmarkConfig)

registry.register("data", dataset_registry)
registry.register("benchmark", benchmark_registry)

def reformat_generic(eval_results: dict, legacy: bool = False) -> dict:
    data = json.loads(eval_results["benchmark_specific_info"])
    while isinstance(data, list) and len(data) == 1:
        data = data[0]
    return data

BENCHMARK_PROCESSORS |= {
    "reformat_generic": reformat_generic,
}

# --- MMLU ---
benchmark_registry.register_logic_config_classes("mmlu")
dataset_registry.register_logic_config_classes("mmlu")

# --- REDDIT BIAS ---
benchmark_registry.register_logic_config_classes("reddit_bias")
dataset_registry.register_logic_config_classes("reddit_bias")

# --- STEREO SET ---
benchmark_registry.register_logic_config_classes("stereo_set")
dataset_registry.register_logic_config_classes("stereo_set")

# --- CROWS PAIRS ---
benchmark_registry.register_logic_config_classes("crows_pairs")
dataset_registry.register_logic_config_classes("crows_pairs")

# --- BBQ ---
benchmark_registry.register_logic_config_classes("bbq")
dataset_registry.register_logic_config_classes("bbq")

# --- DT TOXIC ---
benchmark_registry.register_logic_config_classes("dt_toxic")
dataset_registry.register_logic_config_classes("dt_toxic")

# --- TOXIC RATINGS ---
benchmark_registry.register_logic_config_classes("toxic_ratings")
dataset_registry.register_logic_config_classes("toxic_ratings")

# --- WINO BIAS ---
benchmark_registry.register_logic_config_classes("wino_bias")
dataset_registry.register_logic_config_classes("wino_bias")

# --- DISCRIM EVAL GEN ---
benchmark_registry.register_logic_config_classes("discrim_eval_gen")
dataset_registry.register_logic_config_classes("discrim_eval_gen")