import os
import sys
import platform
import argparse
from pathlib import Path

from src.models.base.model_factory import get_model_from_config
from src.results.file_connector import FileConnector
from src.runner import Runner
from src.utils.initialization import seed_everything
from src.utils.loading import parse_config, patch_config, read_config_from_yaml
from create_bench import set_run_params
from src.configs.global_run_settings import to_string, to_string_extra


print("Node:", platform.node())
print("Python executable:", sys.executable)
print("Conda env name:", os.environ.get("CONDA_DEFAULT_ENV"))
print("Conda env path:", os.environ.get("CONDA_PREFIX"))
print("Virtualenv:", os.environ.get("VIRTUAL_ENV"))

def my_app(model_name: str, bench_name: str, cfg: dict, results_folder: Path = Path("runs")) -> None:
    cfg["data_config"]["type"] = bench_name
    cfg["config"]["seed"] = cfg["config"]["model"]["seed"]
    app = cfg["config"]["benchmark_configs"]
    app[0]["name"] = bench_name
    app[0]["type"] = bench_name
    app[0] |= {"postprocessor" : {"type" : "reformat_generic"}}

    model_name = model_name.split("/")[-1]


    cfg_obj = parse_config(cfg)
    seed_everything(cfg_obj.seed)
    model = get_model_from_config(cfg_obj.model)

    benchmark_name = bench_name
    exp_name = to_string() + to_string_extra()

    with FileConnector(model_name, benchmark_name, exp_name, run_folder=results_folder) as result_handler:
        if result_handler.results_exist():
            return

        result_handler.store_config(cfg_obj)
        runner = Runner(model, cfg_obj, result_handler)

        try:
            runner.run()
        except Exception as e:
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Social Bias Evaluation Framework")
    parser.add_argument("--results_folder", type=str, help="Name of the run folder where results from all the benchmarks are stored")
    parser.add_argument("--model", type=str, help="Name of model to use")
    parser.add_argument("--model_config", type=str, help="Config file form model, applied before --model")
    parser.add_argument("--batch_size", type=int, help="Batch size to use")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    set_run_params(parser, False)

    args = parser.parse_args()
    kwargs = {}
    kwargs["seed"] = args.seed
    model_name = args.model
    model_config = args.model_config
    batch_size = args.batch_size
    bench_name = args.bench

    results_folder = args.results_folder
    if not results_folder:
        results_folder = Path("runs")
    else:
        results_folder = Path(results_folder)

    config_path ="src/empty.yaml" # TODO TO REMOVE, from old Compl-AI fw
    run_name = config_path.split("/")[1].split(".")[0]
    config_dict = read_config_from_yaml(config_path)

    config_dict = patch_config(config_dict, model_name, model_config, batch_size, None, None, **kwargs)
    my_app(model_name, bench_name, config_dict, results_folder=results_folder)