from functools import partial
import yaml
from pydantic import ValidationError
from yamlinclude import YamlIncludeConstructor
from src.config import CONFIG_DIR
from src.configs.base_config import Config

def read_config_from_yaml(path: str) -> dict:
    with open(path, "r") as stream:
        try:
            YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.SafeLoader, base_dir=CONFIG_DIR)
            yaml_obj = yaml.load(stream, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc
    return yaml_obj

def parse_config(config_obj: dict) -> Config:
    try:
        cfg = Config(**config_obj["config"])
        return cfg
    except ValidationError as exc:
        print(exc)
        raise exc

def patch_data_config_prompt_config(model, data_config, *kargs):
    if "prompt_config" in data_config:
        data_config["prompt_config"]["tokenizer_name"] = model
    return data_config

def patch_data_config_add_debug(data_config, subset_size=None):
    data_config["debug"] = True
    if subset_size:
        data_config["subset_size"] = subset_size
    return data_config

def patch_data_configs(config, patch_data_config_fn, subset_size=None):
    if "data_config" in config["benchmark_configs"][0]:
        data_configs = config["benchmark_configs"][0]["data_config"]
        if isinstance(config["benchmark_configs"][0]["data_config"], list):
            config["benchmark_configs"][0]["data_config"] = [patch_data_config_fn(dt_cfg, subset_size) for dt_cfg in data_configs]
        else:
            config["benchmark_configs"][0]["data_config"] = patch_data_config_fn(data_configs, subset_size)

def patch_config(config: dict, model, model_config, batch_size, answers_file, device, seed):
    old_config = config
    config = config["config"]
    if model_config:
        model_config_obj = read_config_from_yaml(model_config)
        config["model"] = model_config_obj
        model_name = model_config_obj["name"]
        patch_data_configs(config, partial(patch_data_config_prompt_config, model_name))
    if model:
        config["model"]["name"] = model
        config["model"]["tokenizer_name"] = model
        patch_data_configs(config, partial(patch_data_config_prompt_config, model))
    if answers_file:
        config["model"]["answers"] = answers_file
    if batch_size:
        config["model"]["batch_size"] = batch_size
    if device:
        config["model"]["device"] = device
    if seed:
        config["model"]["seed"] = seed

    old_config["config"] = config
    return old_config