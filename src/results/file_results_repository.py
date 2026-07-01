import json
import os
from pathlib import Path

from pydantic_core import from_json

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunInfo:
    benchmark_name: str
    model_name: str
    data: dict
    config: dict

class BaseResultsRepository(ABC):
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir

    @abstractmethod
    def list(self) -> list[RunInfo]:
        pass

def node_directories(results_directory) -> list[str]:
    node_dirs = []
    for root, dirs, files in os.walk(results_directory):
        if "config.json" in files:
            node_dirs.append(root)
    return node_dirs

def get_config(run_folder):
    with open(run_folder / "config.json", "r") as config_file:
        content = config_file.read()
    json_config = from_json(content)
    return json_config

def get_run_info(run_path: Path) -> RunInfo:
    config = get_config(run_path)
    results = {}
    if len(results) == 0:
        try:
            alt_res_file = run_path / "final_result.json"
            with open(alt_res_file, "r") as f:
                results = json.load(f)
        except Exception:
            pass
    if "config" in config:
        config = config["config"]
    run_info = RunInfo(
        benchmark_name=(
            config["benchmark_configs"][0]["name"]
            if "name" in config["benchmark_configs"][0]
            else config["benchmark_configs"][0]["type"]
        ),
        model_name=config["model"]["name"],
        data=results,
        config=config,
    )
    return run_info

class FileResultsRepository(BaseResultsRepository):
    def __init__(self, results_dir: Path):
        super().__init__(results_dir)
        self.results_dir = results_dir

    def list(self) -> list[RunInfo]:
        node_dirs = node_directories(self.results_dir)
        run_infos = []
        for node_dir in node_dirs:
            try:
                run_infos.append(get_run_info(Path(node_dir)))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        return run_infos