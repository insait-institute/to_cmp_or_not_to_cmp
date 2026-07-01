import json
import os.path
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic_core import from_json

from src.configs.base_config import Config
from src.results.base_connector import BaseConnector, BenchmarkInfo

def get_timed_run_folder(results_folder: Path, benchmark_name: str, exp_name: str, timestamp: bool = True) -> Path:
    if timestamp:
        current_datetime = datetime.now()
        datetime_string = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
        run_name = benchmark_name + "/" + exp_name + "/" + datetime_string
    folder = results_folder / run_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder

class FileConnector(BaseConnector):
    def __init__(
        self,
        benchmark_name: str,
        exp_name: str,
        create_run: bool = True,
        run_folder: Optional[Path] = None,
        timestamp: bool = True,
        **kwargs
    ):
        super().__init__(benchmark_name, create_run, **kwargs)
        if not run_folder:
            raise ValueError("run_folder is required.")
        self.run_folder = run_folder
        self.run_id = self.get_run_id()
        self.create_run = create_run
        if create_run:
            self.run_folder = get_timed_run_folder(
                self.run_folder, benchmark_name, exp_name, timestamp=timestamp
            )

    def _store_final_result(self, final_result):
        with open(self.run_folder / "final_result.json", "w") as final_result_file:
            final_result_file.write(json.dumps(final_result, indent=4))

    def results_exist(self) -> bool:
        return os.path.exists(self.run_folder / "final_result.json")

    def __enter__(self) -> "FileConnector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        None

    def store_config(self, config: Config) -> None:
        with open(self.run_folder / "config.json", "w") as config_file:
            config_file.write(config.model_dump_json())

    def get_config(self) -> Config:
        with open(self.run_folder / "config.json", "r") as config_file:
            content = config_file.read()
        json_config = from_json(content)
        return Config.model_validate(json_config)

    def log_error(self, exp: Exception) -> None:
        with open(self.run_folder / "error.txt", "w") as f:
            f.write(str(exp))
            f.write(traceback.format_exc())