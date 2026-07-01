import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.configs.base_config import Config
from pydantic import BaseModel, ConfigDict, RootModel

class FinalResult(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )

from typing import Any, Dict, List, Union

class BaseResult(BaseModel):
    name: str
    value: Any
    description: str

Result = RootModel[BaseResult]
Results = RootModel[List[Union["Result", "Results", "ResultsByName"]]]
ResultsByName = RootModel[Dict[str, Union["Result", "Results", "ResultsByName"]]]

Results.model_rebuild()
ResultsByName.model_rebuild()

@dataclass
class BenchmarkInfo:
    benchmark_type: str
    category: str

class BaseConnector(ABC):
    def __init__(self, benchmark_name: BenchmarkInfo, create_run: bool = False, **kwargs):
        self.benchmark_name = benchmark_name
        self.run_id = str(uuid.uuid4())

    def get_run_id(self):
        return self.run_id

    @abstractmethod
    def _store_final_result(self, final_result):
        pass
      
    def add_final_result(self, final_result: FinalResult, runtime: float):
        final_result_info = final_result.model_dump()
        final_result_info = {
            "run_id": self.run_id,
            "time": time.time(),
            "runtime": runtime,
            "benchmark_name": self.benchmark_name,
            "FinalResult": final_result_info,
        }
        return self._store_final_result(final_result_info)

    @abstractmethod
    def get_config(self) -> Config:
        pass

    @abstractmethod
    def store_config(self, config: Config) -> None:
        pass

    @abstractmethod
    def log_error(self, exp: Exception) -> None:
        pass