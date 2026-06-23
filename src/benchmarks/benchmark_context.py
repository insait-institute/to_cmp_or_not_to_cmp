from typing import List, cast
from src.config import dataset_registry
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_data_config import DataConfig
from src.configs.base_model_config import ModelConfig
from src.benchmarks.base_benchmark import BaseData
from src.models.proxy_model import ProxyModel
from src.results.base_connector import BaseConnector

class BenchmarkContext():
    def __init__(self, model: ProxyModel, benchmark_cls, benchmark_config: BenchmarkConfig, data_provider=None, metrics=None, data_handler=None):
        super().__init__()
        self.model = model
        self.model_config = model.config
        self.benchmark_config = benchmark_config
        self.benchmark_cls = benchmark_cls
        self.datasets: dict[str, BaseData] = dict()
        self.dataset_configs: dict[str, DataConfig] = dict()
        self.handler = data_handler
        self.metrics = metrics
        self.datasets["dataset"] = data_provider

    def add_dataset(self, dataset_config: DataConfig):
        dataset_name = dataset_config.type
        data_logic_cls = dataset_registry.get_logic_cls(dataset_name)
        self.datasets[dataset_name] = cast(BaseData, data_logic_cls(dataset_config))
        self.dataset_configs[dataset_name] = dataset_config

    def get_dataset(self) -> BaseData:
        values = self.datasets.values()
        if len(values) == 0:
            raise Exception("Benchmark context doesn't have registered any dataset")
        if len(values) != 1:
            raise Exception("Benchmark context is aware of more than one dataset, therefore it cannot distinguish between them!")
        return list(values)[0]

    def get_metrics(self) -> List:
        return self.metrics

    def get_benchmark_config(self) -> BenchmarkConfig:
        return self.benchmark_config

    def get_dataset_config(self, name: str) -> DataConfig:
        return self.dataset_configs[name]

    def get_data_handler(self) -> BaseConnector:
        return self.handler

    def get_model_config(self) -> ModelConfig:
        return self.model_config