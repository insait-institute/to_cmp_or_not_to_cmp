import json
import logging
import time
from typing import Callable, Iterator, List, Optional

from pydantic import ValidationError

from src.benchmarks.base_benchmark import BaseBenchmark
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_config import Config
from src.benchmarks.benchmark_context import BenchmarkContext
from src.benchmarks.base_benchmark import BaseData
from src.models.base.base_model import BaseModel
from src.models.proxy_model import ProxyModel
from src.registry import BENCHMARK_PROCESSORS, registry
from src.results.base_connector import BaseConnector
from src.results.base_connector import BaseResult, FinalResult, Results

from src.configs.base_data_config import DataConfig
from src.models.proxy_model import ProxyModel


class DataContext():
    """This is what each benchmark gets and contains methods to get all the relevant objects and information dynamically"""

    def __init__(self, model: ProxyModel, config: DataConfig) -> None:
        self.model = model
        self.config = config

    def get_data_config(self) -> DataConfig:
        return self.config

def get_json_results(results: List[BaseResult]) -> str:
    try:
        model_results = Results(results)
        json_result = model_results.model_dump_json()
    except ValidationError:
        json_result = json.dumps(results)
    return json_result

class InternalBenchmarkRepresentation:
    def __init__(self, model: BaseModel, config: BenchmarkConfig, data_handler=None):
        self.model: BaseModel = model
        self.config: BenchmarkConfig = config
        self.data_handler = data_handler
        self.benchmark: BaseBenchmark
        self._get_benchmark()
        self.data_providers: List[BaseData] = []
        self._get_data_providers()
        postprocessor: Optional[Callable] = BENCHMARK_PROCESSORS[config.postprocessor.type]
        assert postprocessor is not None
        self.postprocessor = postprocessor

    def _get_data_providers(self):
        if self.config.data_config:
            if isinstance(self.config.data_config, list):
                for data_config in self.config.data_config:
                    new_data_provider = registry.get("data").get_logic_cls(data_config.type)
                    data_context = DataContext(self.model, data_config)
                    self.data_providers.append(new_data_provider(data_context))
            else:
                new_data_provider = registry.get("data").get_logic_cls(self.config.data_config.type)
                data_context = DataContext(self.model, self.config.data_config)
                self.data_providers.append(new_data_provider(data_context))

    def _get_benchmark(self):
        cfg = self.config
        self.benchmark_cls = registry.get("benchmark").get_logic_cls(cfg.type)

    def _get_metrics(self):
        cfg = self.config
        metrics = []
        for metric_cfg in cfg.metric_configs:
            metric_cls = registry.get("metric").get_logic_cls(metric_cfg.type)
            metrics.append(metric_cls(metric_cfg))

    def _log_intermediate_results(self, results: List[BaseResult]):
        self.json_results = get_json_results(results)
        self.results = results

    def benchmark_generator(self, data_iterator: Iterator[BaseData]):
        data_iterator_empty = True
        for data_provider in data_iterator:
            data_iterator_empty = False
            benchmark_context = BenchmarkContext(
                self.model,
                self.benchmark_cls,
                self.config,
                data_provider=data_provider,
                data_handler=self.data_handler,
            )
            benchmark = self.benchmark_cls(benchmark_context)
            yield benchmark.eval_benchmark(self.model)
        if data_iterator_empty:
            benchmark_context = BenchmarkContext(
                self.model,
                self.benchmark_cls,
                self.config,
                data_handler=self.data_handler,
            )
            benchmark = self.benchmark_cls(benchmark_context)
            yield benchmark.eval_benchmark(self.model)

    def data_generator(self):
        for data_provider in self.data_providers:
            yield data_provider

    def postprocessing(self, results: dict) -> FinalResult:
        return FinalResult(**self.postprocessor(results))

    def to_final_result(self) -> FinalResult:
        data_generator = self.data_generator()
        benchmark_generator = self.benchmark_generator(data_generator)
        intermediate_results = list(benchmark_generator)
        json_results = get_json_results(intermediate_results)
        final_result = self.postprocessing({"benchmark_specific_info": json_results})
        return final_result

class Runner:
    def __init__(self, model: BaseModel, config: Config, result_handler: BaseConnector):
        self.config: Config = config
        self.result_handler = result_handler
        self.model = ProxyModel(model, result_handler)
        self.internal_repr = InternalBenchmarkRepresentation(
            self.model, self.config.benchmark_configs[0], data_handler=self.result_handler
        )

    def run(self):
        start = time.time()
        try:
            final_result = self.internal_repr.to_final_result()
        except Exception as e:
            self.result_handler.log_error(e)
            raise e
        runtime = time.time() - start
        self.result_handler.add_final_result(final_result, runtime=runtime)