#    Copyright 2024 SRI Lab @ ETH Zurich, LatticeFlow AI, INSAIT
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC
from typing import List
from src.benchmarks.benchmark_implementations.utils import prep_bech
from create_bench import BENCHMARK_REGISTRY


from src.benchmarks.benchmark_implementations.utils import common_run

class BaseBenchmark(ABC):
    """
    Base class for benchmarks.

    Each benchmark will be called in the following order:
    1) setup()
    2) run()
    3) teardown()
    """

    def __init__(self, benchmark_context):
        self.benchmark_context = benchmark_context
        self.config = benchmark_context.get_benchmark_config()

    def setup(self):
        pass

    def run(self, model):
        return common_run(model, self.benchmark_context)

    def teardown(self):
        pass

    def eval_benchmark(self, model) -> List:
        """
        Evaluate the benchmark for the given model.

        This method executes the benchmark by calling the setup, run, and teardown methods in order.
        It then computes the results using the metrics defined in the benchmark context.

        Args:
            model (BaseModel): The model to evaluate.

        Returns:
            List[BaseResult]: The computed results for each metric.
        """
        self.setup()

        benchmark_results = self.run(model)
        results: List = []

        self.teardown()

        metrics = self.benchmark_context.get_metrics()

        if not metrics or len(metrics) == 0:
            return [benchmark_results]

        for metric in metrics:
            result = metric.compute_result(benchmark_results)
            results.append(result)

        return results

class BaseData(ABC):

    def __init__(self, data_context):
        self.bench_data = prep_bech(BENCHMARK_REGISTRY[data_context.config.type])

    def get_data(self):
        return self.bench_data