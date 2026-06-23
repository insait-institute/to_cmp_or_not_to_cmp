#    Copyright 2025 Federico Marcuzzi, INSAIT
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


from src.models.base.base_model import BaseModel
import numpy as np
import random
import string

class DummyModel(BaseModel):
    def __init__(self, config):
        self._config = config
        super().__init__(config)

    def perplexities(self, inputs, *args, **kwargs):
        len_inputs = len(inputs)
        return list(np.random.rand(len_inputs) * 100)

    def loglikelihood(self, inputs, *args, **kwargs):
        len_inputs = len(inputs)
        return list(np.random.rand(len_inputs))

    def generate(self, inputs, n=1, *args, **kwargs):
        shape = np.array(inputs).shape
        res = np.array(random.choices(["A", "B", "C", "D", "E", "F"], k=shape[0] * shape[1] * n))
        return list(res.reshape(-1, shape[0], n))

    def generate_chat(self, chats, add_generation_prompt=False, continue_final_message=True, *args, **kwargs):
        return self.generate(self, chats, n=1)
    
    def get_num_gen_tokens(self, inputs):
        len_inputs = len(inputs)
        return list(np.random.randint(0, 1000, size=len_inputs))

    def most_prob_options(self, prompts, anwers, get_soft_max=True, *args, **kwargs):
        shape = np.array(prompts).shape
        return [{"soft_max" : {k : list(np.random.rand(shape[0])) for k in anwers.keys()}}] * shape[1]
    
    def most_prob_options_chat(self, prompts, anwers, get_soft_max=True, *args, **kwargs):
        return self.most_prob_options(self, prompts, anwers)
    
    def reset_statistic(self):
        pass

    def get_statistic(self):
        return {"num_pmt_toks" : 0, "num_gen_toks" : 0, "num_tot_prmt" : 0, "num_tot_gens" : 0}