#    Copyright 2026 Federico Marcuzzi, INSAIT
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


from src.models.base.utils_vllm import *


import torch
import numpy as np

from .base_model import BaseModel
from src.configs.base_model_config import DEVICE, ModelProvider



from transformers import AutoTokenizer
import vllm
from vllm import LLM, SamplingParams


class VLLMCausalLM(BaseModel):
    AUTO_TOKENIZER_CLASS = AutoTokenizer
    AUTO_MODEL_CLASS = LLM

    def __init__(self, config):
        print("[INFO] vllm version:", vllm.__version__)
        super().__init__(config)

        self._batch_size = config.batch_size
        self._max_gen_toks = config.max_gen_toks
        self._quantized = config.quantized
        self._device = ("cuda" if config.device in [DEVICE.AUTO, DEVICE.CUDA] and torch.cuda.is_available() else config.device.value)
        self._generation_args = config.generation_args
        self._dtype = config.dtype
        self._provider = config.provider
        self._seed = config.seed
        self.model_path = config.name
        self.tokenizer_path = config.tokenizer_name

        assert (self._provider == ModelProvider.VLLM)

        self.kvc_allowed_quant = ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
        print(f"[INFO] Model path: {self.model_path}")

        if self._device == "cuda":
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = 1
        print(f"[INFO] Using {self.num_gpus} GPUs for VLLM model: {self.model_path}")

        self._max_length = 4000 # TODO to param
        self.model = self._load_model()

        self.tokenizer = self._load_tokenizer()
        self.tokenizer.model_max_length = self._max_length

        self.sampling_params = self._set_sampling_params(config.generation_args)
        self.num_pmt_toks, self.num_gen_toks, self.num_tot_prmt, self.num_tot_gens = 0, 0, 0, 0

    def loglikelihood(self, prompts):
        return self._loglikelihood(prompts)[0]
    
    def perplexities(self, prompts):
        return self._loglikelihood(prompts)[1]
    
    def most_prob_options(self, input, anwers, get_soft_max=True): # TODO more then 1 prompt per list
        input, shape = to_flat(input)

        if VLLM_VERSION < Version("0.15"):
            get_logits = VLLMLogitsRetriever(self.tokenizer, anwers)
            temp_sampling_params = self._get_temp_updated_sampling_params({"max_tokens" : 1, "detokenize" : True, "logits_processors" : [get_logits]})
            outputs = self.model.generate(input, sampling_params=temp_sampling_params)
        else:
            get_logits = VLLMLogitsRetrieverNew(self.tokenizer, anwers, input)
            list_sp = get_logits.get_sample_params_list(self._generation_args, {"max_tokens" : 1, "detokenize" : True})
            outputs = self.model.generate(input, sampling_params=list_sp)
            get_logits.compute_data()        
    
        for prompts in outputs:
            # statistics:
            self.num_pmt_toks += len(prompts.prompt_token_ids)
            self.num_gen_toks += 1 # here the model generates only one token per prompt.
            self.num_tot_prmt += 1
            self.num_tot_gens += 1

        if get_soft_max:
            outputs = get_logits.get_soft_max()
        else:
            outputs = get_logits.get_all()

        return de_flat_dict(outputs, shape[0])

    def most_prob_options_chat(self, chats, anwers, get_soft_max=True, add_generation_prompt=True, continue_final_message=False, *args, **kwargs):
        tokenized_chats = self.tokenizer.apply_chat_template(chats, tokenize=True, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message)
        tokenized_chats = [{"prompt_token_ids": list(t)} for t in tokenized_chats]
        return self.most_prob_options(tokenized_chats, anwers, get_soft_max, *args, **kwargs)
    
    def generate(self, input, n=1, max_tokens=None):
        input, shape = to_flat(input, n)

        max_tokens = max_tokens if max_tokens is not None and max_tokens > 0 else self._max_gen_toks
        temp_sampling_params = self._get_temp_updated_sampling_params({"max_tokens" : max_tokens})
        outputs = self.model.generate(input, sampling_params=temp_sampling_params)

        list_output = []
        self.num_tot_prmt += len(outputs)
        for prompts in outputs:
            self.num_pmt_toks += len(prompts.prompt_token_ids)
            self.num_tot_gens += len(prompts.outputs)
            list_gen = []
            for gen in prompts.outputs:
                self.num_gen_toks += len(gen.token_ids)
                list_gen.append(gen.text)
            
            list_output.append(list_gen)

        if len(list_output[0]) == 1:
            list_output = [gen[0] for gen in list_output]

        return de_flat_mat(list_output, n, shape)

    
    def generate_chat(self, chats, add_generation_prompt=False, continue_final_message=True, *args, **kwargs):
        tokenized_chats = self.tokenizer.apply_chat_template(chats, tokenize=True, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message)
        tokenized_chats = [{"prompt_token_ids": list(t)} for t in tokenized_chats]
        return self.generate(tokenized_chats, *args, **kwargs)
    
    def reset_statistic(self):
        self.num_pmt_toks, self.num_gen_toks, self.num_tot_prmt, self.num_tot_gens = 0, 0, 0, 0
    
    def get_statistic(self):
        return {"num_pmt_toks" : int(self.num_pmt_toks), "num_gen_toks" : int(self.num_gen_toks), "num_tot_prmt" : int(self.num_tot_prmt), "num_tot_gens" : int(self.num_tot_gens)}
    
    def get_num_gen_tokens(self, texts):
        encodings = self.tokenizer.batch_encode_plus(texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        return [len(ids) for ids in encodings['input_ids']]

    def _load_model(self):
        tokenizer_path = self.tokenizer_path if self.tokenizer_path else self.model_path
        kvargs = {}
        
        if self._quantized in self.kvc_allowed_quant:
            '''
            https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache.html
            The kv_cache_dtype argument specifies the data type for KV cache storage:
            - "auto": Uses the model’s default “unquantized” data type
            - "fp8" or "fp8_e4m3": Supported on CUDA 11.8+ and ROCm (AMD GPU)
            - "fp8_e5m2": Supported on CUDA 11.8+
            '''

            print(f"Using model with quantized kv_cache_dtype: {self._quantized}")
            kvargs = {"kv_cache_dtype": self._quantized, "calculate_kv_scales" : True}
    
        if VLLM_VERSION < Version("0.15"):
            return self.AUTO_MODEL_CLASS(model=self.model_path, tokenizer=tokenizer_path, max_num_seqs=self._batch_size, tensor_parallel_size=self.num_gpus, gpu_memory_utilization=0.8,
                                         seed=self._seed, enforce_eager=False, dtype=self._dtype, device=self._device, **kvargs)
        else:
            return self.AUTO_MODEL_CLASS(model=self.model_path, tokenizer=tokenizer_path, max_num_seqs=self._batch_size, tensor_parallel_size=self.num_gpus, gpu_memory_utilization=0.8, 
                                         seed=self._seed, enforce_eager=False, dtype=self._dtype, logits_processors = [WrappedPerReqLogitsRetriever],
                                         max_model_len=self._max_length, **kvargs)

    def _load_tokenizer(self):
        if self.tokenizer_path:
            return self.AUTO_TOKENIZER_CLASS.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        return self.model.get_tokenizer()

    def _set_sampling_params(self, generation_args):
        if len(generation_args) == 0:
            return SamplingParams()
        else:
            return SamplingParams(**generation_args)
    
    def _loglikelihood(self, prompts):
        sp = SamplingParams(prompt_logprobs=0, temperature=1, max_tokens=1)
        with torch.no_grad():
            model_output = self.model.generate(prompts, sampling_params=sp)
        
        log_likelihoods = []
        perplexities = []
        for request in model_output:
            prompt = request.prompt_logprobs[1:]
            list_logprops = [next(iter(logprobs.values())).logprob for logprobs in prompt]
            sum_logprops = np.sum(list_logprops)
            log_likelihoods.append(sum_logprops)
            n_toks = len(list_logprops)
            perplexities.append(np.exp(-sum_logprops/n_toks))

            # statistics:
            self.num_pmt_toks += np.sum(n_toks)
            self.num_gen_toks += 1 # here the model generates only one token per prompt.
            self.num_tot_prmt += 1
            self.num_tot_gens += 1

        return log_likelihoods, perplexities

    def _get_temp_updated_sampling_params(self, params):
        new_params = self._generation_args.copy()
        new_params.update(params)
        return SamplingParams(**new_params)