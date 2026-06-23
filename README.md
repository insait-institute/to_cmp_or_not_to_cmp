# To Compare, or Not to Compare: On Methodological Practices in Evaluating Social Bias

> #### Federico Marcuzzi, Xuefei Ning, Roy Schwartz, and Iryna Gurevych
This repository includes the code and scripts to reproduce the experiments presented in the arXiv paper [To Compare, or Not to Compare: On Methodological Practices in Evaluating Social Bias](https://arxiv.org/abs/2508.18088) (paper [website](https://insait-institute.github.io/to_cmp_or_not_to_cmp/)).


## Abstract

<p align="center">
  <img src="docs/static/images/example_gh.svg" width="100%">
</p>

As Large Language Models are increasingly deployed in critical applications, robustly evaluating their social biases is paramount. 
However, the current literature suffers from widespread methodological fragmentation, which yields contradictory conclusions.
This stems largely from ignoring the structural framing of benchmark-level evaluations.
To resolve this, we introduce a unified and controllable framework that standardizes heterogeneous benchmarks to systematically contrast isolated demographic assessments with forced-choice comparative settings.
Crucially, this allows us to disentangle the confounding effects of Chain-of-Thought reasoning, neutral fallback options, and other structural artifacts in social bias evaluations.
Our evaluation across multiple model families reveals a massive, systematic paradigm gap: while isolated assessments limit prejudice activation, comparative settings act as aggressive catalysts for latent discrimination, a shift primarily driven by underspecified contexts.
Alarmingly, CoT reasoning exacerbates social biases under comparative settings, and this systemic bias persists as a deterministic prejudice even when models are provided neutral fallback options or claim to answer randomly. 
Finally, we demonstrate that this comparative prejudice is a generalized phenomenon that scales positively with model size.
Ultimately, we offer a crucial methodological guideline: while researchers must leverage comparative settings to robustly audit hidden biases, practitioners cannot safely rely on comparative deployments in ambiguous real-world tasks.

---

## Framework Overview

This repository provides a comprehensive pipeline to evaluate social bias in Large Language Models. The framework is logically structured into three main modules:

1. **Create Benchmark:** Code to recreate the exact datasets used in our paper or to generate new benchmarks using custom, alternative configurations. (See the [Create Benchmark](#create-benchmark) section).
2. **Model Evaluation:** A unified evaluation pipeline to test social bias on any model compatible with the Hugging Face library. (See the [Model Evaluation](#model-evaluation) section).
3. **Result Computation:** Tools to compute the evaluation metrics defined in the paper. (See the [Result Computation](#result-computation) section).

The **[Setup](#setup)** section immediately below provides instructions on how to configure the environment and verify that everything is working correctly.

---

## Setup

**Code:** Clone the repository:

```bash
git clone https://github.com/insait-institute/to_cmp_or_not_to_cmp.git
cd to_cmp_or_not_to_cmp
```

**Environments:** Create the Conda environment needed to run the Social Bias Evaluation Framework using the `environments.yaml` file:

```bash
conda env create -f environments.yaml -n bias_eval
```

**Benchmarks:** Download the necessary datasets to run the evaluation.

> **Note:** You need a Hugging Face token to download most of the datasets. Please ensure your token is set (e.g., via the `HF_TOKEN` environment variable) before proceeding:
> 
> ```bash
> conda activate bias_eval
> python helper_tools/download_datasets.py
> ```

> **Important Note for the `toxic_ratings` dataset:** This specific dataset must be downloaded manually via a separate link because it is provided as a closed copy. Please download it from [https://data.esrg.stanford.edu/study/toxicity-perspectives](https://data.esrg.stanford.edu/study/toxicity-perspectives) and place the files inside the original `benchmarks/` folder.

### Verify Installation

To test the installation of the Social Bias Evaluation Framework on a dummy model, run the following command:

```bash
conda activate bias_eval
cd run_scripts
bash run_test.sh
```

---

## Create Benchmark

To recreate the datasets used in our paper or to generate new benchmarks using alternative configurations, you need to use the `create_bench.py` script. The generated benchmark files will be saved in the `benchmark_exps/` directory.

The available base benchmarks are:
[**StereoSet**](https://github.com/moinnadeem/StereoSet/), [**RedditBias**](https://github.com/umanlp/RedditBias), [**BBQ**](https://github.com/nyu-mll/BBQ), [**DiscrimEvalGen**](https://github.com/aisoc-lab/inference-acceleration-bias), [**DecodingTrust-Toxicity**](https://github.com/AI-secure/DecodingTrust), [**CrowS-Pairs**](https://github.com/nyu-mll/crows-pairs), [**ToxicRatings**](https://data.esrg.stanford.edu/study/toxicity-perspectives), [**WinoBias**](https://github.com/uclanlp/corefBias), and [**MMLU**](https://huggingface.co/datasets/cais/mmlu).

> **Important Note:** Ensure you have followed the steps described in the [Setup](#setup) section before running these commands.

### Usage Examples

Here are two generic examples demonstrating how to generate datasets with different structural configurations:

**Example 1: Generating a benchmark with Chain-of-Thought and a neutral third option**
```bash
# This creates the BBQ ambiguous split benchmark for the 'gender' category.
# It enables Chain-of-Thought (CoT) prompting and adds a third neutral option to the multiple-choice question.
python create_bench.py --bench bbq --category gender --is_cot True --is_third True
```

**Example 2: Generating a subset with prompt variations**
```bash
# This creates the CrowS-Pairs benchmark for the 'race' category.
# It disables Chain-of-Thought (--is_cot False), applies 54 prompt variations (--is_prompt_var), 
# and restricts the creation to a subset of 500 samples for quick testing.
python create_bench.py --bench crows_pairs --category race --is_cot False --is_prompt_var --subset_size 500
```

---

### 🛠️ Benchmark Parameters

The table below describes the configurable parameters used to define and control dataset generation settings during the initialization phase. 

> **Note on Boolean Values:** All boolean flags accept various standard formats (e.g., `1/0`, `True/False`, `yes/no`, `y/n`, `t/f`).

| Parameter | Description | Accepted Values | Default | Notes & Constraints |
| :--- | :--- | :--- | :--- | :--- |
| <nobr>`--bench`</nobr> | Benchmark name. | `bbq`, `crows_pairs`, `discrim_eval`, `dt_toxic`, `mmlu`, `reddit_bias`, `stereo_set`, `toxic_ratings`, `wino_bias` | — | Required during initialization (`init_bench=True`). |
| <nobr>`--category`</nobr> | Demographic category. | *String* (e.g., `"gender"`, `"race"`) | `None` | Depends on the specific benchmark. |
| <nobr>`--is_cot`</nobr> | Use chain-of-thought prompting. | *Boolean* | `True` | Triggers a warning if `--gen_length` is < 1000. Automatically set to `False` if `--is_prob=True`. |
| <nobr>`--is_third`</nobr> | Allow a third neutral option. | *Boolean* | `False` | — |
| <nobr>`--is_prompt_var`</nobr> | Prompt model with 54 prompt variations. | *Boolean* | `False` | — |
| <nobr>`--is_prob`</nobr> | Evaluation based on option token probability distribution. | *Boolean* | `False` | **Note:** If `True`, automatically forces `--is_cot=False`, `--gen_length=None`, and `--n_gens=None`. |
| <nobr>`--is_rand`</nobr> | Enforce random answer. | *Boolean* | `False` | — |
| <nobr>`--is_chat`</nobr> | Use chat templates. | *Boolean* | `False` | — |
| <nobr>`--gen_length`</nobr> | Maximum length of generated text. | *Integer* | `2000` | Automatically disabled (set to `None`) if `--is_prob=True`. |
| <nobr>`--n_gens`</nobr> | Number of generations for each prompt. | *Integer* | `25` | Automatically disabled (set to `None`) if `--is_prob=True`. |
| <nobr>`--is_disamb`</nobr> | Use disambiguated split of BBQ benchmark. | *Boolean* | `False` | — |
| <nobr>`--subset_size`</nobr> | Maximum number of samples to evaluate from the benchmark. | *Integer* | `None` | Useful for testing or debugging on a subset of the dataset. |

---

## Model Evaluation

The model evaluation phase executes inference on a chosen model. This script accepts specific execution arguments **in addition to all structural dataset configuration parameters** defined previously in the [Benchmark Parameters](#️-benchmark-parameters) section (e.g., `--bench`, `--category`, `--is_cot`, etc.). This configuration symmetry ensures that the evaluation pipeline can accurately parse and reference target files on-the-fly.

> **Execution Note:** The parameters listed below represent execution and runtime setups configurable directly via the command line interface. These parameters override any additional model-specific hyperparameters, tokenization configurations, or architecture-level settings loaded from the file path provided to the `--model_config` flag.

### Evaluation Execution Parameters

| Parameter | Description | Accepted Values | Default | Notes & Constraints |
| :--- | :--- | :--- | :--- | :--- |
| <nobr>`--results_folder`</nobr> | Target execution folder path where execution logs are stored. | *String* (path) | `runs` | — |
| <nobr>`--model`</nobr> | Name or HuggingFace path identifier of the target model. | *String* | — | Required parameter. |
| <nobr>`--model_config`</nobr> | Configuration file to pre-load for the model setup. | *String* (path) | — | Applied directly before loading the `--model` argument. |
| <nobr>`--batch_size`</nobr> | Evaluation batch size sizing. | *Integer* | — | Adjust based on available hardware memory limits. |
| <nobr>`--seed`</nobr> | Synchronization random seed for reproducibility tracking. | *Integer* | `42` | Controls general execution initialization environment seeds. |

```bash
# Example command: Evaluating the Llama-3-8B model on the MMLU benchmark without Chain-of-Thought and with a batch size of 16
python evaluate.py --bench mmlu --is_cot False --model "meta-llama/Llama-3-8B" --batch_size 16
```

### Evaluation Output Structure

By default, the results of the framework's evaluation phase are automatically saved into the `runs/` directory. The outputs follow a strict hierarchical structure based on the target benchmark, the specific experiment configuration, and the execution timestamp:

```text
runs/<BENCH>/<EXP_CONF>/<TIMESTAMP>/
```

**Example Output Path:** `runs/mmlu/_no-cot_prob/2026-06-21_15:23:27`

Here, the `<EXP_CONF>` directory name (e.g., `_no-cot_prob`) is procedurally generated from the structural parameters defined during the [Create Benchmark](#create-benchmark) step. 

Inside the `<TIMESTAMP>` folder, the framework generates two primary files:

1. **`config.json`**: This file records the model initialization parameters, inference settings, and framework configuration used for that specific execution.
2. **`final_results.json`**: This file contains the raw results of the evaluation: raw text generations or token probability distributions.

Every `final_results.json` contains a metadata field named `"bench_filename"` (e.g., `"bench_filename": "mmlu_no-cot_prob"`). This string strictly matches the filename of the corresponding benchmark file stored in the `benchmark_exps/` directory used for evaluating the model. Whether this benchmark file was generated beforehand or constructed on-the-fly during execution, it contains all the necessary ground-truth tags and structural metadata required to parse these raw outputs into final bias metrics, as explained in the [Result Computation](#result-computation) section.

---

## Result Computation

The notebook `eval_results.ipynb` show how to compute the social bias metrics defined in the paper. If multiple versions of the same experiment configuration exist for a model within the output directory, the script automatically processes the most recent run that completed successfully.

---

## Citation
```bibtex
```