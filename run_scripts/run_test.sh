#!/bin/bash

cd ..

python3 run.py --model_config model_config/dummy_model.yaml --model DummyModel --results_folder="runs" --bench mmlu
python3 run.py --model_config model_config/dummy_model.yaml --model DummyModel --results_folder="runs" --category gender --bench stereo_set
python3 run.py --model_config model_config/dummy_model.yaml --model DummyModel --results_folder="runs" --category gender --bench reddit_bias
python3 run.py --model_config model_config/dummy_model.yaml --model DummyModel --results_folder="runs" --category gender --bench crows_pairs
python3 run.py --model_config model_config/dummy_model.yaml --model DummyModel --results_folder="runs" --category gender --bench bbq
python3 run.py --model_config model_config/dummy_model.yaml --model DummyModel --results_folder="runs" --category gender --bench wino_bias
python3 run.py --model_config model_config/dummy_model.yaml --model DummyModel --results_folder="runs" --category gender --bench discrim_eval_gen
python3 run.py --model_config model_config/dummy_model.yaml --model DummyModel --results_folder="runs" --category gender --bench dt_toxic
python3 run.py --model_config model_config/dummy_model.yaml --model DummyModel --results_folder="runs" --category gender --bench toxic_ratings