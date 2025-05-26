# Evaluation Scripts Summary

This document provides an overview of three evaluation scripts used to assess the performance in generating suggestions. Each script's inputs and evaluation metrics are described below.

## `eval_stage2.py`
This script performs evaluation for stage 2 of our method. Its target is to evaluate the different LLM ability to select the suggestion Type.
### Input:
- Load our fusion model, used to select suggestion keys.
- Dataset including aligned sensor and pose data.
- Names of various Large Language Model (LLM) APIs used to select suggestion type and generate suggestion content.

### Evaluation Metrics:
- Ability of suggestion type selection
  - precision-S1L(correct stage 1 Loss).
  - recall and F1 scores.
- precision of (key,type) selection (Overall precision of Stage 1 + Stage 2)
- Ability of suggestion content generation
  - ROUGE-L. (Measures the overlap between the LLM-generated text and coach suggestion text. All suggestions for each round are calculated after union.)

## `eval_textonly.py`
This script performs evaluation for text LLM as our baseline. Its target is to evaluate the different LLM ability to select the suggestion key, select the suggestion (key,type), generating content.
### Input:
- Dataset including aligned sensor and pose data.
- Various Large Language Model (LLM) APIs for generating suggestions.

### Evaluation Metrics:
- Ability of suggestion key selection
  - precision
  - recall and F1 scores.
- Ability of suggestion type selection
  - precision-S1L(correct stage 1 Loss)
- Ability of suggestion (key, type) selection
  - precision
- Ability of suggestion content generation
  - ROUGE-L.

## `eval_vllm.py`
This script performs evaluation for video LLM of our baseline. Its target is to evaluate the different vLLM ability to select the suggestion key, select the suggestion (key,type), generating content.
### Input:
- Dataset including aligned sensor and raw video data.
- Various Video Large Language Model (VLLM) APIs for reading videos and generating suggestions.

### Evaluation Metrics:
The same as `eval_textonly.py`

## Usage
Please make sure to set the `FLAG_SENSECOACH=False` in `config.py` when using `stage2.py`. And set `FLAG_SENSECOACH=True` when using baseline evaluation scripts (`eval_textonly.py` and `eval_vllm.py`).

## Other files
- `config.py`: Configuration file for the evaluation scripts.
- `key_dim.py` and `new_key_mapping`: File for predefined key dimensions.
- `load_model.py`: File for loading the model.
- `data_loader_eval.py`: File for loading the dataset.